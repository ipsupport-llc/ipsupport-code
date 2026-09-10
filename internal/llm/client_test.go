package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

func TestChatToolCall(t *testing.T) {
	const resp = `{"choices":[{"message":{"role":"assistant","content":"",
		"tool_calls":[{"id":"call_1","type":"function","function":{
		"name":"calc","arguments":"{\"action\":\"calculate\",\"params\":{\"expression\":\"2+2\"}}"}}]}}]}`

	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		io.WriteString(w, resp)
	}))
	defer srv.Close()

	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "test"})
	msg, err := c.Chat(context.Background(), []Message{User("hi")}, []map[string]any{{"type": "function"}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "calc" {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	if !strings.Contains(msg.ToolCalls[0].Arguments, "expression") {
		t.Errorf("arguments = %q", msg.ToolCalls[0].Arguments)
	}
	if gotBody["model"] != "test" {
		t.Errorf("request model = %v, want test", gotBody["model"])
	}
	if _, ok := gotBody["tools"]; !ok {
		t.Error("request did not include tools")
	}
}

func TestChatContent(t *testing.T) {
	const resp = `{"choices":[{"message":{"role":"assistant","content":"the answer is 4"}}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, resp)
	}))
	defer srv.Close()

	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "test"})
	msg, err := c.Chat(context.Background(), []Message{User("2+2?")}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if msg.Content != "the answer is 4" || len(msg.ToolCalls) != 0 {
		t.Errorf("msg = %+v", msg)
	}
}

func sseServer(t *testing.T, chunks ...string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestChatStreamingContent(t *testing.T) {
	url := sseServer(t,
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"content":" world"}}]}`,
		`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`,
	)
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "Hello world" {
		t.Errorf("content = %q, want 'Hello world'", msg.Content)
	}
	if p, c := cl.Usage(); p != 7 || c != 2 {
		t.Errorf("usage = %d,%d want 7,2 (estimate reconciled to real)", p, c)
	}
}

func TestChatRunawayCapped(t *testing.T) {
	var chunks []string
	for i := 0; i < 50; i++ {
		chunks = append(chunks, `{"choices":[{"delta":{"content":"x"}}]}`)
	}
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t, chunks...), Model: "fake"})
	cl.maxRespTk = 10 // small cap so the 50-token stream trips the runaway guard
	if _, err := cl.Chat(context.Background(), []Message{User("go")}, nil); err == nil || !strings.Contains(err.Error(), "looping") {
		t.Errorf("expected a runaway abort, got %v", err)
	}
}

// A model stuck repeating a single character produces obviously worthless
// output well before the general token-count runaway cap ever fires —
// observed live: 14m52s and 11.2k tokens into a loop that was still short of
// a 32k-token cap. The degenerate-repetition detector catches this in a
// fraction of a second instead.
func TestChatAbortsOnDegenerateRepetition(t *testing.T) {
	var chunks []string
	for i := 0; i < degenerateRunThreshold+10; i++ {
		chunks = append(chunks, `{"choices":[{"delta":{"content":"0"}}]}`)
	}
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t, chunks...), Model: "fake"})
	_, err := cl.Chat(context.Background(), []Message{User("go")}, nil)
	if err == nil || !strings.Contains(err.Error(), "looping") {
		t.Errorf("expected a degenerate-repetition abort, got %v", err)
	}
	var de *degenerateOutputError
	if !errors.As(err, &de) {
		t.Fatalf("error = %v, want a *degenerateOutputError", err)
	}
	if de.r != '0' {
		t.Errorf("detected rune = %q, want '0'", de.r)
	}
}

// Legitimate content that happens to repeat a character — a dashed separator,
// repeated braces/indentation in code — must NOT trip the detector as long as
// it stays under the threshold and isn't the ENTIRE output.
func TestChatDoesNotFlagOrdinaryRepeatedCharacters(t *testing.T) {
	dashes := strings.Repeat("-", degenerateRunThreshold-1)
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t,
		`{"choices":[{"delta":{"content":"a table separator: "}}]}`,
		fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, dashes),
		`{"choices":[{"delta":{"content":" done"}}]}`,
	), Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("go")}, nil)
	if err != nil {
		t.Fatalf("ordinary repeated characters should not abort: %v", err)
	}
	if !strings.HasSuffix(msg.Content, " done") {
		t.Errorf("content = %q, want it to finish normally", msg.Content)
	}
}

func TestReasoningEffortSent(t *testing.T) {
	seen := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen = strings.Contains(string(b), `"reasoning_effort":"low"`)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()
	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m",
		Extra: map[string]any{"reasoning_effort": "low"}})
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if !seen {
		t.Error("extra reasoning param was not merged into the request")
	}
}

func TestContextTracksLastPrompt(t *testing.T) {
	url := sseServer(t,
		`{"choices":[{"delta":{"content":"hi"}}]}`,
		`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":4061,"completion_tokens":1}}`,
	)
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if cl.Context() != 4061 {
		t.Errorf("Context() = %d, want 4061 (last prompt size)", cl.Context())
	}
}

func TestChatStreamingToolCall(t *testing.T) {
	url := sseServer(t,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calc","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"action\":\"calculate\"}"}}]}}]}`,
	)
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, []map[string]any{{"type": "function"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "calc" {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	if !strings.Contains(msg.ToolCalls[0].Arguments, "calculate") {
		t.Errorf("arguments accumulated wrong: %q", msg.ToolCalls[0].Arguments)
	}
}

func TestChatRetriesOnServerError(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n < 2 { // first attempt 500s, like a transient LM Studio hiccup
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, "<html>Internal Server Error</html>")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"recovered"}}]}`)
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	var notified int
	cl.OnRetry = func(_ int, _ time.Duration, _ string) { notified++ }

	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat should have retried past the 500: %v", err)
	}
	if msg.Content != "recovered" || n < 2 {
		t.Errorf("content=%q attempts=%d, want recovered after a retry", msg.Content, n)
	}
	if notified == 0 {
		t.Error("OnRetry was not called — the UI wouldn't show the backoff")
	}
}

// A connection dropped MID-STREAM (not an idle stall) is a transient network
// error — the client must retry it, not abort the whole task.
func TestChatRetriesOnMidStreamDrop(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n < 2 { // first attempt: start streaming, then drop the connection mid-body
			conn, bufrw, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			bufrw.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
			data := "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n"
			fmt.Fprintf(bufrw, "%x\r\n%s\r\n", len(data), data) // one chunk, then NO terminating 0-chunk
			bufrw.Flush()
			conn.Close() // abrupt drop → client sees an unexpected EOF mid-stream
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	var notified int
	cl.OnRetry = func(_ int, _ time.Duration, _ string) { notified++ }

	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat should retry past a mid-stream drop: %v", err)
	}
	if msg.Content != "recovered" || notified == 0 {
		t.Errorf("content=%q notified=%d, want recovered after a retry", msg.Content, notified)
	}
}

// A stream that closes CLEANLY (a normal EOF, not a transport error) before
// "[DONE]" ever arrived is a truncated response, not a completed one — a
// proxy or the server itself cutting the connection mid-generation. Without
// treating this as retriable, whatever partial content had accumulated so
// far was silently accepted as a full, successful answer.
func TestChatRetriesOnPrematureCleanEOF(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n++
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		if n < 2 {
			// A normal (non-hijacked) handler return closes the body cleanly —
			// no [DONE], no transport error, just EOF.
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
			fl.Flush()
			return
		}
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"}}]}\n\n")
		fl.Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat should retry past a premature clean EOF: %v", err)
	}
	if msg.Content != "recovered" {
		t.Errorf("content = %q, want recovered (the truncated first attempt must be discarded, not accepted)", msg.Content)
	}
	if n < 2 {
		t.Errorf("attempts = %d, want at least 2 (a retry)", n)
	}
}

// The idle watchdog must be reset ONLY by real token progress — SSE heartbeats,
// comments, and empty deltas (which proxies emit) must not, or a stalled stream
// that keeps heartbeating would "think" forever without producing a token.
func TestParseStreamTicksOnlyOnProgress(t *testing.T) {
	cl := NewOpenAIClient(config.LLM{Model: "x"})
	ticks := 0
	tick := func() { ticks++ }

	// A comment/heartbeat, a blank data line, and an empty delta — no progress.
	heartbeats := ": ping\n\ndata: \n\ndata: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl.parseStream(strings.NewReader(heartbeats), tick, 0); err != nil {
		t.Fatal(err)
	}
	if ticks != 0 {
		t.Errorf("heartbeats/empty deltas reset the watchdog %d times, want 0", ticks)
	}

	// A real content delta does reset it.
	ticks = 0
	real := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl.parseStream(strings.NewReader(real), tick, 0); err != nil {
		t.Fatal(err)
	}
	if ticks != 1 {
		t.Errorf("a real content delta reset the watchdog %d times, want 1", ticks)
	}
}

// Live() lets the TUI show what the model is currently doing while it's still
// generating — buffered as it streams in, never sent back to the model. Both
// the OpenAI-style "reasoning_content" key and OpenRouter's "reasoning" key
// must accumulate, AND plain content deltas must too: a model with no
// separate reasoning phase at all (most local models — gemma reported live:
// token counter climbing but the panel empty) has nothing else to show.
func TestParseStreamAccumulatesLiveOutput(t *testing.T) {
	cl := NewOpenAIClient(config.LLM{Model: "x"})
	sse := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me \"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"the answer\"}}]}\n\n" +
		"data: [DONE]\n\n"
	msg, err := cl.parseStream(strings.NewReader(sse), func() {}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := cl.Live(); got != "let me thinkthe answer" {
		t.Errorf("Live() = %q, want %q (reasoning then content, in stream order)", got, "let me thinkthe answer")
	}
	if msg.Content != "the answer" {
		t.Errorf("Message.Content = %q, want just the answer (reasoning text must not leak in)", msg.Content)
	}

	// OpenRouter's alternate reasoning key.
	cl2 := NewOpenAIClient(config.LLM{Model: "x"})
	sse2 := "data: {\"choices\":[{\"delta\":{\"reasoning\":\"pondering\"}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl2.parseStream(strings.NewReader(sse2), func() {}, 0); err != nil {
		t.Fatal(err)
	}
	if got := cl2.Live(); got != "pondering" {
		t.Errorf("Live() (OpenRouter key) = %q, want %q", got, "pondering")
	}

	// A model with no reasoning phase at all: plain content deltas alone must
	// still populate Live() — this is the case that was reported empty live.
	cl3 := NewOpenAIClient(config.LLM{Model: "x"})
	sse3 := "data: {\"choices\":[{\"delta\":{\"content\":\"plain answer\"}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl3.parseStream(strings.NewReader(sse3), func() {}, 0); err != nil {
		t.Fatal(err)
	}
	if got := cl3.Live(); got != "plain answer" {
		t.Errorf("Live() (no reasoning phase) = %q, want %q", got, "plain answer")
	}
}

// A retry (or a fresh Chat call) must not mix a dropped attempt's live output
// into the next one's — the live view would show a confusing blend of two
// unrelated turns otherwise.
func TestLiveResetsOnEachSend(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n++
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		if n == 1 {
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"first call thinking\"}}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"first answer\"}}]}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"second call thinking\"}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"second answer\"}}]}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if got := cl.Live(); got != "first call thinkingfirst answer" {
		t.Errorf("after call 1, Live() = %q, want %q", got, "first call thinkingfirst answer")
	}
	if _, err := cl.Chat(context.Background(), []Message{User("hi again")}, nil); err != nil {
		t.Fatal(err)
	}
	if got := cl.Live(); got != "second call thinkingsecond answer" {
		t.Errorf("after call 2, Live() = %q, want %q (must not carry over call 1's text)", got, "second call thinkingsecond answer")
	}
}

// stripChannelTokens must remove leaked Harmony-style control tokens — both
// well-formed and the leading-pipe-dropped form a quantized local model was
// observed to emit — without touching ordinary prose.
func TestStripChannelTokens(t *testing.T) {
	cases := []struct{ in, want string }{
		{"hello world", "hello world"},                                       // no tokens — untouched
		{"<channel|>I'll help you build this.", "I'll help you build this."}, // observed leak (dropped leading pipe)
		// Well-formed pair: both bracketed tokens are stripped. The bare
		// channel name ("final") sitting between them is plain text, not a
		// bracketed token, and isn't touched — a known, accepted gap, since
		// there's no safe way to tell it apart from ordinary prose.
		{"<|channel|>final<|message|>the answer is 4", "finalthe answer is 4"},
		{"a |> b", "a |> b"}, // "|>" alone, no known token name — untouched
	}
	for _, c := range cases {
		if got := stripChannelTokens(c.in); got != c.want {
			t.Errorf("stripChannelTokens(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// A quantized local model was observed emitting a leaked, malformed channel
// tag directly in front of its real answer. The client must not show it.
func TestParseStreamStripsLeakedChannelToken(t *testing.T) {
	cl := NewOpenAIClient(config.LLM{Model: "x"})
	sse := `data: {"choices":[{"delta":{"content":"<channel|>I'll help you build this."}}]}` + "\n\ndata: [DONE]\n\n"
	msg, err := cl.parseStream(strings.NewReader(sse), func() {}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content != "I'll help you build this." {
		t.Errorf("content = %q, want the leaked tag stripped", msg.Content)
	}
}

// Some gateways (an Anthropic→OpenAI converter passing tool_use.input through)
// emit tool-call "arguments" as a raw JSON OBJECT instead of the spec's string.
// That must not drop the arguments — the raw JSON text must reach the tool call.
func TestToolCallObjectArgumentsAccepted(t *testing.T) {
	// Streaming: name in one chunk, OBJECT arguments in another.
	url := sseServer(t,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"file","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":{"action":"list","params":{"path":"."}}}}]}}]}`,
	)
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, []map[string]any{{"type": "function"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "file" {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	if !strings.Contains(msg.ToolCalls[0].Arguments, `"action":"list"`) {
		t.Errorf("object arguments lost: %q", msg.ToolCalls[0].Arguments)
	}

	// Non-streaming JSON response with object arguments.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"",
			"tool_calls":[{"id":"c2","type":"function","function":{"name":"file","arguments":{"action":"read","params":{"path":"x.go"}}}}]}}]}`)
	}))
	defer srv.Close()
	cl2 := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	msg2, err := cl2.Chat(context.Background(), []Message{User("hi")}, []map[string]any{{"type": "function"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg2.ToolCalls) != 1 || !strings.Contains(msg2.ToolCalls[0].Arguments, `"action":"read"`) {
		t.Errorf("non-stream object arguments lost: %+v", msg2.ToolCalls)
	}
}

func TestBackoffGrows(t *testing.T) {
	if backoff(1) != 500*time.Millisecond || backoff(2) != time.Second || backoff(3) != 2*time.Second {
		t.Errorf("backoff = %s,%s,%s want 500ms,1s,2s", backoff(1), backoff(2), backoff(3))
	}
	if backoff(10) != 8*time.Second {
		t.Errorf("backoff cap = %s, want 8s", backoff(10))
	}
}

func TestToolCallArgsSanitized(t *testing.T) {
	const resp = `{"choices":[{"message":{"role":"assistant","content":"",
		"tool_calls":[{"id":"c1","type":"function","function":{"name":"file","arguments":""}}]}}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, resp)
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, []map[string]any{{"type": "function"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Arguments != "{}" {
		t.Errorf("empty args = %q, want {} after sanitizing", msg.ToolCalls[0].Arguments)
	}
}

func TestIdleWatchdogCancelsOnStall(t *testing.T) {
	c := NewOpenAIClient(config.LLM{})
	c.idle = 30 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c.startIdleWatchdog(ctx, cancel) // no ticks → should cancel ctx

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("watchdog did not cancel on a silent stream")
	}
}

func TestIdleWatchdogStaysAliveWithTicks(t *testing.T) {
	c := NewOpenAIClient(config.LLM{})
	c.idle = 60 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tick := c.startIdleWatchdog(ctx, cancel)
	for i := 0; i < 5; i++ {
		time.Sleep(20 * time.Millisecond)
		tick()
	}
	if ctx.Err() != nil {
		t.Error("watchdog cancelled despite regular ticks")
	}
}

// A stalled attempt's live per-delta token estimate must be rolled back before the
// retry re-counts it — otherwise the failed attempt's tokens are double-counted.
func TestChatRollsBackTokensOnStallRetry(t *testing.T) {
	var mu sync.Mutex
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		first := hits == 1
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		if fl != nil {
			fl.Flush()
		}
		if first { // stream 3 deltas (bumps the estimate), then go silent → stall
			for i := 0; i < 3; i++ {
				io.WriteString(w, `data: {"choices":[{"delta":{"content":"x"}}]}`+"\n\n")
			}
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(300 * time.Millisecond)
			return
		}
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"done"}}]}`+"\n\n")
		io.WriteString(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	cl.idle = 40 * time.Millisecond
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatalf("Chat should recover after the stall: %v", err)
	}
	if _, compl := cl.Usage(); compl != 5 {
		t.Errorf("completion tokens = %d, want 5 (the stalled attempt's estimate must be rolled back, not added)", compl)
	}
}

// A per-model idle_timeout_seconds overrides the 90s default.
func TestIdleTimeoutConfigurable(t *testing.T) {
	if c := NewOpenAIClient(config.LLM{IdleTimeoutSeconds: 5}); c.idle != 5*time.Second {
		t.Errorf("idle = %v, want 5s from config", c.idle)
	}
	if c := NewOpenAIClient(config.LLM{}); c.idle != 90*time.Second {
		t.Errorf("idle = %v, want the 90s default", c.idle)
	}
}

// A stalled stream (headers, then silence) must fail fast and be retriable, not
// hang — the "no ping, waits forever" bug.
func TestChatStalledStreamRetriable(t *testing.T) {
	var mu sync.Mutex
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		first := hits == 1
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
		if first {
			time.Sleep(300 * time.Millisecond) // go silent → watchdog should fire
			return
		}
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	cl.idle = 40 * time.Millisecond
	var retried bool
	cl.OnRetry = func(_ int, _ time.Duration, reason string) {
		retried = true
		if !strings.Contains(reason, "stall") {
			t.Errorf("retry reason = %q, want a stall", reason)
		}
	}
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat should recover after the stall: %v", err)
	}
	if !retried || msg.Content != "ok" {
		t.Errorf("retried=%v content=%q, want recovery after a stall", retried, msg.Content)
	}
}

// compile-time guarantee the client satisfies the interface.
var _ Chatter = (*OpenAIClient)(nil)
