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
	"sync/atomic"
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

// temperature and top_p follow the same "only send when explicitly set (>0)"
// convention: a zero value means the server's own default, so it must not
// appear in the request body at all (some hosted models reject an explicit
// default temperature/top_p with a 400).
func TestChatSendsTemperatureAndTopPOnlyWhenSet(t *testing.T) {
	const resp = `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		io.WriteString(w, resp)
	}))
	defer srv.Close()

	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "test"})
	if _, err := c.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, ok := gotBody["temperature"]; ok {
		t.Errorf("unset temperature must be omitted, got %v", gotBody["temperature"])
	}
	if _, ok := gotBody["top_p"]; ok {
		t.Errorf("unset top_p must be omitted, got %v", gotBody["top_p"])
	}

	c2 := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "test", Temperature: 1.0, TopP: 0.95})
	if _, err := c2.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotBody["temperature"] != 1.0 {
		t.Errorf("temperature = %v, want 1.0", gotBody["temperature"])
	}
	if gotBody["top_p"] != 0.95 {
		t.Errorf("top_p = %v, want 0.95", gotBody["top_p"])
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

// A model stuck re-emitting the same SENTENCE (not a single character) is a
// different collapse than the one degenerateRunThreshold catches — observed
// live: the exact same ~90-byte sentence about a "hero's path" simulation
// repeating back to back, dozens of times, in a model's live output.
func TestChatAbortsOnPhraseRepetition(t *testing.T) {
	sentence := `The user wants to build a "hero's path" simulation in Go with visible learning/weights. `
	var chunks []string
	for i := 0; i < 6; i++ {
		chunks = append(chunks, fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, sentence))
	}
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t, chunks...), Model: "fake"})
	_, err := cl.Chat(context.Background(), []Message{User("go")}, nil)
	if err == nil || !strings.Contains(err.Error(), "looping") {
		t.Errorf("expected a phrase-repetition abort, got %v", err)
	}
	var pe *phraseRepeatError
	if !errors.As(err, &pe) {
		t.Fatalf("error = %v, want a *phraseRepeatError", err)
	}
}

// A realistic, short repeated code idiom (the same error-handling boilerplate
// appearing more than once in generated Go code) must NOT trip the detector —
// it's shorter than phraseRepeatMatchLen and/or spaced further apart than
// phraseRepeatWindow in real content, unlike a genuine back-to-back collapse.
func TestChatDoesNotFlagOrdinaryRepeatedCodeIdiom(t *testing.T) {
	idiom := "if err != nil {\n\treturn nil, err\n}\n"
	// Varied (not a single repeated character or a periodic pattern) filler,
	// long enough to push the idiom's 2nd copy outside the lookback window —
	// a homogeneous filler would itself risk tripping the OTHER detector.
	var fb strings.Builder
	for i := 0; fb.Len() < phraseRepeatWindow+50; i++ {
		fmt.Fprintf(&fb, "// unrelated comment line %d with some varying content\n", i)
	}
	filler := fb.String()
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t,
		fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, "func A() (int, error) {\n"+idiom),
		fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, filler),
		fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, "func B() (int, error) {\n"+idiom+"done"),
	), Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("go")}, nil)
	if err != nil {
		t.Fatalf("a short, widely-spaced repeated idiom should not abort: %v", err)
	}
	if !strings.HasSuffix(msg.Content, "done") {
		t.Errorf("content = %q, want it to finish normally", msg.Content)
	}
}

// A model's reasoning phase legitimately drafts its final sentence and then
// the content phase restates that EXACT sentence verbatim as the visible
// answer (a common CoT pattern: reasoning ends "So the answer is: '<sentence>'",
// then content emits "<sentence>"). Before the fix, checkPhraseRepeat ran on
// the single shared live buffer (reasoning+content concatenated), so the
// sentence's appearance in content matched its own earlier appearance in
// reasoning and falsely tripped phraseRepeatError — a legitimate channel
// handoff, not an actual loop. Reasoning and content must be compared only
// within their own channel.
func TestChatDoesNotFlagReasoningToContentHandoff(t *testing.T) {
	sentence := `The migration completes without downtime by draining connections before the cutover, then swapping the pool.`
	reasoning := "Let's work through the tradeoffs of each rollout strategy in detail before settling on one. " +
		"So the answer is: '" + sentence + "'"
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t,
		fmt.Sprintf(`{"choices":[{"delta":{"reasoning_content":%q}}]}`, reasoning),
		fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, sentence),
	), Model: "fake"})
	msg, err := cl.Chat(context.Background(), []Message{User("go")}, nil)
	if err != nil {
		t.Fatalf("reasoning->content handoff repeating the same sentence should not abort: %v", err)
	}
	if msg.Content != sentence {
		t.Errorf("content = %q, want %q", msg.Content, sentence)
	}
}

// DisableLoopDetection (per-connection, config.LLM) opts a provider out of
// both repetition detectors entirely — a capable hosted provider (Claude,
// OpenAI) that a user trusts not to need this can turn it off, while it stays
// on by default for everyone, including a local connection prone to it.
func TestDisableLoopDetectionOptsOutOfBothDetectors(t *testing.T) {
	var chunks []string
	for i := 0; i < degenerateRunThreshold+10; i++ {
		chunks = append(chunks, `{"choices":[{"delta":{"content":"0"}}]}`)
	}
	cl := NewOpenAIClient(config.LLM{BaseURL: sseServer(t, chunks...), Model: "fake", DisableLoopDetection: true})
	if _, err := cl.Chat(context.Background(), []Message{User("go")}, nil); err != nil {
		t.Errorf("character repetition should be ignored when disabled: %v", err)
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

// A server that never reports usage at all (some local runtimes, e.g.
// MLX-based ones, omit it entirely despite stream_options.include_usage)
// must not leave Context() stuck at 0 forever — that would silently disable
// auto-compact, since it would never see the context filling up. Chat falls
// back to a request-size estimate (~4 bytes/token) in that case.
func TestContextFallsBackToEstimateWhenServerReportsNoUsage(t *testing.T) {
	url := sseServer(t, `{"choices":[{"delta":{"content":"hi"}}]}`) // no "usage" field at all
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	longPrompt := strings.Repeat("this is a fairly long user message ", 50)
	if _, err := cl.Chat(context.Background(), []Message{User(longPrompt)}, nil); err != nil {
		t.Fatal(err)
	}
	if cl.Context() <= 0 {
		t.Errorf("Context() = %d, want a positive fallback estimate (not stuck at 0 forever)", cl.Context())
	}
}

// A backend that sends "usage":{} (present, all fields zero) as its final
// chunk before [DONE] must be treated the same as never sending usage at all:
// the promptEstimate fallback must still fire (lastPromptTk/Context() must
// not be disabled), and the zero completion count must not be subtracted from
// the running completion tally (which already holds this call's own 2 real
// per-delta bumps).
func TestChatZeroUsageChunkFallsBackAndDoesNotCorruptCompletionCount(t *testing.T) {
	url := sseServer(t,
		`{"choices":[{"delta":{"content":"Hello"}}]}`,
		`{"choices":[{"delta":{"content":" world"}}]}`,
		`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":0,"completion_tokens":0}}`,
	)
	cl := NewOpenAIClient(config.LLM{BaseURL: url, Model: "fake"})
	longPrompt := strings.Repeat("this is a fairly long user message ", 50)
	if _, err := cl.Chat(context.Background(), []Message{User(longPrompt)}, nil); err != nil {
		t.Fatal(err)
	}
	if cl.Context() <= 0 {
		t.Errorf("Context() = %d, want a positive fallback estimate — an empty/zero usage object must not disable the promptEstimate fallback", cl.Context())
	}
	if _, compl := cl.Usage(); compl != 2 {
		t.Errorf("completion tokens = %d, want 2 (this call's 2 real per-delta bumps) — a zero-usage chunk must not subtract from the completion count", compl)
	}
}

// Usage reconciliation must not be committed to shared state until the
// stream is confirmed fully complete. Exact repro: attempt 1 streams 2
// content deltas then a usage chunk reporting completion=10, but the
// connection is cut before "[DONE]" ever arrives (a truncated, retriable
// stream, same shape as TestChatRetriesOnPrematureCleanEOF) — that usage
// report must never reach c.complTk. The retry then streams its own 2 deltas
// and its own usage=10, completing normally. The only correct final total is
// the accepted retry's own 10, not some combination of both attempts.
func TestChatUsageChunkNotCommittedUntilStreamConfirmedDone(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n++
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		if n < 2 {
			// First attempt: 2 content deltas + a usage chunk reporting
			// completion=10, then a clean EOF with no "[DONE]" — a truncated
			// stream whose usage numbers must never reach shared state.
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"a"}}]}`+"\n\n")
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"b"}}]}`+"\n\n")
			io.WriteString(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`+"\n\n")
			fl.Flush()
			return // clean EOF, no [DONE] — retriable per TestChatRetriesOnPrematureCleanEOF
		}
		// Retry: completes normally with its own usage=10.
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"c"}}]}`+"\n\n")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"d"}}]}`+"\n\n")
		io.WriteString(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatalf("Chat should retry past the truncated stream: %v", err)
	}
	if _, compl := cl.Usage(); compl != 10 {
		t.Errorf("completion tokens = %d, want 10 (only the accepted retry's usage — the discarded attempt's usage chunk must not have been committed)", compl)
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
	if _, err := cl.parseStream(strings.NewReader(heartbeats), tick, 0, new(int), 0); err != nil {
		t.Fatal(err)
	}
	if ticks != 0 {
		t.Errorf("heartbeats/empty deltas reset the watchdog %d times, want 0", ticks)
	}

	// A real content delta does reset it.
	ticks = 0
	real := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl.parseStream(strings.NewReader(real), tick, 0, new(int), 0); err != nil {
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
	msg, err := cl.parseStream(strings.NewReader(sse), func() {}, 0, new(int), 0)
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
	if _, err := cl2.parseStream(strings.NewReader(sse2), func() {}, 0, new(int), 0); err != nil {
		t.Fatal(err)
	}
	if got := cl2.Live(); got != "pondering" {
		t.Errorf("Live() (OpenRouter key) = %q, want %q", got, "pondering")
	}

	// A model with no reasoning phase at all: plain content deltas alone must
	// still populate Live() — this is the case that was reported empty live.
	cl3 := NewOpenAIClient(config.LLM{Model: "x"})
	sse3 := "data: {\"choices\":[{\"delta\":{\"content\":\"plain answer\"}}]}\n\ndata: [DONE]\n\n"
	if _, err := cl3.parseStream(strings.NewReader(sse3), func() {}, 0, new(int), 0); err != nil {
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
	msg, err := cl.parseStream(strings.NewReader(sse), func() {}, 0, new(int), 0)
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

// Two concurrent Chat() calls on the SAME client share one c.complTk counter.
// Before the fix, a retriable failure rolled the counter back to a snapshot
// taken at the start of ITS OWN attempt — an absolute restore, not a per-
// request delta. If a DIFFERENT concurrent Chat() call completed and
// reconciled its own tokens into the counter while the failing attempt was in
// flight, the restore wiped that other call's contribution out too.
//
// Here B starts first (so its pre-attempt snapshot would be 0), streams a
// couple of deltas, then blocks. While B is blocked, A runs an entire
// Chat() to completion, reconciling 100 real completion tokens. Only then is
// B's held connection allowed to fail (a retriable premature EOF, the same
// shape as TestChatRetriesOnPrematureCleanEOF); B retries and reconciles 7
// more. The correct final total is A's 100 + B's 7 = 107. The old snapshot-
// restore code reports 7 — A's 100 silently destroyed by B's rollback.
func TestChatConcurrentRetryRollbackDoesNotClobberOtherCall(t *testing.T) {
	var reqNum int32
	bStreamed := make(chan struct{}) // closed once B's first attempt has streamed its deltas
	letBFail := make(chan struct{})  // closed once A has fully completed — now let B's attempt fail

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		switch atomic.AddInt32(&reqNum, 1) {
		case 1:
			// B's first attempt: stream 2 deltas (bumps B's own live estimate by
			// 2), then hold the connection open — un-reconciled — until A has
			// fully completed elsewhere, then let this handler return, which
			// closes the body cleanly with no "[DONE]" (a retriable premature
			// EOF, per TestChatRetriesOnPrematureCleanEOF).
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"b"}}]}`+"\n\n")
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"b"}}]}`+"\n\n")
			fl.Flush()
			close(bStreamed)
			<-letBFail
		case 2:
			// A: completes immediately, reconciling to 100 real completion tokens.
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"a"}}]}`+"\n\n")
			fl.Flush()
			io.WriteString(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":100}}`+"\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
		case 3:
			// B's retry: succeeds, reconciling to 7 real completion tokens.
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"b2"}}]}`+"\n\n")
			fl.Flush()
			io.WriteString(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":7}}`+"\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
		}
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})

	var bErr, aErr error
	bDone := make(chan struct{})
	go func() {
		_, bErr = cl.Chat(context.Background(), []Message{User("b")}, nil)
		close(bDone)
	}()
	<-bStreamed // B is now blocked, holding its 2-bump estimate un-reconciled

	aDone := make(chan struct{})
	go func() {
		_, aErr = cl.Chat(context.Background(), []Message{User("a")}, nil)
		close(aDone)
	}()
	<-aDone // A has fully completed and reconciled its 100 real tokens

	close(letBFail) // now let B's held attempt fail, retry, and succeed
	<-bDone

	if aErr != nil {
		t.Fatalf("A: Chat failed: %v", aErr)
	}
	if bErr != nil {
		t.Fatalf("B: Chat failed: %v", bErr)
	}
	if _, compl := cl.Usage(); compl != 107 {
		t.Errorf("completion tokens = %d, want 107 (A's 100 + B's 7 — A's contribution must survive B's rollback)", compl)
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
