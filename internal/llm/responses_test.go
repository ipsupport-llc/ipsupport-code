package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// sse writes /responses events the way a server streams them: an event line
// and a data line each.
func sse(w io.Writer, events ...string) {
	for _, e := range events {
		var head struct {
			Type string `json:"type"`
		}
		json.Unmarshal([]byte(e), &head)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", head.Type, e)
	}
}

// A streamed reply with text and a tool call: the call answers to its call_id
// (not the item id), its arguments are the finished ones, and the usage sizes
// the context.
func TestResponsesStreamTextAndToolCall(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w,
			`{"type":"response.created","response":{"status":"in_progress"}}`,
			`{"type":"response.reasoning_summary_text.delta","delta":"thinking"}`,
			`{"type":"response.output_text.delta","delta":"Reading "}`,
			`{"type":"response.output_text.delta","delta":"it."}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file","arguments":""}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"action\":"}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"\"read\"}"}`,
			`{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"action\":\"read\"}"}`,
			`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file","arguments":"{\"action\":\"read\"}"}}`,
			`{"type":"response.completed","response":{"status":"completed","model":"m","usage":{"input_tokens":1234,"output_tokens":20}}}`,
		)
	}))
	defer srv.Close()

	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
	msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if path != "/responses" {
		t.Errorf("path = %q, want /responses", path)
	}
	if msg.Content != "Reading it." || msg.Reasoning != "thinking" {
		t.Errorf("content %q, reasoning %q", msg.Content, msg.Reasoning)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	tc := msg.ToolCalls[0]
	if tc.ID != "call_A" || tc.Name != "file" || tc.Arguments != `{"action":"read"}` {
		t.Errorf("tool call = %+v", tc)
	}
	if msg.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %q", msg.FinishReason)
	}
	if got := c.Context(); got != 1234 {
		t.Errorf("Context() = %d, want the reported input_tokens 1234", got)
	}
	if _, compl := c.Usage(); compl != 20 {
		t.Errorf("completion tokens = %d, want the reported 20", compl)
	}
}

// The request: leading system text as instructions, the rest as input items —
// a past call without the server's item id, its output by call_id — tools
// flattened with strict off, the output cap and reasoning in their /responses
// names, nothing kept on the server.
func TestResponsesRequestBody(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w, `{"type":"response.completed","response":{"status":"completed"}}`)
	}))
	defer srv.Close()

	l := config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses", MaxOutputTokens: 900,
		Extra: map[string]any{"reasoning_effort": "low"}}
	c := NewOpenAIClient(l)
	msgs := []Message{
		System("be brief"),
		User("read a.go"),
		{Role: "assistant", Content: "ok", ToolCalls: []ToolCall{{ID: "call_A", Name: "file", Arguments: `{"action":"read"}`}}},
		ToolResult("call_A", "file", "package a"),
		{Role: "system", Content: "a note"},
	}
	tools := []map[string]any{{"type": "function", "function": map[string]any{"name": "file", "description": "files", "parameters": map[string]any{"type": "object"}}}}
	if _, err := c.Chat(context.Background(), msgs, tools); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if body["instructions"] != "be brief" || body["store"] != false || body["stream"] != true {
		t.Errorf("instructions/store/stream = %v/%v/%v", body["instructions"], body["store"], body["stream"])
	}
	if body["max_output_tokens"] != float64(900) {
		t.Errorf("max_output_tokens = %v", body["max_output_tokens"])
	}
	if _, ok := body["max_tokens"]; ok {
		t.Error("max_tokens is the chat-completions name")
	}
	if r, _ := body["reasoning"].(map[string]any); r["effort"] != "low" {
		t.Errorf("reasoning = %v, want effort low", body["reasoning"])
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Error("reasoning_effort must be moved, not sent as is")
	}
	got, _ := json.Marshal(body["input"])
	want := `[{"content":"read a.go","role":"user","type":"message"},` +
		`{"content":"ok","role":"assistant","type":"message"},` +
		`{"arguments":"{\"action\":\"read\"}","call_id":"call_A","name":"file","type":"function_call"},` +
		`{"call_id":"call_A","output":"package a","type":"function_call_output"},` +
		`{"content":"a note","role":"system","type":"message"}]`
	if string(got) != want {
		t.Errorf("input =\n%s\nwant\n%s", got, want)
	}
	tl, _ := json.Marshal(body["tools"])
	if string(tl) != `[{"description":"files","name":"file","parameters":{"type":"object"},"strict":false,"type":"function"}]` {
		t.Errorf("tools = %s", tl)
	}
}

// A stream that stops before response.completed was cut off: its partial call
// must not run. Retried like a dropped chat stream.
func TestResponsesStreamWithoutCompletionIsAnError(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w,
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file"}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"action\":\"wri"}`,
		)
	}))
	defer srv.Close()

	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses", RetryAttempts: 2})
	_, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err == nil || !strings.Contains(err.Error(), "without completing") {
		t.Fatalf("err = %v, want a cut-off stream", err)
	}
	if hits.Load() != 2 {
		t.Errorf("sent %d times, want a retry", hits.Load())
	}
}

// Errors the server reports in the stream get the chat path's treatment: a
// request the server refuses is not resent with the full budget, a busy server
// is.
func TestResponsesStreamErrors(t *testing.T) {
	for _, tc := range []struct {
		name, event string
		want        int32
	}{
		{"failed too long", `{"type":"response.failed","response":{"status":"failed","error":{"code":"context_length_exceeded","message":"input too long"}}}`, serverErrorAttempts},
		{"error event busy", `{"type":"error","code":"server_overloaded","message":"try again later"}`, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				sse(w, `{"type":"response.output_text.delta","delta":"partial"}`, tc.event)
			}))
			defer srv.Close()
			c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses", RetryAttempts: 3})
			c.OnRetry = func(int, time.Duration, string) {}
			_, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
			if err == nil {
				t.Fatal("a failed response returned as an answer")
			}
			if hits.Load() != tc.want {
				t.Errorf("sent %d times, want %d (%v)", hits.Load(), tc.want, err)
			}
		})
	}
}

// A reply cut at the output cap is returned, marked as cut.
func TestResponsesIncompleteIsLength(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w,
			`{"type":"response.output_text.delta","delta":"half an ans"}`,
			`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
		)
	}))
	defer srv.Close()
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
	msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil || msg.Content != "half an ans" || msg.FinishReason != "length" {
		t.Fatalf("msg %+v, err %v", msg, err)
	}
}

// A whole (non-streamed) /responses reply is read too.
func TestResponsesJSONReply(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"status":"completed","model":"m","output":[
			{"type":"reasoning","id":"rs_1"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]},
			{"type":"function_call","id":"fc_1","call_id":"call_B","name":"calc","arguments":"{}"}],
			"usage":{"input_tokens":50,"output_tokens":5}}`)
	}))
	defer srv.Close()
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
	msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if msg.Content != "done" || len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_B" || c.Context() != 50 {
		t.Errorf("msg %+v, context %d", msg, c.Context())
	}
}

// modelsServer serves a /models list and answers chat or responses requests,
// counting each.
type modelsServer struct {
	models          string
	list, chat, rsp atomic.Int32
	auth            atomic.Value
}

func (s *modelsServer) start(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	return srv
}

func (s *modelsServer) serve(w http.ResponseWriter, r *http.Request) {
	s.auth.Store(r.Header.Get("Authorization"))
	switch r.URL.Path {
	case "/models":
		s.list.Add(1)
		io.WriteString(w, s.models)
	case "/chat/completions":
		s.chat.Add(1)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"chat"}}]}`)
	case "/responses":
		s.rsp.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w, `{"type":"response.output_text.delta","delta":"responses"}`,
			`{"type":"response.completed","response":{"status":"completed"}}`)
	default:
		http.NotFound(w, r)
	}
}

// "auto" asks the model list once, and uses /responses only for a model listed
// as served there and not on /chat/completions.
func TestAutoWireFollowsTheModelList(t *testing.T) {
	for _, tc := range []struct {
		name, models, want string
		reads              int32
	}{
		{"responses only", `{"data":[{"id":"m","supported_endpoints":["/responses","ws:/responses"]}]}`, "responses", 1},
		{"both", `{"data":[{"id":"m","supported_endpoints":["/chat/completions","/responses"]}]}`, "chat", 1},
		{"not said", `{"data":[{"id":"m"}]}`, "chat", 1},
		{"other model", `{"data":[{"id":"x","supported_endpoints":["/responses"]}]}`, "chat", 1},
		{"no list", `nope`, "chat", 2}, // unreadable: tried again on the next request
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &modelsServer{models: tc.models}
			srv := s.start(t)
			c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "auto"})
			for range 2 {
				msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
				if err != nil || msg.Content != tc.want {
					t.Fatalf("answered %q (%v), want %s", msg.Content, err, tc.want)
				}
			}
			if s.list.Load() != tc.reads {
				t.Errorf("model list read %d times, want %d", s.list.Load(), tc.reads)
			}
		})
	}
}

// With no window configured and the key from a command, the client reads the
// model's window before its first request — with that key — so a run started
// before any probe could has it. A static key or a configured window skips
// the read: the caller's own probe covers those.
func TestFoundWindowFromTheModelList(t *testing.T) {
	argv, _ := keyHelper(t, "")
	models := `{"data":[{"id":"m","capabilities":{"limits":{"max_context_window_tokens":264000,"max_prompt_tokens":200000}}}]}`

	s := &modelsServer{models: models}
	srv := s.start(t)
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKeyCmd: argv})
	if _, err := c.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if c.FoundWindow() != 200000 {
		t.Errorf("FoundWindow = %d, want max_prompt_tokens 200000", c.FoundWindow())
	}
	if a, _ := s.auth.Load().(string); !strings.HasPrefix(a, "Bearer key-") {
		t.Errorf("authorization = %q, want the command's key", a)
	}

	for _, l := range []config.LLM{
		{BaseURL: srv.URL, Model: "m", APIKey: "k"},
		{BaseURL: srv.URL, Model: "m", APIKeyCmd: argv, ContextWindow: 32000},
	} {
		s.list.Store(0)
		c := NewOpenAIClient(l)
		if _, err := c.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
			t.Fatalf("Chat: %v", err)
		}
		if s.list.Load() != 0 || c.FoundWindow() != 0 {
			t.Errorf("%+v: list read %d times, found %d — want no read", l, s.list.Load(), c.FoundWindow())
		}
	}
}

// A key command that fails leaves the read for the next request, and the
// request reports the failure.
func TestLookupWaitsForAKey(t *testing.T) {
	argv, _ := keyHelper(t, "fail")
	s := &modelsServer{models: `{"data":[]}`}
	srv := s.start(t)
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKeyCmd: argv, WireAPI: "auto", RetryAttempts: 1})
	_, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err == nil {
		t.Fatal("no error from a failing key command")
	}
	if s.list.Load() != 0 || c.looked {
		t.Errorf("list read %d, looked %v — want neither without a key", s.list.Load(), c.looked)
	}
}

// GitHub Copilot's list carries the window under capabilities.limits.
func TestDetectModelContextReadsCopilotLimits(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[{"id":"a","capabilities":{"limits":{"max_prompt_tokens":64000}}},{"id":"b","capabilities":{"limits":{"max_context_window_tokens":128000}}}]}`)
	}))
	defer srv.Close()
	if got := DetectModelContext(context.Background(), srv.URL, "", "a", nil); got != 64000 {
		t.Errorf("a = %d, want max_prompt_tokens", got)
	}
	if got := DetectModelContext(context.Background(), srv.URL, "", "b", nil); got != 128000 {
		t.Errorf("b = %d, want the whole window when no prompt cap is listed", got)
	}
}

// A finished call item without its arguments keeps the ones streamed before
// it, and a finished-arguments event may carry them as an object.
func TestResponsesKeepsStreamedArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(w,
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file"}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"action\":\"read\"}"}`,
			`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file"}}`,
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_2","call_id":"call_B","name":"calc"}}`,
			`{"type":"response.function_call_arguments.done","item_id":"fc_2","arguments":{"expression":"1+1"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		)
	}))
	defer srv.Close()
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
	msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(msg.ToolCalls) != 2 || msg.ToolCalls[0].Arguments != `{"action":"read"}` || msg.ToolCalls[1].Arguments != `{"expression":"1+1"}` {
		t.Errorf("tool calls = %+v", msg.ToolCalls)
	}
}

// A server that lists the output only in the finished response still gives
// its text and calls — once each, alongside the ones it did stream.
func TestResponsesTakesOutputFromTheFinishedResponse(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []string
		calls  string
	}{
		{"only in the response", []string{
			`{"type":"response.completed","response":{"status":"completed","output":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"reading"}]},
				{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file","arguments":"{\"action\":\"read\"}"},
				{"type":"function_call","call_id":"call_B","name":"calc","arguments":"{}"}]}}`,
		}, "call_A file,call_B calc"},
		{"streamed too", []string{
			`{"type":"response.output_text.delta","delta":"reading"}`,
			`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file","arguments":"{\"action\":\"read\"}"}}`,
			`{"type":"response.completed","response":{"status":"completed","output":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"reading"}]},
				{"type":"function_call","id":"fc_1","call_id":"call_A","name":"file","arguments":"{\"action\":\"read\"}"},
				{"type":"function_call","call_id":"call_X","name":"calc","arguments":"{}"}]}}`,
		}, "call_A file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				for _, e := range tc.events {
					fmt.Fprintf(w, "data: %s\n\n", strings.Join(strings.Fields(e), " "))
				}
			}))
			defer srv.Close()
			c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
			msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			var got []string
			for _, c := range msg.ToolCalls {
				got = append(got, c.ID+" "+c.Name)
			}
			if msg.Content != "reading" || strings.Join(got, ",") != tc.calls {
				t.Errorf("content %q, calls %v, want %s", msg.Content, got, tc.calls)
			}
		})
	}
}

// A model list that fails to load is tried again on the next requests — a
// server briefly down must not pin an auto client to the wrong format — but
// not before every turn forever.
func TestAutoWireRetriesAFailedList(t *testing.T) {
	s := &modelsServer{models: `{"data":[{"id":"m","supported_endpoints":["/responses"]}]}`}
	var down atomic.Int32
	down.Store(1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" && down.Load() > 0 {
			s.list.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		s.serve(w, r)
	}))
	defer srv.Close()
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "auto"})
	if msg, _ := c.Chat(context.Background(), []Message{User("hi")}, nil); msg.Content != "chat" {
		t.Fatalf("first answer %q, want chat while the list is down", msg.Content)
	}
	down.Store(0)
	if msg, _ := c.Chat(context.Background(), []Message{User("hi")}, nil); msg.Content != "responses" {
		t.Fatalf("second answer %q, want responses once the list loads", msg.Content)
	}

	down.Store(1)
	s.list.Store(0)
	c = NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "auto"})
	for range 5 {
		c.Chat(context.Background(), []Message{User("hi")}, nil)
	}
	if n := s.list.Load(); n != lookupTries {
		t.Errorf("list tried %d times, want %d", n, lookupTries)
	}
}

// The runaway cap follows a window found in the model list the way it follows
// a configured one.
func TestRunawayCapFollowsTheFoundWindow(t *testing.T) {
	c := NewOpenAIClient(config.LLM{Model: "m"})
	if c.respCap() != maxResponseTokens(0) {
		t.Fatalf("cap %d with no window", c.respCap())
	}
	c.foundWindow.Store(200000)
	if c.respCap() != maxResponseTokens(200000) {
		t.Errorf("cap %d, want %d for a 200k window", c.respCap(), maxResponseTokens(200000))
	}
}

// A server that gives an item a new id on every event (GitHub Copilot
// reportedly does) still yields each call once: calls are kept by call_id, and
// argument events find theirs by output_index, else the call streaming now.
func TestResponsesCallsSurviveChangingItemIDs(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []string
		want   string
	}{
		{"one call, new id each event", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"e1","call_id":"call_A","name":"file","arguments":""}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"e2","delta":"{\"a\":"}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"e3","delta":"1}"}`,
			`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"e4","arguments":"{\"a\":1}"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"e5","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}}`,
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","id":"e6","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}]}}`,
		}, `call_A file {"a":1}`},
		{"two calls interleaved by index", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"e1","call_id":"call_A","name":"file"}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"e2","call_id":"call_B","name":"calc"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"e3","delta":"{\"b\":2}"}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"e4","delta":"{\"a\":1}"}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		}, `call_A file {"a":1}|call_B calc {"b":2}`},
		{"arguments only, named at the end", []string{
			`{"type":"response.function_call_arguments.delta","item_id":"e1","delta":"{\"a\":1}"}`,
			`{"type":"response.function_call_arguments.done","item_id":"e2","arguments":"{\"a\":1}"}`,
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","id":"e9","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}]}}`,
		}, `call_A file {"a":1}`},
		{"call_id shown only when done", []string{
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"e1","name":"file"}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"e1","delta":"{\"a\":1}"}`,
			`{"type":"response.output_item.done","item":{"type":"function_call","id":"e2","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		}, `call_A file {"a":1}`},
		{"a delta ahead of its call doesn't swallow it", []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"e1","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"e7","delta":""}`,
			`{"type":"response.function_call_arguments.done","output_index":1,"item_id":"e7","arguments":"{\"a\":1}"}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"e8","call_id":"call_B","name":"calc","arguments":"{\"b\":2}"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		}, `call_A file {"a":1}|call_B calc {"b":2}`},
		{"no ids at all", []string{
			`{"type":"response.output_item.added","item":{"type":"function_call","name":"file","arguments":"{\"a\":1}"}}`,
			`{"type":"response.output_item.added","item":{"type":"function_call","name":"calc","arguments":"{\"b\":2}"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		}, ` file {"a":1}| calc {"b":2}`},
		{"no index: the call streaming now", []string{
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"e1","call_id":"call_A","name":"file"}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"e2","delta":"{\"a\":1}"}`,
			`{"type":"response.output_item.added","item":{"type":"function_call","id":"e3","call_id":"call_B","name":"calc"}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"e4","delta":"{\"b\":2}"}`,
			`{"type":"response.output_item.done","item":{"type":"function_call","id":"e5","call_id":"call_A","name":"file","arguments":"{\"a\":1}"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		}, `call_A file {"a":1}|call_B calc {"b":2}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				sse(w, tc.events...)
			}))
			defer srv.Close()
			c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "responses"})
			msg, err := c.Chat(context.Background(), []Message{User("hi")}, nil)
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			var got []string
			for _, c := range msg.ToolCalls {
				got = append(got, c.ID+" "+c.Name+" "+c.Arguments)
			}
			if strings.Join(got, "|") != tc.want {
				t.Errorf("calls %q, want %q", strings.Join(got, "|"), tc.want)
			}
		})
	}
}

// The path a request goes to is the format its body was built in — decided
// once per call, not read again when it is sent.
func TestSendUsesTheFormatTheBodyWasBuiltIn(t *testing.T) {
	s := &modelsServer{}
	srv := s.start(t)
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m"}) // chat format
	msg, err := c.roundTrip(context.Background(), []byte(`{}`), c.tuning.Load(), true)
	if err != nil || msg.Content != "responses" {
		t.Errorf("answered %q (%v), want the /responses path", msg.Content, err)
	}
}

// Requests the user cancelled don't use up the model-list tries.
func TestCancelledLookupsAreNotCounted(t *testing.T) {
	s := &modelsServer{models: `{"data":[{"id":"m","supported_endpoints":["/responses"]}]}`}
	srv := s.start(t)
	c := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", WireAPI: "auto"})
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	for range lookupTries + 1 {
		c.Chat(cancelled, []Message{User("hi")}, nil)
	}
	if msg, _ := c.Chat(context.Background(), []Message{User("hi")}, nil); msg.Content != "responses" {
		t.Errorf("answered %q after cancelled calls, want the list read and responses used", msg.Content)
	}
}
