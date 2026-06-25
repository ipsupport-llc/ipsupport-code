package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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

// compile-time guarantee the client satisfies the interface.
var _ Chatter = (*OpenAIClient)(nil)
