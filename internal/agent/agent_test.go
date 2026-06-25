package agent

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// scriptLLM returns a fixed sequence of replies, ignoring input.
type scriptLLM struct {
	replies []llm.Message
	i       int
}

func (s *scriptLLM) Chat(_ context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	if s.i >= len(s.replies) {
		return llm.Message{Role: "assistant", Content: "(no more replies)"}, nil
	}
	m := s.replies[s.i]
	s.i++
	return m, nil
}

func toolCallReply(id, name, args string) llm.Message {
	return llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: id, Name: name, Arguments: args}}}
}

func toolObservation(msgs []llm.Message) []llm.Message {
	var out []llm.Message
	for _, m := range msgs {
		if m.Role == "tool" {
			out = append(out, m)
		}
	}
	return out
}

func TestRunFiresToolThenFinal(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		{Role: "assistant", Content: "the answer is 4"},
	}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, err := a.Run(context.Background(), "what is 2+2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "the answer is 4" {
		t.Errorf("final = %q", tr.Final)
	}
	if tr.Steps != 2 {
		t.Errorf("steps = %d, want 2", tr.Steps)
	}
	obs := toolObservation(tr.Messages)
	if len(obs) != 1 || !strings.Contains(obs[0].Content, "4") {
		t.Errorf("tool observation = %+v, want result containing 4", obs)
	}
}

func TestRunInjectsPitfall(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "calc", ErrorPattern: "unknown function",
		Context: "using calc", ProvenFix: "only use whitelisted functions",
	})
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"foo(2)"}}`),
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, reg, kb, nil, "", 5)

	tr, _ := a.Run(context.Background(), "compute foo(2)")
	if got := toolObservation(tr.Messages); len(got) == 0 ||
		!strings.Contains(got[0].Content, "only use whitelisted functions") {
		t.Errorf("pitfall hint not injected: %+v", got)
	}
}

func TestParseArgsFoldsTopLevel(t *testing.T) {
	// Small models often omit the "params" wrapper.
	action, params := parseArgs(`{"action":"calculate","expression":"2+2"}`)
	if action != "calculate" {
		t.Errorf("action = %q, want calculate", action)
	}
	if params["expression"] != "2+2" {
		t.Errorf("params = %v, want folded expression", params)
	}
	if _, leaked := params["action"]; leaked {
		t.Error("action leaked into folded params")
	}
}

func TestRunConcurrentToolCallsStayOrdered(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	twoCalls := llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{
		{ID: "c1", Name: "calc", Arguments: `{"action":"calculate","params":{"expression":"2+2"}}`},
		{ID: "c2", Name: "calc", Arguments: `{"action":"calculate","params":{"expression":"10*10"}}`},
	}}
	fake := &scriptLLM{replies: []llm.Message{twoCalls, {Role: "assistant", Content: "done"}}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, _ := a.Run(context.Background(), "two sums")
	obs := toolObservation(tr.Messages)
	if len(obs) != 2 {
		t.Fatalf("observations = %d, want 2", len(obs))
	}
	if obs[0].ToolCallID != "c1" || !strings.Contains(obs[0].Content, "4") {
		t.Errorf("first observation = %+v, want c1=4", obs[0])
	}
	if obs[1].ToolCallID != "c2" || !strings.Contains(obs[1].Content, "100") {
		t.Errorf("second observation = %+v, want c2=100", obs[1])
	}
}

func TestRunWrongToolHint(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc(), tool.NewWeb(nil))
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"search","params":{"query":"x"}}`),
		{Role: "assistant", Content: "ok"},
	}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, _ := a.Run(context.Background(), "search for x")
	if got := toolObservation(tr.Messages); len(got) == 0 ||
		!strings.Contains(got[0].Content, `belongs to tool "web"`) {
		t.Errorf("wrong-tool correction not surfaced: %+v", got)
	}
}
