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

// scriptLLM returns a fixed sequence of replies and records the last messages
// it was given.
type scriptLLM struct {
	replies  []llm.Message
	i        int
	lastMsgs []llm.Message
}

func (s *scriptLLM) Chat(_ context.Context, msgs []llm.Message, _ []map[string]any) (llm.Message, error) {
	s.lastMsgs = msgs
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

func TestSessionMemoryCarriesAcrossRuns(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "the answer is 4"},   // run 1 final
		{Role: "assistant", Content: "we computed 2+2=4"}, // run 2 final
	}}
	a := New(fake, reg, nil, nil, "", 5)

	if _, err := a.Run(context.Background(), "what is 2+2"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), "what did we just do?"); err != nil {
		t.Fatal(err)
	}

	// The second Run's prompt must contain the first goal AND its answer.
	var sawGoal, sawAnswer bool
	for _, m := range fake.lastMsgs {
		if strings.Contains(m.Content, "what is 2+2") {
			sawGoal = true
		}
		if strings.Contains(m.Content, "the answer is 4") {
			sawAnswer = true
		}
	}
	if !sawGoal || !sawAnswer {
		t.Errorf("session memory missing in 2nd run: goal=%v answer=%v msgs=%+v", sawGoal, sawAnswer, fake.lastMsgs)
	}

	a.Reset()
	if a.SessionLen() != 0 {
		t.Errorf("after Reset SessionLen = %d, want 0", a.SessionLen())
	}
}

type recTracer struct{ kinds []string }

func (r *recTracer) Emit(kind string, _ map[string]any) { r.kinds = append(r.kinds, kind) }

func TestNoDuplicateFinalEmit(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	rt := &recTracer{}
	fake := &scriptLLM{replies: []llm.Message{{Role: "assistant", Content: "hi there"}}}
	a := New(fake, reg, nil, rt, "", 5)

	if _, err := a.Run(context.Background(), "say hi"); err != nil {
		t.Fatal(err)
	}
	var assistant, final int
	for _, k := range rt.kinds {
		switch k {
		case "assistant":
			assistant++
		case "final":
			final++
		}
	}
	if final != 1 || assistant != 0 {
		t.Errorf("emitted %v, want exactly 1 final and 0 assistant for a no-tool answer", rt.kinds)
	}
}

func TestCompactSummarizesSession(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "answer A"},
		{Role: "assistant", Content: "answer B"},
		{Role: "assistant", Content: "SUMMARY: we did A and B"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.Run(context.Background(), "task 1")
	a.Run(context.Background(), "task 2")
	if a.SessionLen() != 4 {
		t.Fatalf("SessionLen before compact = %d, want 4", a.SessionLen())
	}

	n, err := a.Compact(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Errorf("compacted %d messages, want 4", n)
	}
	if a.SessionLen() != 2 {
		t.Errorf("SessionLen after compact = %d, want 2 (summary pair)", a.SessionLen())
	}
	var found bool
	for _, m := range a.history {
		if strings.Contains(m.Content, "SUMMARY: we did A and B") {
			found = true
		}
	}
	if !found {
		t.Error("summary not stored in the compacted history")
	}
}

func TestSplitSuggestion(t *testing.T) {
	clean, sug := splitSuggestion("Wrote hello.sh and ran it.\nNEXT: add a test")
	if clean != "Wrote hello.sh and ran it." {
		t.Errorf("clean = %q", clean)
	}
	if sug != "add a test" {
		t.Errorf("suggestion = %q, want 'add a test'", sug)
	}
	if c, s := splitSuggestion("just an answer"); c != "just an answer" || s != "" {
		t.Errorf("no-NEXT case = %q,%q", c, s)
	}
	// A placeholder-shaped suggestion must be unwrapped, not shown with brackets.
	if _, s := splitSuggestion("done\nNEXT: <run the test script>"); s != "run the test script" {
		t.Errorf("bracketed suggestion = %q, want unwrapped", s)
	}
	// A "NEXT:" in the MIDDLE must stay part of the answer, not be extracted.
	mid := "Here is a script:\nNEXT: do X\nand it ends here"
	if c, s := splitSuggestion(mid); c != mid || s != "" {
		t.Errorf("mid-answer NEXT wrongly peeled: clean=%q sug=%q", c, s)
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
