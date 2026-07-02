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

func TestHintsRequireErrorPatternMatch(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "file", ErrorPattern: "missing required param(s): path",
		Context: "file: edit", ProvenFix: "include the path param",
	})
	a := New(&scriptLLM{}, tool.NewRegistry(tool.NewCalc()), kb, nil, "", 5)

	// An unrelated error (e.g. "no action") must NOT surface the path pitfall.
	if h := a.hints("file", `file: no action given — set "action" to one of: read, write`); h != "" {
		t.Errorf("irrelevant hint injected: %q", h)
	}
	// The same error recurring does.
	if h := a.hints("file", "edit failed: missing required param(s): path"); !strings.Contains(h, "include the path param") {
		t.Errorf("relevant hint not injected: %q", h)
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

type recTracer struct {
	kinds        []string
	finalSuggest string
}

func (r *recTracer) Emit(kind string, f map[string]any) {
	r.kinds = append(r.kinds, kind)
	if kind == "final" {
		r.finalSuggest, _ = f["suggest"].(string)
	}
}

func (r *recTracer) has(kind string) bool {
	for _, k := range r.kinds {
		if k == kind {
			return true
		}
	}
	return false
}

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
	// Markdown/bullet-decorated NEXT must still be peeled (the reported bug).
	for _, in := range []string{
		"Done.\n**NEXT:** add a test",
		"Done.\n- NEXT: add a test",
		"Done.\n**NEXT: add a test**",
	} {
		c, s := splitSuggestion(in)
		if c != "Done." || s != "add a test" {
			t.Errorf("decorated NEXT %q → clean=%q sug=%q", in, c, s)
		}
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

func TestParseArgsStringifiedParams(t *testing.T) {
	// The most common malformation: params double-encoded as a JSON string.
	action, params := parseArgs(`{"action":"write","params":"{\"path\":\"main.py\",\"content\":\"x\"}"}`)
	if action != "write" {
		t.Errorf("action = %q, want write", action)
	}
	if params["path"] != "main.py" || params["content"] != "x" {
		t.Errorf("params = %v, want decoded path+content", params)
	}
	// Action nested inside the stringified blob, none at top level.
	action, params = parseArgs(`{"params":"{\"action\":\"write\",\"path\":\"a.txt\",\"content\":\"hi\"}"}`)
	if action != "write" || params["path"] != "a.txt" {
		t.Errorf("nested-action: action=%q params=%v", action, params)
	}
	if _, leaked := params["action"]; leaked {
		t.Error("action leaked into decoded params")
	}
}

func TestLooksLikeRefusal(t *testing.T) {
	refusals := []string{
		"Here are the files:\n```\ncode\n```",
		"I don't have access to your files, copy them manually.",
		"Как языковая модель, я не имею доступа к файловой системе.",
	}
	for _, s := range refusals {
		if !looksLikeRefusal(s) {
			t.Errorf("looksLikeRefusal(%q) = false, want true", s)
		}
	}
	ok := []string{"Added a /health endpoint; tests pass.", "Done — wrote main.py and ran the tests."}
	for _, s := range ok {
		if looksLikeRefusal(s) {
			t.Errorf("looksLikeRefusal(%q) = true, want false", s)
		}
	}
}

// A chat model that dodges an action task (pastes files / "I can't access your
// filesystem") with no tool calls gets nudged once, then proceeds to use tools.
func TestRunNudgesRefusalThenActs(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "I don't have access to your filesystem. Here are the files:\n```py\nx=1\n```"},
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		{Role: "assistant", Content: "done — 4"},
	}}
	a := New(fake, reg, nil, nil, "", 6)
	tr, err := a.Run(context.Background(), "edit the files")
	if err != nil {
		t.Fatal(err)
	}
	if tr.Final != "done — 4" {
		t.Errorf("final = %q, want it to proceed past the refusal", tr.Final)
	}
	if len(toolObservation(tr.Messages)) != 1 {
		t.Error("expected the tool to run after the refusal nudge")
	}
}

// The refusal nudge fires at most once: a model that refuses twice has its second
// refusal accepted as the final answer (no infinite loop).
func TestRunAcceptsRefusalAfterOneNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	refusal := llm.Message{Role: "assistant", Content: "As a language model I cannot modify files. Copy:\n```\nx\n```"}
	fake := &scriptLLM{replies: []llm.Message{refusal, refusal}}
	a := New(fake, reg, nil, nil, "", 6)
	tr, err := a.Run(context.Background(), "edit the files")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(tr.Final, "cannot modify files") {
		t.Errorf("final = %q, want the 2nd refusal accepted", tr.Final)
	}
	if tr.Steps != 2 {
		t.Errorf("steps = %d, want 2 (refuse → nudge → refuse-accept)", tr.Steps)
	}
}

// A model that keeps making the exact same (succeeding) tool call makes no
// progress; the loop guard nudges once, then stops instead of running forever.
func TestRunStopsOnRepeatedIdenticalCalls(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	call := toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`)
	replies := make([]llm.Message, 12)
	for i := range replies {
		replies[i] = call
	}
	a := New(&scriptLLM{replies: replies}, reg, nil, nil, "", 20)
	tr, err := a.Run(context.Background(), "spin on the same call")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(tr.Final, "Stopped") {
		t.Errorf("final = %q, want the loop guard to stop it", tr.Final)
	}
	if tr.Steps >= 20 {
		t.Errorf("steps = %d, want the guard to stop well before maxSteps", tr.Steps)
	}
}

func TestUnwrapEnvelope(t *testing.T) {
	// nested content.text (the reported leak)
	if got := unwrapEnvelope(`{"role":"assistant","content":{"text":"hello there"}}`); got != "hello there" {
		t.Errorf("nested = %q, want 'hello there'", got)
	}
	// content as a plain string
	if got := unwrapEnvelope(`{"role":"assistant","content":"hi"}`); got != "hi" {
		t.Errorf("string content = %q, want 'hi'", got)
	}
	// a normal answer is untouched, even one that contains JSON
	for _, s := range []string{
		"Here is the plan: do X then Y.",
		"```json\n{\"role\":\"assistant\"}\n```", // a JSON code block in prose
		`{"role":"user","content":"x"}`,          // not an assistant envelope
		`{"some":"object","without":"role"}`,     // no role
	} {
		if got := unwrapEnvelope(s); got != s {
			t.Errorf("unwrapEnvelope(%q) = %q, want unchanged", s, got)
		}
	}
}

func TestParseArgsNestedObject(t *testing.T) {
	action, params := parseArgs(`{"action":"edit","params":{"path":"a","find":"x","replace":"y"}}`)
	if action != "edit" || params["find"] != "x" || params["replace"] != "y" {
		t.Errorf("nested object: action=%q params=%v", action, params)
	}
	// Mixed shape: path at the top level, the rest under params — fold them together.
	action, params = parseArgs(`{"action":"edit","path":"main.go","params":{"find":"x","replace":"y"}}`)
	if action != "edit" || params["path"] != "main.go" || params["find"] != "x" {
		t.Errorf("mixed shape dropped a sibling: action=%q params=%v", action, params)
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

// planFileTool is a minimal file-like tool with one read-only and one mutating
// action, for exercising the plan-mode gate.
func planFileTool() tool.Tool {
	return tool.NewDomain(tool.DomainSpec{
		Name: "file", Summary: "files",
		Actions: []tool.Action{
			{Name: "read", Run: func(context.Context, tool.Args) tool.Result { return tool.Ok("content-of-x") }},
			{Name: "write", Mutates: true, Run: func(context.Context, tool.Args) tool.Result { return tool.Ok("wrote") }},
		},
	})
}

func TestPlanModeBlocksMutationAndInjectsDirective(t *testing.T) {
	reg := tool.NewRegistry(planFileTool())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"write","params":{"path":"x","content":"y"}}`),
		{Role: "assistant", Content: "plan: 1. write x"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetPlanMode(true)

	tr, _ := a.Run(context.Background(), "make x")
	obs := toolObservation(tr.Messages)
	if len(obs) == 0 || !strings.Contains(obs[0].Content, "plan mode is ON") {
		t.Fatalf("write was not blocked in plan mode: %+v", obs)
	}
	var sawDirective bool
	for _, m := range fake.lastMsgs {
		if m.Role == "system" && strings.Contains(m.Content, "PLAN MODE is ON") {
			sawDirective = true
		}
	}
	if !sawDirective {
		t.Error("plan directive not injected into the prompt")
	}
}

func TestPlanModeAllowsReadOnly(t *testing.T) {
	reg := tool.NewRegistry(planFileTool())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"read","params":{"path":"x"}}`),
		{Role: "assistant", Content: "the file says X"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetPlanMode(true)

	tr, _ := a.Run(context.Background(), "read x")
	if obs := toolObservation(tr.Messages); len(obs) == 0 || !strings.Contains(obs[0].Content, "content-of-x") {
		t.Errorf("read-only call should run in plan mode: %+v", obs)
	}
}

// A model that keeps failing gets ONE rethink nudge; if it still fails, the run
// stops (bounded) and offers the user a steering suggestion.
func TestRunNudgesThenStops(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`) // empty action → always errors
	fake := &scriptLLM{replies: []llm.Message{bad, bad, bad, bad, bad, bad, bad, bad}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 20)

	tr, _ := a.Run(context.Background(), "do something")
	if tr.Steps > 2*maxStuckTurns+1 {
		t.Errorf("ran %d steps, want it bounded (~2x stuck, after one nudge)", tr.Steps)
	}
	if !strings.Contains(tr.Final, "Stopped") {
		t.Errorf("final = %q, want the stuck stop", tr.Final)
	}
	if !rt.has("nudge") {
		t.Error("expected one rethink nudge before stopping")
	}
	if rt.finalSuggest == "" {
		t.Error("the stop should offer the user a steering suggestion")
	}
}

// If the nudge unsticks the model (it answers), the run recovers instead of
// stopping.
func TestStuckNudgeRecovers(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	fake := &scriptLLM{replies: []llm.Message{
		bad, bad, bad, // 3 fails → nudge
		{Role: "assistant", Content: "I can't do that with calc — here's the answer in words."},
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 12)

	tr, _ := a.Run(context.Background(), "do x")
	if strings.Contains(tr.Final, "Stopped") {
		t.Errorf("should have recovered after the nudge, got: %q", tr.Final)
	}
	if !rt.has("nudge") {
		t.Error("expected a nudge before the recovery")
	}
	var injected bool
	for _, m := range fake.lastMsgs {
		if m.Role == "user" && strings.Contains(m.Content, "repeating the same tool") {
			injected = true
		}
	}
	if !injected {
		t.Error("the nudge message should be in the conversation sent to the model")
	}
}

type cancelLLM struct {
	calls  int
	cancel context.CancelFunc
}

func (c *cancelLLM) Chat(ctx context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	c.calls++
	if c.calls == 1 {
		c.cancel() // simulate the user pressing esc after the first action
		return toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`), nil
	}
	return llm.Message{}, ctx.Err()
}

// Cancelling (esc) mid-run keeps the partial work: the run ends cleanly (no error),
// is marked Cancelled, and is remembered so a follow-up can continue.
func TestRunCancelKeepsProgress(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	ctx, cancel := context.WithCancel(context.Background())
	a := New(&cancelLLM{cancel: cancel}, reg, nil, nil, "", 10)

	tr, err := a.Run(ctx, "do the thing")
	if err != nil {
		t.Fatalf("a cancel should end cleanly, got error: %v", err)
	}
	if !tr.Cancelled {
		t.Error("transcript should be marked Cancelled")
	}
	if a.SessionLen() == 0 {
		t.Error("a cancelled run with work done should be remembered so it can continue")
	}
}

// Progress between failure streaks earns a fresh nudge: a successful tool call
// clears the "already nudged" latch, so a couple of stumbles after good work don't
// insta-stop the run (the reported bug — a success didn't reset the error state).
func TestProgressResetsNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	good := toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`)
	fake := &scriptLLM{replies: []llm.Message{
		bad, bad, bad, // streak 1 → nudge #1
		good,          // progress → clears the nudged latch
		bad, bad, bad, // streak 2 → must nudge again, NOT insta-stop
		good,
		{Role: "assistant", Content: "done"},
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 30)

	tr, _ := a.Run(context.Background(), "do x")
	if strings.Contains(tr.Final, "Stopped") {
		t.Errorf("progress should reset the nudge latch, not stop: %q", tr.Final)
	}
	nudges := 0
	for _, k := range rt.kinds {
		if k == "nudge" {
			nudges++
		}
	}
	if nudges < 2 {
		t.Errorf("expected ≥2 nudges (one per streak, after a progress reset), got %d", nudges)
	}
}

func calcCall() llm.Message {
	return toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`)
}

// The goal loop re-feeds the goal when the judge says it isn't met, then accepts it
// once the judge says DONE. Returns counts the re-feeds; GoalMet records the verdict.
func TestRunGoalLoopRefeedsUntilJudgeSaysDone(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), // act
		{Role: "assistant", Content: "did part 1"},           // finalize #1
		{Role: "assistant", Content: "MORE: still need pt2"}, // judge: not met → re-feed
		calcCall(),                               // act again
		{Role: "assistant", Content: "all done"}, // finalize #2
		{Role: "assistant", Content: "DONE"},     // judge: met
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3)

	tr, err := a.Run(context.Background(), "do part 1 and part 2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "all done" {
		t.Errorf("final = %q, want %q", tr.Final, "all done")
	}
	if tr.Returns != 1 {
		t.Errorf("returns = %d, want 1", tr.Returns)
	}
	if !tr.GoalMet {
		t.Error("GoalMet = false, want true (judge said DONE)")
	}
}

// A tool-less answer never triggers the judge (nothing was done to verify): the loop
// is gated on real progress, so a plain reply finalizes in one Chat call.
func TestRunGoalLoopSkipsToollessAnswer(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{{Role: "assistant", Content: "just an answer"}}}
	a := New(fake, reg, nil, nil, "", 10)
	a.SetGoalLoop(3)

	tr, _ := a.Run(context.Background(), "hi")
	if tr.Returns != 0 {
		t.Errorf("returns = %d, want 0 (no judge for a tool-less answer)", tr.Returns)
	}
	if fake.i != 1 {
		t.Errorf("Chat calls = %d, want 1 (the judge must not run)", fake.i)
	}
}

// When the judge never accepts, the loop stops at the TTL: the last finalize is taken
// as-is (no further judge call) and GoalMet stays false.
func TestRunGoalLoopStopsAtTTL(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	more := llm.Message{Role: "assistant", Content: "MORE: nope"}
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), {Role: "assistant", Content: "f1"}, more, // re-feed 1
		calcCall(), {Role: "assistant", Content: "f2"}, more, // re-feed 2
		calcCall(), {Role: "assistant", Content: "f3"}, // returns==TTL → accept, no judge
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(2)

	tr, _ := a.Run(context.Background(), "loop")
	if tr.Returns != 2 {
		t.Errorf("returns = %d, want 2 (TTL)", tr.Returns)
	}
	if tr.GoalMet {
		t.Error("GoalMet = true, want false (TTL exhausted, never judged done)")
	}
	if tr.Final != "f3" {
		t.Errorf("final = %q, want f3", tr.Final)
	}
}

// The judge must NOT rubber-stamp an unparseable reply as a met goal: it accepts
// the final (so it doesn't trap the loop) but leaves GoalMet false.
func TestRunGoalLoopUnclearJudgeDoesNotMarkMet(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(),
		{Role: "assistant", Content: "I think that's everything"}, // finalize
		{Role: "assistant", Content: "hmm, hard to say really"},   // judge: no DONE/MORE token
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3)

	tr, _ := a.Run(context.Background(), "do the thing")
	if tr.GoalMet {
		t.Error("GoalMet = true on an unparseable judge verdict, want false")
	}
	if tr.Returns != 0 {
		t.Errorf("returns = %d, want 0 (unclear verdict accepts, doesn't re-feed)", tr.Returns)
	}
	if tr.Final != "I think that's everything" {
		t.Errorf("final = %q", tr.Final)
	}
}

// A model that double-encodes params as a JSON string AND puts a param at the top
// level (e.g. path) must not lose the top-level one.
func TestParseArgsStringParamsKeepsSiblings(t *testing.T) {
	action, params := parseArgs(`{"action":"write","path":"x.txt","params":"{\"content\":\"y\"}"}`)
	if action != "write" {
		t.Errorf("action = %q, want write", action)
	}
	if params["path"] != "x.txt" {
		t.Errorf("top-level path lost: %+v", params)
	}
	if params["content"] != "y" {
		t.Errorf("content = %v, want y", params["content"])
	}
}

func TestParseVerdict(t *testing.T) {
	cases := []struct {
		in       string
		want     judgeVerdict
		wantMiss string
	}{
		{"DONE", judgeDone, ""},
		{"The task is DONE.", judgeDone, ""},
		{"MORE: add the tests", judgeMore, "add the tests"},
		{"The plan needs MORE work: no tests yet", judgeMore, "work: no tests yet"},
		{"MOREOVER, it looks done", judgeDone, ""}, // MOREOVER must not match MORE
		{"yeah looks fine to me", judgeUnclear, ""},
		{"", judgeUnclear, ""},
	}
	for _, tc := range cases {
		got, miss := parseVerdict(tc.in)
		if got != tc.want || miss != tc.wantMiss {
			t.Errorf("parseVerdict(%q) = (%d,%q), want (%d,%q)", tc.in, got, miss, tc.want, tc.wantMiss)
		}
	}
}

// After the rethink nudge, a single further unproductive turn stops the run — the
// counter isn't reset on the nudge, so a degenerate model (which may think for
// minutes per turn) can't flail for another full budget before stopping.
func TestStuckStopsOneTurnAfterNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	fake := &scriptLLM{replies: []llm.Message{bad, bad, bad, bad, bad, bad}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 30)

	tr, _ := a.Run(context.Background(), "do x")
	if !tr.Stopped || !strings.Contains(tr.Final, "Stopped") {
		t.Fatalf("want a stuck-stop, got stopped=%v final=%q", tr.Stopped, tr.Final)
	}
	if tr.Steps != 4 { // 3 turns to the nudge, 1 more dud turn to stop
		t.Errorf("steps = %d, want 4", tr.Steps)
	}
	nudges := 0
	for _, k := range rt.kinds {
		if k == "nudge" {
			nudges++
		}
	}
	if nudges != 1 {
		t.Errorf("nudges = %d, want 1", nudges)
	}
}

// An empty-action error must stay a single clean line — no full schema dump, no
// learned hints piled on (that buries the example for a weak model).
func TestEmptyActionErrorStaysTerse(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{Domain: "calc", ErrorPattern: "no action given", Context: "calc", ProvenFix: "provide an action"})
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c", "calc", `{"action":"","params":{}}`),
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, reg, kb, nil, "", 5)

	tr, _ := a.Run(context.Background(), "x")
	obs := toolObservation(tr.Messages)
	if len(obs) == 0 {
		t.Fatal("no observation")
	}
	c := obs[0].Content
	if !strings.Contains(c, "no action given") {
		t.Errorf("want the no-action message, got: %s", c)
	}
	if strings.Contains(c, "usage:") || strings.Contains(c, "Hints from past runs") {
		t.Errorf("empty-action error should stay terse, got:\n%s", c)
	}
}

func TestRunWrongToolHint(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc(), tool.NewWeb(nil, false))
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
