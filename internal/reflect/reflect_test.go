package reflect

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

type fixedLLM struct {
	reply string
	err   error
}

func (f fixedLLM) Chat(_ context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	if f.err != nil {
		return llm.Message{}, f.err
	}
	return llm.Message{Role: "assistant", Content: f.reply}, nil
}

func sampleTranscript() agent.Transcript {
	return agent.Transcript{
		Messages: []llm.Message{
			llm.User("do x"),
			{Role: "tool", Name: "run", Content: "exit 1 permission denied"},
		},
		Final: "done",
	}
}

func TestReflectParsesLessons(t *testing.T) {
	reply := "Here is what I learned:\n" +
		`{"pitfalls":[{"domain":"run","error_pattern":"permission denied","context":"writing to /root","proven_fix":"use sudo"}],` +
		`"facts":["build with: go build ./...","tests live in internal/*_test.go"]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 || l.Pitfalls[0].Domain != "run" || l.Pitfalls[0].ProvenFix != "use sudo" {
		t.Errorf("pitfalls = %+v", l.Pitfalls)
	}
	if len(l.Facts) != 2 || l.Facts[0] != "build with: go build ./..." {
		t.Errorf("facts = %+v", l.Facts)
	}
}

// A pitfall whose domain isn't a real tool name can never be matched by the KB, so
// it's dropped; a valid one is kept (and its domain lower-cased).
func TestReflectDropsUnknownDomain(t *testing.T) {
	reply := `{"pitfalls":[` +
		`{"domain":"shell","error_pattern":"x","context":"c","proven_fix":"do y"},` +
		`{"domain":"File","error_pattern":"z","context":"c","proven_fix":"keep z"}` +
		`],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 || l.Pitfalls[0].Domain != "file" || l.Pitfalls[0].ProvenFix != "keep z" {
		t.Errorf("pitfalls = %+v, want only the valid file lesson (domain lower-cased)", l.Pitfalls)
	}
}

// "exit 1" is the run tool's own generic wrapper prefix on EVERY failed
// command (see run.go's `fmt.Sprintf("exit %d\n%s", ...)`) — a lesson keyed on
// it alone can't discriminate this failure from any other, so it's dropped
// even though it otherwise has a valid domain and a proven fix. A pattern
// that's actually specific to the failure is kept.
func TestReflectDropsGenericExitCodePattern(t *testing.T) {
	reply := `{"pitfalls":[` +
		`{"domain":"run","error_pattern":"exit 1","context":"a python module missing a dependency","proven_fix":"install the required system library"},` +
		`{"domain":"run","error_pattern":"go.mod already exists","context":"go mod init on an existing module","proven_fix":"skip init, cd into the existing module"}` +
		`],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 || l.Pitfalls[0].ErrorPattern != "go.mod already exists" {
		t.Errorf("pitfalls = %+v, want only the specific (non-generic) pattern kept", l.Pitfalls)
	}
}

// Reported live, from a real debug log: a stored lesson read "Provide proper
// path parameter: {\"path\": \"nemotron-extreme-quant/PLAN.md\", ...}" and
// surfaced — in a DIFFERENT project — on a failure whose real cause was the
// encoding of the params blob, not the path. The model quoted that hint back in
// its own reasoning and kept retrying the wrong thing. The prompt forbids
// carrying a value from the run into a lesson; a model that ignores it must not
// be able to poison the store anyway.
func TestReflectDropsLessonsCarryingAPathFromThisRun(t *testing.T) {
	reply := `{"pitfalls":[` +
		`{"domain":"file","error_pattern":"you gave {}","context":"file: write","proven_fix":"Provide proper path parameter: {\"path\": \"nemotron-extreme-quant/PLAN.md\", \"content\": \"...\"}"},` +
		`{"domain":"file","error_pattern":"missing required param(s): path","context":"file: write","proven_fix":"send params as a real JSON object, not a JSON-encoded string"}` +
		`],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 {
		t.Fatalf("pitfalls = %+v, want only the project-neutral lesson kept", l.Pitfalls)
	}
	if strings.Contains(l.Pitfalls[0].ProvenFix, "nemotron") {
		t.Errorf("kept the lesson quoting another project's path: %+v", l.Pitfalls[0])
	}
}

// A run that hit an error and never recovered still has the most useful lesson in
// it — "this approach never worked". kind "avoid" records that; it must survive
// parsing, because rendering it as a fix that worked would push the next model
// straight back into the loop the lesson exists to break.
func TestReflectKeepsAvoidKind(t *testing.T) {
	reply := `{"pitfalls":[` +
		`{"domain":"file","kind":"avoid","error_pattern":"you gave {}","context":"file: write","proven_fix":"send params as a JSON object, never a JSON-encoded string"},` +
		`{"domain":"run","kind":"fix","error_pattern":"permission denied","context":"run: shell","proven_fix":"use sudo"},` +
		`{"domain":"web","error_pattern":"certificate expired","context":"web: get","proven_fix":"use http"}` +
		`],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 3 {
		t.Fatalf("pitfalls = %+v, want all three", l.Pitfalls)
	}
	if l.Pitfalls[0].Kind != knowledge.KindAvoid {
		t.Errorf("kind = %q, want %q", l.Pitfalls[0].Kind, knowledge.KindAvoid)
	}
	// "fix" and an omitted kind both mean the original default. Erring toward
	// "a fix that worked" is deliberate: a dead end mislabeled as a fix reads as
	// bad advice, but a fix mislabeled as a dead end tells the model to stop
	// doing the thing that actually works.
	for _, i := range []int{1, 2} {
		if l.Pitfalls[i].Kind != "" {
			t.Errorf("pitfall %d kind = %q, want \"\" (a fix)", i, l.Pitfalls[i].Kind)
		}
	}
}

func TestReflectParsesObjectAfterBraceProse(t *testing.T) {
	// Prose contains a braced phrase that isn't JSON before the real object.
	reply := "The lessons {for this run} are below:\n" +
		`{"pitfalls":[{"domain":"run","error_pattern":"permission denied","context":"writing /root","proven_fix":"use sudo"}],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 || l.Pitfalls[0].ProvenFix != "use sudo" {
		t.Errorf("pitfalls = %+v, want the real object parsed past the prose braces", l.Pitfalls)
	}
}

func TestReflectFactsOnly(t *testing.T) {
	reply := `{"pitfalls":[],"facts":["run the server with: make dev"]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 0 || len(l.Facts) != 1 || l.Facts[0] != "run the server with: make dev" {
		t.Errorf("lessons = %+v, want facts-only", l)
	}
}

func TestReflectSkipsDecoyObject(t *testing.T) {
	// A model that echoes a format example (empty object) before the real lessons
	// must not lose the real ones to the decoy.
	reply := "Format reminder: {\"pitfalls\": [], \"facts\": []}\nHere is what I learned:\n" +
		`{"pitfalls":[],"facts":["build with: go build ./..."]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Facts) != 1 || l.Facts[0] != "build with: go build ./..." {
		t.Errorf("decoy object won instead of the real lessons: %+v", l)
	}
}

func TestReflectNoJSONIsEmpty(t *testing.T) {
	l, err := New(fixedLLM{reply: "I see nothing durable to learn here."}).
		Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 0 || len(l.Facts) != 0 {
		t.Errorf("want no lessons, got %+v", l)
	}
}

type promptCapture struct{ systems []string }

func (p *promptCapture) Chat(_ context.Context, msgs []llm.Message, _ []map[string]any) (llm.Message, error) {
	for _, m := range msgs {
		if m.Role == "system" {
			p.systems = append(p.systems, m.Content)
		}
	}
	return llm.Message{Role: "assistant", Content: `{"facts":["x"]}`}, nil
}

func (p *promptCapture) asked(what string) bool {
	for _, s := range p.systems {
		if strings.Contains(s, what) {
			return true
		}
	}
	return false
}

// The full prompt asks for both halves in one call. Lite splits that into two
// single-purpose calls — it must still ASK for pitfalls.
//
// Lite used to mean facts only, on the reasoning that a weak model loops on the
// combined ask. Reported live, with the store to prove it: a whole session on a
// local model produced facts and not one lesson, because the lite prompt never
// mentioned pitfalls at all — the lesson mechanism was disabled for exactly the
// models it exists for. A weak model repeating the same broken tool call every
// run is the entire reason the store is there.
func TestReflectLiteStillAsksForPitfallsJustSeparately(t *testing.T) {
	full, lite := &promptCapture{}, &promptCapture{}
	New(full).Reflect(context.Background(), sampleTranscript())
	r := New(lite)
	r.Lite = true
	r.Reflect(context.Background(), sampleTranscript())

	if !full.asked("pitfalls") || len(full.systems) != 1 {
		t.Errorf("full mode: %d call(s), want one combined ask including pitfalls", len(full.systems))
	}
	if !lite.asked("pitfalls") {
		t.Error("lite mode never asks for pitfalls — the lesson store can never fill on a local model")
	}
	if !lite.asked("facts") {
		t.Error("lite mode stopped asking for facts")
	}
	// Two calls, not one: each has a single output shape, which is the whole
	// point of the split for a model that loops on the combined ask.
	if len(lite.systems) != 2 {
		t.Errorf("lite made %d call(s), want 2 (facts and pitfalls asked separately)", len(lite.systems))
	}
	for _, s := range lite.systems {
		if strings.Contains(s, "pitfalls") && strings.Contains(s, `{"facts"`) {
			t.Errorf("a lite call asks for both shapes at once:\n%s", s)
		}
	}
}

// One weak-model reply coming back unusable must not throw away the other half
// that parsed fine — half the lessons beats none.
func TestReflectLiteKeepsTheHalfThatWorked(t *testing.T) {
	r := New(&flakyHalfLLM{})
	r.Lite = true
	l, err := r.Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 || l.Pitfalls[0].ProvenFix != "use sudo" {
		t.Errorf("pitfalls = %+v, want the half that succeeded kept", l.Pitfalls)
	}
}

// flakyHalfLLM fails the first (facts) call and answers the second (pitfalls).
type flakyHalfLLM struct{ n int }

func (f *flakyHalfLLM) Chat(_ context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	f.n++
	if f.n == 1 {
		return llm.Message{}, errors.New("boom")
	}
	return llm.Message{Role: "assistant", Content: `{"pitfalls":[{"domain":"run","kind":"fix","error_pattern":"permission denied","context":"run: shell","proven_fix":"use sudo"}]}`}, nil
}

type recordingLLM struct{ called bool }

func (r *recordingLLM) Chat(_ context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	r.called = true
	return llm.Message{Role: "assistant", Content: "{}"}, nil
}

func TestReflectSkipsWithoutToolUse(t *testing.T) {
	r := &recordingLLM{}
	transcript := agent.Transcript{Messages: []llm.Message{
		llm.User("hi"),
		{Role: "assistant", Content: "Hi there!"},
	}}
	l, err := New(r).Reflect(context.Background(), transcript)
	if err != nil {
		t.Fatal(err)
	}
	if r.called {
		t.Error("reflection called the model for a chat turn with no tool use")
	}
	if len(l.Pitfalls) != 0 || len(l.Facts) != 0 {
		t.Errorf("lessons = %+v, want none", l)
	}
}

// oneLine must clip on a rune boundary so a long line of multi-byte runes never
// writes a broken trailing rune into the knowledge base or the trace dataset.
func TestOneLineRuneSafe(t *testing.T) {
	got := oneLine(strings.Repeat("é", 600)) // 2 bytes each → 500-byte cap lands mid-rune
	if !utf8.ValidString(got) {
		t.Errorf("oneLine produced invalid UTF-8: %q", got)
	}
}

func TestReflectTransportError(t *testing.T) {
	_, err := New(fixedLLM{err: errors.New("boom")}).
		Reflect(context.Background(), sampleTranscript())
	var re *ReflectionError
	if !errors.As(err, &re) {
		t.Errorf("err = %v, want *ReflectionError", err)
	}
}

// The run that most needs an "avoid" lesson is the one the harness STOPPED after
// repeated failures — and that was the one transcript excluded from reflection,
// which made the whole lesson kind unreachable by construction.
func TestReflectMinesAStoppedRunForOneAvoidLesson(t *testing.T) {
	stopped := sampleTranscript()
	stopped.Stopped = true
	reply := `{"pitfalls":[` +
		`{"domain":"file","error_pattern":"you gave {}","context":"file: write","proven_fix":"send params as a JSON object"},` +
		`{"domain":"run","error_pattern":"permission denied","context":"run: shell","proven_fix":"ask the user"}` +
		`],"facts":["this project uses go 1.27"]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), stopped)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(l.Pitfalls) != 1 {
		t.Fatalf("pitfalls = %+v, want exactly one from a stopped run", l.Pitfalls)
	}
	if l.Pitfalls[0].Kind != knowledge.KindAvoid {
		t.Errorf("kind = %q, want %q — nothing in a stopped run was proven to work", l.Pitfalls[0].Kind, knowledge.KindAvoid)
	}
	// A run that ended in repeated failure is the least trustworthy source of
	// "durable truths about this project" there is.
	if len(l.Facts) != 0 {
		t.Errorf("facts = %v, want none from a stopped run", l.Facts)
	}
}

// The stopped-run prompt must ask only for what the transcript SHOWS failing
// repeatedly, and say outright that a guess is worse than nothing — the lesson
// it produces is injected exactly when the model is next struggling.
func TestStoppedRunPromptRefusesToGuess(t *testing.T) {
	stopped := sampleTranscript()
	stopped.Stopped = true
	p := &promptCapture{}
	New(p).Reflect(context.Background(), stopped)
	if len(p.systems) != 1 {
		t.Fatalf("made %d call(s), want 1 narrow pass", len(p.systems))
	}
	sys := p.systems[0]
	for _, want := range []string{"failed more than once", "worse than no lesson"} {
		if !strings.Contains(sys, want) {
			t.Errorf("stopped-run prompt missing %q:\n%s", want, sys)
		}
	}
	if strings.Contains(sys, `"facts"`) {
		t.Error("the stopped-run pass still asks for facts")
	}
}

// Everything the harness injects arrives as role "user". Labelling it "GOAL:"
// told the learning pass our own nudge text was the user's intent — and a
// distilled "fact" could come back reading like our own scaffolding.
func TestSummarizeDoesNotPassOffHarnessTextAsTheGoal(t *testing.T) {
	tr := agent.Transcript{Messages: []llm.Message{
		llm.User("fix the parser"),
		{Role: "tool", Name: "file", Content: "ok"},
		llm.User(agent.GoalReturnForTest("fix the parser", "no tests")),
	}}
	// Counted by LINE: a re-feed restates the goal inside its own body, which is
	// fine — what matters is that the line isn't introduced as the user's ask.
	goals, harness := 0, 0
	for _, line := range strings.Split(summarize(tr), "\n") {
		switch {
		case strings.HasPrefix(line, "GOAL:"):
			goals++
		case strings.HasPrefix(line, "HARNESS"):
			harness++
		}
	}
	if goals != 1 {
		t.Errorf("%d lines presented as the user's goal, want 1:\n%s", goals, summarize(tr))
	}
	if harness != 1 {
		t.Errorf("%d lines labelled as ours, want 1:\n%s", harness, summarize(tr))
	}
}

// The judge's verdict is the sharpest signal in a run, and the learning pass
// never saw it although it sat on the very struct it was handed.
func TestSummarizeCarriesTheJudgesVerdict(t *testing.T) {
	tr := sampleTranscript()
	tr.Returns, tr.GoalMet, tr.Missing = 3, false, "the report has no test results"
	out := summarize(tr)
	if !strings.Contains(out, "JUDGE:") || !strings.Contains(out, "no test results") {
		t.Errorf("the judge's verdict never reaches reflection:\n%s", out)
	}
}

// summarize had no total budget while judgeEvidence capped itself — a long run
// produced a prompt that overran the small local model the lite path targets,
// returned unparseable output, and made the whole pass buy nothing.
func TestSummarizeIsBounded(t *testing.T) {
	tr := agent.Transcript{}
	for i := 0; i < 400; i++ {
		tr.Messages = append(tr.Messages,
			llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{Name: "run", Arguments: strings.Repeat("x", 400)}}},
			llm.Message{Role: "tool", Name: "run", Content: strings.Repeat("y", 400)})
	}
	tr.Final = "THE FINAL ANSWER"
	out := summarize(tr)
	if len(out) > summaryBudget*2 {
		t.Errorf("summary is %d bytes, want it bounded near %d", len(out), summaryBudget)
	}
	// The END of a run is kept: that is where the recovery and the outcome are.
	if !strings.Contains(out, "THE FINAL ANSWER") {
		t.Error("the bound dropped the run's own conclusion")
	}
}

// "The model looked and found nothing" and "we could not read what it said" were
// indistinguishable — both surfaced as an empty Lessons, and neither was logged.
// They need opposite fixes, so they must be told apart.
func TestParsedSeparatesNothingToLearnFromUnreadable(t *testing.T) {
	// A well-formed empty answer IS an answer.
	l, err := New(fixedLLM{reply: `{"pitfalls":[],"facts":[]}`}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatal(err)
	}
	if !l.Parsed {
		t.Error("an explicit empty result reads as unreadable")
	}
	if l.Reply != "" {
		t.Errorf("Reply = %q, want empty when the answer was understood", l.Reply)
	}

	// Prose with no JSON in it at all is not.
	l, err = New(fixedLLM{reply: "I think everything went fine, nothing to add really."}).
		Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatal(err)
	}
	if l.Parsed {
		t.Error("unreadable prose reads as a parsed empty result")
	}
	if !strings.Contains(l.Reply, "nothing to add") {
		t.Errorf("Reply = %q, want the tail of what came back so the log can show it", l.Reply)
	}
}

// A real result is parsed, obviously — but the flag must say so, since the
// caller logs it on every pass.
func TestParsedIsSetOnARealResult(t *testing.T) {
	reply := `{"pitfalls":[{"domain":"run","error_pattern":"permission denied","context":"run: shell","proven_fix":"use sudo"}],"facts":[]}`
	l, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatal(err)
	}
	if !l.Parsed || len(l.Pitfalls) != 1 {
		t.Errorf("lessons = %+v, parsed=%v", l.Pitfalls, l.Parsed)
	}
}
