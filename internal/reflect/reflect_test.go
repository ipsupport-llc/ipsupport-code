package reflect

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
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

type promptCapture struct{ system string }

func (p *promptCapture) Chat(_ context.Context, msgs []llm.Message, _ []map[string]any) (llm.Message, error) {
	for _, m := range msgs {
		if m.Role == "system" {
			p.system = m.Content
		}
	}
	return llm.Message{Role: "assistant", Content: `{"facts":["x"]}`}, nil
}

func TestReflectLiteUsesFactsOnlyPrompt(t *testing.T) {
	full, lite := &promptCapture{}, &promptCapture{}
	New(full).Reflect(context.Background(), sampleTranscript())
	r := New(lite)
	r.Lite = true
	r.Reflect(context.Background(), sampleTranscript())

	if !strings.Contains(full.system, "pitfalls") {
		t.Error("the full prompt should ask for pitfalls")
	}
	if strings.Contains(lite.system, "pitfalls") {
		t.Error("the lite prompt must NOT ask for pitfalls (facts only)")
	}
	if !strings.Contains(lite.system, "facts") {
		t.Error("the lite prompt should ask for facts")
	}
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
