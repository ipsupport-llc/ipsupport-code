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
	reply := "Here are the lessons:\n" +
		`[{"domain":"run","error_pattern":"permission denied","context":"writing to /root","proven_fix":"use sudo"}]`
	ps, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(ps) != 1 || ps[0].Domain != "run" || ps[0].ProvenFix != "use sudo" {
		t.Errorf("lessons = %+v", ps)
	}
}

func TestReflectParsesArrayAfterBracketProse(t *testing.T) {
	// Prose contains a bracketed phrase before the real JSON array.
	reply := "The lessons [for run] are below:\n" +
		`[{"domain":"run","error_pattern":"permission denied","context":"writing /root","proven_fix":"use sudo"}]`
	ps, err := New(fixedLLM{reply: reply}).Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(ps) != 1 || ps[0].ProvenFix != "use sudo" {
		t.Errorf("lessons = %+v, want the real array parsed past the prose brackets", ps)
	}
}

func TestReflectNoJSONIsEmpty(t *testing.T) {
	ps, err := New(fixedLLM{reply: "I see nothing durable to learn here."}).
		Reflect(context.Background(), sampleTranscript())
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(ps) != 0 {
		t.Errorf("want no lessons, got %+v", ps)
	}
}

type recordingLLM struct{ called bool }

func (r *recordingLLM) Chat(_ context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	r.called = true
	return llm.Message{Role: "assistant", Content: "[]"}, nil
}

func TestReflectSkipsWithoutToolUse(t *testing.T) {
	r := &recordingLLM{}
	transcript := agent.Transcript{Messages: []llm.Message{
		llm.User("привет"),
		{Role: "assistant", Content: "Привет!"},
	}}
	lessons, err := New(r).Reflect(context.Background(), transcript)
	if err != nil {
		t.Fatal(err)
	}
	if r.called {
		t.Error("reflection called the model for a chat turn with no tool use")
	}
	if len(lessons) != 0 {
		t.Errorf("lessons = %v, want none", lessons)
	}
}

// oneLine must clip on a rune boundary so a long Cyrillic transcript line never
// writes a broken trailing rune into the knowledge base or the trace dataset.
func TestOneLineRuneSafe(t *testing.T) {
	got := oneLine(strings.Repeat("я", 600)) // 2 bytes each → 500-byte cap lands mid-rune
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
