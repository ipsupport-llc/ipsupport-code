package reflect

import (
	"context"
	"errors"
	"testing"

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

func TestReflectTransportError(t *testing.T) {
	_, err := New(fixedLLM{err: errors.New("boom")}).
		Reflect(context.Background(), sampleTranscript())
	var re *ReflectionError
	if !errors.As(err, &re) {
		t.Errorf("err = %v, want *ReflectionError", err)
	}
}
