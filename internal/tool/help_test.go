package tool

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func TestHelpLessons(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{Domain: "run", ErrorPattern: "permission denied", Context: "shell write to /root", ProvenFix: "use sudo"})
	kb.Add(knowledge.Pitfall{Domain: "run", ErrorPattern: "command not found", Context: "missing binary", ProvenFix: "install it first"})

	r := NewHelp(kb, nil).Call(context.Background(), "lessons", map[string]any{"domain": "run"})
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "permission denied") || !strings.Contains(r.Content, "command not found") {
		t.Errorf("lessons = %q, want both pitfalls", r.Content)
	}
}

func TestHelpUnknownDomain(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	r := NewHelp(kb, nil).Call(context.Background(), "lessons", map[string]any{"domain": "file"})
	if r.IsError || !strings.Contains(r.Content, "No lessons") {
		t.Errorf("unknown domain = %+v, want 'no lessons' message", r)
	}
}

// pitfall.go states the rule outright — "the two must render differently
// wherever a lesson is shown to a model" — and this second display site was
// missed, so every avoid lesson was handed back as "this worked", advising the
// exact approach it was recorded to stop.
func TestHelpLessonsDoNotPresentADeadEndAsAProvenFix(t *testing.T) {
	kb, err := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	if err != nil {
		t.Fatal(err)
	}
	kb.Add(knowledge.Pitfall{Domain: "file", Kind: knowledge.KindAvoid,
		ErrorPattern: "you gave {}", Context: "file: write",
		ProvenFix: "send params as a JSON object, never a JSON-encoded string"})
	kb.Add(knowledge.Pitfall{Domain: "file", ErrorPattern: "no such file",
		Context: "file: read", ProvenFix: "create the parent directory first"})

	res := NewHelp(kb, nil).Call(context.Background(), "lessons", map[string]any{"domain": "file"})
	if res.IsError {
		t.Fatalf("help.lessons errored: %s", res.Content)
	}
	for _, line := range strings.Split(res.Content, "\n") {
		if strings.Contains(line, "never a JSON-encoded string") && strings.Contains(line, "this worked") {
			t.Errorf("a dead end is rendered as a proven fix:\n%s", line)
		}
	}
	if !strings.Contains(res.Content, "did NOT work") {
		t.Errorf("the dead end isn't distinguished at all:\n%s", res.Content)
	}
	// A real fix keeps its original wording.
	if !strings.Contains(res.Content, "this worked: create the parent directory first") {
		t.Errorf("a proven fix lost its wording:\n%s", res.Content)
	}
}
