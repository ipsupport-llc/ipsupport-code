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

	r := NewHelp(kb).Call(context.Background(), "lessons", map[string]any{"domain": "run"})
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "permission denied") || !strings.Contains(r.Content, "command not found") {
		t.Errorf("lessons = %q, want both pitfalls", r.Content)
	}
}

func TestHelpUnknownDomain(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	r := NewHelp(kb).Call(context.Background(), "lessons", map[string]any{"domain": "file"})
	if r.IsError || !strings.Contains(r.Content, "no lessons") {
		t.Errorf("unknown domain = %+v, want 'no lessons' message", r)
	}
}
