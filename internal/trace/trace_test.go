package trace

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileTracerJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "traces.jsonl") // exercises mkdir
	tr, err := NewFileTracer(path, "run1")
	if err != nil {
		t.Fatalf("NewFileTracer: %v", err)
	}
	tr.Emit("goal", map[string]any{"text": "do x"})
	tr.Emit("final", map[string]any{"text": "done"})
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines = %d, want 2", len(lines))
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["kind"] != "goal" || rec["run"] != "run1" || rec["text"] != "do x" {
		t.Errorf("record = %v", rec)
	}
	if rec["time"] == nil {
		t.Error("record missing time")
	}
}

// TestFileTracerPermissions ensures the trace file — which records prompts,
// tool-call arguments, and file-read observations — is created owner-only
// (0600), not group/world readable.
func TestFileTracerPermissions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "traces.jsonl")
	tr, err := NewFileTracer(path, "run1")
	if err != nil {
		t.Fatalf("NewFileTracer: %v", err)
	}
	defer tr.Close()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("trace file perm = %o, want 0600", perm)
	}
}
