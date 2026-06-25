package tool

import (
	"context"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

// approverFunc adapts a function to the Approver interface for tests.
type approverFunc func(kind, detail string) bool

func (f approverFunc) Approve(kind, detail string) bool { return f(kind, detail) }

func yes() Approver { return approverFunc(func(_, _ string) bool { return true }) }
func no() Approver  { return approverFunc(func(_, _ string) bool { return false }) }

func fileToolFor(t *testing.T, dir, def string, ap Approver) Tool {
	t.Helper()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: def, Jail: "."}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	return NewFile(e, ap)
}

func TestFileWriteThenRead(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "allow", yes())
	ctx := context.Background()

	w := tl.Call(ctx, "write", map[string]any{"path": "a/b.txt", "content": "hello"})
	if w.IsError {
		t.Fatalf("write: %s", w.Content)
	}
	r := tl.Call(ctx, "read", map[string]any{"path": "a/b.txt"})
	if r.IsError || r.Content != "hello" {
		t.Errorf("read = %+v, want hello", r)
	}
}

func TestFileJailEscape(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "allow", yes())
	r := tl.Call(context.Background(), "write", map[string]any{"path": "../evil.txt", "content": "x"})
	if !r.IsError {
		t.Errorf("write outside jail should fail, got %+v", r)
	}
}

func TestFileAskDeniedByUser(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "ask", no())
	r := tl.Call(context.Background(), "write", map[string]any{"path": "x.txt", "content": "x"})
	if !r.IsError || !strings.Contains(r.Content, "denied by user") {
		t.Errorf("ask+deny = %+v, want 'denied by user'", r)
	}
}

func TestFileList(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "notes/todo.txt", "content": "x"})

	r := tl.Call(ctx, "list", map[string]any{"path": "notes"})
	if r.IsError || !strings.Contains(r.Content, "todo.txt") {
		t.Errorf("list = %+v, want to include todo.txt", r)
	}
}
