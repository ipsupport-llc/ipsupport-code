package tool

import (
	"context"
	"os"
	"path/filepath"
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

// Creating an empty file (e.g. a Python __init__.py) must work — write with no
// content, not an error. Previously content was required and "" rejected, which
// trapped the model in a retry loop.
func TestFileWriteEmpty(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	r := tl.Call(context.Background(), "write", map[string]any{"path": "pkg/__init__.py"})
	if r.IsError {
		t.Fatalf("empty write should succeed, got: %s", r.Content)
	}
	data, err := os.ReadFile(filepath.Join(dir, "pkg", "__init__.py"))
	if err != nil || len(data) != 0 {
		t.Errorf("file = %q (err %v), want an empty file", data, err)
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

func TestFileEditProducesDiff(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "a.txt", "content": "line1\nline2\nline3\n"})

	r := tl.Call(ctx, "edit", map[string]any{"path": "a.txt", "find": "line2", "replace": "LINE-TWO"})
	if r.IsError {
		t.Fatalf("edit: %s", r.Content)
	}
	if r.Diff == "" {
		t.Fatal("edit produced no diff")
	}
	if !strings.Contains(r.Diff, "-line2") || !strings.Contains(r.Diff, "+") || !strings.Contains(r.Diff, "LINE-TWO") {
		t.Errorf("diff = %q", r.Diff)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "a.txt"))
	if !strings.Contains(string(data), "LINE-TWO") {
		t.Errorf("file not actually edited: %q", data)
	}
}

func TestFileEditFindMissing(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "a.txt", "content": "hello"})

	r := tl.Call(ctx, "edit", map[string]any{"path": "a.txt", "find": "nope", "replace": "x"})
	if !r.IsError || !strings.Contains(r.Content, "not present") {
		t.Errorf("edit with missing find = %+v", r)
	}
}

func TestFileOverwriteSetsDiff(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "a.txt", "content": "old\n"})

	r := tl.Call(ctx, "write", map[string]any{"path": "a.txt", "content": "new\n"})
	if r.IsError || r.Diff == "" {
		t.Errorf("overwrite of existing file should carry a diff: %+v", r)
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
