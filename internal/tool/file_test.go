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

func TestFileSearch(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "a.go", "content": "package main\nfunc Foo() {}\n"})
	tl.Call(ctx, "write", map[string]any{"path": "sub/b.go", "content": "// calls Foo here\nvar x = 1\n"})

	r := tl.Call(ctx, "search", map[string]any{"query": "Foo"})
	if r.IsError {
		t.Fatalf("search: %s", r.Content)
	}
	if !strings.Contains(r.Content, "a.go:2:") || !strings.Contains(r.Content, "sub/b.go:1:") {
		t.Errorf("search missed matches:\n%s", r.Content)
	}
	if r2 := tl.Call(ctx, "search", map[string]any{"query": "zzz-nope"}); !strings.Contains(r2.Content, "no matches") {
		t.Errorf("expected no-matches, got: %s", r2.Content)
	}
}

func TestFileSearchSkipsSymlinks(t *testing.T) {
	dir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt") // outside the jail
	if err := os.WriteFile(outside, []byte("TOPSECRET_TOKEN\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "link.txt")); err != nil {
		t.Skipf("symlinks unsupported here: %v", err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	r := tl.Call(context.Background(), "search", map[string]any{"query": "TOPSECRET_TOKEN"})
	// A real hit would be a "link.txt:1: …" line; the no-match message echoes the
	// query, so assert the symlink simply wasn't matched.
	if strings.Contains(r.Content, "link.txt") {
		t.Errorf("search followed a symlink out of the jail:\n%s", r.Content)
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

func TestFileReadWindow(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "l1\nl2\nl3\nl4\nl5"})
	r := tl.Call(ctx, "read", map[string]any{"path": "f.txt", "offset": 2, "limit": 2})
	if r.IsError || !strings.Contains(r.Content, "l2\nl3") {
		t.Errorf("windowed read = %q, want l2,l3", r.Content)
	}
	if strings.Contains(r.Content, "l5") {
		t.Errorf("window should not include l5: %q", r.Content)
	}
	if !strings.Contains(r.Content, "lines 2") || !strings.Contains(r.Content, "of 5") {
		t.Errorf("missing range header: %q", r.Content)
	}
}

func TestFileFind(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "a.go", "content": "x"})
	tl.Call(ctx, "write", map[string]any{"path": "sub/b.go", "content": "x"})
	tl.Call(ctx, "write", map[string]any{"path": "c.txt", "content": "x"})
	r := tl.Call(ctx, "find", map[string]any{"pattern": "**/*.go"})
	if r.IsError || !strings.Contains(r.Content, "a.go") || !strings.Contains(r.Content, "sub/b.go") {
		t.Errorf("find **/*.go = %q", r.Content)
	}
	if strings.Contains(r.Content, "c.txt") {
		t.Errorf("find should not match c.txt: %q", r.Content)
	}
}

func TestFileMultiEdit(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "alpha beta gamma"})
	r := tl.Call(ctx, "edit", map[string]any{"path": "f.txt",
		"edits": `[{"find":"alpha","replace":"A"},{"find":"gamma","replace":"G"}]`})
	if r.IsError {
		t.Fatalf("multi-edit: %s", r.Content)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "f.txt")); string(data) != "A beta G" {
		t.Errorf("multi-edit result = %q, want 'A beta G'", data)
	}
}

func TestFileEditReplaceAll(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "x x x"})
	tl.Call(ctx, "edit", map[string]any{"path": "f.txt", "find": "x", "replace": "y", "replace_all": true})
	if data, _ := os.ReadFile(filepath.Join(dir, "f.txt")); string(data) != "y y y" {
		t.Errorf("replace_all = %q, want 'y y y'", data)
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
