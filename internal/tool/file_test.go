package tool

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

// approverFunc adapts a function to the Approver interface for tests.
type approverFunc func(kind, detail string) bool

func (f approverFunc) Approve(_ context.Context, kind, detail string) bool { return f(kind, detail) }

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
	return NewFile(e, ap, nil)
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

// The edit action must accept every shape a weak model reaches for: a native JSON
// array, a stringified array, a single object, and top-level find/replace. The
// live failure was a native array being fmt.Sprint'd into Go syntax and rejected.
func TestFileEditAcceptsAllShapes(t *testing.T) {
	ctx := context.Background()
	seed := "alpha beta gamma\n"

	cases := []struct {
		name  string
		edits map[string]any // the params minus path
		want  string
	}{
		{"top-level find/replace", map[string]any{"find": "alpha", "replace": "A"}, "A beta gamma\n"},
		{"stringified array", map[string]any{"edits": `[{"find":"alpha","replace":"A"},{"find":"gamma","replace":"G"}]`}, "A beta G\n"},
		{"native array", map[string]any{"edits": []any{
			map[string]any{"find": "alpha", "replace": "A"},
			map[string]any{"find": "gamma", "replace": "G"},
		}}, "A beta G\n"},
		{"single object", map[string]any{"edits": map[string]any{"find": "beta", "replace": "B"}}, "alpha B gamma\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tl := fileToolFor(t, dir, "allow", yes())
			if w := tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": seed}); w.IsError {
				t.Fatal(w.Content)
			}
			params := map[string]any{"path": "f.txt"}
			for k, v := range tc.edits {
				params[k] = v
			}
			if r := tl.Call(ctx, "edit", params); r.IsError {
				t.Fatalf("edit (%s) errored: %s", tc.name, r.Content)
			}
			got, _ := os.ReadFile(filepath.Join(dir, "f.txt"))
			if string(got) != tc.want {
				t.Errorf("result = %q, want %q", got, tc.want)
			}
		})
	}
}

// An edit call with only {path} must fail early with a clear message — before the
// approval prompt / snapshot / read — not a late "empty find in edit #1".
func TestFileEditMissingFindFailsEarly(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "x"})

	r := tl.Call(ctx, "edit", map[string]any{"path": "f.txt"})
	if !r.IsError {
		t.Fatal("edit with no find/replace should error")
	}
	if !strings.Contains(r.Content, "find") || !strings.Contains(r.Content, "replace") {
		t.Errorf("error should name find+replace, got: %s", r.Content)
	}
}

// read and search must not surface secret files (.env / *secret*) — otherwise the
// model could slurp credentials and exfiltrate them via the web tool.
func TestFileReadAndSearchSkipSecrets(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "app.go", "content": "TOKEN_MARKER = env"})
	os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN_MARKER=supersecret\n"), 0o644)

	if r := tl.Call(ctx, "read", map[string]any{"path": ".env"}); !r.IsError {
		t.Errorf("read(.env) should be blocked, got: %q", r.Content)
	}
	r := tl.Call(ctx, "search", map[string]any{"query": "TOKEN_MARKER"})
	if r.IsError {
		t.Fatalf("search: %s", r.Content)
	}
	if strings.Contains(r.Content, ".env") || strings.Contains(r.Content, "supersecret") {
		t.Errorf("search surfaced a secret file:\n%s", r.Content)
	}
	if !strings.Contains(r.Content, "app.go") {
		t.Errorf("search should still find the normal file:\n%s", r.Content)
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

// search must skip FIFOs (named pipes) — os.ReadFile on one blocks forever
// without a writer on the other end, and search's ctx is discarded so
// cancellation can't unblock it. A FIFO reports Size() == 0, so only the size
// filter let it through; the fix adds a regular-file check.
func TestFileSearchSkipsFIFO(t *testing.T) {
	dir := t.TempDir()
	if err := mkfifo(filepath.Join(dir, "pipe")); err != nil {
		t.Skipf("mkfifo unsupported here: %v", err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	done := make(chan Result, 1)
	go func() {
		done <- tl.Call(context.Background(), "search", map[string]any{"query": "anything"})
	}()

	select {
	case r := <-done:
		if r.IsError {
			t.Errorf("search errored: %s", r.Content)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("search hung reading a FIFO")
	}
}

// read must reject FIFOs (named pipes) instead of hanging — os.Open itself
// blocks forever on a FIFO with no writer, before any read even starts. This
// is a separate code path from search (its own os.Open call in readPlain),
// so search's fix doesn't cover it.
func TestFileReadSkipsFIFO(t *testing.T) {
	dir := t.TempDir()
	if err := mkfifo(filepath.Join(dir, "pipe")); err != nil {
		t.Skipf("mkfifo unsupported here: %v", err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	done := make(chan Result, 1)
	go func() {
		done <- tl.Call(context.Background(), "read", map[string]any{"path": "pipe"})
	}()

	select {
	case r := <-done:
		if !r.IsError {
			t.Errorf("read of a FIFO should be rejected, got: %s", r.Content)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("read hung opening a FIFO")
	}
}

// Same as TestFileReadSkipsFIFO, but through the windowed (offset/limit) read
// path — readWindow has its own os.Open call, distinct from readPlain's.
func TestFileReadWindowSkipsFIFO(t *testing.T) {
	dir := t.TempDir()
	if err := mkfifo(filepath.Join(dir, "pipe")); err != nil {
		t.Skipf("mkfifo unsupported here: %v", err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	done := make(chan Result, 1)
	go func() {
		done <- tl.Call(context.Background(), "read", map[string]any{"path": "pipe", "offset": 1})
	}()

	select {
	case r := <-done:
		if !r.IsError {
			t.Errorf("windowed read of a FIFO should be rejected, got: %s", r.Content)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("windowed read hung opening a FIFO")
	}
}

// write, edit and append must not hang on a FIFO nobody reads: opening one for
// writing blocks until a reader appears, with the file lock held, wedging the
// tool for good. (append to a pipe something reads still works — see the
// snapshot test in cmd/agent.)
func TestFileWritesRefuseFIFO(t *testing.T) {
	dir := t.TempDir()
	if err := mkfifo(filepath.Join(dir, "pipe")); err != nil {
		t.Skipf("mkfifo unsupported here: %v", err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	for _, c := range []struct {
		action string
		params map[string]any
	}{
		{"write", map[string]any{"path": "pipe", "content": "x"}},
		{"append", map[string]any{"path": "pipe", "content": "x"}},
		{"edit", map[string]any{"path": "pipe", "find": "a", "replace": "b"}},
	} {
		done := make(chan Result, 1)
		go func() { done <- tl.Call(context.Background(), c.action, c.params) }()
		select {
		case r := <-done:
			if !r.IsError {
				t.Errorf("%s to a FIFO should be refused, got: %s", c.action, r.Content)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("%s hung on a FIFO", c.action)
		}
	}
}

// write and edit replace a file whole — a new file renamed over the old — so a
// failure part-way can't leave half of it. The file keeps its permissions and
// nothing is left behind beside it.
func TestFileWriteReplacesWholeAndKeepsMode(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "script.sh")
	if err := os.WriteFile(p, []byte("old\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "script.sh", "content": "new\n"}); r.IsError {
		t.Fatal(r.Content)
	}
	if after, err := os.Stat(p); err == nil && runtime.GOOS != "windows" && os.SameFile(before, after) {
		t.Error("write rewrote the file in place — a failure part-way would leave half of it")
	}
	if r := tl.Call(context.Background(), "edit", map[string]any{"path": "script.sh", "find": "new", "replace": "newer"}); r.IsError {
		t.Fatal(r.Content)
	}
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(p); string(data) != "newer\n" {
		t.Errorf("content = %q", data)
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm() != 0o755 {
		t.Errorf("mode = %v, want 0755 kept", fi.Mode().Perm())
	}
	if ents, _ := os.ReadDir(dir); len(ents) != 1 {
		t.Errorf("left beside it: %v", ents)
	}
}

// A writable file in a directory we can't create in still gets written — in
// place, as before, since there is nowhere to put the new copy.
func TestFileWriteInAReadOnlyDirectory(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs POSIX directory permissions and a non-root user")
	}
	root := t.TempDir()
	dir := filepath.Join(root, "ro")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(p, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o755) })
	tl := fileToolFor(t, root, "allow", yes())
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "ro/f.txt", "content": "new"}); r.IsError {
		t.Fatalf("write failed: %s", r.Content)
	}
	if data, _ := os.ReadFile(p); string(data) != "new" {
		t.Errorf("content = %q", data)
	}
}

// Reading a binary file (a compiled executable, in this case) as "text" used
// to just dump its raw bytes back — huge, useless to the model, and capable
// of corrupting the terminal that renders the tool-call output (embedded
// control/escape bytes). It must be rejected with a clear, actionable error
// instead.
func TestFileReadRejectsBinary(t *testing.T) {
	dir := t.TempDir()
	blob := append([]byte("\x7fELF"), bytes.Repeat([]byte{0, 1, 2, 3}, 100)...)
	if err := os.WriteFile(filepath.Join(dir, "app"), blob, 0o755); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	r := tl.Call(context.Background(), "read", map[string]any{"path": "app"})
	if !r.IsError {
		t.Errorf("read of a binary file should be rejected, got: %s", r.Content)
	}
	if !strings.Contains(r.Content, "binary") {
		t.Errorf("error should explain it's binary, got: %s", r.Content)
	}
}

// Same as TestFileReadRejectsBinary, but through the windowed (offset/limit)
// read path — readWindow has its own os.Open call, distinct from readPlain's,
// but both go through the shared looksBinaryFile check in read() first.
func TestFileReadWindowRejectsBinary(t *testing.T) {
	dir := t.TempDir()
	blob := append([]byte("\x7fELF"), bytes.Repeat([]byte{0, 1, 2, 3}, 100)...)
	if err := os.WriteFile(filepath.Join(dir, "app"), blob, 0o755); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	r := tl.Call(context.Background(), "read", map[string]any{"path": "app", "offset": 1})
	if !r.IsError {
		t.Errorf("windowed read of a binary file should be rejected, got: %s", r.Content)
	}
}

// A plain text file — even one containing high-bit / non-ASCII bytes (UTF-8
// content, e.g. non-English text) — must NOT be flagged as binary. Only a
// genuine NUL byte should trip the heuristic.
func TestFileReadDoesNotFlagUTF8AsBinary(t *testing.T) {
	dir := t.TempDir()
	content := "hello — привет — 日本語\nsecond line\n"
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())

	r := tl.Call(context.Background(), "read", map[string]any{"path": "notes.txt"})
	if r.IsError {
		t.Fatalf("plain UTF-8 text was rejected as binary: %s", r.Content)
	}
	if r.Content != content {
		t.Errorf("content = %q, want %q", r.Content, content)
	}
}

// os.MkdirAll is a silent no-op on an existing directory — mkdir must tell
// the model that explicitly instead of always saying "created", which reads
// as a fresh start and hides the one fact (this already exists) that would
// let a stuck model skip a step it doesn't need to repeat.
func TestFileMkdirReportsAlreadyExists(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()

	r1 := tl.Call(ctx, "mkdir", map[string]any{"path": "rl_hero_go"})
	if r1.IsError || !strings.Contains(r1.Content, "created directory") {
		t.Fatalf("first mkdir = %+v, want a 'created' result", r1)
	}

	r2 := tl.Call(ctx, "mkdir", map[string]any{"path": "rl_hero_go"})
	if r2.IsError {
		t.Errorf("mkdir on an existing directory should not be an error: %+v", r2)
	}
	if !strings.Contains(r2.Content, "already exists") {
		t.Errorf("second mkdir = %q, want it to say the directory already exists", r2.Content)
	}
	if strings.Contains(r2.Content, "created directory") {
		t.Errorf("second mkdir must not claim it was created: %q", r2.Content)
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

// A write's deny-write check (against .env / *secret* / …) only runs once,
// against the path as it resolved BEFORE the (possibly slow, human) approval
// prompt. If the target is swapped to a symlink pointing at a denied file
// WHILE approval is pending, the post-approval Resolve() call follows that
// symlink — and, absent a second deny-write check, the jail-confinement check
// alone lets it through (the real .env is still inside the workspace), so the
// write lands on the protected file. Simulate the swap happening during the
// async approval wait by doing it inside the approver's callback itself.
func TestFileWriteRejectsSymlinkSwapToDeniedFileDuringApproval(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, ".env")
	const secret = "TOKEN=supersecret\n"
	if err := os.WriteFile(envPath, []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}

	newPath := filepath.Join(dir, "new.txt")
	swap := approverFunc(func(_, _ string) bool {
		// Simulate the target being swapped mid-approval: "new.txt" becomes a
		// symlink to the deny-listed .env.
		if err := os.Symlink(envPath, newPath); err != nil {
			t.Fatal(err)
		}
		return true
	})
	// Build the policy engine straight from config.Default() (not the
	// fileToolFor helper, which replaces the whole FilePolicy and drops its
	// DenyWrite floor) so the real .env/*secret* deny-write floor is active,
	// matching how config.Load() always unions it in for a real workspace.
	c := config.Default()
	c.Workspace = dir
	c.File.Default = "ask"
	c.File.Jail = "."
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	tl := NewFile(e, swap, nil)

	r := tl.Call(context.Background(), "write", map[string]any{"path": "new.txt", "content": "PWNED"})
	if !r.IsError {
		t.Fatalf("write via symlink swap to .env should be rejected, got: %+v", r)
	}

	got, err := os.ReadFile(envPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != secret {
		t.Errorf(".env content = %q, want it unchanged (%q) — the write landed on the protected file", got, secret)
	}
}

// The re-check added above must not break the ordinary case: an "ask" write
// with no symlink shenanigans, approved by the user, still lands normally.
func TestFileWriteApprovedByUserStillSucceeds(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "ask", yes())
	r := tl.Call(context.Background(), "write", map[string]any{"path": "ok.txt", "content": "hello"})
	if r.IsError {
		t.Fatalf("approved write should succeed: %s", r.Content)
	}
	data, err := os.ReadFile(filepath.Join(dir, "ok.txt"))
	if err != nil || string(data) != "hello" {
		t.Errorf("file = %q (err %v), want %q", data, err, "hello")
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

// The windowed read streams the file line by line instead of loading it whole
// (codex review finding #19); this must still reproduce strings.Split(content,
// "\n") semantics exactly, including its trailing empty element for content
// that ends in "\n".
func TestFileReadWindowMatchesStringsSplitOnTrailingNewline(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "a\nb\nc\n"})
	r := tl.Call(ctx, "read", map[string]any{"path": "f.txt", "offset": 1, "limit": 0})
	if r.IsError || !strings.Contains(r.Content, "of 4") {
		t.Errorf("windowed read of trailing-newline file = %q, want total 4 (matches strings.Split)", r.Content)
	}
}

// A window near the end of a large file must not require the whole file in
// memory at once — the reader streams to the offset and stops collecting
// lines once past the limit, only counting the rest.
func TestFileReadWindowNearEndOfLargeFile(t *testing.T) {
	tl := fileToolFor(t, t.TempDir(), "allow", yes())
	ctx := context.Background()
	var b strings.Builder
	for i := 1; i <= 5000; i++ {
		fmt.Fprintf(&b, "line%d\n", i)
	}
	tl.Call(ctx, "write", map[string]any{"path": "big.txt", "content": b.String()})
	r := tl.Call(ctx, "read", map[string]any{"path": "big.txt", "offset": 4990, "limit": 5})
	if r.IsError {
		t.Fatalf("read error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "line4990") || !strings.Contains(r.Content, "line4994") || strings.Contains(r.Content, "line4995") {
		t.Errorf("windowed read near EOF of a large file = %q", r.Content)
	}
	if !strings.Contains(r.Content, "of 5001") { // 5000 real lines + trailing empty split element
		t.Errorf("missing/incorrect total in header: %q", r.Content)
	}
}

// The plain read path (no offset/limit) must stream a bounded read instead of
// os.ReadFile-ing the whole file into memory before textutil.Clip trims it
// back down — otherwise a multi-gigabyte file in the workspace gets fully
// loaded despite the advertised maxReadBytes cap. This creates a file far
// larger than the cap and checks two things: the returned content is clipped
// exactly as before (same output contract), and — via a runtime.MemStats
// delta around the call — that the read allocates only a small, bounded
// amount of memory rather than one proportional to the file's size (the old
// code would allocate roughly 2x the file size: the ReadFile buffer plus the
// string(data) conversion).
func TestFileReadPlainPathIsBounded(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()

	const fileSize = 50 * maxReadBytes // ~9.5MiB, far bigger than the cap
	const pattern = "0123456789"
	content := strings.Repeat(pattern, fileSize/len(pattern))
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	r := tl.Call(ctx, "read", map[string]any{"path": "big.txt"})
	runtime.ReadMemStats(&after)

	if r.IsError {
		t.Fatalf("read: %s", r.Content)
	}
	wantFooter := fmt.Sprintf("\n…[truncated; %d bytes total]", len(content))
	if !strings.HasSuffix(r.Content, wantFooter) {
		t.Errorf("missing/incorrect truncation footer in result (len %d)", len(r.Content))
	}
	if gotBody := strings.TrimSuffix(r.Content, wantFooter); gotBody != content[:maxReadBytes] {
		t.Errorf("clipped content mismatch: got %d bytes, want the first %d bytes of the file", len(gotBody), maxReadBytes)
	}

	if delta := after.TotalAlloc - before.TotalAlloc; delta > fileSize/2 {
		t.Errorf("read allocated %d bytes for a %d-byte file — looks like the plain path loaded the whole file into memory instead of streaming a bounded read", delta, fileSize)
	}
}

// The windowed read (offset given, no limit) must stream a bounded
// accumulation instead of collecting every line from offset to EOF into
// `window` before textutil.Clip trims it back down — otherwise a large file
// gets almost entirely loaded into memory despite the advertised maxReadBytes
// cap. This creates a file far larger than the cap and checks two things: the
// returned content is clipped exactly as before (same output contract), and —
// via a runtime.MemStats delta around the call — that the read allocates only
// a small, bounded amount of memory rather than one proportional to the
// file's size (the old code would allocate roughly the whole remaining file).
func TestFileReadWindowOffsetNoLimitIsBounded(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()

	const fileSize = 60 * maxReadBytes // ~12MB, far bigger than the cap
	const pattern = "0123456789"
	content := strings.Repeat(pattern, fileSize/len(pattern))
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	r := tl.Call(ctx, "read", map[string]any{"path": "big.txt", "offset": 1})
	runtime.ReadMemStats(&after)

	if r.IsError {
		t.Fatalf("read: %s", r.Content)
	}
	// The whole file is one giant "line" (no newlines), so the window holds a
	// single element and the header reports 1 of 1.
	if !strings.Contains(r.Content, "of 1)") {
		t.Errorf("missing/incorrect header: %q", firstLine(r.Content))
	}
	body := strings.TrimPrefix(r.Content, firstLine(r.Content)+"\n")
	wantFooter := "\n…[truncated]"
	if !strings.HasSuffix(body, wantFooter) {
		t.Errorf("missing truncation marker in result (len %d)", len(body))
	}
	gotBody := strings.TrimSuffix(body, wantFooter)
	if gotBody != content[:len(gotBody)] || len(gotBody) == 0 || len(gotBody) > maxReadBytes {
		t.Errorf("clipped content mismatch: got %d bytes, want a non-empty prefix of the file capped at %d", len(gotBody), maxReadBytes)
	}

	if delta := after.TotalAlloc - before.TotalAlloc; delta > fileSize/2 {
		t.Errorf("windowed read allocated %d bytes for a %d-byte file — looks like it accumulated the whole file into `window` instead of a bounded read", delta, fileSize)
	}
}

// firstLine returns s up to (not including) its first newline.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// A windowed read with BOTH offset and limit given must not pay to allocate a
// fresh string for every line beyond the requested window just to keep
// counting the file's total line count for the header — a plain
// bufio.Reader.ReadString('\n') call for each of those lines allocates and
// immediately discards it. This creates a small window near the START of a
// large file (so most of the file lies beyond the window) and checks that the
// content stops at the requested window (never reaching the file's tail)
// while allocation stays small — proving the read isn't materializing a full
// line for every one of the many lines past the window just to reach EOF.
func TestFileReadWindowOffsetAndLimitIsBounded(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()

	const nLines = 400_000 // long lines keep the file far bigger than maxReadBytes
	var b strings.Builder
	for i := 1; i <= nLines; i++ {
		fmt.Fprintf(&b, "line-%08d-abcdefghijklmnopqrstuvwxyz\n", i)
	}
	content := b.String()
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	r := tl.Call(ctx, "read", map[string]any{"path": "big.txt", "offset": 5, "limit": 3})
	runtime.ReadMemStats(&after)

	if r.IsError {
		t.Fatalf("read: %s", r.Content)
	}
	if !strings.Contains(r.Content, "line-00000005") || !strings.Contains(r.Content, "line-00000007") {
		t.Errorf("windowed read missing requested lines: %q", r.Content)
	}
	if strings.Contains(r.Content, "line-00000008") {
		t.Errorf("window should stop at the requested limit: %q", r.Content)
	}
	if !strings.Contains(r.Content, fmt.Sprintf("of %d", nLines+1)) {
		t.Errorf("missing/incorrect total in header: %q", r.Content)
	}

	// Each line is ~40 bytes; if the fix worked, allocation stays on the order
	// of a handful of lines/buffers, nowhere near one allocation per line for
	// all 400,000 lines (which the pre-fix ReadString-per-discarded-line
	// behavior would rack up).
	const perLineAllocIfUnfixed = 40
	if delta := after.TotalAlloc - before.TotalAlloc; delta > nLines*perLineAllocIfUnfixed/10 {
		t.Errorf("windowed read with offset+limit allocated %d bytes over %d lines — looks like it's still allocating a string per line past the requested window", delta, nLines)
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

// On a file with Windows line endings the model writes "\n": a multi-line find
// never matched, and a multi-line replace left mixed endings.
func TestFileEditKeepsCRLF(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w.txt")
	if err := os.WriteFile(p, []byte("one\r\ntwo\r\nthree\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	if r := tl.Call(context.Background(), "edit", map[string]any{"path": "w.txt", "find": "one\ntwo", "replace": "uno\ndos\nmas"}); r.IsError {
		t.Fatalf("edit: %s", r.Content)
	}
	if data, _ := os.ReadFile(p); string(data) != "uno\r\ndos\r\nmas\r\nthree\r\n" {
		t.Errorf("result = %q, want every line ending CRLF", data)
	}
}

// A find that occurs more than once edited the first silently, often not the
// one meant. It is refused, saying how many there are; replace_all still works.
func TestFileEditRefusesAnAmbiguousFind(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.go", "content": "if a { return nil }\nif b { return nil }\n"})
	r := tl.Call(ctx, "edit", map[string]any{"path": "f.go", "find": "return nil", "replace": "return err"})
	if !r.IsError || !strings.Contains(r.Content, "2 times") {
		t.Errorf("ambiguous find: %+v, want refused with the count", r)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "f.go")); strings.Contains(string(data), "err") {
		t.Errorf("the file changed anyway: %q", data)
	}
}

// A new file is created the way anything else is, under the user's umask —
// not widened to 0644 by a chmod.
func TestFileWriteNewFileFollowsUmask(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX modes")
	}
	old := setUmask(0o077)
	defer setUmask(old)
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "new.txt", "content": "x"}); r.IsError {
		t.Fatal(r.Content)
	}
	if fi, err := os.Stat(filepath.Join(dir, "new.txt")); err != nil || fi.Mode().Perm() != 0o600 {
		t.Errorf("mode = %v, want 0600 under umask 077", fi.Mode().Perm())
	}
}

// A whole-file write keeps a UTF-8 byte order mark the file had: the model
// never sees it, so it dropped it silently. Not into an empty write, though.
func TestFileWriteKeepsABOM(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "b.csv")
	if err := os.WriteFile(p, []byte("\ufeffa,b\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "b.csv", "content": "a,b,c\n"}); r.IsError {
		t.Fatal(r.Content)
	}
	if data, _ := os.ReadFile(p); string(data) != "\ufeffa,b,c\n" {
		t.Errorf("result = %q, want the BOM kept", data)
	}
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "b.csv", "content": ""}); r.IsError {
		t.Fatal(r.Content)
	}
	if data, _ := os.ReadFile(p); len(data) != 0 {
		t.Errorf("empty write left %q, want an empty file", data)
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

// Parallel sub-agents share a workspace by default (a fan-out of `agent`
// calls with no explicit dir all get the SAME file tool instance), so two of
// them editing the same file concurrently is real — an unlocked read-then-
// write there lets a later write silently discard an earlier one, with both
// calls reporting success. Every concurrent edit here targets a disjoint
// substring, so none should ever be lost.
func TestFileEditConcurrentEditsAllApply(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	tl.Call(ctx, "write", map[string]any{"path": "f.txt", "content": "A B C D"})

	const n = 4
	markers := []string{"A", "B", "C", "D"}
	var wg sync.WaitGroup
	results := make([]Result, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = tl.Call(ctx, "edit", map[string]any{
				"path": "f.txt", "find": markers[i], "replace": markers[i] + "1",
			})
		}(i)
	}
	wg.Wait()

	for i, r := range results {
		if r.IsError {
			t.Errorf("edit %d failed: %s", i, r.Content)
		}
	}
	data, err := os.ReadFile(filepath.Join(dir, "f.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range markers {
		if !strings.Contains(string(data), m+"1") {
			t.Errorf("final content = %q, missing replacement for %q (a concurrent edit was lost)", data, m)
		}
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

// A model that omits "action" (violating the required+enum schema) still gets a
// READ-ONLY intent recovered from the params — but never a guessed mutation.
func TestDispatchInfersReadOnlyAction(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "hello.txt"), []byte("hi there"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(fileToolFor(t, dir, "allow", yes()))

	// No action + a path → inferred read.
	if res := r.Dispatch(context.Background(), "file", "", map[string]any{"path": "hello.txt"}); res.IsError || !strings.Contains(res.Content, "hi there") {
		t.Errorf("empty action + path should infer read: %+v", res)
	}
	// No action + a query → inferred search (read-only).
	if res := r.Dispatch(context.Background(), "file", "", map[string]any{"query": "hi"}); res.IsError || !strings.Contains(res.Content, "hello.txt") {
		t.Errorf("empty action + query should infer search: %+v", res)
	}
	// No action + content → must NOT guess a write; and nothing is written.
	res := r.Dispatch(context.Background(), "file", "", map[string]any{"path": "new.txt", "content": "x"})
	if !res.IsError || !strings.Contains(res.Content, "no action given") {
		t.Errorf("empty action + content must not infer a mutation: %+v", res)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.txt")); err == nil {
		t.Error("a file was written from an inferred (guessed) action")
	}
}

// write and edit loaded the existing file whole to diff or edit it, with no
// bound: a multi-gigabyte data file exhausted memory (found by review). Above
// maxEditBytes, edit refuses and write replaces without a diff.
func TestWriteAndEditBoundWhatTheyLoad(t *testing.T) {
	dir := t.TempDir()
	tl := fileToolFor(t, dir, "allow", yes())
	big := filepath.Join(dir, "data.bin")
	f, _ := os.Create(big)
	f.WriteString("needle")
	f.Truncate(maxEditBytes + 1<<20) // sparse: instant, no disk
	f.Close()
	ctx := context.Background()
	if r := tl.Call(ctx, "edit", map[string]any{"path": "data.bin", "find": "needle", "replace": "thread"}); !r.IsError || !strings.Contains(r.Content, "too large") {
		t.Errorf("edit of a %d MB file: %+v", (maxEditBytes+1<<20)>>20, r)
	}
	if r := tl.Call(ctx, "write", map[string]any{"path": "data.bin", "content": "x"}); r.IsError {
		t.Errorf("write over a large file failed: %s", r.Content)
	}
}

// search and find walk the tree with no regard for cancellation: over a big
// tree (a sub-agent jailed to ~) a cancelled task's walk ran on to the end
// (found by review). A cancelled context stops the walk.
func TestSearchAndFindStopWhenCancelled(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 50; i++ {
		os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%02d.txt", i)), []byte("needle"), 0o644)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if r := tl.Call(ctx, "search", map[string]any{"query": "needle"}); strings.Count(r.Content, "needle") == 50 {
		t.Errorf("search walked the whole tree after cancellation")
	}
	if r := tl.Call(ctx, "find", map[string]any{"pattern": "*.txt"}); strings.Count(r.Content, ".txt") == 50 {
		t.Errorf("find walked the whole tree after cancellation")
	}
}

// Search and find skip caches and virtualenvs (dot-directories), but not the
// ones holding the project's own files — a CI workflow in .github was
// invisible. A search that finds nothing says what it skipped.
func TestFileSearchLooksInCIDirectories(t *testing.T) {
	dir := t.TempDir()
	for p, body := range map[string]string{
		".github/workflows/ci.yml": "run: make deploy-target",
		".cache/x.txt":             "deploy-target",
	} {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	tl := fileToolFor(t, dir, "allow", yes())
	ctx := context.Background()
	r := tl.Call(ctx, "search", map[string]any{"query": "deploy-target"})
	if !strings.Contains(r.Content, "ci.yml") || strings.Contains(r.Content, ".cache") {
		t.Errorf("search = %q, want the workflow found and the cache skipped", r.Content)
	}
	if r := tl.Call(ctx, "find", map[string]any{"pattern": "**/*.yml"}); !strings.Contains(r.Content, ".github/workflows/ci.yml") {
		t.Errorf("find = %q, want the workflow", r.Content)
	}
	if r := tl.Call(ctx, "search", map[string]any{"query": "nowhere-at-all"}); !strings.Contains(r.Content, "starting with a dot were skipped") {
		t.Errorf("empty search = %q, want it to say what was skipped", r.Content)
	}
}

// A whole-file replace must not cut links: through a symlink the target is what
// gets written, and a file with another hard link is written in place so both
// names see the new content.
func TestFileWriteKeepsLinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX links")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "real.yaml")
	if err := os.WriteFile(target, []byte("a: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("real.yaml", filepath.Join(dir, "conf.yaml")); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(target, filepath.Join(dir, "hard.yaml")); err != nil {
		t.Fatal(err)
	}
	tl := fileToolFor(t, dir, "allow", yes())
	if r := tl.Call(context.Background(), "write", map[string]any{"path": "conf.yaml", "content": "a: 2\n"}); r.IsError {
		t.Fatal(r.Content)
	}
	if fi, err := os.Lstat(filepath.Join(dir, "conf.yaml")); err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Error("the symlink was replaced by a file")
	}
	for _, name := range []string{"real.yaml", "hard.yaml"} {
		if data, _ := os.ReadFile(filepath.Join(dir, name)); string(data) != "a: 2\n" {
			t.Errorf("%s = %q, want the new content", name, data)
		}
	}
}
