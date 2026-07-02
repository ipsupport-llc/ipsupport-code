package tool

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

func gitToolFor(t *testing.T, dir string, ap Approver) Tool {
	t.Helper()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	return NewGit(e, ap)
}

func initRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init"}, {"config", "user.email", "t@example.com"}, {"config", "user.name", "tester"},
	} {
		if out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

func TestGitStatusAddCommitLog(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()

	if r := tl.Call(ctx, "status", nil); r.IsError {
		t.Fatalf("status: %s", r.Content)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r := tl.Call(ctx, "add", map[string]any{"paths": "a.txt"}); r.IsError {
		t.Fatalf("add: %s", r.Content)
	}
	if r := tl.Call(ctx, "commit", map[string]any{"message": "first commit"}); r.IsError {
		t.Fatalf("commit: %s", r.Content)
	}
	r := tl.Call(ctx, "log", map[string]any{"n": 5})
	if r.IsError || !strings.Contains(r.Content, "first commit") {
		t.Errorf("log = %+v, want the commit", r)
	}
}

func TestGitInit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir() // a bare, non-repo workspace
	tl := gitToolFor(t, dir, yes())
	if r := tl.Call(context.Background(), "init", nil); r.IsError {
		t.Fatalf("init: %s", r.Content)
	}
	if _, err := os.Stat(filepath.Join(dir, ".git")); err != nil {
		t.Errorf("expected a .git directory after init: %v", err)
	}
}

// A leading-dash ref must be rejected on show/branch/checkout so it can't be read
// as a git flag (e.g. show "-O…" or checkout "-f").
func TestGitRejectsLeadingDashRef(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()
	for _, tc := range []struct{ action, key, val string }{
		{"show", "ref", "-O/etc/passwd"},
		{"checkout", "ref", "-f"},
		{"branch", "name", "-D"},
	} {
		r := tl.Call(ctx, tc.action, map[string]any{tc.key: tc.val})
		if !r.IsError || !strings.Contains(r.Content, "leading dash") {
			t.Errorf("%s(%s=%q) = %+v, want a leading-dash rejection", tc.action, tc.key, tc.val, r)
		}
	}
}

func TestGitMutatingDeniedByUser(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, no())
	r := tl.Call(context.Background(), "commit", map[string]any{"message": "x"})
	if !r.IsError || !strings.Contains(r.Content, "denied") {
		t.Errorf("commit with deny = %+v, want denied", r)
	}
}
