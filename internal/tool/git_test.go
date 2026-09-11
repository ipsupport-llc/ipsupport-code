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

// "checkout -- ref" made ref a PATHSPEC, not a revision — a real branch name
// always failed with "pathspec did not match any files", so checkout could
// never switch branches at all.
func TestGitCheckoutSwitchesBranch(t *testing.T) {
	dir := initRepo(t)
	if out, err := exec.Command("git", "-C", dir, "commit", "--allow-empty", "-m", "root").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "branch", "feature").CombinedOutput(); err != nil {
		t.Fatalf("branch: %v\n%s", err, out)
	}
	tl := gitToolFor(t, dir, yes())
	if r := tl.Call(context.Background(), "checkout", map[string]any{"ref": "feature"}); r.IsError {
		t.Fatalf("checkout feature: %s", r.Content)
	}
	out, err := exec.Command("git", "-C", dir, "branch", "--show-current").CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(out)); got != "feature" {
		t.Errorf("current branch = %q, want feature", got)
	}
}

// git show/diff read a path's content directly (at a given revision, or a
// working-tree diff) — the same jail/secret-file checks file.read enforces
// on that path must apply here too, or they're a wide-open bypass.
func TestGitShowDiffRespectFilePolicy(t *testing.T) {
	dir := initRepo(t)
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=hunter2"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", ".env").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add env").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()

	if r := tl.Call(ctx, "show", map[string]any{"ref": "HEAD:.env"}); !r.IsError || strings.Contains(r.Content, "hunter2") {
		t.Errorf("show HEAD:.env = %+v, want blocked as a secret, not the raw content", r)
	}
	// A blob reference for an ordinary tracked file must still work.
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", "main.go").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add main.go").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if r := tl.Call(ctx, "show", map[string]any{"ref": "HEAD:main.go"}); r.IsError || !strings.Contains(r.Content, "package main") {
		t.Errorf("show HEAD:main.go = %+v, want the file content", r)
	}

	if r := tl.Call(ctx, "diff", map[string]any{"path": ".env"}); !r.IsError {
		t.Errorf("diff --path .env = %+v, want blocked as a secret", r)
	}
}

// diff with no path (the whole repo) or path "." both used to bypass the
// secret-file check entirely: checkPathPolicy only ran when "path" was a
// literal file name, so an omitted path skipped it outright and "." resolved
// to the repo root DIRECTORY, which IsSecret never matches — either way the
// raw content of an uncommitted-modified .env came back in the diff. Each
// changed file must now be checked individually and secret ones excluded,
// while an ordinary changed file still comes through.
func TestGitDiffExcludesSecretFilesFromWholeRepoDiff(t *testing.T) {
	dir := initRepo(t)
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", ".env", "main.go").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "initial").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	// Modify both, uncommitted, so a plain "git diff" covers each.
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=hunter2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()

	for _, tc := range []struct {
		name string
		args map[string]any
	}{
		{"no path", map[string]any{}},
		{`path "."`, map[string]any{"path": "."}},
	} {
		r := tl.Call(ctx, "diff", tc.args)
		if r.IsError {
			t.Fatalf("diff (%s) = %+v, want ok", tc.name, r)
		}
		if strings.Contains(r.Content, "hunter2") {
			t.Errorf("diff (%s) = %+v, want the .env secret excluded, not leaked", tc.name, r)
		}
		if !strings.Contains(r.Content, "func main") {
			t.Errorf("diff (%s) = %+v, want main.go's ordinary diff still included", tc.name, r)
		}
	}
}

// git's own ":[<n>:]<path>" index-stage syntax (e.g. ":0:.env") names the SAME
// path as "HEAD:.env" but bypassed the naive first-colon split (which read
// ":0:.env" as path "0:.env", never matching the secret-file glob) and read
// the blob straight out of the index regardless of file.read's restriction on
// the identical path.
func TestGitShowRejectsIndexStageSecretBypass(t *testing.T) {
	dir := initRepo(t)
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=hunter2"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", ".env").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add env").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()

	if r := tl.Call(ctx, "show", map[string]any{"ref": ":0:.env"}); !r.IsError || strings.Contains(r.Content, "hunter2") {
		t.Errorf("show :0:.env = %+v, want blocked as a secret, not the raw content", r)
	}
}

// show's "<rev>:<path>" blob syntax always resolves <path> against the
// REPOSITORY ROOT, never the configured jail dir or cwd. When the jail is a
// subdirectory of a larger repo, checkPathPolicy(path) used to validate
// <jail>/<path> (which doesn't exist) while git actually read
// <repo-root>/<path> — a completely different, out-of-jail file — with the
// jail check passing on the wrong path. A file outside the jail but inside
// the repo must now be rejected, while a file genuinely inside the jail must
// still work.
func TestGitShowRejectsRevPathOutsideJailWhenJailIsSubdir(t *testing.T) {
	dir := initRepo(t) // dir is the repo root
	subdir := filepath.Join(dir, "sub")
	if err := os.MkdirAll(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "outside.txt"), []byte("TOP-SECRET-ROOT-CONTENT"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subdir, "infile.txt"), []byte("inside the jail"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", "outside.txt", "sub/infile.txt").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add files").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}

	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "sub"} // jail = a SUBDIRECTORY of the repo
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	tl := NewGit(e, yes())
	ctx := context.Background()

	// The exploit: HEAD:outside.txt reads a file OUTSIDE the jail (at the
	// repo root) — must now be rejected, not leak its content.
	if r := tl.Call(ctx, "show", map[string]any{"ref": "HEAD:outside.txt"}); !r.IsError || strings.Contains(r.Content, "TOP-SECRET-ROOT-CONTENT") {
		t.Errorf("show HEAD:outside.txt = %+v, want rejected as outside the jail, not the leaked content", r)
	}

	// Normal usage of a file genuinely inside the jail must still work.
	if r := tl.Call(ctx, "show", map[string]any{"ref": "HEAD:sub/infile.txt"}); r.IsError || !strings.Contains(r.Content, "inside the jail") {
		t.Errorf("show HEAD:sub/infile.txt = %+v, want the file content", r)
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
