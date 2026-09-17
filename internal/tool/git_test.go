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
	return NewGit(e, ap, 0, false)
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

// A tracked file literally named "*" is a valid pathspec GLOB when passed
// bare, and independently glob-matches every other file in the same
// directory — including ".env", which checkPathPolicy already excluded from
// "allowed" as a secret. Without marking "allowed" paths literal, git's own
// pathspec expansion re-includes .env's real content in the diff anyway,
// contradicting the tool's own "excluded" accounting.
func TestGitDiffWildcardNamedFileDoesNotGlobMatchSecret(t *testing.T) {
	dir := initRepo(t)
	if err := os.WriteFile(filepath.Join(dir, "*"), []byte("star old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=hunter2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", "*", ".env").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "initial").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	// Modify both, uncommitted, so a plain "git diff" covers each.
	if err := os.WriteFile(filepath.Join(dir, "*"), []byte("star new\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("API_KEY=hunter2_v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tl := gitToolFor(t, dir, yes())
	r := tl.Call(context.Background(), "diff", nil)
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if strings.Contains(r.Content, "hunter2") {
		t.Errorf("diff = %+v, want .env's secret content excluded, not leaked via the \"*\"-named file's glob pathspec", r)
	}
	if !strings.Contains(r.Content, "star new") {
		t.Errorf("diff = %+v, want the \"*\"-named file's own change included", r)
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
	tl := NewGit(e, yes(), 0, false)
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

// diff's enumeration step (git diff --name-only) always reports paths
// relative to the REPO ROOT, never to the current effective directory. A
// "/cd sub" moves cmd.Dir to the subdirectory for both the enumeration call
// and the real diff that reuses its output as pathspecs — so a name like
// "sub/x" got re-applied against a cwd that was ALREADY "sub", i.e. looked
// for the nonexistent "sub/sub/x", and the real diff for a genuinely
// modified file silently came back empty.
func TestGitDiffFromSubdirResolvesRepoRootRelativePaths(t *testing.T) {
	dir := initRepo(t)
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	subFile := filepath.Join(dir, "sub", "x")
	if err := os.WriteFile(subFile, []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", "sub/x").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add sub/x").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if err := os.WriteFile(subFile, []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Simulate "/cd sub": the tool's effective directory becomes the
	// subdirectory while the workspace/jail stay at the repo root.
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.SetWorkdir("sub"); err != nil {
		t.Fatal(err)
	}
	tl := NewGit(e, yes(), 0, false)

	r := tl.Call(context.Background(), "diff", nil)
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if !strings.Contains(r.Content, "v2") {
		t.Errorf("diff from cwd=sub = %+v, want sub/x's change included, not silently dropped", r)
	}
}

// git C-quotes non-ASCII filenames in --name-only's default output (e.g.
// "é.txt" becomes a literal quoted "\303\251.txt"); passed straight back as a
// pathspec, git does not unquote it and the real diff comes back empty.
func TestGitDiffHandlesNonASCIIFilename(t *testing.T) {
	dir := initRepo(t)
	name := "é.txt"
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", name).CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add non-ascii file").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if err := os.WriteFile(path, []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tl := gitToolFor(t, dir, yes())
	r := tl.Call(context.Background(), "diff", nil)
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if !strings.Contains(r.Content, "v2") {
		t.Errorf("diff = %+v, want the non-ASCII filename's change included, not silently dropped", r)
	}
}

// A legitimately whitespace-leading filename (plain ASCII spaces aren't
// special, so git does NOT quote it) used to be corrupted by a bare
// strings.TrimSpace on the enumerated name, turning it into a pathspec that
// doesn't exist and silently dropping its diff.
func TestGitDiffHandlesWhitespaceLeadingFilename(t *testing.T) {
	dir := initRepo(t)
	name := " lead.txt"
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", name).CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add whitespace-leading file").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if err := os.WriteFile(path, []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tl := gitToolFor(t, dir, yes())
	r := tl.Call(context.Background(), "diff", nil)
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if !strings.Contains(r.Content, "v2") {
		t.Errorf("diff = %+v, want the whitespace-leading filename's change included, not silently dropped", r)
	}
}

// A .gitattributes diff/textconv driver bound in the local git config must
// never be executed by the "diff" action: it's a read-only action with no
// approval gate, so a configured driver would otherwise let plain "git diff"
// run an arbitrary subprocess. Approver denies everything, to make clear the
// driver's non-execution has nothing to do with approval (diff never asks).
func TestGitDiffDoesNotRunTextconvDriver(t *testing.T) {
	dir := initRepo(t)

	if err := os.WriteFile(filepath.Join(dir, ".gitattributes"), []byte("data.bin diff=marker\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "data.bin"), []byte("v1\x00binary\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", ".gitattributes", "data.bin").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add data.bin").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}

	marker := filepath.Join(t.TempDir(), "marker")
	driver := filepath.Join(dir, "textconv-driver.sh")
	script := "#!/bin/sh\ntouch " + marker + "\ncat \"$1\"\n"
	if err := os.WriteFile(driver, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "config", "diff.marker.textconv", driver).CombinedOutput(); err != nil {
		t.Fatalf("config: %v\n%s", err, out)
	}

	if err := os.WriteFile(filepath.Join(dir, "data.bin"), []byte("v2\x00binary\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	tl := gitToolFor(t, dir, no())
	r := tl.Call(context.Background(), "diff", map[string]any{"path": "data.bin"})
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if !strings.Contains(r.Content, "data.bin") {
		t.Errorf("diff = %+v, want a sensible (binary-fallback) diff mentioning data.bin", r)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Error("diff ran the configured textconv driver (marker file created): approval/plan-mode bypass via .gitattributes")
	}
}

// A local .git/config binding core.fsmonitor to an arbitrary executable must
// never be executed by the "status" action: it's a read-only action with no
// approval gate, so a configured fsmonitor hook would otherwise let plain
// "git status" run an arbitrary subprocess. Approver denies everything, to
// make clear the hook's non-execution has nothing to do with approval
// (status never asks in the first place).
func TestGitStatusDoesNotRunFsmonitorHook(t *testing.T) {
	dir := initRepo(t)

	marker := filepath.Join(t.TempDir(), "marker")
	hook := filepath.Join(dir, "fsmonitor-hook.sh")
	script := "#!/bin/sh\ntouch " + marker + "\necho 1\nexit 0\n"
	if err := os.WriteFile(hook, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "config", "core.fsmonitor", hook).CombinedOutput(); err != nil {
		t.Fatalf("config: %v\n%s", err, out)
	}

	tl := gitToolFor(t, dir, no())
	r := tl.Call(context.Background(), "status", nil)
	if r.IsError {
		t.Fatalf("status: %s", r.Content)
	}
	if !strings.Contains(r.Content, "##") {
		t.Errorf("status = %+v, want the usual --short --branch output (\"##\" branch line)", r)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Error("status ran the configured fsmonitor hook (marker file created): approval bypass via core.fsmonitor")
	}
}

// Same as above but for "diff", which has TWO git invocations (the
// --name-only enumeration and the real diff) — both must suppress
// core.fsmonitor. An actual pending change is required so the real diff
// invocation is reached too, not just the enumeration.
func TestGitDiffDoesNotRunFsmonitorHook(t *testing.T) {
	dir := initRepo(t)
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "add", "a.txt").CombinedOutput(); err != nil {
		t.Fatalf("add: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "commit", "-m", "add a.txt").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("v2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	marker := filepath.Join(t.TempDir(), "marker")
	hook := filepath.Join(dir, "fsmonitor-hook.sh")
	script := "#!/bin/sh\ntouch " + marker + "\necho 1\nexit 0\n"
	if err := os.WriteFile(hook, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "config", "core.fsmonitor", hook).CombinedOutput(); err != nil {
		t.Fatalf("config: %v\n%s", err, out)
	}

	tl := gitToolFor(t, dir, no())
	r := tl.Call(context.Background(), "diff", nil)
	if r.IsError {
		t.Fatalf("diff: %s", r.Content)
	}
	if !strings.Contains(r.Content, "v2") {
		t.Errorf("diff = %+v, want the usual diff output", r)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Error("diff ran the configured fsmonitor hook (marker file created): approval bypass via core.fsmonitor")
	}
}

// core.fsmonitor suppression now lives in run() itself, applied to every
// action, not just status/diff (which used to each carry their own copy).
// "checkout" never had a guard at all before that centralization, and it
// genuinely consults fsmonitor (unlike e.g. log/show/branch, which don't
// touch the working tree or index and never invoke the hook regardless) —
// verify it's covered too, proving the fix reaches a mutating action beyond
// the two read-only ones that were patched individually in earlier rounds.
func TestGitCheckoutDoesNotRunFsmonitorHook(t *testing.T) {
	dir := initRepo(t)
	if out, err := exec.Command("git", "-C", dir, "commit", "--allow-empty", "-m", "root").CombinedOutput(); err != nil {
		t.Fatalf("commit: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "-C", dir, "branch", "feature").CombinedOutput(); err != nil {
		t.Fatalf("branch: %v\n%s", err, out)
	}

	marker := filepath.Join(t.TempDir(), "marker")
	hook := filepath.Join(dir, "fsmonitor-hook.sh")
	script := "#!/bin/sh\ntouch " + marker + "\nprintf '1\\0'\n"
	if err := os.WriteFile(hook, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", dir, "config", "core.fsmonitor", hook).CombinedOutput(); err != nil {
		t.Fatalf("config: %v\n%s", err, out)
	}

	tl := gitToolFor(t, dir, yes())
	r := tl.Call(context.Background(), "checkout", map[string]any{"ref": "feature"})
	if r.IsError {
		t.Fatalf("checkout feature: %s", r.Content)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Error("checkout ran the configured fsmonitor hook (marker file created): centralized suppression in run() didn't cover checkout")
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

// offlineGitTool is the same tool with offline mode on.
func offlineGitTool(t *testing.T, dir string) Tool {
	t.Helper()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	return NewGit(e, yes(), 0, true)
}

// "ext::sh -c <cmd>" is a valid git URL whose transport is an arbitrary command:
// cloning one executes it. Nothing about the string looks like a command, which
// is exactly why the model would pass it along from a page it read.
func TestCloneRejectsRemoteHelperURLs(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, yes())
	marker := filepath.Join(dir, "pwned")

	r := tl.Call(context.Background(), "clone", map[string]any{
		"url": "ext::sh -c touch% " + marker,
	})

	if !r.IsError {
		t.Fatalf("clone accepted an ext:: URL: %s", r.Content)
	}
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("the ext:: transport actually ran — it created its marker file")
	}
	if !strings.Contains(r.Content, "remote-helper") {
		t.Errorf("error should say why:\n%s", r.Content)
	}
}

// A clone writes a whole tree, so its destination is jailed like any other write.
// Cloned from a LOCAL repo on purpose: an unreachable URL fails on its own and
// would let this pass with no jail at all.
func TestCloneDestinationStaysInTheWorkspace(t *testing.T) {
	src := initRepo(t)
	if out, err := exec.Command("git", "-C", src, "commit", "--allow-empty", "-m", "seed").CombinedOutput(); err != nil {
		t.Fatalf("seed commit: %v\n%s", err, out)
	}
	ws := t.TempDir()
	tl := gitToolFor(t, ws, yes())
	ctx := context.Background()

	r := tl.Call(ctx, "clone", map[string]any{"url": src, "dir": "../escaped"})

	if !r.IsError {
		t.Fatalf("clone accepted a destination outside the workspace: %s", r.Content)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(ws), "escaped")); err == nil {
		t.Fatal("it cloned outside the jail")
	}
	// The same clone INSIDE the workspace works — so what was refused was the
	// destination, not the clone.
	if r := tl.Call(ctx, "clone", map[string]any{"url": src, "dir": "copy"}); r.IsError {
		t.Fatalf("clone into the workspace: %s", r.Content)
	}
	if _, err := os.Stat(filepath.Join(ws, "copy", ".git")); err != nil {
		t.Errorf("the in-workspace clone didn't land: %v", err)
	}
}

func TestCloneDirDefaultsToTheRepoName(t *testing.T) {
	for _, tc := range []struct{ url, want string }{
		{"https://github.com/o/repo.git", "repo"},
		{"https://github.com/o/repo", "repo"},
		{"https://github.com/o/repo.git/", "repo"},
		{"git@github.com:o/repo.git", "repo"},
		{"ssh://git@host:22/o/repo.git", "repo"},
	} {
		if got := repoDirFromURL(tc.url); got != tc.want {
			t.Errorf("repoDirFromURL(%q) = %q, want %q", tc.url, got, tc.want)
		}
	}
}

// Offline mode is the user saying "no internet right now". The actions that need
// a server must say so rather than hang until the timeout kills them.
func TestOfflineRefusesNetworkGitActions(t *testing.T) {
	dir := initRepo(t)
	tl := offlineGitTool(t, dir)
	ctx := context.Background()

	for _, action := range []string{"clone", "fetch", "pull", "push"} {
		params := map[string]any{}
		if action == "clone" {
			params["url"] = "https://example.invalid/x.git"
		}
		r := tl.Call(ctx, action, params)
		if !r.IsError || !strings.Contains(r.Content, "offline") {
			t.Errorf("%s while offline = %q, want an offline refusal", action, r.Content)
		}
	}
	// …and the local actions still work.
	if r := tl.Call(ctx, "status", nil); r.IsError {
		t.Errorf("status must still work offline: %s", r.Content)
	}
	if r := tl.Call(ctx, "remote", nil); r.IsError {
		t.Errorf("listing remotes touches no server, it must work offline: %s", r.Content)
	}
}

// pull is fast-forward only: a diverged branch fails cleanly instead of leaving
// a conflicted merge the agent then has to reason about mid-task.
func TestPullIsFastForwardOnly(t *testing.T) {
	dir := initRepo(t)
	var asked string
	tl := gitToolFor(t, dir, approverFunc(func(_, detail string) bool {
		asked = detail
		return false // deny: we only care what it was asked to approve
	}))

	tl.Call(context.Background(), "pull", map[string]any{"remote": "origin"})

	if !strings.Contains(asked, "--ff-only") {
		t.Errorf("approval asked for %q, want a --ff-only pull", asked)
	}
}

// push publishes. The approval must show where, and a leading dash must never
// reach git as an option.
func TestPushApprovalShowsTheDestination(t *testing.T) {
	dir := initRepo(t)
	var asked string
	tl := gitToolFor(t, dir, approverFunc(func(_, detail string) bool {
		asked = detail
		return false
	}))
	ctx := context.Background()

	tl.Call(ctx, "push", map[string]any{"remote": "origin", "branch": "main"})
	if !strings.Contains(asked, "origin") || !strings.Contains(asked, "main") {
		t.Errorf("approval detail = %q, want the remote and branch being published to", asked)
	}
	if strings.Contains(asked, "--force") {
		t.Errorf("approval detail = %q, force-push must not be reachable", asked)
	}

	if r := tl.Call(ctx, "push", map[string]any{"remote": "--delete"}); !r.IsError {
		t.Error("a leading-dash remote must be refused, not passed to git as an option")
	}
}

func TestRemoteListsAndAdds(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()

	if r := tl.Call(ctx, "remote", map[string]any{"name": "origin", "url": "https://example.invalid/x.git"}); r.IsError {
		t.Fatalf("remote add: %s", r.Content)
	}
	r := tl.Call(ctx, "remote", nil)
	if r.IsError || !strings.Contains(r.Content, "example.invalid") {
		t.Fatalf("remote list = %q, want the remote just added", r.Content)
	}
	// Adding the same name again repoints it rather than failing.
	if r := tl.Call(ctx, "remote", map[string]any{"name": "origin", "url": "https://other.invalid/y.git"}); r.IsError {
		t.Fatalf("repointing an existing remote: %s", r.Content)
	}
	if r := tl.Call(ctx, "remote", nil); !strings.Contains(r.Content, "other.invalid") {
		t.Errorf("remote list = %q, want the new URL", r.Content)
	}
	if r := tl.Call(ctx, "remote", map[string]any{"name": "origin"}); !r.IsError {
		t.Error("a name with no url should say what's missing, not silently do nothing")
	}
}

// The policy engine stops the file tool READING a secrets file, but nothing
// stopped git staging one — and now that push exists, a staged secret is one
// commit away from leaving the machine. "git add ." is the case that matters:
// it names no secret at all while staging every one of them.
func TestAddRefusesToStageSecrets(t *testing.T) {
	dir := initRepo(t)
	tl := gitToolFor(t, dir, yes())
	ctx := context.Background()
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN=hunter2"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main"), 0o644); err != nil {
		t.Fatal(err)
	}

	if r := tl.Call(ctx, "add", map[string]any{"paths": ".env"}); !r.IsError {
		t.Error("staged a secrets file by name")
	}
	if r := tl.Call(ctx, "add", map[string]any{"paths": "."}); !r.IsError {
		t.Errorf("'git add .' staged the secrets file alongside everything else: %s", r.Content)
	}
	staged := exec.Command("git", "-C", dir, "diff", "--staged", "--name-only")
	out, _ := staged.CombinedOutput()
	if strings.Contains(string(out), ".env") {
		t.Errorf("the secrets file reached the index anyway:\n%s", out)
	}
	// An ordinary file still stages fine.
	if r := tl.Call(ctx, "add", map[string]any{"paths": "main.go"}); r.IsError {
		t.Errorf("add main.go: %s", r.Content)
	}
}
