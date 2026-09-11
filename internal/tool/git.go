package tool

import (
	"context"
	"errors"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

type gitTool struct {
	pol *policy.Engine
	ap  Approver
}

// NewGit returns the git tool. It runs git directly (argv, no shell) in the
// workspace; read-only actions run freely, mutating ones ask for approval.
func NewGit(p *policy.Engine, ap Approver) Tool {
	g := &gitTool{pol: p, ap: ap}
	return NewDomain(DomainSpec{
		Name:    "git",
		Summary: "Git in the workspace. Mutating actions (init/add/commit/branch/checkout) ask approval.",
		NotHere: "NOT here — non-git shell → run; files → file.",
		Actions: []Action{
			{Name: "init", Mutates: true, Run: g.initRepo, Note: "(start a repo in the workspace)"},
			{Name: "status", Run: g.status},
			{Name: "diff", Params: []Param{Opt("path", "str", ""), Opt("staged", "bool", "")}, Run: g.diff},
			{Name: "log", Params: []Param{Opt("n", "int", "15")}, Run: g.log},
			{Name: "show", Params: []Param{Opt("ref", "str", "HEAD")}, Run: g.show},
			{Name: "add", Mutates: true, Params: []Param{Req("paths", "str")}, Note: "(space-separated)", Run: g.add},
			{Name: "commit", Mutates: true, Params: []Param{Req("message", "str")}, Run: g.commit},
			{Name: "branch", Mutates: true, Params: []Param{Opt("name", "str", "")}, Note: "(none=list, name=create)", Run: g.branch},
			{Name: "checkout", Mutates: true, Params: []Param{Req("ref", "str")}, Run: g.checkout},
		},
	})
}

func (g *gitTool) initRepo(ctx context.Context, _ Args) Result {
	return g.run(ctx, "init", true, "init")
}

func (g *gitTool) status(ctx context.Context, _ Args) Result {
	return g.run(ctx, "status", false, "status", "--short", "--branch")
}

func (g *gitTool) diff(ctx context.Context, a Args) Result {
	staged := a.Bool("staged")
	p := a.Str("path")
	if p != "" {
		if err := g.checkPathPolicy(p); err != nil {
			return Err(err.Error())
		}
	}

	// checkPathPolicy(p) above only catches a path that IS the secret file
	// itself. An omitted path (whole-repo diff) or "." (which resolves to the
	// repo root DIRECTORY, never matched by the secret-file glob) can still
	// span a tracked secret file among the changed ones, so find the actual
	// changed files first and check each individually before diffing them.
	//
	// -z NUL-delimits the enumerated names instead of newlines, which also
	// leaves them raw and unquoted: git otherwise C-quotes non-ASCII/special
	// filenames in --name-only output, and a quoted name passed back to git
	// as a pathspec is NOT unquoted, silently matching nothing.
	//
	// --no-textconv --no-ext-diff: without them, a workspace's .gitattributes
	// can bind a diff/textconv driver to an arbitrary local command, which
	// plain git diff would execute as a subprocess with no approval gate at
	// all (diff is a read-only action here) — these flags force git to fall
	// back to its own built-in comparison instead of running that command.
	nameArgs := []string{"diff", "--no-textconv", "--no-ext-diff", "--name-only", "-z"}
	if staged {
		nameArgs = append(nameArgs, "--staged")
	}
	if p != "" {
		nameArgs = append(nameArgs, "--", p)
	}
	namesRes := g.run(ctx, "diff", false, nameArgs...)
	if namesRes.IsError {
		return namesRes
	}

	var allowed []string
	blocked := 0
	if namesRes.Content != "(ok, no output)" { // run()'s sentinel for empty output
		// --name-only output is always repository-root-relative, regardless of
		// the current effective directory (which cmd.Dir uses below for both
		// this call and the real diff). Resolving each name against the repo
		// root into an absolute path — rather than reusing it as-is — keeps
		// the pathspec correct even when a "/cd" has moved cwd into a subdir.
		root, err := g.repoRoot(ctx)
		if err != nil {
			return Err(err.Error())
		}
		// NUL-delimited entries need no trimming (that's the point of -z); a
		// bare TrimSpace here would corrupt a legitimately whitespace-leading
		// filename that git does NOT quote (plain spaces aren't special).
		for _, name := range strings.Split(namesRes.Content, "\x00") {
			if name == "" { // trailing delimiter after the last entry
				continue
			}
			abs := filepath.Join(root, name)
			if err := g.checkPathPolicy(abs); err != nil {
				blocked++
				continue
			}
			allowed = append(allowed, abs)
		}
	}

	if len(allowed) == 0 {
		if blocked > 0 {
			return Ok(strconv.Itoa(blocked) + " file(s) excluded (secrets/credentials); no other changes")
		}
		return Ok("(ok, no output)")
	}

	// --no-textconv/--no-ext-diff: this is a read-only action with no
	// approval gate, but a .gitattributes diff/textconv driver (or an
	// ext-diff command) configured in the local .git/config would otherwise
	// let plain "git diff" execute an arbitrary subprocess to produce the
	// diff content. Suppress both so diff can never run configured commands.
	args := []string{"diff", "--no-textconv", "--no-ext-diff"}
	if staged {
		args = append(args, "--staged")
	}
	args = append(args, "--")
	// ":(literal)" forces each path to match itself exactly. Without it, a
	// bare path is a pathspec GLOB: a tracked file literally named e.g. "*"
	// would make git match every other file too — including one excluded
	// above as a secret — and leak its content into this "read-only" diff.
	for _, p := range allowed {
		args = append(args, ":(literal)"+p)
	}
	res := g.run(ctx, "diff", false, args...)
	if blocked > 0 && !res.IsError {
		res.Content += "\n…[" + strconv.Itoa(blocked) + " secret/credential file(s) excluded from this diff]"
	}
	return res
}

// checkPathPolicy applies the SAME jail + secret-file checks file.read does to
// a path git is about to read the (possibly historical) content of — git's
// own argv access has no separate confinement, so show/diff would otherwise
// read straight through file.read's restrictions on the identical path.
func (g *gitTool) checkPathPolicy(path string) error {
	abs, err := g.pol.Resolve(path)
	if err != nil {
		return err
	}
	if g.pol.IsSecret(abs) {
		return errors.New("reading " + path + " is blocked (it looks like a secrets/credentials file)")
	}
	return nil
}

// repoRoot returns the repository's absolute top-level directory. It's used
// to resolve --name-only's repo-root-relative output into pathspecs that are
// unambiguous regardless of the current effective directory.
func (g *gitTool) repoRoot(ctx context.Context) (string, error) {
	res := g.run(ctx, "diff", false, "rev-parse", "--show-toplevel")
	if res.IsError {
		return "", errors.New(res.Content)
	}
	return res.Content, nil
}

func (g *gitTool) log(ctx context.Context, a Args) Result {
	n := a.Int("n", 15)
	if n < 1 {
		n = 15
	}
	return g.run(ctx, "log", false, "log", "--oneline", "-n", strconv.Itoa(n))
}

func (g *gitTool) show(ctx context.Context, a Args) Result {
	ref := a.Str("ref")
	if ref == "" {
		ref = "HEAD"
	}
	if strings.HasPrefix(ref, "-") { // a leading dash would be read as a git flag
		return Err("invalid ref (leading dash): " + ref)
	}
	if path, ok := blobPath(ref); ok {
		// <rev>:<path> blob/tree syntax returns the file's raw content at that
		// revision — --stat has no effect on a blob reference, so this is a
		// direct read of the path's content and must respect the same jail +
		// secret-file checks file.read enforces on the identical path. Unlike
		// an ordinary path, git ALWAYS resolves <path> here against the
		// REPOSITORY ROOT, never cwd or the configured jail — so when the
		// jail is a subdirectory of a larger repo, checking it the way
		// checkPathPolicy does (jail-relative) validates a different file
		// than the one git actually reads.
		if err := g.checkRevPathPolicy(ctx, path); err != nil {
			return Err(err.Error())
		}
	}
	return g.run(ctx, "show", false, "show", "--stat", ref)
}

// blobPath extracts the <path> portion of a git "<rev>:<path>" blob/tree ref,
// including its index-stage variants ":<path>" and ":<n>:<path>" (stage 0-3,
// see gitrevisions(7)). ":0:.env" names the SAME path as "HEAD:.env", but a
// naive split on the first colon reads it as path "0:.env" — which never
// matches the secret-file glob (it isn't ".env") — letting git read the real
// .env blob straight out of the index while the policy check looks at the
// wrong string. Parsed per git's own grammar so the same path is checked
// whichever form selected it.
func blobPath(ref string) (path string, ok bool) {
	if strings.HasPrefix(ref, ":") {
		rest := ref[1:]
		if len(rest) >= 2 && rest[0] >= '0' && rest[0] <= '3' && rest[1] == ':' {
			rest = rest[2:]
		}
		return rest, rest != ""
	}
	_, p, cut := strings.Cut(ref, ":")
	return p, cut && p != ""
}

// checkRevPathPolicy applies the jail + secret-file checks to <path> from a
// "<rev>:<path>" blob/tree ref, resolved against the ACTUAL git repository
// root (not the configured jail) since that's what git itself resolves path
// against. The resulting absolute path is then checked against the jail same
// as any other read, so a path outside the jail (even if inside the repo) is
// rejected.
func (g *gitTool) checkRevPathPolicy(ctx context.Context, path string) error {
	root, err := g.repoRoot(ctx)
	if err != nil {
		return err
	}
	abs, err := g.pol.Resolve(filepath.Join(root, path))
	if err != nil {
		return err
	}
	if g.pol.IsSecret(abs) {
		return errors.New("reading " + path + " is blocked (it looks like a secrets/credentials file)")
	}
	return nil
}

func (g *gitTool) add(ctx context.Context, a Args) Result {
	return g.run(ctx, "add", true, append([]string{"add", "--"}, strings.Fields(a.Str("paths"))...)...)
}

func (g *gitTool) commit(ctx context.Context, a Args) Result {
	return g.run(ctx, "commit", true, "commit", "-m", a.Str("message"))
}

func (g *gitTool) branch(ctx context.Context, a Args) Result {
	if name := a.Str("name"); name != "" {
		if strings.HasPrefix(name, "-") { // a leading dash would be read as a git flag
			return Err("invalid branch name (leading dash): " + name)
		}
		return g.run(ctx, "branch", true, "branch", "--", name)
	}
	return g.run(ctx, "branch", false, "branch")
}

func (g *gitTool) checkout(ctx context.Context, a Args) Result {
	ref := a.Str("ref")
	if strings.HasPrefix(ref, "-") { // e.g. "-f" → "git checkout -f" force-discards changes
		return Err("invalid ref (leading dash): " + ref)
	}
	// NOT "checkout -- ref": everything after "--" is a pathspec, not a
	// revision, so that always failed to switch branches at all (or, worse,
	// if a tracked file happened to share the branch's name, silently
	// discarded that file's uncommitted changes instead). The leading-dash
	// guard above already fully defends against flag injection, so "--"
	// bought no safety here, only breakage.
	return g.run(ctx, "checkout", true, "checkout", ref)
}

func (g *gitTool) run(ctx context.Context, action string, mutating bool, args ...string) Result {
	if mutating && !g.ap.Approve(ctx, "git", "git "+strings.Join(args, " ")) {
		return Err("git " + action + " denied by user")
	}
	dir, err := g.pol.Resolve(".")
	if err != nil {
		return Err(err.Error())
	}

	cctx, cancel := context.WithTimeout(ctx, defaultRunTimeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, "git", args...)
	cmd.Dir = dir
	// Kill the WHOLE process group on timeout/cancel, and bound Wait so a
	// hook, pager, or external diff/merge tool that outlives git itself (and
	// keeps holding the shared stdout/stderr pipe) can't hang this call past
	// its timeout — the same class of wedge the run tool already guards
	// against (internal/procgroup).
	procgroup.Set(cmd)
	out := textutil.NewBoundedWriter(maxRunOutput)
	cmd.Stdout, cmd.Stderr = out, out
	runErr := cmd.Run()

	body := strings.TrimRight(out.String(), "\n")
	if out.Truncated {
		body += "\n…[truncated]"
	}

	if runErr != nil {
		var ee *exec.ExitError
		switch {
		case errors.Is(runErr, exec.ErrWaitDelay):
			// git itself exited fine; we just stopped waiting on a lingering
			// child (hook/pager) still holding the output pipe. What we
			// captured is git's complete output — treat it as success.
			runErr = nil
		case cctx.Err() == context.DeadlineExceeded:
			return Err("git " + action + " timed out")
		case errors.As(runErr, &ee):
			if body == "" {
				body = runErr.Error()
			}
			return Result{Content: "git " + action + " failed:\n" + body, IsError: true}
		default:
			return Fail("git", action, "could not run git (is it installed?): "+runErr.Error(), runErr)
		}
	}
	if body == "" {
		body = "(ok, no output)"
	}
	return Ok(body)
}
