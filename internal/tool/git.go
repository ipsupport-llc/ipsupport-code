package tool

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

type gitTool struct {
	pol     *policy.Engine
	ap      Approver
	maxOut  int // per-call output cap in bytes (see OutputBudget)
	offline bool
}

// gitNetTimeout bounds the actions that talk to a server. The 60s a local action
// gets is barely a clone's warm-up on a real repository, and the cost of being
// generous is a wait, not a wedge: the process group is still killed on expiry.
const gitNetTimeout = 5 * time.Minute

// NewGit returns the git tool. It runs git directly (argv, no shell) in the
// workspace; read-only actions run freely, mutating ones ask for approval. When
// offline is true the actions that need a server refuse, the same way the web
// tool does.
func NewGit(p *policy.Engine, ap Approver, ctxWindow int, offline bool) Tool {
	g := &gitTool{pol: p, ap: ap, maxOut: OutputBudget(ctxWindow), offline: offline}
	return NewDomain(DomainSpec{
		Name:    "git",
		Summary: "Git in the workspace. Mutating actions (add/commit/clone/pull/push…) ask approval.",
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
			{Name: "clone", Mutates: true, Params: []Param{Req("url", "str"), Opt("dir", "str", "")},
				Note: "(into the workspace)", Run: g.clone},
			{Name: "fetch", Mutates: true, Params: []Param{Opt("remote", "str", "origin")}, Run: g.fetch},
			{Name: "pull", Mutates: true, Params: []Param{Opt("remote", "str", ""), Opt("branch", "str", "")},
				Note: "(ff-only)", Run: g.pull},
			{Name: "push", Mutates: true, Params: []Param{Opt("remote", "str", ""), Opt("branch", "str", ""), Opt("set_upstream", "bool", "")},
				Note: "(PUBLISHES; never force)", Run: g.push},
			{Name: "remote", Mutates: true, Params: []Param{Opt("name", "str", ""), Opt("url", "str", "")},
				Note: "(none=list)", Run: g.remote},
		},
	})
}

// checkRemoteURL rejects the two shapes that turn a remote into something other
// than a remote: git's remote-helper syntax, where "ext::sh -c …" runs its
// argument as the transport — an arbitrary command, from a plain-looking URL —
// and a leading dash, which git reads as an option instead of an address.
func checkRemoteURL(u string) error {
	switch {
	case u == "":
		return errors.New("url is required")
	case strings.HasPrefix(u, "-"):
		return errors.New("invalid url (leading dash): " + u)
	case strings.Contains(u, "::"):
		return errors.New("remote-helper URLs are not allowed (ext:: runs a command as the transport) — use https://, ssh:// or git@host:path")
	}
	return nil
}

// checkGitName guards a remote name, branch or refspec passed straight to git.
func checkGitName(what, v string) error {
	switch {
	case strings.HasPrefix(v, "-"):
		return errors.New("invalid " + what + " (leading dash): " + v)
	case strings.Contains(v, "::"):
		return errors.New("invalid " + what + " (remote-helper syntax): " + v)
	}
	return nil
}

// repoDirFromURL is the directory git itself would clone into, worked out here
// so the destination can be jailed BEFORE git runs rather than discovered after.
func repoDirFromURL(u string) string {
	s := strings.TrimSuffix(strings.TrimRight(u, "/"), ".git")
	if i := strings.LastIndexAny(s, "/:"); i >= 0 {
		s = s[i+1:]
	}
	return s
}

func (g *gitTool) clone(ctx context.Context, a Args) Result {
	url := strings.TrimSpace(a.Str("url"))
	if err := checkRemoteURL(url); err != nil {
		return Err(err.Error())
	}
	dir := strings.TrimSpace(a.Str("dir"))
	if dir == "" {
		dir = repoDirFromURL(url)
	}
	if dir == "" {
		return Err("could not work out a directory name from " + url + " — pass dir explicitly")
	}
	if strings.HasPrefix(dir, "-") {
		return Err("invalid dir (leading dash): " + dir)
	}
	// A clone writes a whole tree to disk, so its destination goes through the
	// same jail the file tool enforces — resolved here, and handed to git as an
	// absolute path, so a "../.." in dir is refused before git ever runs.
	abs, err := g.pol.Resolve(dir)
	if err != nil {
		return Err(err.Error())
	}
	return g.runNet(ctx, "clone", "clone", "--", url, abs)
}

func (g *gitTool) fetch(ctx context.Context, a Args) Result {
	remote := strings.TrimSpace(a.Str("remote"))
	if remote == "" {
		remote = "origin"
	}
	if err := checkGitName("remote", remote); err != nil {
		return Err(err.Error())
	}
	return g.runNet(ctx, "fetch", "fetch", "--", remote)
}

func (g *gitTool) pull(ctx context.Context, a Args) Result {
	remote, branch := strings.TrimSpace(a.Str("remote")), strings.TrimSpace(a.Str("branch"))
	if branch != "" && remote == "" {
		return Err("a branch needs a remote too (e.g. remote=origin branch=main)")
	}
	// --ff-only: a plain pull can start a MERGE, and a conflicted merge leaves a
	// working tree the agent then has to understand mid-task. Failing cleanly on
	// a diverged branch is the better answer — fetch and merge deliberately.
	args := []string{"pull", "--ff-only"}
	for what, v := range map[string]string{"remote": remote, "branch": branch} {
		if v != "" {
			if err := checkGitName(what, v); err != nil {
				return Err(err.Error())
			}
		}
	}
	if remote != "" {
		args = append(args, "--", remote)
		if branch != "" {
			args = append(args, branch)
		}
	}
	return g.runNet(ctx, "pull", args...)
}

func (g *gitTool) push(ctx context.Context, a Args) Result {
	remote, branch := strings.TrimSpace(a.Str("remote")), strings.TrimSpace(a.Str("branch"))
	if branch != "" && remote == "" {
		return Err("a branch needs a remote too (e.g. remote=origin branch=main)")
	}
	for what, v := range map[string]string{"remote": remote, "branch": branch} {
		if v != "" {
			if err := checkGitName(what, v); err != nil {
				return Err(err.Error())
			}
		}
	}
	// No --force, and no way to ask for one: a force-push destroys history on a
	// server, which is the one git mistake an approval prompt cannot undo.
	args := []string{"push"}
	if a.Bool("set_upstream") {
		args = append(args, "--set-upstream")
	}
	if remote != "" {
		args = append(args, "--", remote)
		if branch != "" {
			args = append(args, branch)
		}
	}
	return g.runNet(ctx, "push", args...)
}

// remote lists remotes, or adds one (repointing it if the name already exists).
// It touches no server, so it works offline and keeps the local timeout.
func (g *gitTool) remote(ctx context.Context, a Args) Result {
	name, url := strings.TrimSpace(a.Str("name")), strings.TrimSpace(a.Str("url"))
	if name == "" && url == "" {
		return g.run(ctx, "remote", false, "remote", "-v")
	}
	if name == "" || url == "" {
		return Err("adding a remote needs both name and url (pass neither to list them)")
	}
	if err := checkGitName("remote", name); err != nil {
		return Err(err.Error())
	}
	if err := checkRemoteURL(url); err != nil {
		return Err(err.Error())
	}
	// Probe first (read-only, no approval) so the one approved call is the right
	// one: "remote add" fails on an existing name, "set-url" on a missing one.
	verb := "add"
	if probe := g.run(ctx, "remote", false, "remote", "get-url", "--", name); !probe.IsError {
		verb = "set-url"
	}
	return g.run(ctx, "remote", true, "remote", verb, "--", name, url)
}

// runNet runs an action that talks to a server.
func (g *gitTool) runNet(ctx context.Context, action string, args ...string) Result {
	if g.offline {
		return Err("offline mode is ON — git " + action + " needs the network. This is temporary: run /offline off when you're back online.")
	}
	// protocol.ext.allow=never is the second lock behind checkRemoteURL: a
	// .gitmodules entry or a server redirect can name an ext:: URL that never
	// passed through the url parameter at all, and ext:: runs its argument as a
	// command. It's a global -c flag, so it's kept out of the approval text.
	flags := []string{"-c", "protocol.ext.allow=never"}
	// Without these, a private repo with no usable credentials does not fail —
	// it BLOCKS, on a username prompt or an ssh passphrase prompt, against a
	// stdin nobody is typing into, until the timeout kills it minutes later.
	env := []string{"GIT_TERMINAL_PROMPT=0"}
	if os.Getenv("GIT_SSH_COMMAND") == "" { // don't stomp a deliberately configured one
		env = append(env, "GIT_SSH_COMMAND=ssh -o BatchMode=yes")
	}
	return g.runWith(ctx, action, true, gitNetTimeout, env, flags, args...)
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
	// This is a second, separate "git diff" invocation from the enumeration
	// above, so it needs its own copy of these action-specific flags.
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
	paths := strings.Fields(a.Str("paths"))
	if len(paths) == 0 {
		return Err("paths is required (space-separated)")
	}
	// Check what would actually be STAGED, not what was typed. This matters now
	// that push exists: a secrets file the policy engine will not let the file
	// tool read can still be committed and published, and "git add ." — by far
	// the most common call — names no secret at all while staging every one of
	// them. --dry-run prints exactly the paths the real add would take, the same
	// enumerate-then-check shape diff already uses.
	dry := append([]string{"add", "--dry-run", "--ignore-missing", "--"}, paths...)
	if listed := g.run(ctx, "add", false, dry...); !listed.IsError {
		var blocked []string
		for _, line := range strings.Split(listed.Content, "\n") {
			name, ok := addedPath(line)
			if !ok {
				continue
			}
			root, err := g.repoRoot(ctx)
			if err != nil {
				return Err(err.Error())
			}
			if abs, err := g.pol.Resolve(filepath.Join(root, name)); err != nil || g.pol.IsSecret(abs) {
				blocked = append(blocked, name)
			}
		}
		if len(blocked) > 0 {
			return Err("refusing to stage " + strings.Join(blocked, ", ") +
				" — it looks like a secrets/credentials file, and a staged secret is one commit away from being pushed")
		}
	}
	return g.run(ctx, "add", true, append([]string{"add", "--"}, paths...)...)
}

// addedPath pulls the path out of one "git add --dry-run" line, which reads
// `add 'some/path'` (or `remove 'some/path'`). Anything else is not a staging.
func addedPath(line string) (string, bool) {
	rest, ok := strings.CutPrefix(strings.TrimSpace(line), "add ")
	if !ok {
		return "", false
	}
	rest = strings.TrimSpace(rest)
	if len(rest) < 2 || rest[0] != '\'' || rest[len(rest)-1] != '\'' {
		return "", false
	}
	return rest[1 : len(rest)-1], true
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
	return g.runWith(ctx, action, mutating, defaultRunTimeout, nil, nil, args...)
}

// runWith is run with the knobs the networked actions need: a longer timeout,
// extra environment, and extra global "-c" flags. The flags are deliberately not
// part of the approval text — what the user is asked to approve is the command
// they would have typed, not our hardening.
func (g *gitTool) runWith(ctx context.Context, action string, mutating bool, timeout time.Duration, extraEnv, gitFlags []string, args ...string) Result {
	if mutating && !g.ap.Approve(ctx, "git", "git "+strings.Join(args, " ")) {
		return Err("git " + action + " denied by user")
	}
	dir, err := g.pol.Resolve(".")
	if err != nil {
		return Err(err.Error())
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	// -c core.fsmonitor=: applied to EVERY action, mutating or not. A
	// workspace's local .git/config can bind core.fsmonitor to an arbitrary
	// executable (a legitimate perf feature for large repos) that git would
	// otherwise invoke as a subprocess on actions like status/diff that
	// consult it — with no approval gate, since those are read-only. The
	// override is a harmless no-op for actions that never consult fsmonitor
	// (log, show, add, commit, branch, checkout, init), so it's applied here
	// once for all of them rather than requiring each action to remember it.
	// It must precede the subcommand (git -c is a global option, unlike
	// diff's --no-textconv/--no-ext-diff, which are the diff subcommand's own
	// flags and stay action-specific since they'd be invalid on e.g. commit).
	gitArgs := append(append([]string{"-c", "core.fsmonitor="}, gitFlags...), args...)
	cmd := exec.CommandContext(cctx, "git", gitArgs...)
	cmd.Dir = dir
	if len(extraEnv) > 0 {
		cmd.Env = append(os.Environ(), extraEnv...)
	}
	// Kill the WHOLE process group on timeout/cancel, and bound Wait so a
	// hook, pager, or external diff/merge tool that outlives git itself (and
	// keeps holding the shared stdout/stderr pipe) can't hang this call past
	// its timeout — the same class of wedge the run tool already guards
	// against (internal/procgroup).
	procgroup.Set(cmd)
	out := textutil.NewBoundedWriter(g.maxOut)
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
