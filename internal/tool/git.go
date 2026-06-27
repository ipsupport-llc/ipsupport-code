package tool

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"strconv"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
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
	args := []string{"diff"}
	if a.Bool("staged") {
		args = append(args, "--staged")
	}
	if p := a.Str("path"); p != "" {
		args = append(args, "--", p)
	}
	return g.run(ctx, "diff", false, args...)
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
	return g.run(ctx, "show", false, "show", "--stat", ref)
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
	return g.run(ctx, "checkout", true, "checkout", "--", ref)
}

func (g *gitTool) run(ctx context.Context, action string, mutating bool, args ...string) Result {
	if mutating && !g.ap.Approve("git", "git "+strings.Join(args, " ")) {
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
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	runErr := cmd.Run()

	body := strings.TrimRight(out.String(), "\n")
	if clipped, truncated := textutil.Clip(body, maxRunOutput); truncated {
		body = clipped + "\n…[truncated]"
	}

	if runErr != nil {
		var ee *exec.ExitError
		switch {
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
