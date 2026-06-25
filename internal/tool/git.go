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
func NewGit(p *policy.Engine, a Approver) Tool { return &gitTool{pol: p, ap: a} }

func (*gitTool) Name() string { return "git" }
func (*gitTool) Actions() []string {
	return []string{"status", "diff", "log", "show", "add", "commit", "branch", "checkout"}
}

func (*gitTool) Description() string {
	return strings.TrimSpace(`Run common git operations in the workspace. Read-only actions run freely; mutating ones (add/commit/branch/checkout) ask for approval.
Actions:
  - status:   {}                              short status + branch
  - diff:     {"path"?: str, "staged"?: bool}
  - log:      {"n"?: int=15}                  oneline history
  - show:     {"ref"?: str="HEAD"}            commit details (--stat)
  - add:      {"paths": str}                  stage paths (space-separated)
  - commit:   {"message": str}
  - branch:   {"name"?: str}                  no name = list; name = create
  - checkout: {"ref": str}                    switch branch/commit
NOT here — non-git shell → run; read/write files → file.`)
}

func (g *gitTool) Call(ctx context.Context, action string, params map[string]any) Result {
	switch action {
	case "status":
		return g.run(ctx, action, false, "status", "--short", "--branch")
	case "diff":
		args := []string{"diff"}
		if b, _ := params["staged"].(bool); b {
			args = append(args, "--staged")
		}
		if p := Str(params, "path"); p != "" {
			args = append(args, "--", p)
		}
		return g.run(ctx, action, false, args...)
	case "log":
		n := Int(params, "n", 15)
		if n < 1 {
			n = 15
		}
		return g.run(ctx, action, false, "log", "--oneline", "-n", strconv.Itoa(n))
	case "show":
		ref := Str(params, "ref")
		if ref == "" {
			ref = "HEAD"
		}
		return g.run(ctx, action, false, "show", "--stat", ref)
	case "add":
		if err := Require(params, "paths"); err != nil {
			return Err(err.Error())
		}
		return g.run(ctx, action, true, append([]string{"add", "--"}, strings.Fields(Str(params, "paths"))...)...)
	case "commit":
		if err := Require(params, "message"); err != nil {
			return Err(err.Error())
		}
		return g.run(ctx, action, true, "commit", "-m", Str(params, "message"))
	case "branch":
		if name := Str(params, "name"); name != "" {
			return g.run(ctx, action, true, "branch", name)
		}
		return g.run(ctx, action, false, "branch")
	case "checkout":
		if err := Require(params, "ref"); err != nil {
			return Err(err.Error())
		}
		return g.run(ctx, action, true, "checkout", Str(params, "ref"))
	}
	return Err("git: unknown action " + action)
}

func (g *gitTool) run(ctx context.Context, action string, mutating bool, args ...string) Result {
	if mutating && !g.ap.Approve("git", "git "+strings.Join(args, " ")) {
		return Err("git " + action + " denied by user")
	}
	dir, err := g.pol.Resolve(".")
	if err != nil {
		return Err(err.Error())
	}

	cctx, cancel := context.WithTimeout(ctx, runTimeout)
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
