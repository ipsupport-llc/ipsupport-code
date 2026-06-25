package tool

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

const (
	runTimeout   = 60 * time.Second
	maxRunOutput = 50_000
)

type runTool struct {
	pol *policy.Engine
	ap  Approver
}

// NewRun returns the run tool: a single `shell` action gated by the policy
// engine, executed with a timeout and a jail-confined working directory.
func NewRun(p *policy.Engine, a Approver) Tool { return &runTool{pol: p, ap: a} }

func (*runTool) Name() string      { return "run" }
func (*runTool) Actions() []string { return []string{"shell"} }

func (*runTool) Description() string {
	return strings.TrimSpace(`Run a shell command (sh -c) in the workspace; returns combined stdout+stderr and the exit code. Gated by the workspace permission policy.
Actions:
  - shell: {"command": str, "cwd"?: str}
Use for builds, tests, git, package managers — anything not covered by the other tools.
NOT here — read/write files → file; web/search/fetch → web; arithmetic → calc.`)
}

func (r *runTool) Call(ctx context.Context, action string, params map[string]any) Result {
	if action != "shell" {
		return Err("run: unknown action " + action)
	}
	if err := Require(params, "command"); err != nil {
		return Err(err.Error())
	}
	command := Str(params, "command")

	switch r.pol.Run(command) {
	case policy.Deny:
		return Err("command denied by workspace policy: " + command)
	case policy.Ask:
		if !r.ap.Approve("run", command) {
			return Err("command denied by user: " + command)
		}
	}

	cwd := Str(params, "cwd")
	if cwd == "" {
		cwd = "."
	}
	dir, err := r.pol.Resolve(cwd)
	if err != nil {
		return Err(err.Error())
	}

	cctx, cancel := context.WithTimeout(ctx, runTimeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, "sh", "-c", command)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()

	body := strings.TrimRight(out.String(), "\n")
	if len(body) > maxRunOutput {
		body = body[:maxRunOutput] + "\n…[truncated]"
	}

	exit := 0
	if runErr != nil {
		var ee *exec.ExitError
		switch {
		case cctx.Err() == context.DeadlineExceeded:
			return Err(fmt.Sprintf("command timed out after %s: %s", runTimeout, command))
		case errors.As(runErr, &ee):
			exit = ee.ExitCode()
		default:
			return Fail("run", "shell", "failed to start command: "+runErr.Error(), runErr)
		}
	}

	result := fmt.Sprintf("exit %d\n%s", exit, body)
	if exit != 0 {
		return Result{Content: result, IsError: true}
	}
	return Ok(result)
}
