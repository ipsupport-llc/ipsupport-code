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
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

const (
	defaultRunTimeout = 60 * time.Second
	maxRunTimeout     = 60 * time.Minute // ceiling for a per-call override
	maxRunOutput      = 50_000
)

type runTool struct {
	pol     *policy.Engine
	ap      Approver
	timeout time.Duration // default per-command wall-clock limit
}

// NewRun returns the run tool: a single `shell` action gated by the policy
// engine, executed with a timeout and a jail-confined working directory. The
// default timeout comes from config (run.timeout_seconds); 0 falls back to 60s.
func NewRun(p *policy.Engine, ap Approver, defaultTimeout time.Duration) Tool {
	if defaultTimeout <= 0 {
		defaultTimeout = defaultRunTimeout
	}
	r := &runTool{pol: p, ap: ap, timeout: defaultTimeout}
	return NewDomain(DomainSpec{
		Name:    "run",
		Summary: "Run a shell command (sh -c) in the workspace; returns combined stdout+stderr and the exit code. Gated by the workspace permission policy.",
		Details: "Use for builds, tests, package managers — anything not covered by the other tools.",
		NotHere: "NOT here — read/write files → file; web/search/fetch → web; arithmetic → calc.",
		Actions: []Action{{
			Name:    "shell",
			Mutates: true,
			Params:  []Param{Req("command", "str"), Opt("cwd", "str", ""), Opt("timeout", "int", "")},
			Note:    "(timeout = seconds before the command is killed; raise it for slow builds/tests)",
			Run:     r.shell,
		}},
	})
}

func (r *runTool) shell(ctx context.Context, a Args) Result {
	command := a.Str("command")

	switch r.pol.Run(command) {
	case policy.Deny:
		return Err("command denied by workspace policy: " + command)
	case policy.Ask:
		if !r.ap.Approve("run", command) {
			return Err("command denied by user: " + command)
		}
	}

	cwd := a.Str("cwd")
	if cwd == "" {
		cwd = "."
	}
	dir, err := r.pol.Resolve(cwd)
	if err != nil {
		return Err(err.Error())
	}

	timeout := r.timeout
	if s := a.Int("timeout", 0); s > 0 { // per-call override, capped
		if timeout = time.Duration(s) * time.Second; timeout > maxRunTimeout {
			timeout = maxRunTimeout
		}
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, "sh", "-c", command)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()

	body := strings.TrimRight(out.String(), "\n")
	if clipped, truncated := textutil.Clip(body, maxRunOutput); truncated {
		body = clipped + "\n…[truncated]"
	}

	exit := 0
	if runErr != nil {
		var ee *exec.ExitError
		switch {
		case cctx.Err() == context.DeadlineExceeded:
			return Err(fmt.Sprintf("command timed out after %s (raise the run timeout, or pass a larger `timeout`): %s", timeout, command))
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
