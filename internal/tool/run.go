package tool

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

const (
	defaultRunTimeout = 60 * time.Second
	maxRunTimeout     = 60 * time.Minute // ceiling for a per-call override
	maxRunOutput      = 50_000           // ceiling for one call's captured output, in bytes
	minRunOutput      = 4_000            // floor: below this even a short error message gets cut
)

// outputShare is the fraction of the model's context window one tool call may
// spend on its own output, and charsPerToken the rough conversion used to get
// there. Reported live, with the debug log to prove it: on a 32.8k-token window,
// two curl commands that each dumped a whole HTML page took the run's context
// from 5k to 44k tokens in two steps — past the window, after which the model
// degenerated into echoing its own prompt and the client's loop detector killed
// the run. maxRunOutput alone could not prevent that: 50 000 bytes is nothing
// against a 200k window and roughly 40% of a 32.8k one, so the cap has to be
// read against the window it is spending, not as an absolute.
const (
	outputShare   = 0.12
	charsPerToken = 4
)

// OutputBudget is the per-call output cap for the run and git tools, scaled to
// the active model's context window. A zero or unknown window keeps the old
// absolute cap — better to behave exactly as before than to guess.
func OutputBudget(ctxWindow int) int {
	if ctxWindow <= 0 {
		return maxRunOutput
	}
	n := int(float64(ctxWindow*charsPerToken) * outputShare)
	if n > maxRunOutput {
		n = maxRunOutput
	}
	if n < minRunOutput {
		n = minRunOutput
	}
	return n
}

// CmdWrapper rewrites the exec (name, args) before it runs — used to wrap a
// command in an OS sandbox. It knows nothing about the sandbox itself; the app
// injects a closure. nil means run the command directly.
type CmdWrapper func(name string, args []string) (string, []string)

type runTool struct {
	pol     *policy.Engine
	ap      Approver
	timeout time.Duration // default per-command wall-clock limit
	wrap    CmdWrapper    // optional OS-sandbox wrapper (nil = run directly)
	maxOut  int           // per-call output cap in bytes (see OutputBudget)
}

// NewRun returns the run tool: a single `shell` action gated by the policy
// engine, executed with a timeout and a jail-confined working directory. The
// default timeout comes from config (run.timeout_seconds); 0 falls back to 60s.
func NewRun(p *policy.Engine, ap Approver, defaultTimeout time.Duration, ctxWindow int, wrap ...CmdWrapper) Tool {
	if defaultTimeout <= 0 {
		defaultTimeout = defaultRunTimeout
	}
	r := &runTool{pol: p, ap: ap, timeout: defaultTimeout, maxOut: OutputBudget(ctxWindow)}
	if len(wrap) > 0 {
		r.wrap = wrap[0]
	}
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
		if !r.ap.Approve(ctx, "run", command) {
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
	if s := a.Int("timeout", 0); s > 0 { // per-call override, capped (clamp before
		maxSecs := int(maxRunTimeout / time.Second) // multiplying so a huge value
		if s > maxSecs {                            // can't overflow to a negative
			s = maxSecs // duration → instant timeout
		}
		timeout = time.Duration(s) * time.Second
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	name, args := "sh", []string{"-c", command}
	if r.wrap != nil { // wrap in the OS sandbox (confines writes to the workspace)
		name, args = r.wrap(name, args)
	}
	cmd := exec.CommandContext(cctx, name, args...)
	cmd.Dir = dir
	// Kill the WHOLE process group on timeout/cancel, and bound Wait so a child
	// that outlives the shell while holding the output pipe (a dev server, a
	// backgrounded process) can't hang this tool call forever.
	procgroup.Set(cmd)
	out := textutil.NewBoundedWriter(r.maxOut)
	cmd.Stdout = out
	cmd.Stderr = out
	runErr := cmd.Run()

	body := strings.TrimRight(out.String(), "\n")
	if out.Truncated {
		body += "\n…[truncated]"
	}

	exit := 0
	if runErr != nil {
		var ee *exec.ExitError
		switch {
		case errors.Is(runErr, exec.ErrWaitDelay):
			// The shell itself exited fine; we just stopped waiting for a lingering
			// child that still held the output pipe (e.g. something backgrounded).
			// The output we have is the shell's complete output — a success.
			runErr = nil
		case cctx.Err() == context.DeadlineExceeded:
			// Not silent: tell the model it was killed, how to allow longer (the
			// exact param + an example), and hand back whatever ran before the kill.
			msg := fmt.Sprintf("command timed out after %s and was killed: %s\n"+
				"If it just needs more time, re-run with a larger timeout — add \"timeout\": %d (seconds) to the params.",
				timeout, command, int(timeout.Seconds())*4)
			if body != "" {
				msg += "\n--- output captured before the timeout ---\n" + body
			}
			return Err(msg)
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
