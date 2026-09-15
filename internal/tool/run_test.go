package tool

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

func runToolFor(t *testing.T, dir, def string, ap Approver, deny []string) Tool {
	t.Helper()
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "."} // jail for cwd
	c.Run = config.RunPolicy{Default: def, Deny: deny}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	return NewRun(e, ap, 0, 0)
}

func TestRunEcho(t *testing.T) {
	tl := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "echo hi"})
	if r.IsError || !strings.Contains(r.Content, "hi") {
		t.Errorf("echo = %+v, want output containing hi", r)
	}
}

func TestRunAppliesCmdWrapper(t *testing.T) {
	c := config.Default()
	c.Workspace = t.TempDir()
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	c.Run = config.RunPolicy{Default: "allow"}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	var gotName string
	var gotArgs []string
	wrap := func(name string, args []string) (string, []string) {
		gotName, gotArgs = name, args
		return "sh", []string{"-c", "echo WRAPPED"} // rewrite the command entirely
	}
	tl := NewRun(e, yes(), 0, 0, wrap)
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "echo original"})
	if r.IsError || !strings.Contains(r.Content, "WRAPPED") || strings.Contains(r.Content, "original") {
		t.Errorf("wrapper not applied: %+v", r)
	}
	if gotName != "sh" || len(gotArgs) != 2 || gotArgs[1] != "echo original" {
		t.Errorf("wrapper saw %s %v, want sh [-c echo original]", gotName, gotArgs)
	}
}

// A command whose CHILD outlives the shell while holding the output pipe (a
// backgrounded process, a dev server at timeout) must not hang the tool call:
// WaitDelay bounds the pipe wait. Without it this test blocks ~20s.
func TestRunReturnsWhenChildHoldsPipe(t *testing.T) {
	tl := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	start := time.Now()
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "sleep 20 & echo started"})
	if d := time.Since(start); d > 15*time.Second {
		t.Fatalf("tool call blocked %s on a pipe-holding child, want a bounded return", d)
	}
	if r.IsError || !strings.Contains(r.Content, "started") {
		t.Errorf("result = %+v, want the shell's own output", r)
	}
}

func TestRunPerCallTimeout(t *testing.T) {
	tl := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "sleep 3", "timeout": 1})
	if !r.IsError || !strings.Contains(r.Content, "timed out") {
		t.Errorf("sleep 3 with timeout=1 = %+v, want a timeout error", r)
	}
}

func TestRunConfigTimeout(t *testing.T) {
	c := config.Default()
	c.Workspace = t.TempDir()
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	c.Run = config.RunPolicy{Default: "allow"}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	tl := NewRun(e, yes(), 1*time.Second, 0) // 1s default from config
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "sleep 3"})
	if !r.IsError || !strings.Contains(r.Content, "timed out") {
		t.Errorf("sleep 3 with 1s default = %+v, want a timeout error", r)
	}
}

func TestRunDeniedNotExecuted(t *testing.T) {
	dir := t.TempDir()
	tl := runToolFor(t, dir, "ask", yes(), []string{"touch*"})
	sentinel := filepath.Join(dir, "created.txt")

	r := tl.Call(context.Background(), "shell", map[string]any{"command": "touch " + sentinel})
	if !r.IsError {
		t.Errorf("denied command = %+v, want error", r)
	}
	if _, err := os.Stat(sentinel); err == nil {
		t.Error("denied command was executed (sentinel file created)")
	}
}

func TestRunAskDeniedByUser(t *testing.T) {
	tl := runToolFor(t, t.TempDir(), "ask", no(), nil)
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "echo hi"})
	if !r.IsError || !strings.Contains(r.Content, "denied by user") {
		t.Errorf("ask+deny = %+v, want 'denied by user'", r)
	}
}

// Reported live, with the debug log to prove it: on a 32.8k-token window two
// curl commands that each dumped a whole HTML page took a run's context from 5k
// to 44k tokens in two steps — past the window, after which the model
// degenerated into echoing its own prompt. A flat 50 000-byte cap cannot prevent
// that: it is nothing against a 200k window and roughly 40% of a 32.8k one. The
// cap has to be read against the window it is spending.
func TestOutputBudgetScalesToTheContextWindow(t *testing.T) {
	small, large := OutputBudget(32_800), OutputBudget(200_000)
	if small >= large {
		t.Errorf("small window got %d and large got %d — the cap must scale", small, large)
	}
	if large != maxRunOutput {
		t.Errorf("large window = %d, want it capped at the %d ceiling", large, maxRunOutput)
	}
	// One call must not be able to eat the small window on its own: the live
	// failure was two calls totalling ~38k tokens against a 32.8k window.
	if tokens := small / charsPerToken; tokens > 32_800/4 {
		t.Errorf("one call may spend %d tokens of a 32.8k window — too much", tokens)
	}
	// An unknown window behaves exactly as before rather than guessing.
	if got := OutputBudget(0); got != maxRunOutput {
		t.Errorf("unknown window = %d, want the old absolute cap %d", got, maxRunOutput)
	}
	// A tiny window still leaves room for a real error message.
	if got := OutputBudget(1_000); got != minRunOutput {
		t.Errorf("tiny window = %d, want the %d floor", got, minRunOutput)
	}
}
