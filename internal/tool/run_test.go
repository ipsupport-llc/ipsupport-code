package tool

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	return NewRun(e, ap)
}

func TestRunEcho(t *testing.T) {
	tl := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := tl.Call(context.Background(), "shell", map[string]any{"command": "echo hi"})
	if r.IsError || !strings.Contains(r.Content, "hi") {
		t.Errorf("echo = %+v, want output containing hi", r)
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
