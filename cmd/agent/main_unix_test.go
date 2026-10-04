//go:build !windows

package main

import (
	"bufio"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func mkfifo(path string) error { return syscall.Mkfifo(path, 0o600) }

// Process exit must not leave a background job's subprocess running: shutdownJobs
// (called from cleanup in main.go) cancels every active job before the process
// exits. For an external-agent job in particular, cancelling only the Go-side
// context isn't enough on its own unless the subprocess is actually torn down —
// so this asserts the underlying OS process is dead too (see procgroup.Set in
// spawnExternalAgent), not just that our bookkeeping says the job is done.
func TestShutdownJobsKillsExternalAgentProcess(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	pidFile := filepath.Join(cfg.Workspace, "child.pid")
	cfg.Agents = map[string]config.AgentProfile{
		"longrun": {Kind: "external", Command: "sh", Args: []string{"-c", "echo $$ > " + pidFile + "; sleep 30"}},
	}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	if _, err := a.spawnAgentBackground(context.Background(), "longrun", "go", ""); err != nil {
		t.Fatal(err)
	}

	var pid int
	for i := 0; i < 100; i++ { // wait for the child to write its own PID
		if b, err := os.ReadFile(pidFile); err == nil {
			if p, perr := strconv.Atoi(strings.TrimSpace(string(b))); perr == nil && p > 0 {
				pid = p
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if pid == 0 {
		t.Fatal("external agent never wrote its PID — test setup broken")
	}
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("child process %d not alive before shutdown: %v", pid, err)
	}

	a.shutdownJobs()

	if n := a.jobsPending(); n != 0 {
		t.Errorf("jobsPending after shutdownJobs = %d, want 0 (job context should be cancelled)", n)
	}
	if err := syscall.Kill(pid, 0); err == nil {
		t.Errorf("child process %d still alive after shutdownJobs — external-agent subprocess was not killed", pid)
	}
}
