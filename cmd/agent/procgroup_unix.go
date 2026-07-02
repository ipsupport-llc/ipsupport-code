//go:build !windows

package main

import (
	"os/exec"
	"syscall"
	"time"
)

// setProcGroup makes cmd the leader of a fresh process group and arranges for
// cancellation to kill the WHOLE group. External CLI agents fork helpers of their
// own; killing only the direct child on esc/timeout would leave orphans running
// (and, for API-backed CLIs, still burning tokens).
func setProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 5 * time.Second // don't hang Wait forever on a stuck pipe reader
}
