//go:build !windows

package procgroup

import (
	"os/exec"
	"syscall"
	"time"
)

// Set makes cmd the leader of a fresh process group, arranges for cancellation
// to kill the WHOLE group, and bounds Wait so a grandchild holding the stdout
// pipe can't hang the caller forever. Without it, killing only the direct child
// (e.g. `npm run dev` at the shell timeout) leaves its children running AND
// holding the pipe — cmd.Run() then never returns and the task wedges.
func Set(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 5 * time.Second // don't hang Wait forever on a stuck pipe holder
}
