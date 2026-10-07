//go:build windows

package procgroup

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"time"
)

// Set on Windows: there is no process group to signal, so cancelling kills the
// whole tree with taskkill /T — which walks it by parent PID, anchored at the
// shell while it is still alive — before the shell itself. A plain Kill left a
// dev server started under npm holding its port after a timeout. WaitDelay
// still bounds Wait so a child holding the output pipe can't hang the caller.
func Set(cmd *exec.Cmd) {
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		_ = exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid)).Run()
		if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			return err
		}
		return nil
	}
}
