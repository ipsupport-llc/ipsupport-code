//go:build windows

package procgroup

import (
	"os/exec"
	"time"
)

// Set on Windows: no job-object tree kill wired up yet, but WaitDelay still
// bounds Wait so a child holding the output pipe can't hang the caller forever.
func Set(cmd *exec.Cmd) {
	cmd.WaitDelay = 5 * time.Second
}
