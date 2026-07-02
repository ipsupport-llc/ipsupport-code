//go:build windows

package main

import "os/exec"

// setProcGroup is a no-op on Windows: CommandContext kills the direct child, and
// a full job-object tree kill isn't wired up yet.
func setProcGroup(cmd *exec.Cmd) {}
