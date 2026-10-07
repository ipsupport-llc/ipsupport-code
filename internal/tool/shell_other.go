//go:build !windows

package tool

import (
	"context"
	"os"
	"os/exec"
)

// Shell picks the shell `run` uses on Windows (see shell_windows.go). Elsewhere
// it is always sh, and the setting is ignored.
type Shell string

func (Shell) desc() string { return "sh -c" }

func (Shell) argv(line string) (string, []string) { return "sh", []string{"-c", line} }

func (Shell) prepare(*exec.Cmd, string) {}

// Command runs one command line through the shell.
func (s Shell) Command(ctx context.Context, line string) *exec.Cmd {
	name, args := s.argv(line)
	return exec.CommandContext(ctx, name, args...)
}

// Interactive is the shell /shell opens: the user's own.
func (Shell) Interactive() string {
	if s := os.Getenv("SHELL"); s != "" {
		return s
	}
	return "/bin/sh"
}
