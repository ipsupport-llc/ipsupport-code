//go:build !windows

package tool

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
)

// Shell picks the shell `run` uses on Windows (see shell_windows.go). Elsewhere
// it is always sh, and the setting is ignored.
type Shell string

func (Shell) desc() string { return "sh -c" }

func (Shell) argv(line string) (string, []string) { return "sh", []string{"-c", line} }

func (Shell) prepare(*exec.Cmd, string) {}

// PromptNote tells the model which shell its commands run in, for the system
// prompt — the sh actually installed: on Debian/Ubuntu it is dash, where bash
// syntax fails.
func (Shell) PromptNote() string {
	note := " Shell commands (the run tool) execute in sh -c, POSIX sh"
	if p, err := exec.LookPath("sh"); err == nil {
		if real, err := filepath.EvalSymlinks(p); err == nil && filepath.Base(real) != "sh" {
			note += " (here " + filepath.Base(real) + ")"
		}
	}
	return note + ": no bash-only syntax ([[ ]], arrays, {a,b}, source) — wrap it in bash -c '…' when you need it."
}

// CleanOutput turns what the shell printed into plain text; sh needs nothing.
func (Shell) CleanOutput(out string) string { return out }

// Command runs one command line through the shell.
func (s Shell) Command(ctx context.Context, line string) *exec.Cmd {
	name, args := s.argv(line)
	return exec.CommandContext(ctx, name, args...)
}

// InteractiveCommand opens Interactive.
func (s Shell) InteractiveCommand(ctx context.Context) *exec.Cmd {
	return exec.CommandContext(ctx, s.Interactive())
}

// Interactive is the shell /shell opens: the user's own.
func (Shell) Interactive() string {
	if s := os.Getenv("SHELL"); s != "" {
		return s
	}
	return "/bin/sh"
}
