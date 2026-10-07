//go:build windows

package tool

import (
	"context"
	"os"
	"os/exec"
	"syscall"
)

// Shell picks the shell `run` uses on Windows, which has no sh (config
// run.shell):
//
//	""           pwsh if installed, else Windows PowerShell (the default)
//	"pwsh"       PowerShell 7
//	"powershell" Windows PowerShell 5.1 — no && between commands
//	"cmd"        cmd.exe
//
// An unknown value is the default.
type Shell string

func (s Shell) powerShellExe() string {
	switch s {
	case "pwsh":
		return "pwsh.exe"
	case "powershell":
		return "powershell.exe"
	}
	if p, err := exec.LookPath("pwsh.exe"); err == nil {
		return p
	}
	return "powershell.exe"
}

func (s Shell) desc() string {
	if s == "cmd" {
		return "cmd /c"
	}
	return "PowerShell"
}

func (s Shell) argv(line string) (string, []string) {
	if s == "cmd" {
		// /d skips AutoRun; /s takes the rest of the line as is between its
		// outer quotes. The line itself goes in raw, by prepare.
		return "cmd.exe", []string{"/d", "/s", "/c", line}
	}
	return s.powerShellExe(), powerShellArgs(line)
}

// prepare finishes a command argv built: cmd.exe gets its command line
// verbatim — Go quotes arguments by the C runtime's rules, which cmd.exe does
// not follow, and would mangle any line with quotes in it; PowerShell gets an
// environment without color codes.
func (s Shell) prepare(cmd *exec.Cmd, line string) {
	if s == "cmd" {
		if cmd.SysProcAttr == nil {
			cmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		cmd.SysProcAttr.CmdLine = `cmd.exe /d /s /c "` + line + `"`
		return
	}
	cmd.Env = append(os.Environ(), powerShellEnv...)
}

// Command runs one command line through the shell.
func (s Shell) Command(ctx context.Context, line string) *exec.Cmd {
	name, args := s.argv(line)
	cmd := exec.CommandContext(ctx, name, args...)
	s.prepare(cmd, line)
	return cmd
}

// Interactive is the shell /shell opens.
func (s Shell) Interactive() string {
	if s == "cmd" {
		return "cmd.exe"
	}
	return s.powerShellExe()
}
