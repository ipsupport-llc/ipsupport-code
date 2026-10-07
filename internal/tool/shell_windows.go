//go:build windows

package tool

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/windows/registry"
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
	switch {
	case s == "cmd":
		return "cmd /c"
	case strings.EqualFold(filepath.Base(s.powerShellExe()), "powershell.exe"):
		return "Windows PowerShell 5.1: no &&, chain with ;"
	}
	return "PowerShell"
}

// PromptNote tells the model which shell its commands run in — the one
// configured and installed, not an assumption — for the system prompt.
func (s Shell) PromptNote() string {
	if s == "cmd" {
		return " Shell commands (the run tool) execute in cmd.exe: write cmd syntax, not sh."
	}
	return " Shell commands (the run tool) execute in " + s.desc() + " — write PowerShell, not sh/bash."
}

// CleanOutput turns what the shell printed into plain text: Windows
// PowerShell 5.1 writes its errors as CLIXML (see decodeCLIXML).
func (s Shell) CleanOutput(out string) string {
	if s == "cmd" {
		return out
	}
	return decodeCLIXML(out)
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
//
// Both get PATH as the system has it now (freshPath), so a tool installed
// while the agent runs is found without restarting it.
func (s Shell) prepare(cmd *exec.Cmd, line string) {
	env := withEnv(os.Environ(), "Path", freshPath())
	if s == "cmd" {
		if cmd.SysProcAttr == nil {
			cmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		cmd.SysProcAttr.CmdLine = `cmd.exe /d /s /c "` + line + `"`
		cmd.Env = env
		return
	}
	cmd.Env = append(env, powerShellEnv...)
}

// freshPath is this process's PATH plus what the registry's has gained since
// it started. Windows hands a process its environment once, at launch; an
// installer that adds itself to PATH writes the registry, which only a new
// process reads — so without this, node or python installed mid-session stays
// "not found" until the agent is restarted from a new terminal.
func freshPath() string {
	var fresh []string
	for _, k := range []struct {
		root registry.Key
		path string
	}{
		{registry.LOCAL_MACHINE, `SYSTEM\CurrentControlSet\Control\Session Manager\Environment`},
		{registry.CURRENT_USER, `Environment`},
	} {
		key, err := registry.OpenKey(k.root, k.path, registry.QUERY_VALUE)
		if err != nil {
			continue
		}
		v, _, err := key.GetStringValue("Path")
		key.Close()
		if err != nil {
			continue
		}
		if x, err := registry.ExpandString(v); err == nil { // REG_EXPAND_SZ: %SystemRoot%…
			v = x
		}
		fresh = append(fresh, v)
	}
	return mergePath(os.Getenv("PATH"), fresh...)
}

// Command runs one command line through the shell.
func (s Shell) Command(ctx context.Context, line string) *exec.Cmd {
	name, args := s.argv(line)
	cmd := exec.CommandContext(ctx, name, args...)
	s.prepare(cmd, line)
	return cmd
}

// InteractiveCommand opens Interactive with the same fresh PATH as run, so
// /shell finds what run finds.
func (s Shell) InteractiveCommand(ctx context.Context) *exec.Cmd {
	cmd := exec.CommandContext(ctx, s.Interactive())
	cmd.Env = withEnv(os.Environ(), "Path", freshPath())
	return cmd
}

// Interactive is the shell /shell opens.
func (s Shell) Interactive() string {
	if s == "cmd" {
		return "cmd.exe"
	}
	return s.powerShellExe()
}
