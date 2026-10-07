package tool

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The PowerShell `run` uses on Windows, run for real wherever pwsh is
// installed (CI's Linux runners have it): the exit code of the last command,
// plain-text errors, and a line that reaches PowerShell exactly as written.
func TestPowerShellBehavesLikeShC(t *testing.T) {
	pwsh, err := exec.LookPath("pwsh")
	if err != nil {
		t.Skip("pwsh not installed")
	}
	run := func(line string) (string, int) {
		cmd := exec.Command(pwsh, powerShellArgs(line)...)
		cmd.Env = append(os.Environ(), powerShellEnv...)
		out, err := cmd.CombinedOutput()
		code := 0
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		} else if err != nil {
			t.Fatal(err)
		}
		return string(out), code
	}
	for _, c := range []struct {
		line, want string
		code       int
	}{
		{`Write-Output "привет 'x' ""y"" & | ; $null"`, `привет 'x' "y" & | ; `, 0},
		{`pwsh -NoProfile -Command "exit 7"`, "", 7},
		{`Get-Item /no/such/path`, "Cannot find path", 1},
		{`Get-Item /no/such/path; Write-Output after`, "after", 0},
		{`Write-Output a && Write-Output b`, "a\nb", 0},
	} {
		out, code := run(c.line)
		if code != c.code || !strings.Contains(out, c.want) {
			t.Errorf("%s\n  exit %d, output %q; want exit %d containing %q", c.line, code, out, c.code, c.want)
		}
		if strings.Contains(out, "\x1b[") || strings.Contains(out, "CLIXML") {
			t.Errorf("%s\n  output is not plain text: %q", c.line, out)
		}
	}
}
