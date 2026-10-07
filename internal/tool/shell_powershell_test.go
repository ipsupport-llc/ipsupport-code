package tool

import (
	"os"
	"os/exec"
	"runtime"
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

// Windows PowerShell 5.1 ignores -OutputFormat Text under -EncodedCommand and
// serializes errors as CLIXML; the model must get them back as text. The
// sample is what it printed for `node -v && python --version 2>&1`.
func TestCLIXMLIsDecodedToText(t *testing.T) {
	out := "#< CLIXML\r\n" + `<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04">` +
		`<S S="Error">At line:4 char:9_x000D__x000A_</S>` +
		`<S S="Error">+ node -v &amp;&amp; python --version 2&gt;&amp;1_x000D__x000A_</S>` +
		`<S S="Error">The token '&amp;&amp;' is not a valid statement separator in this version._x000D__x000A_</S>` +
		`<S S="Error">    + FullyQualifiedErrorId : InvalidEndOfLine_x000D__x000A_</S>` +
		`<S S="Error"> _x000D__x000A_</S></Objs>`
	got := decodeCLIXML("v20.1.0\n" + out)
	for _, want := range []string{"v20.1.0", "+ node -v && python --version 2>&1\n", "The token '&&' is not a valid statement separator"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	for _, bad := range []string{"<Objs", "_x000D_", "&amp;", "CLIXML", "\r"} {
		if strings.Contains(got, bad) {
			t.Errorf("still holds %q:\n%s", bad, got)
		}
	}
	if plain := "no xml here & <b>"; decodeCLIXML(plain) != plain {
		t.Error("plain output was changed")
	}
}

// The prompt names the shell commands really run in.
func TestPromptNoteNamesTheShell(t *testing.T) {
	note := Shell("").PromptNote()
	want := "PowerShell"
	if runtime.GOOS != "windows" {
		want = "sh -c"
	}
	if !strings.Contains(note, want) {
		t.Fatalf("PromptNote() = %q, want it to name %q", note, want)
	}
}
