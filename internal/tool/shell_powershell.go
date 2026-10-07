package tool

import (
	"encoding/base64"
	"unicode/utf16"
)

// powerShellArgs runs a command line in PowerShell. -OutputFormat Text: with
// -EncodedCommand and redirected output, errors otherwise arrive serialized as
// CLIXML.
func powerShellArgs(line string) []string {
	return []string{"-NoProfile", "-NonInteractive", "-OutputFormat", "Text", "-EncodedCommand", encodePowerShell(line)}
}

// powerShellEnv is added to PowerShell's environment: PowerShell 7 writes
// redirected errors with color codes, and TERM=dumb is what turns that off
// (NO_COLOR and $PSStyle alone do not).
var powerShellEnv = []string{"TERM=dumb", "NO_COLOR=1"}

// powerShellScript wraps a command line so it behaves like sh -c: UTF-8 plain
// text output (no progress records, no color codes — PowerShell 7 colors even
// redirected errors), and the exit code of the last command — PowerShell on its
// own exits 0 or 1, losing a native program's code.
func powerShellScript(line string) string {
	return "$ProgressPreference = 'SilentlyContinue'\n" +
		"if ($PSStyle) { $PSStyle.OutputRendering = 'PlainText' }\n" +
		"try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }\n" +
		line + "\n" +
		"$__ok = $?; $__code = $LASTEXITCODE\n" +
		"if (-not $__ok) { if ($__code) { exit $__code } else { exit 1 } }\n" +
		"exit 0\n"
}

// encodePowerShell is the -EncodedCommand form of a command line: base64 of
// UTF-16LE, so no quoting rule anywhere between here and PowerShell can
// change what it runs.
func encodePowerShell(line string) string {
	u := utf16.Encode([]rune(powerShellScript(line)))
	b := make([]byte, 2*len(u))
	for i, c := range u {
		b[2*i], b[2*i+1] = byte(c), byte(c>>8)
	}
	return base64.StdEncoding.EncodeToString(b)
}
