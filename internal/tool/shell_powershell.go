package tool

import (
	"encoding/base64"
	"html"
	"regexp"
	"strconv"
	"strings"
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

var (
	cliXMLBlock  = regexp.MustCompile(`(?s)(?:#< CLIXML\r?\n)?<Objs Version="[^"]*" xmlns="http://schemas\.microsoft\.com/powershell/2004/04">(.*?)</Objs>`)
	cliXMLString = regexp.MustCompile(`(?s)<S S="[^"]*">(.*?)</S>`)
	cliXMLEscape = regexp.MustCompile(`_x([0-9A-Fa-f]{4})_`)
)

// decodeCLIXML turns the CLIXML that Windows PowerShell 5.1 writes for its
// error stream under -EncodedCommand back into the text it stands for: 5.1
// ignores -OutputFormat Text there, and a model reading the XML has to dig
// the error out of entities and _x000D__x000A_ escapes. Anything else in the
// output is left as it is.
func decodeCLIXML(out string) string {
	if !strings.Contains(out, "<Objs ") {
		return out
	}
	return cliXMLBlock.ReplaceAllStringFunc(out, func(block string) string {
		var b strings.Builder
		for _, m := range cliXMLString.FindAllStringSubmatch(block, -1) {
			b.WriteString(cliXMLEscape.ReplaceAllStringFunc(html.UnescapeString(m[1]), func(e string) string {
				n, _ := strconv.ParseUint(e[2:6], 16, 16)
				return string(rune(n))
			}))
		}
		return strings.ReplaceAll(b.String(), "\r\n", "\n")
	})
}
