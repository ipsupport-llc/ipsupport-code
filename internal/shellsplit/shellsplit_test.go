package shellsplit

import (
	"reflect"
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
