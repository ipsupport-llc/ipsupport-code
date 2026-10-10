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
	"sh <<EOF\n# $(rm -rf src/legacy)\nEOF",
	"timeout 5 sh <<EOF\nrm -rf x\nEOF", "ssh deploy@host <<'EOF'\nrm -rf /srv/app\nEOF",
	"echo $(printf x)#not-comment; rm -rf src/legacy", "cat <<EO\\\nF\nbody\nEOF\nrm -rf victim",
	"cat <<EOF\nbody\nEO\\\nF\nrm -rf victim", "sh -c 'echo hi; rm -rf /tmp/q'", "bash -lc \"cd x && rm -rf build\"",
	"pwsh -Command \"Remove-Item -Recurse x\"", "cmd /c \"rd /s /q build\"", "cmd /c rd /s /q build & dir",
	"python3 -c 'import shutil; shutil.rmtree(\"/\")'", "eval 'rm -rf x'", "nice -n 10 bash <<EOF\nmake\nEOF",
	"echo " + strings.Repeat("$(", 40) + "rm -rf x" + strings.Repeat(")", 40),
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
			g := golden{Dialect: k.d, Line: k.line, Parts: p.Commands, Groups: p.Groups, Code: p.Code}
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
	return len(p.Commands) == 1 && p.Commands[0] == s && len(p.Groups) == 0 && p.Code == s
}
