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

	"github.com/ipsupport-llc/ipsupport-code/internal/risk"
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

var update = flag.Bool("update", false, "rewrite testdata/golden.jsonl from the corpus")

type golden struct {
	Dialect Dialect  `json:"dialect"`
	Line    string   `json:"line"`
	Parts   []string `json:"parts"`
	Code    string   `json:"code"`
}

type key struct {
	d    Dialect
	line string
}

const goldenPath = "testdata/golden.jsonl"

// DialectForOS is the dialect a dataset row's "os" names.
func DialectForOS(os string) Dialect {
	if os == "windows" {
		return PowerShell
	}
	return Sh
}

func corpus(t *testing.T) map[key]bool {
	t.Helper()
	lines := map[key]bool{}
	for _, f := range []string{"../../scripts/risk_dataset.jsonl", "../../scripts/risk_eval.jsonl"} {
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
				// Workspace: the scorer reads a path into it as the project's
				// own (risk.Localize) before cutting the line.
				Workspace string `json:"workspace"`
			}
			if json.Unmarshal(sc.Bytes(), &r) != nil || r.Tool != "run" || r.Action != "shell" {
				continue
			}
			if c, ok := r.Params["command"].(string); ok {
				lines[key{DialectForOS(r.OS), risk.Localize(c, r.Workspace)}] = true
			}
		}
		fh.Close()
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
}

func TestGolden(t *testing.T) {
	if *update {
		var out []golden
		for k := range corpus(t) {
			if trivial(k) {
				continue
			}
			g := golden{Dialect: k.d, Line: k.line, Parts: Split(k.d, k.line), Code: Code(k.d, k.line)}
			if g.Parts == nil {
				g.Parts = []string{}
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
		os.MkdirAll(filepath.Dir(goldenPath), 0o755)
		if err := os.WriteFile(goldenPath, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d lines", len(out))
	}
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
		have[key{g.Dialect, g.Line}] = true
	}
	missing := 0
	for k := range corpus(t) {
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
	s := strings.TrimSpace(k.line)
	p := Split(k.d, k.line)
	return len(p) == 1 && p[0] == s && Code(k.d, k.line) == s
}
