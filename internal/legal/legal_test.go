package legal

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Every module linked on any platform a release ships for has its notice:
// a new dependency fails here until `go run ./internal/legal/gen` is run.
func TestNoticesCoverEveryLinkedModule(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	root, _ := filepath.Abs("../..")
	summary := Summary()
	for _, goos := range []string{"linux", "darwin", "windows"} {
		cmd := exec.Command("go", "list", "-deps", "-f",
			`{{if not .Standard}}{{with .Module}}{{if not .Main}}{{.Path}} {{.Version}}{{end}}{{end}}{{end}}`, "./cmd/agent")
		cmd.Dir = root
		cmd.Env = append(cmd.Environ(), "GOOS="+goos)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("go list (%s): %v", goos, err)
		}
		for _, mod := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			if mod != "" && !strings.Contains(summary, mod+" ") {
				t.Errorf("%s (%s) has no notice — run: go run ./internal/legal/gen", mod, goos)
			}
			// The table row alone is not the notice: MIT and BSD require the text.
			if mod != "" && !strings.Contains(Full(), "---- "+mod+" (") {
				t.Errorf("%s (%s) has no license text — run: go run ./internal/legal/gen", mod, goos)
			}
		}
	}
	if !strings.Contains(Full(), marker) || strings.Contains(Summary(), "Permission is hereby granted") {
		t.Error("the summary and the full texts are not separated")
	}
}
