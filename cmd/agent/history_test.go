package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func TestSessionArchiverAndHistorySourceRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sessions", "default.archive.jsonl")
	ar := &sessionArchiver{path: path}
	ar.Archive("build the world module", "created world.go (actions this turn — files touched: world.go;)")
	ar.Archive("add the brain", "created brain.go (actions this turn — files touched: brain.go;)")

	src := historySource{path: path}
	recent := src.Recent(1)
	if len(recent) != 1 || !strings.Contains(recent[0], "brain.go") {
		t.Errorf("Recent(1) = %+v, want just the last (brain.go) entry", recent)
	}

	all := src.Recent(10)
	if len(all) != 2 {
		t.Fatalf("Recent(10) = %d entries, want 2", len(all))
	}

	found := src.Search("world.go")
	if len(found) != 1 || !strings.Contains(found[0], "world.go") {
		t.Errorf("Search(world.go) = %+v, want the world.go entry only", found)
	}

	if got := src.Search("nothing matches this"); len(got) != 0 {
		t.Errorf("Search with no match = %+v, want empty", got)
	}
}

func TestHasArchivedHistory(t *testing.T) {
	a := &app{workspace: t.TempDir(), cfg: config.Config{Name: "default"}}
	if a.hasArchivedHistory() {
		t.Error("a fresh workspace should have no archived history yet")
	}
	(&sessionArchiver{path: a.archivePath()}).Archive("g", "e")
	if !a.hasArchivedHistory() {
		t.Error("after one archived entry, hasArchivedHistory should be true")
	}
}

// A session that STARTS with an empty archive must still get the `history`
// tool once something has actually been archived — even though wire() only
// checked hasArchivedHistory() once, at the start, before there was anything
// to recall. The first task's own remember() call archives an entry; a
// second, perfectly ordinary task (no /config, no session switch, no other
// unrelated action) must then see the tool, proving the app rechecks on its
// own rather than needing some unrelated trigger to rewire.
func TestHistoryToolAppearsAfterFirstArchiveWithNoUnrelatedAction(t *testing.T) {
	var toolsPerRequest [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &body)
		var names []string
		for _, tl := range body.Tools {
			names = append(names, tl.Function.Name)
		}
		toolsPerRequest = append(toolsPerRequest, names)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`)
	}))
	defer srv.Close()

	kb, _ := knowledge.Open("")
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.LLM.BaseURL = srv.URL
	cfg.LLM.Model = "fake"
	cfg.Run.Default, cfg.File.Default, cfg.File.Jail = "allow", "allow", "."
	cfg.ReflectDisabled = true // isolate the tools sent for the task itself from the reflection pass's own call
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	a.windowDetected = true // skip runOne's context-window probe — an unrelated network call to the same fake server
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	// Fresh session, nothing archived yet: the history tool must be absent.
	a.runOne(context.Background(), "first task")
	if len(toolsPerRequest) != 1 {
		t.Fatalf("requests after the first task = %d, want 1", len(toolsPerRequest))
	}
	if slices.Contains(toolsPerRequest[0], "history") {
		t.Fatalf("tools on the first task = %v, want no %q (archive is still empty)", toolsPerRequest[0], "history")
	}
	if !a.hasArchivedHistory() {
		t.Fatal("the first task should have archived an entry via remember()")
	}

	// Second, ordinary task — nothing unrelated happened in between.
	a.runOne(context.Background(), "second task")
	if len(toolsPerRequest) != 2 {
		t.Fatalf("requests after the second task = %d, want 2", len(toolsPerRequest))
	}
	if !slices.Contains(toolsPerRequest[1], "history") {
		t.Errorf("tools on the second task = %v, want %q present now that an entry is archived", toolsPerRequest[1], "history")
	}
}
