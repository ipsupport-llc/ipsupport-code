package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
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
