package knowledge

import (
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// The KB is shared by the main agent and parallel sub-agents; concurrent
// Add/Query/Save must be race-free (run with -race). Also asserts the atomic Save
// round-trips.
func TestConcurrentAccessIsRaceFree(t *testing.T) {
	kb, err := Open(filepath.Join(t.TempDir(), "k.json"))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			kb.Add(Pitfall{Domain: "file", ErrorPattern: fmt.Sprintf("e%d", n), ProvenFix: "x"})
			kb.Query("file", "e", 5)
			_ = kb.Save()
		}(i)
	}
	wg.Wait()
	if kb.Count() != 20 {
		t.Errorf("count = %d, want 20", kb.Count())
	}
}

func TestPurgeByAgeAndRecurrence(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	kb := &KB{now: func() time.Time { return base }}
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "old", ProvenFix: "x"})

	kb.now = func() time.Time { return base.AddDate(0, 0, 100) } // 100 days later
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "new", ProvenFix: "y"})

	// purge lessons last seen >30 days ago: "old" (day 0) goes, "new" (day 100) stays
	if n := kb.Purge(30); n != 1 || kb.Count() != 1 {
		t.Errorf("Purge(30) dropped %d, count %d; want 1 dropped, 1 kept", n, kb.Count())
	}
	// a recurrence bumps LastSeen, keeping the lesson fresh
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "new", ProvenFix: "y"}) // dup → bump
	if n := kb.Purge(1); n != 0 {
		t.Errorf("a just-seen lesson should survive Purge(1), dropped %d", n)
	}
}

func TestAddDedupe(t *testing.T) {
	kb := &KB{}
	p := Pitfall{Domain: "file", ErrorPattern: "Permission denied", Context: "writing /root", ProvenFix: "use run.shell with sudo"}
	if !kb.Add(p) {
		t.Error("first Add should report a new lesson")
	}
	if kb.Add(p) {
		t.Error("second Add of same lesson should report duplicate")
	}
	all := kb.All()
	if len(all) != 1 {
		t.Fatalf("len = %d, want 1", len(all))
	}
	if all[0].Hits < 2 {
		t.Errorf("Hits = %d, want >= 2 after duplicate", all[0].Hits)
	}
}

func TestQueryRankAndDomain(t *testing.T) {
	kb := &KB{}
	kb.Add(Pitfall{Domain: "run", ErrorPattern: "permission denied sudo", Context: "shell"})
	kb.Add(Pitfall{Domain: "run", ErrorPattern: "command not found", Context: "shell"})
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "permission denied", Context: "write"})

	got := kb.Query("run", "permission denied when running command", 5)
	if len(got) == 0 {
		t.Fatal("expected matches")
	}
	if got[0].ErrorPattern != "permission denied sudo" {
		t.Errorf("top = %q, want the permission-denied entry", got[0].ErrorPattern)
	}
	for _, p := range got {
		if p.Domain != "run" {
			t.Errorf("Query leaked a cross-domain pitfall: %+v", p)
		}
	}
}

func TestSaveOpenRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "knowledge.json") // exercises mkdir
	kb := &KB{path: path}
	kb.Add(Pitfall{Domain: "web", ErrorPattern: "429", Context: "rate limit", ProvenFix: "back off"})
	if err := kb.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	kb2, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	all := kb2.All()
	if len(all) != 1 || all[0].Domain != "web" || all[0].ProvenFix != "back off" {
		t.Errorf("round-trip mismatch: %+v", all)
	}
}
