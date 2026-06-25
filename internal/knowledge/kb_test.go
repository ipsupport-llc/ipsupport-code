package knowledge

import (
	"path/filepath"
	"testing"
)

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
