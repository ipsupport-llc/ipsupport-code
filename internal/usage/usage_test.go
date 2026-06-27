package usage

import (
	"path/filepath"
	"testing"
)

func TestStoreAddAndAggregate(t *testing.T) {
	s, _ := Open("")
	s.Add("2026-06-27", "grok", "grok-4.3", 100, 200)
	s.Add("2026-06-27", "grok", "grok-4.3", 10, 20) // folds into same bucket
	s.Add("2026-06-26", "local", "qwen", 5, 5)
	s.Add("2026-06-27", "local", "qwen", 1, 1)
	s.Add("2026-06-27", "x", "z", 0, 0) // non-positive → ignored

	days := s.ByDay()
	if len(days) != 2 || days[0].Key != "2026-06-27" {
		t.Fatalf("ByDay = %+v, want most-recent-first with 2 days", days)
	}
	if got := days[0].Tokens(); got != 100+200+10+20+1+1 {
		t.Errorf("2026-06-27 total = %d, want 332", got)
	}
	models := s.ByModel()
	if len(models) != 2 || models[0].Key != "grok/grok-4.3" {
		t.Fatalf("ByModel = %+v, want grok/grok-4.3 first (largest)", models)
	}
}

func TestStoreTotalSincePurgeClear(t *testing.T) {
	s, _ := Open("")
	s.Add("2026-06-01", "p", "m", 10, 10)
	s.Add("2026-06-20", "p", "m", 20, 20)
	s.Add("2026-06-27", "p", "m", 30, 30)

	if got := s.Total().Tokens(); got != 120 {
		t.Errorf("Total = %d, want 120", got)
	}
	if got := s.TotalSince("2026-06-20").Tokens(); got != 100 {
		t.Errorf("TotalSince(06-20) = %d, want 100 (20+20+30+30)", got)
	}
	if n := s.Purge("2026-06-20"); n != 1 { // drops only 2026-06-01
		t.Errorf("Purge older than 06-20 removed %d, want 1", n)
	}
	if got := s.Total().Tokens(); got != 100 {
		t.Errorf("after purge Total = %d, want 100", got)
	}
	s.Clear()
	if got := s.Total().Tokens(); got != 0 {
		t.Errorf("after Clear Total = %d, want 0", got)
	}
}

func TestStorePersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	s, _ := Open(path)
	s.Add("2026-06-27", "grok", "grok-4.3", 100, 200)
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}
	s2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m := s2.ByModel()
	if len(m) != 1 || m[0].Tokens() != 300 {
		t.Errorf("reloaded ledger = %+v, want one row of 300 tokens", m)
	}
}
