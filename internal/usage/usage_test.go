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

func TestPricing(t *testing.T) {
	// built-in: gpt-4o = $2.50 in / $10 out per 1M
	if got := CostUSD("gpt-4o", 1_000_000, 1_000_000, nil); got != 12.50 {
		t.Errorf("gpt-4o cost = %v, want 12.50", got)
	}
	// :free is always $0 even if a substring would otherwise match
	if got := CostUSD("nvidia/nemotron:free", 5_000_000, 5_000_000, nil); got != 0 {
		t.Errorf(":free cost = %v, want 0", got)
	}
	// unknown model → $0 (no estimate)
	if got := CostUSD("some-unknown-model", 1_000_000, 0, nil); got != 0 {
		t.Errorf("unknown cost = %v, want 0", got)
	}
	// override wins over the built-in table
	ov := map[string]Price{"gpt-4o": {In: 1, Out: 1}}
	if got := CostUSD("gpt-4o", 1_000_000, 0, ov); got != 1 {
		t.Errorf("override cost = %v, want 1", got)
	}
	// CostSince sums per-entry by model
	s, _ := Open("")
	s.Add("2026-06-27", "openai", "gpt-4o", 1_000_000, 0)         // $2.50
	s.Add("2026-06-27", "openrouter", "x:free", 9_000_000, 9_000) // $0
	if got := s.CostSince("", nil); got != 2.50 {
		t.Errorf("CostSince = %v, want 2.50", got)
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
