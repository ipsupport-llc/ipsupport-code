package usage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestStoreConcurrentAdd(t *testing.T) {
	s, _ := Open("") // run with -race: the mutex must make this safe
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); s.Add("2026-06-27", "p", "m", 1, 1, 0) }()
	}
	wg.Wait()
	if got := s.Total().Tokens(); got != 128 {
		t.Errorf("concurrent Add total = %d, want 128", got)
	}
}

func TestStoreAddAndAggregate(t *testing.T) {
	s, _ := Open("")
	s.Add("2026-06-27", "grok", "grok-4.3", 100, 200, 0)
	s.Add("2026-06-27", "grok", "grok-4.3", 10, 20, 0) // folds into same bucket
	s.Add("2026-06-26", "local", "qwen", 5, 5, 0)
	s.Add("2026-06-27", "local", "qwen", 1, 1, 0)
	s.Add("2026-06-27", "x", "z", 0, 0, 0) // non-positive → ignored

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
	s.Add("2026-06-01", "p", "m", 10, 10, 0)
	s.Add("2026-06-20", "p", "m", 20, 20, 0)
	s.Add("2026-06-27", "p", "m", 30, 30, 0)

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
	s.Add("2026-06-27", "openai", "gpt-4o", 1_000_000, 0, 0)         // $2.50
	s.Add("2026-06-27", "openrouter", "x:free", 9_000_000, 9_000, 0) // $0
	if got := s.CostSince("", nil); got != 2.50 {
		t.Errorf("CostSince = %v, want 2.50", got)
	}
}

// Two separate ipsupport-code processes can share the same global usage store.
// Each opens its own *Store from the same file; a naive "overwrite with my
// in-memory snapshot" Save would let whichever one saves last silently discard
// the other's update. Save must instead merge its own delta onto a fresh
// re-read of the file (codex review finding #22).
func TestStoreSaveMergesConcurrentProcessesInsteadOfClobbering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	a, _ := Open(path)
	a.Add("2026-06-27", "grok", "grok-4.3", 100, 100, 0)
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A second process opens the same file (sees a's entry) and adds its own.
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add("2026-06-27", "openai", "gpt-4o", 50, 50, 0)

	// Back on the first process: it adds more of its own and saves again —
	// without ever having seen b's update.
	a.Add("2026-06-27", "grok", "grok-4.3", 10, 10, 0)
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// Now b saves. A blind overwrite would erase a's second Add (and a's own
	// row would still be present since b loaded it at Open, but a's newest 10+10
	// would be lost); the merge must fold b's own delta onto the fresh file
	// instead, preserving what a already persisted.
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := final.Total().Tokens(); got != 320 { // (100+10)*2 + 50*2 = 220+100
		t.Errorf("merged total = %d, want 320 (a's two adds + b's add, nothing clobbered)", got)
	}
	models := final.ByModel()
	if len(models) != 2 {
		t.Fatalf("ByModel = %+v, want 2 rows (grok survives, openai survives)", models)
	}
}

func TestStorePersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	s, _ := Open(path)
	s.Add("2026-06-27", "grok", "grok-4.3", 100, 200, 0)
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

// The previous merge-on-Save fix only narrows the cross-process race, it
// doesn't close it: two processes' Save calls can still interleave, since the
// only serialization is an in-process mutex. Here "a" does a large save (many
// distinct pending entries, so its own read-merge-write takes a while) and "b"
// does a tiny one that finishes almost instantly. Without a lock around the
// whole cycle, b's fresh read (taken while a is still merging) misses a's
// not-yet-written update, and b's write then gets clobbered by a's own
// (later) write — silently losing b's entry. Save must hold a cross-process
// file lock for its entire read-merge-write cycle so neither process's read
// can land inside the other's read-to-write window.
func TestSaveLocksAcrossWholeReadMergeWriteCycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")

	a, _ := Open(path)
	for i := 0; i < 2000; i++ { // pads a's own merge so its write lands well after its read
		a.Add(fmt.Sprintf("2020-01-%02d", 1+i%28), fmt.Sprintf("p%d", i), "m", 1, 1, 0)
	}

	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add("2026-06-27", "b-process", "m", 5, 5, 0)

	var wg sync.WaitGroup
	var errA error
	wg.Add(1)
	go func() {
		defer wg.Done()
		errA = a.Save()
	}()
	time.Sleep(10 * time.Millisecond) // let a pass its own fresh read and start its slow merge
	if err := b.Save(); err != nil {
		t.Fatalf("b.Save: %v", err)
	}
	wg.Wait()
	if errA != nil {
		t.Fatalf("a.Save: %v", errA)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(final.entries) != 2001 {
		t.Fatalf("final entries = %d, want 2001 (a's 2000 + b's 1, nothing clobbered)", len(final.entries))
	}
	found := false
	for _, e := range final.entries {
		if e.Provider == "b-process" {
			found = true
		}
	}
	if !found {
		t.Error("b's entry was clobbered by a's concurrent (larger) save — Save must lock the file across the whole read-merge-write cycle")
	}
}

// If Save's own write fails (disk full, permission error, ...), any pending
// Add deltas must survive so the NEXT Save replays them. Clearing pending
// unconditionally would let a concurrent Add — whose own eventual Save only
// knows about ITS OWN delta — silently and permanently drop the failed one,
// even from memory: that next Save re-reads the file fresh (still missing the
// failed write) and assigns s.entries wholesale to fresh-plus-its-own-pending,
// discarding the earlier delta entirely.
func TestSaveKeepsPendingWhenWriteFails(t *testing.T) {
	dir := t.TempDir()
	goodPath := filepath.Join(dir, "usage.json")
	s, _ := Open(goodPath)
	s.Add("2026-06-27", "p1", "m", 10, 10, 0) // delta 1 — its save below will fail

	// Force the write to fail deterministically (no reliance on permission
	// bits, which root/CI can bypass): point path at a file inside a directory
	// component that is itself a plain file, so atomicfile.Write's MkdirAll
	// can never succeed.
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	s.path = filepath.Join(blocker, "usage.json")
	if err := s.Save(); err == nil {
		t.Fatal("Save should have failed: parent path component is a file, not a dir")
	}

	// A concurrent Add lands while delta 1 is still only pending, not durable.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.Add("2026-06-27", "p2", "m", 5, 5, 0) // delta 2
	}()
	wg.Wait()

	s.path = goodPath // the process recovers / retries against the real path
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	final, err := Open(goodPath)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, m := range final.ByModel() {
		seen[m.Key] = true
	}
	if !seen["p1/m"] || !seen["p2/m"] {
		t.Errorf("ByModel keys = %v, want both p1/m and p2/m (the failed save's delta must survive to the next successful Save)", seen)
	}
}

// A Save() call with nothing newly pending (e.g. an idle/periodic flush, or
// simply a second Save with no Add in between) used to skip the merge-read
// entirely — that branch only ran when pending was non-empty — and fall
// through to an unconditional write of s.entries. That write has no fresh
// read backing it, so it silently clobbers whatever a concurrent process
// wrote to the shared file since this Store last read it. Save must instead
// skip the write altogether when there's nothing of its own to persist.
func TestSaveWithNothingPendingDoesNotClobberConcurrentWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Add("2026-06-27", "grok", "grok-4.3", 10, 0, 0)
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A second process opens the same file (sees a's entry), adds its own, and
	// saves — correctly merging onto the fresh file (disk total now 12).
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add("2026-06-27", "grok", "grok-4.3", 2, 0, 0)
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	// Back on the first process: it Saves again with NOTHING newly pending (no
	// Add since its last Save). A blind overwrite would write a's stale
	// in-memory snapshot (10) over b's already-persisted total (12).
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := final.Total().Tokens(); got != 12 {
		t.Errorf("final on-disk total = %d, want 12 (b's update must survive a's empty-pending Save)", got)
	}
}

// The real call sites (/usage purge, /usage retain) always Save()
// unconditionally after Purge, regardless of how many entries it actually
// dropped. A no-op Purge forces pending back to nil (see Purge), so that
// Save() runs with nothing pending and !overwrite — the same empty-pending
// gap as above, just reached via Purge instead of a plain second Save.
func TestSaveAfterNoopPurgeDoesNotClobberConcurrentWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Add("2026-06-27", "grok", "grok-4.3", 10, 0, 0)
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A concurrent process writes its own entry to the shared file.
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add("2026-06-27", "grok", "grok-4.3", 2, 0, 0)
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	// Back on the first process: a no-op purge (nothing old enough to drop),
	// immediately followed by an unconditional Save — mirroring /usage purge
	// and /usage retain.
	if n := a.Purge("2000-01-01"); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := final.Total().Tokens(); got != 12 {
		t.Errorf("final on-disk total = %d, want 12 (b's update must survive a's no-op-purge Save)", got)
	}
}

// A no-op Purge (nothing actually expired, the common case — the retention
// check runs unconditionally at startup) must not leave overwrite stuck true
// for the rest of the process's life: that would make every later Save() skip
// the merge-read and clobber whatever a concurrent process wrote to the
// shared file in the meantime.
func TestNoopPurgeDoesNotPoisonMergeSafety(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	s, _ := Open(path)
	s.Add("2026-06-27", "grok", "grok-4.3", 10, 10, 0)
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}

	// Startup retention check: cutoff is in the far past, so nothing is old
	// enough to expire — a no-op purge.
	if n := s.Purge("2000-01-01"); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}

	// A concurrent process writes its own entry straight to the shared file,
	// bypassing this process's in-memory state entirely.
	onDisk, err := readEntries(path)
	if err != nil {
		t.Fatal(err)
	}
	addEntry(&onDisk, "2026-06-27", "concurrent-process", "m", 99, 99, 0)
	data, err := json.MarshalIndent(onDisk, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	// Ordinary use later in the same process: an Add followed by a Save.
	s.Add("2026-06-27", "grok", "grok-4.3", 1, 1, 0)
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range final.ByModel() {
		if m.Key == "concurrent-process/m" {
			found = true
		}
	}
	if !found {
		t.Error("concurrent process's entry was clobbered — a no-op Purge left overwrite stuck true, so the next Save skipped its merge-read")
	}
}

// A no-op Purge cleared pending unconditionally, even though it only sets
// overwrite when it actually removed something. So an Add's delta, still only
// pending (not yet Saved), was wiped by a Purge that removed nothing —
// leaving overwrite=false and pending=nil. The next Save (correctly, per the
// empty-pending skip above) then saw nothing of its own to persist and wrote
// nothing at all: the delta survived only in s.entries in memory, and the
// ledger file was never even created.
func TestNoopPurgeDoesNotDiscardEarlierAddDelta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("2026-06-27", "grok", "grok-4.3", 10, 5, 0) // delta not yet saved

	// Retention check runs before the caller gets around to Save — cutoff is
	// in the far past, so nothing is old enough to expire, a no-op purge.
	if n := s.Purge("2000-01-01"); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}

	if err := s.Save(); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("ledger file was never created — the Add's delta was lost: %v", err)
	}
	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := final.Total().Tokens(); got != 15 {
		t.Errorf("on-disk total = %d, want 15 (the Add's delta from before the no-op Purge must survive)", got)
	}
}

// Duration accumulates across multiple Add calls into the same bucket (like
// Prompt/Completion already do), and TokensPerSec derives from the total —
// not just the last call's duration.
func TestDurationAccumulatesAndDerivesTokensPerSec(t *testing.T) {
	s, err := Open("")
	if err != nil {
		t.Fatal(err)
	}
	s.Add("2026-06-27", "local", "qwen", 0, 100, 2*time.Second)
	s.Add("2026-06-27", "local", "qwen", 0, 100, 2*time.Second) // same bucket — folds in

	models := s.ByModel()
	if len(models) != 1 {
		t.Fatalf("got %d model buckets, want 1", len(models))
	}
	m := models[0]
	if m.DurationMS != 4000 {
		t.Fatalf("DurationMS = %d, want 4000 (2s + 2s)", m.DurationMS)
	}
	if got, want := m.TokensPerSec(), 50.0; got != want { // 200 completion tokens / 4s
		t.Errorf("TokensPerSec = %v, want %v", got, want)
	}
}

// An entry with no duration recorded (e.g. from before Entry.DurationMS
// existed, or a caller that couldn't measure one) must report 0 tok/s, not
// divide by zero or fabricate a rate.
func TestTokensPerSecZeroWithoutDuration(t *testing.T) {
	s, err := Open("")
	if err != nil {
		t.Fatal(err)
	}
	s.Add("2026-06-27", "local", "qwen", 0, 100, 0)
	if got := s.ByModel()[0].TokensPerSec(); got != 0 {
		t.Errorf("TokensPerSec = %v, want 0 (no duration recorded)", got)
	}
}
