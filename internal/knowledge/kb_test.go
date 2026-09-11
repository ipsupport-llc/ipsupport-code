package knowledge

import (
	"encoding/json"
	"fmt"
	"os"
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

// A pitfall keyed on a bare "exit N" pattern (the run tool's own generic
// wrapper prefix on every failed command) must never surface — it "matches"
// (and misleads on) any unrelated failure. internal/reflect already refuses
// to STORE one, but a KB that already had one from before that guard existed
// (this test bypasses Add's caller entirely, mirroring a pitfall loaded
// straight off disk by Open) must still not SURFACE it.
func TestQuerySkipsGenericErrorPattern(t *testing.T) {
	kb := &KB{pitfalls: []Pitfall{
		{Domain: "run", ErrorPattern: "exit 1", Context: "a python module missing a dependency", ProvenFix: "install the required system library"},
		{Domain: "run", ErrorPattern: "go.mod already exists", Context: "go mod init on an existing module", ProvenFix: "skip init, cd into the existing module"},
	}}
	got := kb.Query("run", "go: /Users/roman220/rl_hero_go/go.mod already exists", 5)
	if len(got) != 1 || got[0].ErrorPattern != "go.mod already exists" {
		t.Errorf("Query = %+v, want only the specific (non-generic) pattern", got)
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

// Two separate ipsupport-code processes can share the same global KB. Each
// opens its own *KB from the same file; a naive "overwrite with my in-memory
// snapshot" Save would let whichever one saves last silently discard the
// other's learned lesson. Save must instead merge its own pending Add()s onto
// a fresh re-read of the file (codex review finding #22).
func TestSaveMergesConcurrentProcessesInsteadOfClobbering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")
	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Add(Pitfall{Domain: "file", ErrorPattern: "from a", ProvenFix: "x"})
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A second process opens the same file (sees a's lesson) and adds its own.
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add(Pitfall{Domain: "file", ErrorPattern: "from b", ProvenFix: "y"})

	// Back on the first process: it learns another lesson and saves again —
	// without ever having seen b's addition.
	a.Add(Pitfall{Domain: "file", ErrorPattern: "from a again", ProvenFix: "z"})
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// Now b saves. A blind overwrite would erase a's second lesson; the merge
	// must fold b's own pending add onto the fresh file instead.
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	all := final.All()
	if len(all) != 3 {
		t.Fatalf("merged lessons = %+v, want 3 (nothing clobbered)", all)
	}
}

// The previous merge-on-Save fix only narrows the cross-process race, it
// doesn't close it: two processes' Save calls can still interleave, since the
// only serialization is an in-process mutex. Here "a" does a large save (many
// distinct pending lessons, so its own read-merge-write takes a while) and "b"
// does a tiny one that finishes almost instantly. Without a lock around the
// whole cycle, b's fresh read (taken while a is still merging) misses a's
// not-yet-written update, and b's write then gets clobbered by a's own
// (later) write — silently losing b's lesson. Save must hold a cross-process
// file lock for its entire read-merge-write cycle so neither process's read
// can land inside the other's read-to-write window.
func TestSaveLocksAcrossWholeReadMergeWriteCycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")

	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2000; i++ { // pads a's own merge so its write lands well after its read
		a.Add(Pitfall{Domain: fmt.Sprintf("d%d", i), ErrorPattern: "e", ProvenFix: "x"})
	}

	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add(Pitfall{Domain: "b-process", ErrorPattern: "e", ProvenFix: "y"})

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
	all := final.All()
	if len(all) != 2001 {
		t.Fatalf("final lessons = %d, want 2001 (a's 2000 + b's 1, nothing clobbered)", len(all))
	}
	found := false
	for _, p := range all {
		if p.Domain == "b-process" {
			found = true
		}
	}
	if !found {
		t.Error("b's lesson was clobbered by a's concurrent (larger) save — Save must lock the file across the whole read-merge-write cycle")
	}
}

// If Save's own write fails (disk full, permission error, ...), any pending
// Add lessons must survive so the NEXT Save replays them. Clearing pending
// unconditionally would let a concurrent Add — whose own eventual Save only
// knows about ITS OWN lesson — silently and permanently drop the failed one,
// even from memory: that next Save re-reads the file fresh (still missing the
// failed write) and assigns k.pitfalls wholesale to fresh-plus-its-own-
// pending, discarding the earlier lesson entirely.
func TestSaveKeepsPendingWhenWriteFails(t *testing.T) {
	dir := t.TempDir()
	goodPath := filepath.Join(dir, "knowledge.json")
	kb, err := Open(goodPath)
	if err != nil {
		t.Fatal(err)
	}
	kb.Add(Pitfall{Domain: "d1", ErrorPattern: "from failed save", ProvenFix: "x"}) // lesson 1 — its save below will fail

	// Force the write to fail deterministically (no reliance on permission
	// bits, which root/CI can bypass): point path at a file inside a directory
	// component that is itself a plain file, so atomicfile.Write's MkdirAll
	// can never succeed.
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	kb.path = filepath.Join(blocker, "knowledge.json")
	if err := kb.Save(); err == nil {
		t.Fatal("Save should have failed: parent path component is a file, not a dir")
	}

	// A concurrent Add lands while lesson 1 is still only pending, not durable.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		kb.Add(Pitfall{Domain: "d2", ErrorPattern: "concurrent add", ProvenFix: "y"}) // lesson 2
	}()
	wg.Wait()

	kb.path = goodPath // the process recovers / retries against the real path
	if err := kb.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	final, err := Open(goodPath)
	if err != nil {
		t.Fatal(err)
	}
	all := final.All()
	if len(all) != 2 {
		t.Fatalf("lessons after recovery = %+v, want 2 (the failed save's lesson must survive to the next successful Save)", all)
	}
}

// A Save() call with nothing newly pending (e.g. an idle/periodic flush, or
// simply a second Save with no Add in between) used to skip the merge-read
// entirely — that branch only ran when pending was non-empty — and fall
// through to an unconditional write of k.pitfalls. That write has no fresh
// read backing it, so it silently clobbers whatever a concurrent process
// wrote to the shared file since this KB last read it. Save must instead
// skip the write altogether when there's nothing of its own to persist.
func TestSaveWithNothingPendingDoesNotClobberConcurrentWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")
	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Add(Pitfall{Domain: "file", ErrorPattern: "from a", ProvenFix: "x"})
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A second process opens the same file (sees a's lesson), learns its own,
	// and saves — correctly merging onto the fresh file (disk now has 2).
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add(Pitfall{Domain: "file", ErrorPattern: "from b", ProvenFix: "y"})
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	// Back on the first process: it Saves again with NOTHING newly pending (no
	// Add since its last Save). A blind overwrite would write a's stale
	// in-memory snapshot (1 lesson) over b's already-persisted 2.
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(final.All()); got != 2 {
		t.Errorf("final on-disk lessons = %d, want 2 (b's lesson must survive a's empty-pending Save)", got)
	}
}

// The real call sites (/knowledge purge, /knowledge retain) always Save()
// unconditionally after Purge, regardless of how many lessons it actually
// dropped. A no-op Purge forces pending back to nil (see Purge), so that
// Save() runs with nothing pending and !overwrite — the same empty-pending
// gap as above, just reached via Purge instead of a plain second Save.
func TestSaveAfterNoopPurgeDoesNotClobberConcurrentWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")
	a, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	a.Add(Pitfall{Domain: "file", ErrorPattern: "from a", ProvenFix: "x"})
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	// A concurrent process writes its own lesson to the shared file.
	b, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b.Add(Pitfall{Domain: "file", ErrorPattern: "from b", ProvenFix: "y"})
	if err := b.Save(); err != nil {
		t.Fatal(err)
	}

	// Back on the first process: a no-op purge (nothing old enough to drop),
	// immediately followed by an unconditional Save — mirroring /knowledge
	// purge and /knowledge retain.
	if n := a.Purge(3650); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}
	if err := a.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(final.All()); got != 2 {
		t.Errorf("final on-disk lessons = %d, want 2 (b's lesson must survive a's no-op-purge Save)", got)
	}
}

// A no-op Purge (nothing actually old enough to drop, the common case — the
// retention check runs unconditionally at startup) must not leave overwrite
// stuck true for the rest of the process's life: that would make every later
// Save() skip the merge-read and clobber whatever a concurrent process wrote
// to the shared file in the meantime.
func TestNoopPurgeDoesNotPoisonMergeSafety(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")
	kb, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "from kb", ProvenFix: "x"})
	if err := kb.Save(); err != nil {
		t.Fatal(err)
	}

	// Startup retention check: everything was just added, so nothing is older
	// than 3650 days — a no-op purge.
	if n := kb.Purge(3650); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}

	// A concurrent process writes its own lesson straight to the shared file,
	// bypassing this process's in-memory state entirely.
	onDisk, err := readPitfalls(path)
	if err != nil {
		t.Fatal(err)
	}
	onDisk = append(onDisk, Pitfall{Domain: "concurrent", ErrorPattern: "from other process", ProvenFix: "y", Hits: 1, Added: "2026-01-01", LastSeen: "2026-01-01"})
	data, err := json.MarshalIndent(onDisk, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	// Ordinary use later in the same process: an Add followed by a Save.
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "another one", ProvenFix: "z"})
	if err := kb.Save(); err != nil {
		t.Fatal(err)
	}

	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, p := range final.All() {
		if p.Domain == "concurrent" {
			found = true
		}
	}
	if !found {
		t.Error("concurrent process's lesson was clobbered — a no-op Purge left overwrite stuck true, so the next Save skipped its merge-read")
	}
}

// A no-op Purge cleared pending unconditionally, even though it only sets
// overwrite when it actually removed something. So an Add's lesson, still
// only pending (not yet Saved), was wiped by a Purge that removed nothing —
// leaving overwrite=false and pending=nil. The next Save (correctly, per the
// empty-pending skip above) then saw nothing of its own to persist and wrote
// nothing at all: the lesson survived only in k.pitfalls in memory, and the
// store file was never even created.
func TestNoopPurgeDoesNotDiscardEarlierAddLesson(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.json")
	kb, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	kb.Add(Pitfall{Domain: "file", ErrorPattern: "from kb", ProvenFix: "x"}) // lesson not yet saved

	// Retention check runs before the caller gets around to Save — everything
	// was just added, so nothing is old enough to drop, a no-op purge.
	if n := kb.Purge(3650); n != 0 {
		t.Fatalf("Purge = %d, want 0 (no-op)", n)
	}

	if err := kb.Save(); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("store file was never created — the Add's lesson was lost: %v", err)
	}
	final, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	all := final.All()
	if len(all) != 1 || all[0].Domain != "file" {
		t.Errorf("on-disk lessons = %+v, want the one Add'd lesson from before the no-op Purge to survive", all)
	}
}
