// Package usage is a small persistent token-usage ledger: prompt/completion
// tokens spent, bucketed by day, provider, and model. It lets /usage show a
// history beyond the current session.
package usage

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"sort"
	"sync"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
)

// Entry is one (day, provider, model) bucket of token counts.
type Entry struct {
	Date       string `json:"date"` // YYYY-MM-DD
	Provider   string `json:"provider"`
	Model      string `json:"model"`
	Prompt     int    `json:"prompt"`
	Completion int    `json:"completion"`
}

// Store is the ledger, persisted to a JSON file (in-memory if path is ""). Safe
// for concurrent use (sub-agents record from parallel tool calls).
type Store struct {
	mu      sync.Mutex
	path    string
	entries []Entry
	// pending holds the Add() deltas not yet folded into the on-disk file — Save
	// replays them onto a freshly re-read copy of the file instead of blindly
	// overwriting it with entries, so a separate ipsupport-code process sharing
	// the same global usage store can't have its update silently lost. overwrite
	// (set by Purge/Clear) skips that merge: those are a deliberate replace of
	// the whole ledger, not a delta.
	pending   []Entry
	overwrite bool
}

// Open loads the ledger, or starts empty if the file is absent. A blank path
// yields an in-memory store that never persists.
func Open(path string) (*Store, error) {
	s := &Store{path: path}
	if path == "" {
		return s, nil
	}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, &s.entries); err != nil {
			return s, err
		}
	case errors.Is(err, fs.ErrNotExist):
		// empty ledger
	default:
		return s, err
	}
	return s, nil
}

// Add folds tokens into the (date, provider, model) bucket. A no-op for a
// non-positive delta so a turn that reported no usage doesn't create a row.
func (s *Store) Add(date, provider, model string, prompt, completion int) {
	if prompt <= 0 && completion <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	addEntry(&s.entries, date, provider, model, prompt, completion)
	addEntry(&s.pending, date, provider, model, prompt, completion)
}

// addEntry folds (date, provider, model, prompt, completion) into an existing
// bucket in list, or appends a new one.
func addEntry(list *[]Entry, date, provider, model string, prompt, completion int) {
	for i := range *list {
		e := &(*list)[i]
		if e.Date == date && e.Provider == provider && e.Model == model {
			e.Prompt += prompt
			e.Completion += completion
			return
		}
	}
	*list = append(*list, Entry{date, provider, model, prompt, completion})
}

// readEntries loads the ledger at path without mutating a Store — a missing
// file is an empty ledger, not an error.
func readEntries(path string) ([]Entry, error) {
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		var out []Entry
		if err := json.Unmarshal(data, &out); err != nil {
			return nil, err
		}
		return out, nil
	case errors.Is(err, fs.ErrNotExist):
		return nil, nil
	default:
		return nil, err
	}
}

// Save writes the ledger (no-op for an in-memory store). Before writing, it
// re-reads the file and replays this Store's own pending Add() deltas onto
// that fresh copy instead of blindly overwriting it with entries — otherwise
// two separate ipsupport-code processes sharing the same global usage store
// could have one's update silently lost to the other's last write (the mutex
// only protects against races WITHIN one process). Purge/Clear set overwrite,
// skipping the merge: those are a deliberate replace of the whole ledger. The
// write itself is atomic (temp + rename) so a crash mid-write can't truncate
// the file either.
func (s *Store) Save() error {
	if s.path == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.overwrite && len(s.pending) > 0 {
		if onDisk, err := readEntries(s.path); err == nil {
			for _, p := range s.pending {
				addEntry(&onDisk, p.Date, p.Provider, p.Model, p.Prompt, p.Completion)
			}
			s.entries = onDisk
		}
		// on a read error, fall back to writing our own in-memory state — no
		// worse than the previous unconditional-overwrite behavior.
	}
	s.pending = nil
	s.overwrite = false
	data, err := json.MarshalIndent(s.entries, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(s.path, data, 0o644)
}

// Total is an aggregated row for display.
type Total struct {
	Key        string // a date, or "provider/model"
	Prompt     int
	Completion int
}

// Tokens is the combined prompt+completion count.
func (t Total) Tokens() int { return t.Prompt + t.Completion }

// TotalSince sums all entries on or after cutoff (an ISO YYYY-MM-DD date; ISO
// dates compare correctly as strings). cutoff "" sums everything.
func (s *Store) TotalSince(cutoff string) Total {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalSince(cutoff)
}

// totalSince is TotalSince without locking (callers hold the lock).
func (s *Store) totalSince(cutoff string) Total {
	t := Total{Key: cutoff}
	for _, e := range s.entries {
		if e.Date >= cutoff {
			t.Prompt += e.Prompt
			t.Completion += e.Completion
		}
	}
	return t
}

// Total is the all-time total.
func (s *Store) Total() Total {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalSince("")
}

// Purge drops entries older than cutoff (Date < cutoff) and returns how many were
// removed. The caller persists with Save.
func (s *Store) Purge(cutoff string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.entries[:0]
	removed := 0
	for _, e := range s.entries {
		if e.Date < cutoff {
			removed++
			continue
		}
		kept = append(kept, e)
	}
	s.entries = kept
	s.pending = nil
	s.overwrite = true
	return removed
}

// Clear drops the entire ledger. The caller persists with Save.
func (s *Store) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = nil
	s.pending = nil
	s.overwrite = true
}

// ByDay returns per-day totals, most recent day first.
func (s *Store) ByDay() []Total {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.aggregate(func(e Entry) string { return e.Date }, true)
}

// ByModel returns per provider/model totals, largest first.
func (s *Store) ByModel() []Total {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.aggregate(func(e Entry) string { return e.Provider + "/" + e.Model }, false)
}

// aggregate sums entries by key (callers hold the lock); byKeyDesc sorts on the
// key (for dates), otherwise on token count.
func (s *Store) aggregate(keyOf func(Entry) string, byKeyDesc bool) []Total {
	m := map[string]*Total{}
	for _, e := range s.entries {
		k := keyOf(e)
		t := m[k]
		if t == nil {
			t = &Total{Key: k}
			m[k] = t
		}
		t.Prompt += e.Prompt
		t.Completion += e.Completion
	}
	out := make([]Total, 0, len(m))
	for _, t := range m {
		out = append(out, *t)
	}
	sort.Slice(out, func(i, j int) bool {
		if byKeyDesc {
			return out[i].Key > out[j].Key
		}
		if out[i].Tokens() != out[j].Tokens() {
			return out[i].Tokens() > out[j].Tokens()
		}
		return out[i].Key < out[j].Key
	})
	return out
}
