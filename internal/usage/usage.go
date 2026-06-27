// Package usage is a small persistent token-usage ledger: prompt/completion
// tokens spent, bucketed by day, provider, and model. It lets /usage show a
// history beyond the current session.
package usage

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// Entry is one (day, provider, model) bucket of token counts.
type Entry struct {
	Date       string `json:"date"` // YYYY-MM-DD
	Provider   string `json:"provider"`
	Model      string `json:"model"`
	Prompt     int    `json:"prompt"`
	Completion int    `json:"completion"`
}

// Store is the ledger, persisted to a JSON file (in-memory if path is "").
type Store struct {
	path    string
	entries []Entry
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
	for i := range s.entries {
		e := &s.entries[i]
		if e.Date == date && e.Provider == provider && e.Model == model {
			e.Prompt += prompt
			e.Completion += completion
			return
		}
	}
	s.entries = append(s.entries, Entry{date, provider, model, prompt, completion})
}

// Save writes the ledger (no-op for an in-memory store).
func (s *Store) Save() error {
	if s.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s.entries, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o644)
}

// Total is an aggregated row for display.
type Total struct {
	Key        string // a date, or "provider/model"
	Prompt     int
	Completion int
}

// Tokens is the combined prompt+completion count.
func (t Total) Tokens() int { return t.Prompt + t.Completion }

// ByDay returns per-day totals, most recent day first.
func (s *Store) ByDay() []Total {
	return s.aggregate(func(e Entry) string { return e.Date }, true)
}

// ByModel returns per provider/model totals, largest first.
func (s *Store) ByModel() []Total {
	return s.aggregate(func(e Entry) string { return e.Provider + "/" + e.Model }, false)
}

// aggregate sums entries by key; byKeyDesc sorts on the key (for dates),
// otherwise on token count.
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
