package knowledge

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
)

const dateFmt = "2006-01-02"

// KnowledgeError wraps a host-level failure of the knowledge store (read, parse,
// or write). Model-recoverable conditions never surface as this type.
type KnowledgeError struct {
	Op   string
	Path string
	Err  error
}

func (e *KnowledgeError) Error() string {
	return fmt.Sprintf("knowledge: %s %q: %v", e.Op, e.Path, e.Err)
}
func (e *KnowledgeError) Unwrap() error { return e.Err }

// KB is an in-memory view of the pitfall store plus its backing file path. The
// same *KB is shared by the main agent and parallel sub-agents (which record
// lessons and Query from concurrent tool calls), so it guards its state with mu.
type KB struct {
	mu       sync.Mutex
	path     string
	pitfalls []Pitfall
	now      func() time.Time // overridable clock for tests
}

// Open loads the store at path. A missing file is an empty store, not an error.
func Open(path string) (*KB, error) {
	kb := &KB{path: path, now: time.Now}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, &kb.pitfalls); err != nil {
			return kb, &KnowledgeError{Op: "parse", Path: path, Err: err}
		}
	case errors.Is(err, fs.ErrNotExist):
		// empty store
	default:
		return kb, &KnowledgeError{Op: "read", Path: path, Err: err}
	}
	// Back-fill timestamps on pre-dated entries so age-based pruning has a baseline
	// (they start aging from now). Persisted on the next Save.
	today := kb.today()
	for i := range kb.pitfalls {
		if kb.pitfalls[i].LastSeen == "" {
			kb.pitfalls[i].LastSeen = today
		}
		if kb.pitfalls[i].Added == "" {
			kb.pitfalls[i].Added = today
		}
	}
	return kb, nil
}

func (k *KB) today() string {
	if k.now == nil {
		k.now = time.Now
	}
	return k.now().Format(dateFmt)
}

// Add inserts a pitfall, de-duplicating on (domain, normalized error pattern).
// A duplicate bumps Hits and back-fills any empty fields; returns true only when
// a genuinely new lesson was stored.
func (k *KB) Add(p Pitfall) bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	today := k.today()
	key := dedupeKey(p)
	for i := range k.pitfalls {
		if dedupeKey(k.pitfalls[i]) == key {
			k.pitfalls[i].Hits++
			k.pitfalls[i].LastSeen = today // recurred → keep it fresh
			if k.pitfalls[i].ProvenFix == "" {
				k.pitfalls[i].ProvenFix = p.ProvenFix
			}
			if k.pitfalls[i].Context == "" {
				k.pitfalls[i].Context = p.Context
			}
			return false
		}
	}
	if p.Hits == 0 {
		p.Hits = 1
	}
	p.Added, p.LastSeen = today, today
	k.pitfalls = append(k.pitfalls, p)
	return true
}

// Purge drops lessons last seen more than maxAgeDays ago (by LastSeen), returning
// how many were removed. maxAgeDays <= 0 is a no-op. Undated entries are kept
// (Open back-fills them, so they age from first load).
func (k *KB) Purge(maxAgeDays int) int {
	if maxAgeDays <= 0 {
		return 0
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	cutoff := k.now().AddDate(0, 0, -maxAgeDays)
	kept := k.pitfalls[:0]
	dropped := 0
	for _, p := range k.pitfalls {
		t, err := time.Parse(dateFmt, p.LastSeen)
		if err == nil && t.Before(cutoff) {
			dropped++
			continue
		}
		kept = append(kept, p)
	}
	k.pitfalls = kept
	return dropped
}

// Clear removes every lesson, returning how many were dropped.
func (k *KB) Clear() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	n := len(k.pitfalls)
	k.pitfalls = nil
	return n
}

// Count reports how many lessons are stored.
func (k *KB) Count() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	return len(k.pitfalls)
}

// Query returns pitfalls for a domain. With a non-empty errText it keeps only
// keyword-overlapping lessons, ranked by overlap then Hits. With an empty
// errText it returns every lesson in the domain ranked by Hits (used by the
// help tool). limit <= 0 means no cap.
func (k *KB) Query(domain, errText string, limit int) []Pitfall {
	k.mu.Lock()
	defer k.mu.Unlock()
	type scored struct {
		p Pitfall
		s int
	}
	toks := tokens(errText)
	var out []scored
	for _, p := range k.pitfalls {
		if p.Domain != domain {
			continue
		}
		if len(toks) == 0 {
			out = append(out, scored{p, 0})
			continue
		}
		hay := norm(p.ErrorPattern + " " + p.Context)
		s := 0
		for tk := range toks {
			if strings.Contains(hay, tk) {
				s++
			}
		}
		if s > 0 {
			out = append(out, scored{p, s})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].s != out[j].s {
			return out[i].s > out[j].s
		}
		return out[i].p.Hits > out[j].p.Hits
	})
	res := make([]Pitfall, 0, len(out))
	for _, sc := range out {
		res = append(res, sc.p)
	}
	if limit > 0 && len(res) > limit {
		res = res[:limit]
	}
	return res
}

// Save writes the store to disk as pretty JSON, creating the parent directory. The
// write is atomic (temp + rename) so a crash mid-write can't truncate the lessons
// file — this is the agent's persistent memory.
func (k *KB) Save() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	data, err := json.MarshalIndent(k.pitfalls, "", "  ")
	if err != nil {
		return &KnowledgeError{Op: "marshal", Path: k.path, Err: err}
	}
	if err := atomicfile.Write(k.path, data, 0o644); err != nil {
		return &KnowledgeError{Op: "write", Path: k.path, Err: err}
	}
	return nil
}

// All returns a copy of the stored pitfalls.
func (k *KB) All() []Pitfall {
	k.mu.Lock()
	defer k.mu.Unlock()
	return append([]Pitfall(nil), k.pitfalls...)
}

func dedupeKey(p Pitfall) string { return p.Domain + "\x00" + norm(p.ErrorPattern) }

func norm(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), " ")
}

// tokens is the set of distinct word tokens (>=3 runes) in s, lowercased.
func tokens(s string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, w := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len([]rune(w)) >= 3 {
			out[w] = struct{}{}
		}
	}
	return out
}
