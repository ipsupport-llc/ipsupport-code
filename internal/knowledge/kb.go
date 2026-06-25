package knowledge

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

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

// KB is an in-memory view of the pitfall store plus its backing file path.
type KB struct {
	path     string
	pitfalls []Pitfall
}

// Open loads the store at path. A missing file is an empty store, not an error.
func Open(path string) (*KB, error) {
	kb := &KB{path: path}
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
	return kb, nil
}

// Add inserts a pitfall, de-duplicating on (domain, normalized error pattern).
// A duplicate bumps Hits and back-fills any empty fields; returns true only when
// a genuinely new lesson was stored.
func (k *KB) Add(p Pitfall) bool {
	key := dedupeKey(p)
	for i := range k.pitfalls {
		if dedupeKey(k.pitfalls[i]) == key {
			k.pitfalls[i].Hits++
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
	k.pitfalls = append(k.pitfalls, p)
	return true
}

// Query returns pitfalls for a domain. With a non-empty errText it keeps only
// keyword-overlapping lessons, ranked by overlap then Hits. With an empty
// errText it returns every lesson in the domain ranked by Hits (used by the
// help tool). limit <= 0 means no cap.
func (k *KB) Query(domain, errText string, limit int) []Pitfall {
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

// Save writes the store to disk as pretty JSON, creating the parent directory.
func (k *KB) Save() error {
	if dir := filepath.Dir(k.path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return &KnowledgeError{Op: "mkdir", Path: dir, Err: err}
		}
	}
	data, err := json.MarshalIndent(k.pitfalls, "", "  ")
	if err != nil {
		return &KnowledgeError{Op: "marshal", Path: k.path, Err: err}
	}
	if err := os.WriteFile(k.path, data, 0o644); err != nil {
		return &KnowledgeError{Op: "write", Path: k.path, Err: err}
	}
	return nil
}

// All returns a copy of the stored pitfalls.
func (k *KB) All() []Pitfall { return append([]Pitfall(nil), k.pitfalls...) }

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
