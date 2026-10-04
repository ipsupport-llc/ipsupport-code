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
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
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
	// pending is the log of every change not yet on disk — adds, retrieval
	// bumps, deletes, purges — each as the edit it made to pitfalls. Save
	// replays them, in order, onto a freshly re-read copy of the file instead
	// of writing pitfalls over it, so a separate ipsupport-code process sharing
	// the store can't have its lessons silently lost — not to an Add, and not
	// to a Purge or Delete either, which used to replace the whole file with
	// this process's stale snapshot.
	pending []kbOp
	now     func() time.Time // overridable clock for tests
}

// kbOp is one change to a list of lessons, applied once to the in-memory list
// when it happens and again to the fresh on-disk list at Save.
type kbOp func(list *[]Pitfall)

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
	for _, p := range kb.pitfalls {
		if p.LastSeen == "" || p.Added == "" {
			kb.apply(func(list *[]Pitfall) {
				for i := range *list {
					if (*list)[i].LastSeen == "" {
						(*list)[i].LastSeen = today
					}
					if (*list)[i].Added == "" {
						(*list)[i].Added = today
					}
				}
			})
			break
		}
	}
	return kb, nil
}

// apply makes a change now and records it for Save to replay. Callers hold mu
// (Open's caller has the only reference).
func (k *KB) apply(op kbOp) {
	op(&k.pitfalls)
	k.pending = append(k.pending, op)
}

// removeWhere drops every lesson drop reports true for, returning how many.
func removeWhere(list *[]Pitfall, drop func(Pitfall) bool) int {
	kept := (*list)[:0]
	n := 0
	for _, p := range *list {
		if drop(p) {
			n++
			continue
		}
		kept = append(kept, p)
	}
	*list = kept
	return n
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
	isNew := false
	k.apply(func(list *[]Pitfall) {
		isNew = mergeOne(list, p, today) // isNew is read only from the first, in-memory, run
	})
	return isNew
}

// mergeOne merges p into a pitfalls list by dedupeKey — a duplicate bumps Hits
// and back-fills any empty ProvenFix/Context, an original starts at Hits>=1.
// Returns true only when p was a genuinely new lesson, not a duplicate.
func mergeOne(list *[]Pitfall, p Pitfall, today string) bool {
	key := dedupeKey(p)
	for i := range *list {
		if dedupeKey((*list)[i]) == key {
			(*list)[i].Hits++
			(*list)[i].LastSeen = today // recurred → keep it fresh
			if (*list)[i].ProvenFix == "" {
				(*list)[i].ProvenFix = p.ProvenFix
			}
			if (*list)[i].Context == "" {
				(*list)[i].Context = p.Context
			}
			// A later run that actually RECOVERED supersedes an earlier dead end
			// for the same failure. Only Hits and LastSeen used to move, so the
			// first answer stood forever: once an "avoid" hypothesis was stored,
			// no amount of later experience proving what does work could replace
			// it. The reverse never happens — a fix already demonstrated is not
			// unlearned by a later run failing to reproduce it.
			if (*list)[i].Kind == KindAvoid && p.Kind != KindAvoid && strings.TrimSpace(p.ProvenFix) != "" {
				(*list)[i].Kind, (*list)[i].ProvenFix = p.Kind, p.ProvenFix
				if strings.TrimSpace(p.Context) != "" {
					(*list)[i].Context = p.Context
				}
			}
			return false
		}
	}
	if p.Hits == 0 {
		p.Hits = 1
	}
	p.Added, p.LastSeen = today, today
	*list = append(*list, p)
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
	stale := func(p Pitfall) bool {
		t, err := time.Parse(dateFmt, p.LastSeen)
		return err == nil && t.Before(cutoff)
	}
	// Recorded only when it removed something, so a no-op purge leaves Save
	// with nothing to write. On replay it removes only what is stale on disk:
	// a lesson another session saved since is fresh, and stays.
	dropped := removeWhere(&k.pitfalls, stale)
	if dropped > 0 {
		k.pending = append(k.pending, func(list *[]Pitfall) { removeWhere(list, stale) })
	}
	return dropped
}

// Clear removes every lesson, returning how many were dropped.
func (k *KB) Clear() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	n := len(k.pitfalls)
	k.apply(func(list *[]Pitfall) { *list = nil })
	return n
}

// Sorted returns every lesson in a stable, content-derived order (domain, then
// normalized error pattern). Listing and deleting a lesson by its position both
// go through this: the store's own slice order depends on insertion history and
// on Purge compaction, so a number printed by one command could otherwise point
// at a different lesson by the time the next command runs.
func (k *KB) Sorted() []Pitfall {
	out := k.All()
	sort.Slice(out, func(i, j int) bool {
		if out[i].Domain != out[j].Domain {
			return out[i].Domain < out[j].Domain
		}
		return norm(out[i].ErrorPattern) < norm(out[j].ErrorPattern)
	})
	return out
}

// Delete removes the one lesson matching p's (domain, normalized error pattern),
// reporting whether anything was removed. Pass a Pitfall straight from Sorted.
func (k *KB) Delete(p Pitfall) bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	key := dedupeKey(p)
	match := func(q Pitfall) bool { return dedupeKey(q) == key }
	if removeWhere(&k.pitfalls, match) == 0 {
		return false
	}
	k.pending = append(k.pending, func(list *[]Pitfall) { removeWhere(list, match) })
	return true
}

// DropWhere removes every lesson for which drop reports true, returning how many
// went. Used to retire lessons a later rule made unstorable (see
// IsProjectSpecific) without waiting for them to age out.
func (k *KB) DropWhere(drop func(Pitfall) bool) int {
	k.mu.Lock()
	defer k.mu.Unlock()
	dropped := removeWhere(&k.pitfalls, drop)
	if dropped > 0 {
		k.pending = append(k.pending, func(list *[]Pitfall) { removeWhere(list, drop) })
	}
	return dropped
}

// MarkUsed records that a lesson was actually SURFACED to the model, bumping its
// Hits and freshness.
//
// Hits used to move only in Add — i.e. only when the reflection pass re-derived
// the same lesson from a fresh failure. That made the ranking measure exactly
// the wrong thing: a lesson that WORKS stops the failure recurring, so it is
// never re-derived, its Hits and LastSeen freeze, and Purge eventually deletes
// it; a lesson that never helps keeps being re-derived, climbs the ranking and
// stays fresh forever. Counting retrieval instead means usefulness, not
// futility, is what keeps a lesson alive.
//
// Not persisted on its own: the next Save writes it, which is enough for a
// ranking signal and keeps a hot path off the disk.
func (k *KB) MarkUsed(p Pitfall) {
	k.mu.Lock()
	defer k.mu.Unlock()
	key, today := dedupeKey(p), k.today()
	k.apply(func(list *[]Pitfall) {
		for i := range *list {
			if dedupeKey((*list)[i]) == key {
				(*list)[i].Hits++
				(*list)[i].LastSeen = today
				return
			}
		}
	})
}

// Count reports how many lessons are stored. Nil-safe, like All.
func (k *KB) Count() int {
	if k == nil {
		return 0
	}
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
		if IsGenericErrorPattern(p.ErrorPattern) {
			continue // e.g. "exit 1" — matches (and misleads on) any unrelated failure
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

// readPitfalls loads the store at path without mutating a KB — a missing file
// is an empty store, not an error.
func readPitfalls(path string) ([]Pitfall, error) {
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		var out []Pitfall
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

// Save writes the store to disk as pretty JSON, creating the parent directory.
// Before writing, it re-reads the file and replays this KB's own pending
// changes onto that fresh copy instead of blindly overwriting it with
// pitfalls — otherwise two separate ipsupport-code processes sharing the same
// same store could have one's learned lesson silently lost to the other's last
// write (the mutex only protects against races WITHIN one process). With
// nothing pending there is nothing of this KB's own to persist, so Save skips the write
// entirely instead of falling through to an unconditional write of
// k.pitfalls: that write would have no fresh read backing it, so a stale
// in-memory snapshot (e.g. after an idle flush, or a no-op Purge, with no Add
// in between) could silently clobber a fresher file a different process wrote
// since this one last read it. A file lock around the whole read-merge-write
// cycle keeps two such processes from interleaving (one's read landing before
// the other's write, so each only ever merges its own delta) the same way the
// in-process mutex keeps two goroutines from interleaving; an in-memory KB
// (path=="") skips it, since there's nothing on disk to serialize around. The
// write itself is atomic (temp + rename) so a crash mid-write can't truncate
// the lessons file either. pending is only cleared once that write
// actually succeeds — if it fails, they're left intact so the next Save
// replays them instead of silently losing them (a concurrent Add's own Save
// wouldn't otherwise know to replay a lesson it never recorded).
func (k *KB) Save() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if len(k.pending) == 0 {
		return nil // nothing of ours to persist — see doc comment above
	}
	if k.path != "" {
		unlock, err := filelock.Lock(k.path)
		if err != nil {
			return &KnowledgeError{Op: "lock", Path: k.path, Err: err}
		}
		defer unlock()
	}
	if onDisk, err := readPitfalls(k.path); err == nil {
		for _, op := range k.pending {
			op(&onDisk)
		}
		k.pitfalls = onDisk
	}
	// on a read error, fall back to writing our own in-memory state — no
	// worse than the previous unconditional-overwrite behavior.
	data, err := json.MarshalIndent(k.pitfalls, "", "  ")
	if err != nil {
		return &KnowledgeError{Op: "marshal", Path: k.path, Err: err}
	}
	if err := atomicfile.Write(k.path, data, 0o644); err != nil {
		return &KnowledgeError{Op: "write", Path: k.path, Err: err}
	}
	k.pending = nil
	return nil
}

// All returns a copy of the stored pitfalls. A nil KB reports none rather than
// panicking: callers that merely DISPLAY what is known (the /config panel) must
// not be the reason a session dies when the store failed to open.
func (k *KB) All() []Pitfall {
	if k == nil {
		return nil
	}
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
