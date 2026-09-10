package tool

import (
	"context"
	"strings"
)

// HistorySource is the read-only view of the session's durable task record —
// every (goal, final answer + actions digest) this session has ever produced,
// kept in full even after auto-compact folds the live session memory into a
// short recap. It lets the model recover a fact compaction has since
// paraphrased away, instead of guessing.
type HistorySource interface {
	// Recent returns up to n of the most recently archived entries, oldest
	// first. n<=0 means the source's own default.
	Recent(n int) []string
	// Search returns archived entries whose goal or answer contains query
	// (case-insensitive substring), oldest match first, capped at a small
	// count so a broad query can't flood the model's context.
	Search(query string) []string
}

// NewHistory exposes the session's full task-by-task record to the model, so
// it can recall something the compacted session summary no longer states —
// e.g. which files an earlier task actually created. Only registered once
// there's something archived to recall (see the caller) — it costs nothing in
// the catalog for a session that doesn't need it yet.
func NewHistory(src HistorySource) Tool {
	return NewDomain(DomainSpec{
		Name:    "history",
		Summary: "Recall earlier tasks THIS session ran, including ones a context-compaction summary has since shortened.",
		Details: "Use when you suspect something you already did (a file you created, a decision you made) isn't in the current context anymore.",
		NotHere: "NOT here — the CURRENT content of a file → file.read (this is what happened, not what's on disk now).",
		Actions: []Action{
			{
				Name:   "recent",
				Note:   "(the last N tasks this session ran; default 5)",
				Params: []Param{Opt("n", "int", "5")},
				Run: func(_ context.Context, a Args) Result {
					return Ok(joinHistoryEntries(src.Recent(a.Int("n", 5))))
				},
			},
			{
				Name:   "search",
				Note:   "(archived tasks whose goal or answer mentions query)",
				Params: []Param{Req("query", "str")},
				Run: func(_ context.Context, a Args) Result {
					return Ok(joinHistoryEntries(src.Search(a.Str("query"))))
				},
			},
		},
	})
}

func joinHistoryEntries(entries []string) string {
	if len(entries) == 0 {
		return "(no match)"
	}
	return strings.Join(entries, "\n---\n")
}
