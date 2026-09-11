package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// archiveRecord is one durable session-history entry: what was asked and what
// actually happened, kept forever regardless of context-window compaction.
type archiveRecord struct {
	Time  string `json:"time"`
	Goal  string `json:"goal"`
	Entry string `json:"entry"`
}

// sessionArchiver appends every (goal, final+digest) pair remember() commits
// to session history as a durable JSONL record — see agent.Archiver. Unlike
// the session file (a live mirror of the agent's rolling history, so it
// shrinks along with an auto-compact), this file is append-only: nothing here
// is ever rewritten or dropped, so the `history` tool can always recall a
// fact a compacted summary no longer states.
type sessionArchiver struct {
	mu   sync.Mutex
	path string
}

func (s *sessionArchiver) Archive(goal, entry string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return
	}
	f, err := os.OpenFile(s.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	data, err := json.Marshal(archiveRecord{Time: time.Now().UTC().Format(time.RFC3339), Goal: goal, Entry: entry})
	if err != nil {
		return
	}
	_, _ = f.Write(append(data, '\n'))
}

// archivePath is the durable per-session record — see sessionArchiver. It
// sits alongside sessionPath, named after the same session identity.
func (a *app) archivePath() string {
	return filepath.Join(a.workspace, ".agent", "sessions", slugName(a.cfg.Name)+".archive.jsonl")
}

// hasArchivedHistory reports whether there's anything for the `history` tool
// to recall yet — it's only worth its catalog space once there is.
func (a *app) hasArchivedHistory() bool {
	fi, err := os.Stat(a.archivePath())
	return err == nil && fi.Size() > 0
}

// maybeRewireHistoryTool re-checks hasArchivedHistory() at the start of a
// task. wire() only decides whether the `history` tool belongs in the tool
// list once, at wiring time (see historyToolOn) — but a session that STARTS
// with an empty archive gets its first entry archived only after its first
// task completes (Archive() is called from remember(), right before Run()
// returns). Without this check, nothing would ever notice the archive went
// from empty to non-empty, and the tool would stay absent for the rest of the
// running process.
//
// This must run at a task BOUNDARY, not from inside Archive() itself:
// Archive() runs before remember() appends the just-finished turn to the
// agent's live history, so rewiring synchronously from there would rebuild
// a.ag (via wire()) from a history snapshot missing that turn.
//
// It also must run on the same goroutine as the UI's Update/View loop, never
// from inside the task's own goroutine: wire() reassigns a.client and a.ag
// with no lock, and the UI reads both live while a task is in flight. That's
// why runTask/runLoop (tui.go) call this synchronously, before building the
// tea.Cmd closure Bubble Tea runs the task on — not runTaskStreaming itself,
// which executes ON that closure's goroutine.
func (a *app) maybeRewireHistoryTool() {
	if !a.historyToolOn && a.hasArchivedHistory() {
		_ = a.wire()
	}
}

// historySource adapts the on-disk archive to tool.HistorySource.
type historySource struct{ path string }

func (h historySource) records() []archiveRecord {
	f, err := os.Open(h.path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []archiveRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		var rec archiveRecord
		if json.Unmarshal(sc.Bytes(), &rec) == nil {
			out = append(out, rec)
		}
	}
	return out
}

func (h historySource) Recent(n int) []string {
	if n <= 0 {
		n = 5
	}
	recs := h.records()
	if len(recs) > n {
		recs = recs[len(recs)-n:]
	}
	return formatArchiveRecords(recs)
}

// maxHistoryMatches caps a search result so a broad query can't flood the
// model's context with the entire archive.
const maxHistoryMatches = 10

func (h historySource) Search(query string) []string {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}
	var matched []archiveRecord
	for _, r := range h.records() {
		if strings.Contains(strings.ToLower(r.Goal), q) || strings.Contains(strings.ToLower(r.Entry), q) {
			matched = append(matched, r)
		}
	}
	if len(matched) > maxHistoryMatches {
		matched = matched[len(matched)-maxHistoryMatches:]
	}
	return formatArchiveRecords(matched)
}

func formatArchiveRecords(recs []archiveRecord) []string {
	out := make([]string, len(recs))
	for i, r := range recs {
		out[i] = fmt.Sprintf("goal: %s\nanswer: %s", r.Goal, r.Entry)
	}
	return out
}
