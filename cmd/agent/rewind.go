package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// rewindCommand is the REPL/text path: bare lists checkpoints, /rewind <n> applies
// the n-th (1 = most recent). The TUI uses an interactive picker instead.
func (a *app) rewindCommand(rest string) []string {
	rows := a.rewindRows()
	if len(rows) == 0 {
		return []string{"nothing to rewind — no turns yet this session"}
	}
	arg := strings.TrimSpace(rest)
	if arg == "" {
		out := []string{"rewind to (run /rewind <n>):"}
		for i, r := range rows {
			out = append(out, fmt.Sprintf("  %d  %-46s %d file(s)", i+1, oneLine(r.goal, 46), r.files))
		}
		return out
	}
	n, err := strconv.Atoi(arg)
	if err != nil || n < 1 || n > len(rows) {
		return []string{fmt.Sprintf("usage: /rewind <1..%d>", len(rows))}
	}
	return a.applyRewind(rows[n-1].idx)
}

// checkpoint is the state captured at the START of one turn, so /rewind can
// restore to before it: how long the session memory was then, and the prior
// content of every file the turn (or its sub-agents) went on to change.
type checkpoint struct {
	goal    string
	histLen int
	files   map[string]fileSnap // absolute path → prior state
}

type fileSnap struct {
	content []byte
	existed bool
	tooBig  bool // larger than the snapshot cap — recorded but not restorable
}

const (
	maxSnapBytes   = 1 << 20 // don't copy files larger than 1 MiB into a checkpoint
	maxCheckpoints = 50      // keep the most recent N turns
)

// beginCheckpoint opens a checkpoint for a new turn (call before the agent runs).
func (a *app) beginCheckpoint(goal string) {
	a.ckptMu.Lock()
	defer a.ckptMu.Unlock()
	cp := &checkpoint{goal: oneLine(goal, 60), histLen: a.ag.SessionLen(), files: map[string]fileSnap{}}
	a.checkpoints = append(a.checkpoints, cp)
	if len(a.checkpoints) > maxCheckpoints {
		a.checkpoints = a.checkpoints[len(a.checkpoints)-maxCheckpoints:]
	}
	a.curCkpt = cp
}

// endCheckpoint closes the running turn's checkpoint.
func (a *app) endCheckpoint() {
	a.ckptMu.Lock()
	a.curCkpt = nil
	a.ckptMu.Unlock()
}

// snapFile records a file's prior content the first time it's touched in a turn.
// Wired into the file tool (and sub-agents), so it fires for every edit/write.
func (a *app) snapFile(abs string) {
	a.ckptMu.Lock()
	defer a.ckptMu.Unlock()
	cp := a.curCkpt
	if cp == nil {
		return
	}
	if _, done := cp.files[abs]; done {
		return
	}
	info, err := os.Stat(abs)
	if err != nil {
		cp.files[abs] = fileSnap{existed: false} // file is being created
		return
	}
	if info.Size() > maxSnapBytes {
		cp.files[abs] = fileSnap{existed: true, tooBig: true}
		return
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		cp.files[abs] = fileSnap{existed: false}
		return
	}
	cp.files[abs] = fileSnap{existed: true, content: data}
}

// rewindRow is one selectable checkpoint for the picker (newest first).
type rewindRow struct {
	idx   int
	goal  string
	files int
}

func (a *app) rewindRows() []rewindRow {
	a.ckptMu.Lock()
	defer a.ckptMu.Unlock()
	rows := make([]rewindRow, 0, len(a.checkpoints))
	for i := len(a.checkpoints) - 1; i >= 0; i-- {
		rows = append(rows, rewindRow{idx: i, goal: a.checkpoints[i].goal, files: len(a.checkpoints[i].files)})
	}
	return rows
}

// applyRewind restores to before checkpoint idx: every file changed from that turn
// onward is reverted to its earliest pre-state (created files deleted), and the
// session memory is trimmed back. Side effects (shell, git, network) can't be undone.
func (a *app) applyRewind(idx int) []string {
	a.ckptMu.Lock()
	if idx < 0 || idx >= len(a.checkpoints) {
		a.ckptMu.Unlock()
		return []string{"nothing to rewind to"}
	}
	seen := map[string]fileSnap{} // earliest pre-state per path across [idx:]
	for i := idx; i < len(a.checkpoints); i++ {
		for p, s := range a.checkpoints[i].files {
			if _, ok := seen[p]; !ok {
				seen[p] = s
			}
		}
	}
	histLen := a.checkpoints[idx].histLen
	a.checkpoints = a.checkpoints[:idx]
	a.ckptMu.Unlock()

	restored, deleted, skipped := 0, 0, 0
	for p, s := range seen {
		switch {
		case s.tooBig:
			skipped++
		case s.existed:
			if os.WriteFile(p, s.content, 0o644) == nil {
				restored++
			}
		default: // created from the target turn onward → remove it
			if os.Remove(p) == nil {
				deleted++
			}
		}
	}
	hist := a.ag.History()
	if histLen > len(hist) {
		histLen = len(hist)
	}
	a.ag.SetHistory(hist[:histLen])
	a.saveSession()

	out := []string{fmt.Sprintf("rewound — restored %d file(s), removed %d, trimmed the conversation", restored, deleted)}
	if skipped > 0 {
		out = append(out, fmt.Sprintf("  (%d file(s) too large to snapshot were left as-is)", skipped))
	}
	out = append(out, "  ⚠ shell commands, git, and network actions were NOT undone")
	return out
}
