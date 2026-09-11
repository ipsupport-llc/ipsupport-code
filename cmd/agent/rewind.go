package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	udiff "github.com/aymanbagabas/go-udiff"
)

// rewindPreviewItem is one change rewinding to a step would make.
type rewindPreviewItem struct {
	Rel  string // path relative to the workspace
	Kind string // restore | delete | toobig
	Diff string // unified diff (current → restored), for Kind == restore
}

// rewindPreview computes what rewinding to checkpoint idx would do: per affected
// file the change (diff to restore / delete a created file / skip a too-big one)
// and how many conversation messages get trimmed. Used to preview in the picker.
func (a *app) rewindPreview(idx int) ([]rewindPreviewItem, int) {
	a.ckptMu.Lock()
	if idx < 0 || idx >= len(a.checkpoints) {
		a.ckptMu.Unlock()
		return nil, 0
	}
	seen := map[string]fileSnap{}
	for i := idx; i < len(a.checkpoints); i++ {
		for p, s := range a.checkpoints[i].files {
			if _, ok := seen[p]; !ok {
				seen[p] = s
			}
		}
	}
	histLen := a.checkpoints[idx].histLen
	a.ckptMu.Unlock()

	trimmed := a.ag.SessionLen() - histLen
	if trimmed < 0 {
		trimmed = 0
	}
	paths := make([]string, 0, len(seen))
	for p := range seen {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	var items []rewindPreviewItem
	for _, p := range paths {
		rel := p
		if r, err := filepath.Rel(a.workspace, p); err == nil {
			rel = r
		}
		s := seen[p]
		switch {
		case s.tooBig:
			items = append(items, rewindPreviewItem{Rel: rel, Kind: "toobig"})
		case !s.existed:
			items = append(items, rewindPreviewItem{Rel: rel, Kind: "delete"})
		default:
			cur, _ := os.ReadFile(p)
			if string(cur) == string(s.content) {
				continue // unchanged since the checkpoint — nothing to restore
			}
			items = append(items, rewindPreviewItem{Rel: rel, Kind: "restore",
				Diff: udiff.Unified(rel, rel, string(cur), string(s.content))})
		}
	}
	return items, trimmed
}

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

// beginCheckpoint opens a checkpoint for a new turn (call before the agent
// runs) and returns it, so the caller's endCheckpoint can only ever close its
// OWN checkpoint (see endCheckpoint) — a force-detached task's deferred close
// firing late must not wipe curCkpt out from under a different task that
// started in the meantime.
func (a *app) beginCheckpoint(goal string) *checkpoint {
	a.ckptMu.Lock()
	defer a.ckptMu.Unlock()
	cp := &checkpoint{goal: oneLine(goal, 60), histLen: a.ag.SessionLen(), files: map[string]fileSnap{}}
	a.checkpoints = append(a.checkpoints, cp)
	if len(a.checkpoints) > maxCheckpoints {
		a.checkpoints = a.checkpoints[len(a.checkpoints)-maxCheckpoints:]
	}
	a.curCkpt = cp
	return cp
}

// endCheckpoint closes cp — but ONLY if it's still the live one. A detached
// (orphaned) task's goroutine can return long after a new task has begun its
// own checkpoint; without this check its deferred close would clear the NEW
// task's curCkpt, silently stopping snapFile from recording its edits.
func (a *app) endCheckpoint(cp *checkpoint) {
	a.ckptMu.Lock()
	if a.curCkpt == cp {
		a.curCkpt = nil
	}
	a.ckptMu.Unlock()
}

// resetCheckpoints drops all checkpoints and closes any open one — a
// checkpoint's histLen indexes a specific session's conversation, so it's
// meaningless once /new or /sessions switches to a different thread.
func (a *app) resetCheckpoints() {
	a.ckptMu.Lock()
	a.checkpoints = nil
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
		if os.IsNotExist(err) {
			cp.files[abs] = fileSnap{existed: false} // file is being created
		} else {
			// Some other stat failure (permission, I/O) — the file may well
			// already exist. Treat it like "too big to snapshot" so rewind
			// leaves it alone instead of risking deleting a file that was
			// already there before this turn.
			cp.files[abs] = fileSnap{existed: true, tooBig: true}
		}
		return
	}
	if info.Size() > maxSnapBytes {
		cp.files[abs] = fileSnap{existed: true, tooBig: true}
		return
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		// Stat just succeeded, so the file definitely exists — a read
		// failure here means "can't snapshot", not "doesn't exist".
		cp.files[abs] = fileSnap{existed: true, tooBig: true}
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

// ancestorRedirected reports whether p's parent directory has been swapped for
// a symlink (or sits beneath one) since it was captured: os.Lstat(p) only ever
// inspects the LEAF component of p, so a parent directory replaced with a
// symlink to somewhere else entirely is invisible to it — WriteFile/Remove
// below would follow that ancestor symlink transparently and act on whatever
// it now points at instead of the checkpointed path. An error from
// EvalSymlinks (parent simply doesn't exist) is left for the caller's own
// existing fallback to handle, not treated as redirection.
func ancestorRedirected(p string) bool {
	dir := filepath.Dir(p)
	real, err := filepath.EvalSymlinks(dir)
	return err == nil && real != dir
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
	a.ckptMu.Unlock()

	restored, deleted, skipped := 0, 0, 0
	var failed []string
	for p, s := range seen {
		switch {
		case s.tooBig:
			skipped++
		case s.existed:
			// p was recorded at checkpoint time and never re-validated since —
			// if it's been swapped for a symlink (or anything non-regular) in
			// the meantime, os.WriteFile would follow it and clobber whatever
			// it now points at instead of restoring the checkpointed file. A
			// Lstat error here means the path is simply gone, which is fine:
			// WriteFile below recreates it as a plain file.
			if ancestorRedirected(p) {
				failed = append(failed, p+" (parent directory redirected — refusing to restore through it)")
			} else if fi, err := os.Lstat(p); err == nil && !fi.Mode().IsRegular() {
				failed = append(failed, p+" (no longer a plain file — refusing to restore through it)")
			} else if err := os.WriteFile(p, s.content, 0o644); err == nil {
				restored++
			} else {
				failed = append(failed, p)
			}
		default: // created from the target turn onward → remove it
			if ancestorRedirected(p) {
				failed = append(failed, p+" (parent directory redirected — refusing to remove through it)")
			} else if err := os.Remove(p); err == nil || os.IsNotExist(err) {
				deleted++
			} else {
				failed = append(failed, p)
			}
		}
	}
	hist := a.ag.History()
	if histLen > len(hist) {
		histLen = len(hist)
	}
	a.ag.SetHistory(hist[:histLen])
	a.saveSession()

	// Only discard the checkpoint once its file changes actually applied — a
	// failure partway through (disk full, permissions) should stay retryable
	// instead of silently vanishing along with the only copy of the prior
	// file content.
	if len(failed) == 0 {
		a.ckptMu.Lock()
		a.checkpoints = a.checkpoints[:idx]
		a.ckptMu.Unlock()
	}

	out := []string{fmt.Sprintf("rewound — restored %d file(s), removed %d, trimmed the conversation", restored, deleted)}
	if skipped > 0 {
		out = append(out, fmt.Sprintf("  (%d file(s) too large to snapshot were left as-is)", skipped))
	}
	if len(failed) > 0 {
		sort.Strings(failed)
		out = append(out, fmt.Sprintf("  ⚠ %d file(s) failed to restore, checkpoint kept for retry: %s", len(failed), strings.Join(failed, ", ")))
	}
	out = append(out, "  ⚠ shell commands, git, and network actions were NOT undone")
	return out
}
