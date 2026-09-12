package main

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// alwaysApprove is a trivial Approver for tests that don't exercise "ask" policy.
type alwaysApprove struct{}

func (alwaysApprove) Approve(_ context.Context, _, _ string) bool { return true }

// TestApplyRewindRefusesToRestoreThroughSymlink reproduces a TOCTOU: a file is
// checkpointed, then — after the checkpoint but before /rewind runs — swapped
// for a symlink pointing outside the workspace. os.WriteFile follows symlinks
// (no O_NOFOLLOW), so restoring must not blindly write to the checkpointed
// path: it would silently clobber whatever the link points at instead of the
// file that was actually snapshotted.
func TestApplyRewindRefusesToRestoreThroughSymlink(t *testing.T) {
	ws := t.TempDir()
	a := &app{workspace: ws, cfg: config.Config{Name: "default"}, ag: agent.New(nil, nil, nil, nil, "", 0)}

	victim := filepath.Join(ws, "victim.txt")
	if err := os.WriteFile(victim, []byte("original content\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	cp := a.beginCheckpoint("test")
	a.snapFile(victim)
	a.endCheckpoint(cp)

	outside := filepath.Join(t.TempDir(), "attacker.txt") // outside the workspace
	attackerContent := []byte("do not touch\n")
	if err := os.WriteFile(outside, attackerContent, 0o644); err != nil {
		t.Fatal(err)
	}

	// Something replaces the checkpointed path with a symlink to the outside
	// file before /rewind runs.
	if err := os.Remove(victim); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, victim); err != nil {
		t.Fatal(err)
	}

	out := a.applyRewind(0)

	got, err := os.ReadFile(outside)
	if err != nil {
		t.Fatalf("outside file vanished: %v", err)
	}
	if string(got) != string(attackerContent) {
		t.Fatalf("outside file was overwritten by restore: got %q, want %q", got, attackerContent)
	}

	fi, err := os.Lstat(victim)
	if err != nil {
		t.Fatalf("checkpointed path vanished: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("checkpointed path is no longer a symlink — restore wrote through it")
	}

	if joined := strings.Join(out, "\n"); !strings.Contains(joined, "victim.txt") {
		t.Errorf("rewind output doesn't clearly flag the skipped file, got: %v", out)
	}
}

// TestApplyRewindRefusesToRestoreThroughAncestorSymlink reproduces a different
// TOCTOU: instead of the checkpointed path ITSELF becoming a symlink (see
// above), its PARENT DIRECTORY is replaced with a symlink to an external
// location that happens to contain a same-named file. os.Lstat(p) only
// inspects the leaf, so it reports "still a plain file" — but os.WriteFile
// resolves the parent symlink transparently at the OS level and clobbers the
// external file instead of restoring the checkpointed one.
func TestApplyRewindRefusesToRestoreThroughAncestorSymlink(t *testing.T) {
	ws := t.TempDir()
	a := &app{workspace: ws, cfg: config.Config{Name: "default"}, ag: agent.New(nil, nil, nil, nil, "", 0)}

	sub := filepath.Join(ws, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	victim := filepath.Join(sub, "file.txt")
	if err := os.WriteFile(victim, []byte("original content\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	cp := a.beginCheckpoint("test")
	a.snapFile(victim)
	a.endCheckpoint(cp)

	outsideDir := t.TempDir() // outside the workspace
	outsideFile := filepath.Join(outsideDir, "file.txt")
	attackerContent := []byte("do not touch\n")
	if err := os.WriteFile(outsideFile, attackerContent, 0o644); err != nil {
		t.Fatal(err)
	}

	// Something replaces the checkpointed file's PARENT DIRECTORY with a
	// symlink to the outside directory before /rewind runs.
	if err := os.RemoveAll(sub); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outsideDir, sub); err != nil {
		t.Fatal(err)
	}

	out := a.applyRewind(0)

	got, err := os.ReadFile(outsideFile)
	if err != nil {
		t.Fatalf("outside file vanished: %v", err)
	}
	if string(got) != string(attackerContent) {
		t.Fatalf("outside file was overwritten by restore: got %q, want %q", got, attackerContent)
	}

	fi, err := os.Lstat(sub)
	if err != nil {
		t.Fatalf("checkpointed path's parent vanished: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("parent directory is no longer a symlink — restore wrote through it")
	}

	if joined := strings.Join(out, "\n"); !strings.Contains(joined, "file.txt") {
		t.Errorf("rewind output doesn't clearly flag the skipped file, got: %v", out)
	}
}

// TestApplyRewindRefusesToDeleteThroughAncestorSymlink is the delete-branch
// counterpart: a file created during the turn (so rewind must remove it) has
// its parent directory replaced with a symlink to an external location before
// /rewind runs. os.Remove follows that ancestor symlink just as transparently
// as WriteFile does, so without the ancestor check it deletes the external
// file instead of leaving it alone.
func TestApplyRewindRefusesToDeleteThroughAncestorSymlink(t *testing.T) {
	ws := t.TempDir()
	a := &app{workspace: ws, cfg: config.Config{Name: "default"}, ag: agent.New(nil, nil, nil, nil, "", 0)}

	sub := filepath.Join(ws, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	victim := filepath.Join(sub, "file.txt") // doesn't exist yet at checkpoint time

	cp := a.beginCheckpoint("test")
	a.snapFile(victim) // records existed:false — rewind will try to remove it
	if err := os.WriteFile(victim, []byte("created by the turn\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	a.endCheckpoint(cp)

	outsideDir := t.TempDir() // outside the workspace
	outsideFile := filepath.Join(outsideDir, "file.txt")
	protectedContent := []byte("do not delete\n")
	if err := os.WriteFile(outsideFile, protectedContent, 0o644); err != nil {
		t.Fatal(err)
	}

	// Something replaces the checkpointed file's PARENT DIRECTORY with a
	// symlink to the outside directory before /rewind runs.
	if err := os.RemoveAll(sub); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outsideDir, sub); err != nil {
		t.Fatal(err)
	}

	out := a.applyRewind(0)

	got, err := os.ReadFile(outsideFile)
	if err != nil {
		t.Fatalf("outside file was deleted by rewind: %v", err)
	}
	if string(got) != string(protectedContent) {
		t.Fatalf("outside file content changed: got %q, want %q", got, protectedContent)
	}

	fi, err := os.Lstat(sub)
	if err != nil {
		t.Fatalf("checkpointed path's parent vanished: %v", err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("parent directory is no longer a symlink — delete acted through it")
	}

	if joined := strings.Join(out, "\n"); !strings.Contains(joined, "file.txt") {
		t.Errorf("rewind output doesn't clearly flag the skipped file, got: %v", out)
	}
}

// TestSnapFileSkipsFIFO reproduces the checkpoint-snapshot hang: snapFile
// stats then os.ReadFile's a file's prior content before a mutation —
// unguarded, this blocks forever reading a FIFO with no writer, the same
// class of bug internal/tool/file.go's read() was fixed against earlier
// (see !info.Mode().IsRegular() there). This exercises the REAL wiring: the
// file tool's Snapshotter callback is a.snapFile in production (see
// tool.NewFile(pol, gatedApprover{a}, a.snapFile) in main.go), so this
// constructs the file tool the same way and drives it through the "append"
// action (which, unlike "write"/"edit", does no OTHER read of prior content —
// isolating the hang to snapFile's own read, not a second unrelated one).
//
// Actually writing to a FIFO also needs a reader on the other end (opening it
// for write blocks until one shows up) — a background goroutine drains it,
// standing in for a real consumer already reading from the pipe. Without the
// fix, snapFile's own read deadlocks before that write-side open is ever
// reached, so the drain goroutine never unblocks either.
func TestSnapFileSkipsFIFO(t *testing.T) {
	ws := t.TempDir()
	fifoPath := filepath.Join(ws, "pipe")
	if err := syscall.Mkfifo(fifoPath, 0o600); err != nil {
		t.Skipf("mkfifo unsupported here: %v", err)
	}

	c := config.Default()
	c.Workspace = ws
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	pol, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}

	a := &app{workspace: ws, cfg: config.Config{Name: "default"}, ag: agent.New(nil, nil, nil, nil, "", 0)}
	tl := tool.NewFile(pol, alwaysApprove{}, a.snapFile)

	go func() {
		f, err := os.Open(fifoPath)
		if err != nil {
			return
		}
		defer f.Close()
		io.Copy(io.Discard, f)
	}()

	cp := a.beginCheckpoint("test")

	done := make(chan tool.Result, 1)
	go func() {
		done <- tl.Call(context.Background(), "append", map[string]any{"path": "pipe", "content": "x"})
	}()

	// endCheckpoint is deliberately NOT deferred here: if snapFile is broken
	// (holds ckptMu forever), calling it would block right along with it —
	// this select must be able to time out on its own regardless.
	select {
	case r := <-done:
		a.endCheckpoint(cp)
		if r.IsError {
			t.Errorf("append to FIFO errored: %s", r.Content)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("append hung snapshotting a FIFO")
	}
}
