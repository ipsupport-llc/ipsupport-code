package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

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
