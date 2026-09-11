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
