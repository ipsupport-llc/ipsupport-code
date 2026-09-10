package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// The exact scenario an operator hit live: the /config panel's file-writes
// toggle persists to the WORKSPACE file (SaveWorkspacePolicy), while a bare
// `config set` (no --local) persists to the GLOBAL one — and the workspace
// file always wins the merge. Without a warning, the global `set` reports
// success and silently has zero effect.
func TestShadowWarningFiresForGlobalSetShadowedByWorkspace(t *testing.T) {
	ws := t.TempDir()
	wsPath := filepath.Join(ws, ".agent", "config.json")
	if err := config.SetFileValue(wsPath, 0o644, "file.default", "deny"); err != nil {
		t.Fatal(err)
	}

	w := shadowWarning(ws, false, "file.default")
	if !strings.Contains(w, wsPath) {
		t.Errorf("shadowWarning = %q, want it to name the shadowing workspace file", w)
	}
}

// A key the workspace file does NOT define isn't shadowed — no warning.
func TestShadowWarningSilentWhenNotShadowed(t *testing.T) {
	ws := t.TempDir()
	if w := shadowWarning(ws, false, "file.default"); w != "" {
		t.Errorf("shadowWarning = %q, want empty (no workspace override exists)", w)
	}
}

// A --local write is applied directly to the workspace file, which always
// wins the merge — it can never be shadowed, so this never warns (it must not
// even need to look at any file to know that).
func TestShadowWarningNeverFiresForLocalWrites(t *testing.T) {
	if w := shadowWarning(t.TempDir(), true, "file.default"); w != "" {
		t.Errorf("shadowWarning(wroteLocal=true) = %q, want always empty", w)
	}
}
