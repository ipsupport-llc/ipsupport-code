package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestUpdateCheckDefaultTrue(t *testing.T) {
	if !Default().UpdateCheck {
		t.Fatal("UpdateCheck should default to true")
	}
}

// setThenLoad sets a key in a fresh file and returns the config that Load would
// see for that workspace (global path is isolated via HOME).
func TestSetFileValueTypesAndNesting(t *testing.T) {
	isolate(t)
	path := GlobalPath()

	// bool, typed int, string-that-isn't-JSON, and a nested path.
	must := func(k, v string) {
		if err := SetFileValue(path, 0o600, k, v); err != nil {
			t.Fatalf("set %s=%s: %v", k, v, err)
		}
	}
	must("update_check", "false")
	must("goal_max_returns", "8")
	must("channel", "nightly") // bare word → string
	must("run.default", "allow")

	cfg, err := Load(t.TempDir()) // empty workspace: only the global file applies
	if err != nil {
		t.Fatal(err)
	}
	if cfg.UpdateCheck {
		t.Error("update_check not persisted as false")
	}
	if cfg.GoalMaxReturns != 8 {
		t.Errorf("goal_max_returns = %d, want 8", cfg.GoalMaxReturns)
	}
	if cfg.Channel != "nightly" {
		t.Errorf("channel = %q, want nightly", cfg.Channel)
	}
	if cfg.Run.Default != "allow" {
		t.Errorf("run.default = %q, want allow", cfg.Run.Default)
	}
	// The deny floor survives an unrelated run.* edit.
	found := false
	for _, d := range cfg.Run.Deny {
		if d == "rm -rf*" {
			found = true
		}
	}
	if !found {
		t.Error("deny floor lost after setting run.default")
	}
}

func TestSetFileValueRejectsBadTypeAndTypo(t *testing.T) {
	isolate(t)
	path := GlobalPath()

	if err := SetFileValue(path, 0o600, "goal_max_returns", "abc"); err == nil {
		t.Error("expected error setting int field to a string")
	}
	if err := SetFileValue(path, 0o600, "goal_max_retunrs", "8"); err == nil {
		t.Error("expected error on unknown (misspelled) key")
	}
	// A rejected set must not create/clobber the file.
	if _, err := os.Stat(path); err == nil {
		t.Error("file written despite rejected set")
	}
}

func TestSetFileValueAbortsOnCorrupt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte("{ not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := SetFileValue(path, 0o600, "update_check", "false"); err == nil {
		t.Fatal("expected refusal to edit a corrupt file")
	}
	// The corrupt file is left untouched (not clobbered).
	data, _ := os.ReadFile(path)
	if string(data) != "{ not json" {
		t.Error("corrupt file was modified")
	}
}

func TestUnsetFileValue(t *testing.T) {
	isolate(t)
	path := GlobalPath()
	if err := SetFileValue(path, 0o600, "update_check", "false"); err != nil {
		t.Fatal(err)
	}
	if err := UnsetFileValue(path, 0o600, "update_check"); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.UpdateCheck {
		t.Error("update_check should fall back to the default (true) after unset")
	}
	// Unsetting an absent key is a no-op, not an error.
	if err := UnsetFileValue(path, 0o600, "offline"); err != nil {
		t.Errorf("unset of absent key returned error: %v", err)
	}
}

func TestLookupPathAndFlatten(t *testing.T) {
	cfg := Default()
	cfg.Channel = "nightly"

	if v, ok := LookupPath(cfg, "update_check"); !ok || v != true {
		t.Errorf("LookupPath update_check = %v, %v; want true", v, ok)
	}
	if v, ok := LookupPath(cfg, "llm.model"); !ok || v != "qwen2.5-7b-instruct" {
		t.Errorf("LookupPath llm.model = %v, %v", v, ok)
	}
	if _, ok := LookupPath(cfg, "llm.nope"); ok {
		t.Error("LookupPath should miss on an unknown nested key")
	}

	lines := Flatten(cfg)
	want := map[string]bool{"channel\t\"nightly\"": false, "update_check\ttrue": false, "llm.model\t\"qwen2.5-7b-instruct\"": false}
	for _, l := range lines {
		if _, ok := want[l]; ok {
			want[l] = true
		}
	}
	for k, seen := range want {
		if !seen {
			t.Errorf("Flatten missing line: %q", k)
		}
	}
}
