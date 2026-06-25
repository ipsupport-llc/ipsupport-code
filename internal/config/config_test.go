package config

import (
	"os"
	"path/filepath"
	"testing"
)

// isolate points HOME at a temp dir so configHome()/GlobalPath() never touch the
// real user config during tests.
func isolate(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
}

func TestLoadNoFileReturnsDefaults(t *testing.T) {
	isolate(t)
	dir := t.TempDir()

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.LLM.MaxSteps != 12 {
		t.Errorf("MaxSteps = %d, want 12", cfg.LLM.MaxSteps)
	}
	if cfg.LLM.BaseURL != "http://localhost:1234/v1" {
		t.Errorf("BaseURL = %q", cfg.LLM.BaseURL)
	}
	if cfg.Run.Default != "ask" {
		t.Errorf("Run.Default = %q, want ask", cfg.Run.Default)
	}
	want, _ := filepath.Abs(dir)
	if cfg.Workspace != want {
		t.Errorf("Workspace = %q, want %q", cfg.Workspace, want)
	}
	if cfg.KBPath == "" || cfg.TracePath == "" {
		t.Errorf("KBPath=%q TracePath=%q, want non-empty defaults", cfg.KBPath, cfg.TracePath)
	}
}

func TestLoadMergesPartial(t *testing.T) {
	isolate(t)
	dir := writeWorkspaceConfig(t, `{"run":{"allow":["ls*","git status"]}}`)

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Run.Allow) != 2 || cfg.Run.Allow[0] != "ls*" {
		t.Errorf("Run.Allow = %v, want [ls* git status]", cfg.Run.Allow)
	}
	if cfg.LLM.MaxSteps != 12 {
		t.Errorf("MaxSteps = %d, want 12 (default preserved)", cfg.LLM.MaxSteps)
	}
	if cfg.Run.Default != "ask" {
		t.Errorf("Run.Default = %q, want ask (default preserved)", cfg.Run.Default)
	}
}

// The footgun the deny floor fixes: a workspace config that sets its OWN run.deny
// must NOT be able to drop the protective guards.
func TestLoadUnionsDenyFloor(t *testing.T) {
	isolate(t)
	dir := writeWorkspaceConfig(t, `{"run":{"deny":["my-own-rule*"]}}`)

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !contains(cfg.Run.Deny, "my-own-rule*") {
		t.Errorf("user deny rule missing: %v", cfg.Run.Deny)
	}
	for _, must := range []string{"rm -rf*", "sudo*", "shutdown*"} {
		if !contains(cfg.Run.Deny, must) {
			t.Errorf("protective deny %q dropped by user config: %v", must, cfg.Run.Deny)
		}
	}
}

func TestGlobalConfigMerged(t *testing.T) {
	isolate(t)
	if err := SaveGlobal("renamed-bot", LLM{BaseURL: "http://host:9999/v1", Model: "custom", MaxSteps: 7}); err != nil {
		t.Fatal(err)
	}
	if !GlobalExists() {
		t.Fatal("GlobalExists() = false after SaveGlobalLLM")
	}
	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.BaseURL != "http://host:9999/v1" || cfg.LLM.Model != "custom" || cfg.LLM.MaxSteps != 7 {
		t.Errorf("global config not applied: %+v", cfg.LLM)
	}
	if cfg.Name != "renamed-bot" {
		t.Errorf("name = %q, want renamed-bot", cfg.Name)
	}
}

func writeWorkspaceConfig(t *testing.T, js string) string {
	t.Helper()
	dir := t.TempDir()
	agentDir := filepath.Join(dir, ".agent")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "config.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
