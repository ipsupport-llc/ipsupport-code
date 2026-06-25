package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadNoFileReturnsDefaults(t *testing.T) {
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
	dir := t.TempDir()
	agentDir := filepath.Join(dir, ".agent")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	js := `{"run":{"allow":["ls*","git status"]}}`
	if err := os.WriteFile(filepath.Join(agentDir, "config.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Run.Allow) != 2 || cfg.Run.Allow[0] != "ls*" {
		t.Errorf("Run.Allow = %v, want [ls* git status]", cfg.Run.Allow)
	}
	// Untouched defaults must survive a partial merge.
	if cfg.LLM.MaxSteps != 12 {
		t.Errorf("MaxSteps = %d, want 12 (default preserved)", cfg.LLM.MaxSteps)
	}
	if cfg.Run.Default != "ask" {
		t.Errorf("Run.Default = %q, want ask (default preserved)", cfg.Run.Default)
	}
	if len(cfg.Run.Deny) == 0 {
		t.Error("Run.Deny defaults were zeroed by partial merge")
	}
}
