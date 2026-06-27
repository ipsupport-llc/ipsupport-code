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

// /permissions writes the relaxed policy here; it must survive a reload, and the
// deny floor must still be re-unioned on top of it.
func TestSaveWorkspacePolicyRoundTrips(t *testing.T) {
	isolate(t)
	dir := t.TempDir()

	cfg, _ := Load(dir)
	cfg.File.Default = "allow" // as /permissions files on would set
	if err := SaveWorkspacePolicy(dir, cfg.Run, cfg.File); err != nil {
		t.Fatalf("SaveWorkspacePolicy: %v", err)
	}

	reloaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.File.Default != "allow" {
		t.Errorf("File.Default = %q, want allow (persisted)", reloaded.File.Default)
	}
	if !contains(reloaded.File.DenyWrite, "**/.env*") {
		t.Errorf("deny floor lost after save/reload: %v", reloaded.File.DenyWrite)
	}
}

func TestResolveProvider(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	// template-only: base_url/model from the template, key from the env var.
	l, ok := ResolveProvider(Config{}, "openai")
	if !ok || l.BaseURL != "https://api.openai.com/v1" || l.Model != "gpt-4o-mini" || l.APIKey != "sk-env" {
		t.Fatalf("openai template = %+v ok=%v", l, ok)
	}
	if l.MaxSteps == 0 {
		t.Error("MaxSteps default should be filled")
	}
	if l.Temperature != 0 {
		t.Errorf("Temperature = %v, want 0 left unset (client omits it so hosted models accept their default)", l.Temperature)
	}
	// a user preset overrides the template (key + model), base_url still filled.
	cfg := Config{Providers: map[string]LLM{"openai": {APIKey: "sk-preset", Model: "gpt-4o"}}}
	if l, _ := ResolveProvider(cfg, "openai"); l.APIKey != "sk-preset" || l.Model != "gpt-4o" || l.BaseURL != "https://api.openai.com/v1" {
		t.Errorf("preset merge = %+v", l)
	}
	if _, ok := ResolveProvider(Config{}, "nope"); ok {
		t.Error("unknown provider should be false")
	}
}

func TestSaveGlobalPreservesOtherKeys(t *testing.T) {
	isolate(t)
	// stash providers (with a key) + a channel in the global file
	if err := SaveProviders("openrouter", map[string]LLM{"openrouter": {APIKey: "sk-keep"}}); err != nil {
		t.Fatal(err)
	}
	if err := SaveChannel("nightly"); err != nil {
		t.Fatal(err)
	}
	// a /rename or /model-on-local would call SaveGlobal — it must NOT wipe them
	if err := SaveGlobal("bob", Default().LLM); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Name != "bob" {
		t.Errorf("name = %q, want bob", cfg.Name)
	}
	if cfg.Providers["openrouter"].APIKey != "sk-keep" {
		t.Errorf("provider key lost after SaveGlobal: %+v", cfg.Providers)
	}
	if cfg.Channel != "nightly" {
		t.Errorf("channel lost after SaveGlobal: %q", cfg.Channel)
	}
}

func TestSaveProvidersRoundTrip(t *testing.T) {
	isolate(t)
	if err := SaveProviders("openai", map[string]LLM{"openai": {APIKey: "sk-x", Model: "gpt-4o"}}); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider != "openai" || cfg.Providers["openai"].APIKey != "sk-x" {
		t.Errorf("providers not persisted: provider=%q providers=%+v", cfg.Provider, cfg.Providers)
	}
	if fi, _ := os.Stat(GlobalPath()); fi.Mode().Perm() != 0o600 {
		t.Errorf("config perms = %o, want 600 (holds API keys)", fi.Mode().Perm())
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
