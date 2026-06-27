// Package config loads the agent's runtime configuration. Settings come from
// two JSON files merged over safe defaults: a machine-level user config
// (~/.config/ipsupport-code/config.json — the LM Studio endpoint, written by
// first-run setup) and a per-workspace config (<workspace>/.agent/config.json —
// the permission policy). A protective deny floor is unioned in afterwards so a
// workspace config can never drop the most dangerous-command guards.
package config

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
)

// LLM describes the OpenAI-compatible endpoint (LM Studio by default). The same
// shape points at OpenAI or a LiteLLM proxy by changing BaseURL/APIKey.
type LLM struct {
	BaseURL     string  `json:"base_url"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	MaxSteps    int     `json:"max_steps"`
	APIKey      string  `json:"api_key,omitempty"`
	// ContextWindow is the model's context size in tokens; auto-compact triggers
	// as the prompt approaches it. 0 disables auto-compact.
	ContextWindow int `json:"context_window,omitempty"`
}

// RunPolicy gates shell execution. Resolution per command: a Deny glob (matched
// anywhere) blocks, an Allow glob (whole command) auto-runs, otherwise Default.
type RunPolicy struct {
	Default string   `json:"default"`
	Allow   []string `json:"allow"`
	Deny    []string `json:"deny"`
}

// FilePolicy gates file writes and confines all file ops to Jail (empty = no
// jail). AllowWrite/DenyWrite are globs applied to writes; reads are jail-checked
// only.
type FilePolicy struct {
	Default    string   `json:"default"`
	Jail       string   `json:"jail"`
	AllowWrite []string `json:"allow_write"`
	DenyWrite  []string `json:"deny_write"`
}

// Config is the merged runtime configuration.
type Config struct {
	Name      string     `json:"name,omitempty"` // display name (renameable)
	LLM       LLM        `json:"llm"`
	Run       RunPolicy  `json:"run"`
	File      FilePolicy `json:"file"`
	KBPath     string     `json:"kb_path,omitempty"`
	TracePath  string     `json:"trace_path,omitempty"`
	SkillsPath string     `json:"skills_path,omitempty"`
	Workspace  string     `json:"-"` // resolved absolute workspace root
}

// Non-overridable safety floor, always unioned into the resolved policy.
var (
	runDenyFloor  = []string{"rm -rf*", "sudo*", "mkfs*", "dd if=*", ":(){*", "shutdown*", "reboot*"}
	fileDenyFloor = []string{".git", ".git/**", "**/*secret*", "**/.env*"}
)

// Default returns the baseline configuration: LM Studio on localhost, ask-by-
// default policies, a jail at the workspace root, and the protective deny floor.
func Default() Config {
	return Config{
		Name: "ipsupport-code",
		LLM: LLM{
			BaseURL:       "http://localhost:1234/v1",
			Model:         "qwen2.5-7b-instruct",
			Temperature:   0.2,
			MaxSteps:      12,
			ContextWindow: 8192,
		},
		Run: RunPolicy{
			Default: "ask",
			Allow:   []string{},
			Deny:    append([]string{}, runDenyFloor...),
		},
		File: FilePolicy{
			Default:    "ask",
			Jail:       ".",
			AllowWrite: []string{},
			DenyWrite:  append([]string{}, fileDenyFloor...),
		},
	}
}

// configHome is ~/.config/ipsupport-code on both Linux and macOS (matching the
// operator's expectation), falling back to a workspace-local .agent dir.
func configHome() string {
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".config", "ipsupport-code")
	}
	return ".agent"
}

// GlobalPath is the machine-level user config file.
func GlobalPath() string { return filepath.Join(configHome(), "config.json") }

// GlobalExists reports whether the user config file is present.
func GlobalExists() bool {
	_, err := os.Stat(GlobalPath())
	return err == nil
}

// DefaultKBPath is the global knowledge-base location.
func DefaultKBPath() string { return filepath.Join(configHome(), "knowledge.json") }

// DefaultTracePath is the global decision-trace (training dataset) location.
func DefaultTracePath() string { return filepath.Join(configHome(), "traces.jsonl") }

// DefaultSkillsPath is the global skills directory.
func DefaultSkillsPath() string { return filepath.Join(configHome(), "skills") }

// SaveGlobal writes the machine-level settings (display name + LLM connection)
// to the user config file, creating its directory.
func SaveGlobal(name string, l LLM) error {
	if name == "" {
		name = "ipsupport-code"
	}
	if err := os.MkdirAll(configHome(), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(map[string]any{"name": name, "llm": l}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(GlobalPath(), data, 0o644)
}

// SaveWorkspacePolicy persists the run/file permission policy to the workspace
// config (<workspace>/.agent/config.json) so /permissions changes survive a
// restart. Any other keys already in that file are preserved.
func SaveWorkspacePolicy(workspace string, run RunPolicy, file FilePolicy) error {
	path := filepath.Join(workspace, ".agent", "config.json")
	raw := map[string]json.RawMessage{}
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &raw)
	}
	for key, val := range map[string]any{"run": run, "file": file} {
		b, err := json.Marshal(val)
		if err != nil {
			return err
		}
		raw[key] = b
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// Load merges the user config then the workspace config over Default(), unions
// the deny floor, and fills in default paths. Absent files are not errors.
func Load(workspace string) (Config, error) {
	cfg := Default()

	if err := mergeFile(&cfg, GlobalPath()); err != nil {
		return cfg, err
	}

	abs, err := filepath.Abs(workspace)
	if err != nil {
		return cfg, err
	}
	if err := mergeFile(&cfg, filepath.Join(abs, ".agent", "config.json")); err != nil {
		return cfg, err
	}

	cfg.Workspace = abs
	cfg.Run.Deny = union(cfg.Run.Deny, runDenyFloor)
	cfg.File.DenyWrite = union(cfg.File.DenyWrite, fileDenyFloor)
	if cfg.KBPath == "" {
		cfg.KBPath = DefaultKBPath()
	}
	if cfg.TracePath == "" {
		cfg.TracePath = DefaultTracePath()
	}
	if cfg.SkillsPath == "" {
		cfg.SkillsPath = DefaultSkillsPath()
	}
	return cfg, nil
}

// mergeFile unmarshals a JSON config file over cfg (keys absent from the file
// keep their current value). A missing file is a no-op.
func mergeFile(cfg *Config, path string) error {
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		return json.Unmarshal(data, cfg)
	case errors.Is(err, fs.ErrNotExist):
		return nil
	default:
		return err
	}
}

// union appends floor entries not already present in base.
func union(base, floor []string) []string {
	seen := make(map[string]struct{}, len(base))
	for _, b := range base {
		seen[b] = struct{}{}
	}
	for _, f := range floor {
		if _, ok := seen[f]; !ok {
			base = append(base, f)
			seen[f] = struct{}{}
		}
	}
	return base
}
