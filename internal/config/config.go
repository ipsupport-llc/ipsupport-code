// Package config loads the agent's runtime configuration: the LM Studio
// endpoint, the workspace permission policy, and the paths of the persisted
// knowledge base and decision trace. Configuration lives in the workspace at
// .agent/config.json and is merged over safe defaults; an absent file is not an
// error — the defaults stand.
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
}

// RunPolicy gates shell execution. Resolution per command: a Deny glob blocks,
// an Allow glob auto-runs, otherwise Default applies (ask|allow|deny).
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
	LLM       LLM        `json:"llm"`
	Run       RunPolicy  `json:"run"`
	File      FilePolicy `json:"file"`
	KBPath    string     `json:"kb_path,omitempty"`
	TracePath string     `json:"trace_path,omitempty"`
	Workspace string     `json:"-"` // resolved absolute workspace root
}

// Default returns the baseline configuration: LM Studio on localhost, ask-by-
// default policies, a jail at the workspace root, and a protective deny list so
// the most destructive commands are blocked even before any config exists.
func Default() Config {
	return Config{
		LLM: LLM{
			BaseURL:     "http://localhost:1234/v1",
			Model:       "qwen2.5-7b-instruct",
			Temperature: 0.2,
			MaxSteps:    12,
		},
		Run: RunPolicy{
			Default: "ask",
			Allow:   []string{},
			Deny: []string{
				"rm -rf*", "sudo*", "mkfs*", "dd if=*",
				"* > /dev/sd*", ":(){*", "shutdown*", "reboot*",
			},
		},
		File: FilePolicy{
			Default:    "ask",
			Jail:       ".",
			AllowWrite: []string{},
			DenyWrite:  []string{".git/**", "**/*secret*", "**/.env*"},
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

// DefaultKBPath is the global knowledge-base location.
func DefaultKBPath() string { return filepath.Join(configHome(), "knowledge.json") }

// DefaultTracePath is the global decision-trace (training dataset) location.
func DefaultTracePath() string { return filepath.Join(configHome(), "traces.jsonl") }

// Load reads <workspace>/.agent/config.json and merges it over Default(). A
// missing file yields the defaults. The merge is a JSON unmarshal over a
// pre-populated Default value, so keys absent from the file keep their default
// (including nested fields and the protective deny list).
func Load(workspace string) (Config, error) {
	cfg := Default()
	abs, err := filepath.Abs(workspace)
	if err != nil {
		return cfg, err
	}

	data, err := os.ReadFile(filepath.Join(abs, ".agent", "config.json"))
	switch {
	case err == nil:
		if err := json.Unmarshal(data, &cfg); err != nil {
			return cfg, err
		}
	case errors.Is(err, fs.ErrNotExist):
		// no workspace config — defaults stand
	default:
		return cfg, err
	}

	cfg.Workspace = abs // json:"-", but set after unmarshal regardless
	if cfg.KBPath == "" {
		cfg.KBPath = DefaultKBPath()
	}
	if cfg.TracePath == "" {
		cfg.TracePath = DefaultTracePath()
	}
	return cfg, nil
}
