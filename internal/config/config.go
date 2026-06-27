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
	"sort"
)

// LLM describes the OpenAI-compatible endpoint (LM Studio by default). The same
// shape points at OpenAI or a LiteLLM proxy by changing BaseURL/APIKey.
type LLM struct {
	BaseURL     string  `json:"base_url"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	MaxSteps    int     `json:"max_steps"`
	APIKey      string  `json:"api_key,omitempty"`
	// Type selects extra capabilities: "lmstudio" uses LM Studio's native API
	// (rich model list, context detection); anything else is plain OpenAI-compat.
	Type string `json:"type,omitempty"`
	// ContextWindow is the model's context size in tokens; auto-compact triggers
	// as the prompt approaches it. 0 disables auto-compact.
	ContextWindow int `json:"context_window,omitempty"`
}

// LMStudio reports whether this connection speaks LM Studio's native API.
func (l LLM) LMStudio() bool { return l.Type == "lmstudio" }

// AgentProfile is a named sub-agent target for the `agent` tool: which provider
// and model to run the sub-agent on, plus an optional role prompt (e.g. "you are
// a strict code reviewer").
type AgentProfile struct {
	Provider string `json:"provider"`
	Model    string `json:"model,omitempty"`
	Prompt   string `json:"prompt,omitempty"`
}

// RunPolicy gates shell execution. Resolution per command: a Deny glob (matched
// anywhere) blocks, an Allow glob (whole command) auto-runs, otherwise Default.
type RunPolicy struct {
	Default string   `json:"default"`
	Allow   []string `json:"allow"`
	Deny    []string `json:"deny"`
	// TimeoutSeconds is the default wall-clock limit for a shell command. 0 uses
	// the built-in default (60s). A command may pass a larger per-call `timeout`.
	TimeoutSeconds int `json:"timeout_seconds,omitempty"`
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
	Name      string         `json:"name,omitempty"`      // display name (renameable)
	Channel   string         `json:"channel,omitempty"`   // update channel: stable | nightly
	Provider  string         `json:"provider,omitempty"`  // active provider ("" / "local" = LLM below)
	LLM       LLM            `json:"llm"`                 // the local connection (the "local" provider)
	Providers map[string]LLM `json:"providers,omitempty"` // external provider presets
	Run       RunPolicy      `json:"run"`
	File      FilePolicy     `json:"file"`
	KBPath    string         `json:"kb_path,omitempty"`
	TracePath string         `json:"trace_path,omitempty"`
	UsagePath string         `json:"usage_path,omitempty"`
	// UsageRetentionDays auto-drops usage-ledger entries older than N days on
	// startup. 0 keeps history forever.
	UsageRetentionDays int `json:"usage_retention_days,omitempty"`
	// Prices overrides the built-in per-model price estimates for /usage cost:
	// model-id substring → [input, output] USD per 1M tokens.
	Prices map[string][2]float64 `json:"prices,omitempty"`
	// Agents are named sub-agent profiles for the `agent` tool (delegate a task to
	// another model/provider): name → {provider, model, optional role prompt}.
	Agents     map[string]AgentProfile `json:"agents,omitempty"`
	SkillsPath string                  `json:"skills_path,omitempty"`
	Workspace  string                  `json:"-"` // resolved absolute workspace root
}

// ProviderTemplates are built-in OpenAI-compatible providers: base URL (and a
// sensible default model) are known, so the user only needs to add an API key.
var ProviderTemplates = map[string]LLM{
	"openai":     {BaseURL: "https://api.openai.com/v1", Model: "gpt-4o-mini"},
	"anthropic":  {BaseURL: "https://api.anthropic.com/v1", Model: "claude-3-5-sonnet-latest"}, // OpenAI-compat endpoint
	"grok":       {BaseURL: "https://api.x.ai/v1", Model: "grok-2-latest"},
	"groq":       {BaseURL: "https://api.groq.com/openai/v1", Model: "llama-3.3-70b-versatile"},
	"openrouter": {BaseURL: "https://openrouter.ai/api/v1", Model: "openai/gpt-4o-mini"},
}

// providerEnvKey maps a provider to the env var its API key falls back to.
var providerEnvKey = map[string]string{
	"openai":     "OPENAI_API_KEY",
	"anthropic":  "ANTHROPIC_API_KEY",
	"grok":       "XAI_API_KEY",
	"groq":       "GROQ_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
}

// KnownProviders lists the built-in template names (sorted), for help/listing.
func KnownProviders() []string {
	names := make([]string, 0, len(ProviderTemplates))
	for n := range ProviderTemplates {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ResolveProvider returns the connection for an external provider name, merging
// a user preset over the built-in template and falling back to the env var for
// the key. Not for "local" (that's cfg.LLM). false if the name is unknown.
func ResolveProvider(cfg Config, name string) (LLM, bool) {
	p, hasPreset := cfg.Providers[name]
	tmpl, hasTmpl := ProviderTemplates[name]
	if !hasPreset && !hasTmpl {
		return LLM{}, false
	}
	if hasTmpl {
		if p.BaseURL == "" {
			p.BaseURL = tmpl.BaseURL
		}
		if p.Model == "" {
			p.Model = tmpl.Model
		}
	}
	if p.APIKey == "" {
		if env := providerEnvKey[name]; env != "" {
			p.APIKey = os.Getenv(env)
		}
	}
	// Leave Temperature at 0 unless the user set one: the client omits a zero
	// temperature so hosted models that only accept their default (OpenAI gpt-5.x,
	// chat-latest) keep working. Only MaxSteps falls back to the baseline.
	if p.MaxSteps == 0 {
		p.MaxSteps = Default().LLM.MaxSteps
	}
	return p, true
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
		Name:    "ipsupport-code",
		Channel: "stable",
		LLM: LLM{
			BaseURL:       "http://localhost:1234/v1",
			Model:         "qwen2.5-7b-instruct",
			Temperature:   0.2,
			MaxSteps:      12,
			Type:          "lmstudio",
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

// DefaultUsagePath is the global token-usage ledger location.
func DefaultUsagePath() string { return filepath.Join(configHome(), "usage.json") }

// DefaultSkillsPath is the global skills directory.
func DefaultSkillsPath() string { return filepath.Join(configHome(), "skills") }

// SystemPromptPath is the optional global system-prompt override file; if it (or
// a workspace .agent/system.md) exists, it replaces the built-in base prompt.
func SystemPromptPath() string { return filepath.Join(configHome(), "system.md") }

// SaveGlobal writes the machine-level settings (display name + LLM connection)
// to the user config file, creating its directory.
func SaveGlobal(name string, l LLM) error {
	if name == "" {
		name = "ipsupport-code"
	}
	// Merge, don't overwrite: the global file also holds providers (with keys),
	// channel, prices, agents, retention — clobbering them on a /model or /rename
	// would silently lose the user's keys/config.
	return mergeGlobalKeys(map[string]any{"name": name, "llm": l})
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

// mergeGlobalKeys writes the given top-level keys into the user config,
// preserving everything else already there. The file is 0600 — it can hold API
// keys.
func mergeGlobalKeys(kv map[string]any) error {
	raw := map[string]json.RawMessage{}
	if data, err := os.ReadFile(GlobalPath()); err == nil {
		_ = json.Unmarshal(data, &raw)
	}
	for k, v := range kv {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		raw[k] = b
	}
	if err := os.MkdirAll(configHome(), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(GlobalPath(), data, 0o600)
}

// SaveChannel persists the update channel (stable|nightly).
func SaveChannel(channel string) error { return mergeGlobalKeys(map[string]any{"channel": channel}) }

// SaveUsageRetention persists the usage-ledger retention window (days; 0 = keep
// forever).
func SaveUsageRetention(days int) error {
	return mergeGlobalKeys(map[string]any{"usage_retention_days": days})
}

// SaveProviders persists the active provider name and the external provider
// presets (which may include API keys — the file is written 0600).
func SaveProviders(provider string, providers map[string]LLM) error {
	return mergeGlobalKeys(map[string]any{"provider": provider, "providers": providers})
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
	if cfg.UsagePath == "" {
		cfg.UsagePath = DefaultUsagePath()
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
