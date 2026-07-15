// Package config loads the agent's runtime configuration. Settings come from
// two JSON files merged over safe defaults: a machine-level user config
// (~/.config/ipsupport-code/config.json — the LM Studio endpoint, written by
// first-run setup) and a per-workspace config (<workspace>/.agent/config.json —
// the permission policy). A protective deny floor is unioned in afterwards so a
// workspace config can never drop the most dangerous-command guards.
package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/mcp"
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
	// IdleTimeoutSeconds bounds how long a request waits with NO response/stream
	// data before it's aborted and retried. 0 uses the default (90s). Raise it for a
	// hosted reasoning model that can think silently (no streamed deltas) for longer.
	IdleTimeoutSeconds int `json:"idle_timeout_seconds,omitempty"`
	// Extra is resolved per request (NOT persisted here): extra top-level body
	// params merged into the chat request — used for per-model reasoning controls
	// (see Config.Reasoning), whose shape varies by provider.
	Extra map[string]any `json:"-"`
}

// LMStudio reports whether this connection speaks LM Studio's native API.
func (l LLM) LMStudio() bool { return l.Type == "lmstudio" }

// AgentProfile is a named sub-agent target for the `agent` tool: which provider
// and model to run the sub-agent on. The task always comes from the delegating
// assistant; Prompt is an optional, persistent role note for that model.
type AgentProfile struct {
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
	Prompt   string `json:"prompt,omitempty"`

	// External CLI agent fields (Kind "external"): run Command with Args in the
	// target dir as an autonomous local coding agent (codex, claude, aider…).
	// {task} in an arg is replaced by the task text (appended if no placeholder).
	// An empty Kind is a normal LLM profile, so old configs keep working.
	Kind    string   `json:"kind,omitempty"`
	Command string   `json:"command,omitempty"`
	Args    []string `json:"args,omitempty"`
	Timeout int      `json:"timeout,omitempty"` // seconds per launch; 0 = default (15 min)
}

// SpawnPolicy gates the `agent` tool. Default is the approval mode before a
// sub-agent runs (ask | allow — ask guards against runaway spawns, even local
// ones that still cost compute); Exec controls whether sub-agents get the run
// (shell) tool — off by default, the sharpest capability to hand an autonomous
// sub-agent.
type SpawnPolicy struct {
	Default string `json:"default"` // ask (default) | allow
	Exec    bool   `json:"exec"`    // give sub-agents the run (shell) tool
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
	// KnowledgeRetentionDays auto-drops learned lessons last seen over N days ago
	// on startup. 0 keeps them forever.
	KnowledgeRetentionDays int `json:"knowledge_retention_days,omitempty"`
	// GoalMaxReturns is the TTL for goal pursuit: if the model finalizes but the
	// goal isn't done (a plan still has open items), it's re-fed the goal and tries
	// again, up to this many returns. 0 disables (one run). Default 6.
	GoalMaxReturns int `json:"goal_max_returns,omitempty"`
	// GoalMaxSteps is the hard step backstop for one goal pursuit. Default 80.
	GoalMaxSteps int `json:"goal_max_steps,omitempty"`
	// GoalNudge, when true (default), gives the model ONE push if it re-reads the
	// goal after a re-feed but then finishes without doing any work — instead of
	// silently giving up. Set false to accept a no-progress finish immediately.
	GoalNudge bool `json:"goal_nudge"`
	// SessionBudgetUSD caps estimated spend for one run of the process: once this
	// session's estimated cost reaches it, new tasks are refused until it's raised.
	// 0 disables the guard.
	SessionBudgetUSD float64 `json:"session_budget_usd,omitempty"`
	// Prices overrides the built-in per-model price estimates for /usage cost:
	// model-id substring → [input, output] USD per 1M tokens.
	Prices map[string][2]float64 `json:"prices,omitempty"`
	// Reasoning holds per-model reasoning controls, merged as raw params into the
	// chat request body. Keyed by "<provider>" (default for all its models) or
	// "<provider>/<model>" (overrides the provider default). The value is the
	// provider's own shape — e.g. {"reasoning_effort":"low"} (openai),
	// {"reasoning":{"effort":"low"}} (openrouter),
	// {"chat_template_kwargs":{"enable_thinking":false}} (qwen/lmstudio).
	Reasoning map[string]json.RawMessage `json:"reasoning,omitempty"`
	// Agents are named sub-agent profiles for the `agent` tool (delegate a task to
	// another model/provider): name → {provider, model, optional role prompt}.
	Agents map[string]AgentProfile `json:"agents,omitempty"`
	// Spawn gates the `agent` tool: approval mode before a sub-agent runs, and
	// whether sub-agents may run shell commands.
	Spawn SpawnPolicy `json:"spawn"`
	// Offline disables all internet egress (web tool, startup update check) and
	// makes anything that would reach the net fail fast with a clear message.
	// Local model calls (e.g. LM Studio on localhost) are unaffected.
	Offline bool `json:"offline,omitempty"`
	// Sandbox selects an OS-level containment layer for shell (`run`) commands,
	// UNDER the policy engine: off (default) | seatbelt (macOS) | landlock (Linux)
	// | auto (the platform's mechanism). It confines writes to the workspace even
	// for an allowed command; unsupported platforms run unconfined.
	Sandbox string `json:"sandbox,omitempty"`
	// UpdateCheck controls the best-effort startup "newer build available" notice.
	// On by default; set it false to silence the check WITHOUT going fully Offline
	// (which also disables the web tool) — for pinned or package-managed installs
	// that own their own upgrades. No omitempty: false must round-trip and show in
	// `config list`.
	UpdateCheck bool `json:"update_check"`
	// ReflectDisabled skips the post-task reflection pass (lesson distillation).
	// Reflection is on by default; turn it off when a weak model loops there.
	ReflectDisabled bool `json:"reflect_disabled,omitempty"`
	// ReflectProfile, when set, runs the reflection pass on this agent profile
	// (a capable model) instead of the current one. Empty = the current model.
	ReflectProfile string `json:"reflect_profile,omitempty"`
	// McpServers are MCP servers the `mcp` proxy tool can use: name → stdio launch
	// spec. Empty = the mcp tool isn't offered.
	McpServers map[string]mcp.Server `json:"mcp_servers,omitempty"`
	SkillsPath string                `json:"skills_path,omitempty"`
	Workspace  string                `json:"-"` // resolved absolute workspace root
}

// ProviderTemplates are built-in OpenAI-compatible providers: base URL (and a
// sensible default model) are known, so the user only needs to add an API key.
var ProviderTemplates = map[string]LLM{
	"openai":     {BaseURL: "https://api.openai.com/v1", Model: "gpt-4o-mini"},
	"anthropic":  {BaseURL: "https://api.anthropic.com/v1", Model: "claude-3-5-sonnet-latest"}, // OpenAI-compat endpoint
	"grok":       {BaseURL: "https://api.x.ai/v1", Model: "grok-2-latest"},
	"groq":       {BaseURL: "https://api.groq.com/openai/v1", Model: "llama-3.3-70b-versatile"},
	"openrouter": {BaseURL: "https://openrouter.ai/api/v1", Model: "openai/gpt-4o-mini"},
	"zai":        {BaseURL: "https://api.z.ai/api/paas/v4", Model: "glm-5.2"}, // Z.ai (Zhipu GLM), OpenAI-compat endpoint
}

// providerEnvKey maps a provider to the env var its API key falls back to.
var providerEnvKey = map[string]string{
	"openai":     "OPENAI_API_KEY",
	"anthropic":  "ANTHROPIC_API_KEY",
	"grok":       "XAI_API_KEY",
	"groq":       "GROQ_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"zai":        "ZAI_API_KEY",
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

// IsCustomProvider reports whether name is a user-defined provider (present in the
// config's providers map) rather than a built-in template. A custom provider is
// self-describing (the user supplied base_url) and may be keyless — e.g. a local
// Ollama/vLLM/second-LM-Studio endpoint — so the UI shouldn't demand an API key.
func IsCustomProvider(cfg Config, name string) bool {
	_, hasPreset := cfg.Providers[name]
	_, hasTmpl := ProviderTemplates[name]
	return hasPreset && !hasTmpl
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
		Spawn:          SpawnPolicy{Default: "ask", Exec: false},
		UpdateCheck:    true,
		GoalMaxReturns: 6,
		GoalMaxSteps:   80,
		GoalNudge:      true,
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

// SnippetsPath is the global prompt-snippets store (/snip save+recall).
func SnippetsPath() string { return filepath.Join(configHome(), "snippets.json") }

// LogPath is where logs go while the TUI owns the screen (writing them to stderr
// would corrupt the alt-screen). Tail it to watch retries/warnings live.
func LogPath() string { return filepath.Join(configHome(), "agent.log") }

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
	return mergeJSONFile(filepath.Join(workspace, ".agent", "config.json"), 0o644,
		map[string]any{"run": run, "file": file})
}

// mergeGlobalKeys writes the given top-level keys into the user config,
// preserving everything else already there. The file is 0600 — it can hold API
// keys.
func mergeGlobalKeys(kv map[string]any) error {
	return mergeJSONFile(GlobalPath(), 0o600, kv)
}

// mergeJSONFile sets the given top-level keys in a JSON object file, preserving
// every other key. Two safety properties matter because this file holds API keys
// and provider presets:
//   - If the existing file can't be parsed, ABORT (don't write). The old code
//     ignored the parse error and wrote back only the new key, silently wiping
//     providers/keys — and one truncated write then cascaded into total loss.
//   - Write atomically (temp + rename) so an interrupted save can never leave a
//     half-written, unparseable file behind.
func mergeJSONFile(path string, perm os.FileMode, kv map[string]any) error {
	raw := map[string]json.RawMessage{}
	if data, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(data, &raw); err != nil {
			return fmt.Errorf("refusing to save: %s is not valid JSON (%v) — fix or remove it so your saved settings aren't lost", path, err)
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	for k, v := range kv {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		raw[k] = b
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, data, perm)
}

// SaveChannel persists the update channel (stable|nightly).
func SaveChannel(channel string) error { return mergeGlobalKeys(map[string]any{"channel": channel}) }

// SaveUsageRetention persists the usage-ledger retention window (days; 0 = keep
// forever).
func SaveUsageRetention(days int) error {
	return mergeGlobalKeys(map[string]any{"usage_retention_days": days})
}

// SaveSpawn persists the sub-agent spawn policy (approval mode + exec) globally.
func SaveSpawn(s SpawnPolicy) error { return mergeGlobalKeys(map[string]any{"spawn": s}) }

// SaveOffline persists the offline-mode flag globally.
func SaveOffline(off bool) error { return mergeGlobalKeys(map[string]any{"offline": off}) }

// SaveReflectCfg persists the reflection flag + profile globally.
func SaveReflectCfg(disabled bool, profile string) error {
	return mergeGlobalKeys(map[string]any{"reflect_disabled": disabled, "reflect_profile": profile})
}

// SaveReasoning persists the per-model reasoning params map globally.
func SaveReasoning(m map[string]json.RawMessage) error {
	return mergeGlobalKeys(map[string]any{"reasoning": m})
}

// SaveKnowledgeRetention persists the lesson-retention window (days) globally.
func SaveKnowledgeRetention(days int) error {
	return mergeGlobalKeys(map[string]any{"knowledge_retention_days": days})
}

// SaveGoalMaxReturns persists the goal-pursuit return TTL globally.
func SaveGoalMaxReturns(n int) error {
	return mergeGlobalKeys(map[string]any{"goal_max_returns": n})
}

// SaveSessionBudget persists the per-session spend cap (USD) globally.
func SaveSessionBudget(usd float64) error {
	return mergeGlobalKeys(map[string]any{"session_budget_usd": usd})
}

// SaveAgents persists the sub-agent profile map globally.
func SaveAgents(agents map[string]AgentProfile) error {
	return mergeGlobalKeys(map[string]any{"agents": agents})
}

// SaveProviders persists the active provider name and the external provider
// presets (which may include API keys — the file is written 0600).
func SaveProviders(provider string, providers map[string]LLM) error {
	return mergeGlobalKeys(map[string]any{"provider": provider, "providers": providers})
}

// ---- dotted-path get/set/unset for the `config` subcommand --------------
//
// These operate on a config file as a generic JSON object so any key (including
// nested policy like run.default or llm.model) is reachable without a
// field-by-field flag. Writes reuse the same safety properties as the typed
// savers: refuse to touch a corrupt file, and write atomically.

// SetFileValue sets the dotted-path key in the JSON config file at path to the
// value parsed from raw (valid JSON becomes its typed form — false→bool, 8→int,
// [..]/{..}→array/object; anything else is the literal string), preserving every
// other key. It refuses to write if the existing file is unparseable, and
// validates that the result is a well-formed Config with no unknown keys (so a
// typo like `goal_max_retunrs` is rejected rather than silently persisted).
func SetFileValue(path string, perm os.FileMode, dotted, raw string) error {
	segs, err := splitPath(dotted)
	if err != nil {
		return err
	}
	root, err := readObject(path)
	if err != nil {
		return err
	}
	if err := setPath(root, segs, parseValue(raw)); err != nil {
		return err
	}
	if err := validateConfigObject(root); err != nil {
		return err
	}
	return writeObject(path, perm, root)
}

// UnsetFileValue removes the dotted-path key from the config file at path. A key
// that isn't present is a no-op (not an error).
func UnsetFileValue(path string, perm os.FileMode, dotted string) error {
	segs, err := splitPath(dotted)
	if err != nil {
		return err
	}
	root, err := readObject(path)
	if err != nil {
		return err
	}
	m := root
	for _, s := range segs[:len(segs)-1] {
		next, ok := m[s].(map[string]any)
		if !ok {
			return nil // parent path absent → nothing to unset
		}
		m = next
	}
	delete(m, segs[len(segs)-1])
	return writeObject(path, perm, root)
}

// LookupPath returns the effective value at a dotted path in cfg (its marshaled
// view), and whether that path exists.
func LookupPath(cfg Config, dotted string) (any, bool) {
	segs, err := splitPath(dotted)
	if err != nil {
		return nil, false
	}
	var cur any = toObject(cfg)
	for _, s := range segs {
		obj, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		if cur, ok = obj[s]; !ok {
			return nil, false
		}
	}
	return cur, true
}

// Flatten returns the effective config as sorted "dotted.path\tvalue" lines
// (value as compact JSON), for `config list`.
func Flatten(cfg Config) []string {
	var out []string
	var walk func(prefix string, v any)
	walk = func(prefix string, v any) {
		if obj, ok := v.(map[string]any); ok && len(obj) > 0 {
			for k, vv := range obj {
				child := k
				if prefix != "" {
					child = prefix + "." + k
				}
				walk(child, vv)
			}
			return
		}
		b, _ := json.Marshal(v)
		out = append(out, prefix+"\t"+string(b))
	}
	walk("", toObject(cfg))
	sort.Strings(out)
	return out
}

func splitPath(dotted string) ([]string, error) {
	dotted = strings.TrimSpace(dotted)
	if dotted == "" {
		return nil, errors.New("empty key")
	}
	segs := strings.Split(dotted, ".")
	for _, s := range segs {
		if s == "" {
			return nil, fmt.Errorf("invalid key %q: empty path segment", dotted)
		}
	}
	return segs, nil
}

// parseValue interprets a CLI value: valid JSON keeps its type, otherwise it's
// the literal string (so `nightly` or a path needn't be quoted).
func parseValue(raw string) any {
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err == nil {
		return v
	}
	return raw
}

func setPath(root map[string]any, segs []string, val any) error {
	m := root
	for i, s := range segs[:len(segs)-1] {
		next, ok := m[s]
		if !ok {
			child := map[string]any{}
			m[s] = child
			m = child
			continue
		}
		child, ok := next.(map[string]any)
		if !ok {
			return fmt.Errorf("cannot set %s: %s is not an object", strings.Join(segs, "."), strings.Join(segs[:i+1], "."))
		}
		m = child
	}
	m[segs[len(segs)-1]] = val
	return nil
}

// readObject loads the config file as a generic JSON object, aborting (rather
// than clobbering) if it exists but can't be parsed. A missing file yields an
// empty object.
func readObject(path string) (map[string]any, error) {
	root := map[string]any{}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, &root); err != nil {
			return nil, fmt.Errorf("refusing to edit: %s is not valid JSON (%v) — fix or remove it first so your settings aren't lost", path, err)
		}
	case errors.Is(err, fs.ErrNotExist):
		// new file
	default:
		return nil, err
	}
	return root, nil
}

func writeObject(path string, perm os.FileMode, root map[string]any) error {
	data, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, data, perm)
}

// validateConfigObject rejects a proposed config object that wouldn't decode
// into a Config — a wrong type (goal_max_returns "abc") or an unknown key (a
// typo). Map contents (provider names, agent names) stay free; only struct
// fields are checked.
func validateConfigObject(root map[string]any) error {
	data, err := json.Marshal(root)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var c Config
	if err := dec.Decode(&c); err != nil {
		return fmt.Errorf("invalid config after edit: %w", err)
	}
	return nil
}

// toObject renders cfg as a generic JSON object for path lookup/flattening.
func toObject(cfg Config) map[string]any {
	var m map[string]any
	data, _ := json.Marshal(cfg)
	_ = json.Unmarshal(data, &m)
	return m
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
