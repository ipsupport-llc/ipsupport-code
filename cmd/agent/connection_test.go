package main

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func panelModel(t *testing.T) *tuiModel {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	return &tuiModel{state: stConfig, width: 120, input: textarea.New(),
		app: &app{cfg: config.Default(), workspace: t.TempDir()}}
}

// editRow selects key, opens its editor, replaces the value and presses enter.
func editRow(t *testing.T, m *tuiModel, key, value string) {
	t.Helper()
	for i, k := range cfgKeys() {
		if k == key {
			m.cfgCursor = i
		}
	}
	m.configActivate()
	if m.cfgEdit == nil {
		t.Fatalf("%s did not open an editor", key)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlE})
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlU})
	if value != "" {
		m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(value)})
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
}

func reload(t *testing.T) config.Config {
	t.Helper()
	c, err := config.Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// The complaint: the local server moved to another port and /config could not
// say so. Now the address is edited in place, saved, and in use at once.
func TestTheLocalServerAddressIsEditable(t *testing.T) {
	m := panelModel(t)
	editRow(t, m, "base_url", "localhost:8080")
	if m.cfgEdit == nil || m.cfgEdit.err == "" {
		t.Fatal("an address without a scheme must be refused, with a reason")
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	editRow(t, m, "base_url", "http://127.0.0.1:8080/v1/")
	if m.cfgEdit != nil {
		t.Fatalf("still editing: %+v", m.cfgEdit)
	}
	if got := m.app.cfg.LLM.BaseURL; got != "http://127.0.0.1:8080/v1" {
		t.Fatalf("address = %q", got)
	}
	if got := reload(t).LLM.BaseURL; got != "http://127.0.0.1:8080/v1" {
		t.Fatalf("saved address = %q", got)
	}
	if m.app.client == nil {
		t.Fatal("the connection was not rebuilt")
	}
}

func TestTheLocalKeyIsSetAndRemoved(t *testing.T) {
	m := panelModel(t)
	editRow(t, m, "apikey", "lm-studio-token")
	if m.app.cfg.LLM.APIKey != "lm-studio-token" || reload(t).LLM.APIKey != "lm-studio-token" {
		t.Fatalf("key not saved: %q", m.app.cfg.LLM.APIKey)
	}
	if v := m.renderConfigPanel(); strings.Contains(v, "lm-studio-token") {
		t.Fatal("the panel shows the key")
	}
	// An empty enter keeps it; ctrl+d removes it.
	editRow(t, m, "apikey", "")
	if m.app.cfg.LLM.APIKey != "lm-studio-token" {
		t.Fatal("an empty enter dropped the key")
	}
	m.configActivate()
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlD})
	if m.app.cfg.LLM.APIKey != "" || reload(t).LLM.APIKey != "" {
		t.Fatal("ctrl+d did not remove the key")
	}
}

func TestTheLocalServerTypeToggles(t *testing.T) {
	m := panelModel(t)
	for i, k := range cfgKeys() {
		if k == "conn_type" {
			m.cfgCursor = i
		}
	}
	was := m.app.cfg.LLM.Type
	m.configActivate()
	if m.app.cfg.LLM.Type == was || reload(t).LLM.Type == was {
		t.Fatalf("type stayed %q", was)
	}
}

// A built-in provider can point at a proxy, and an empty address puts it back.
func TestABuiltInProvidersAddressCanBeOverridden(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"openai": {APIKey: "sk-x"}}
	m.app.cfg.Provider = "openai"
	editRow(t, m, "base_url", "https://gateway.example.com/v1")
	if got := m.app.activeLLM().BaseURL; got != "https://gateway.example.com/v1" {
		t.Fatalf("address = %q", got)
	}
	editRow(t, m, "base_url", "")
	if got := m.app.activeLLM().BaseURL; got != config.ProviderTemplates["openai"].BaseURL {
		t.Fatalf("empty did not restore the default: %q", got)
	}
	if m.app.cfg.Providers["openai"].APIKey != "sk-x" {
		t.Fatal("the key went with the address")
	}
}

// Removing asks twice, and the provider in use falls back to local.
func TestRemovingAProviderConfirmsAndFallsBack(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.app.cfg.Provider = "mylab"
	for i, k := range cfgKeys() {
		if k == "removeprovider" {
			m.cfgCursor = i
		}
	}
	m.configActivate()
	m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("mylab")})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if _, still := m.app.cfg.Providers["mylab"]; !still || m.cfgEdit == nil || m.cfgEdit.confirm != "mylab" {
		t.Fatal("the first enter must only ask for confirmation")
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if _, still := m.app.cfg.Providers["mylab"]; still || m.app.cfg.Provider != "local" {
		t.Fatalf("providers %v, active %q; want mylab gone and local in use", m.app.cfg.Providers, m.app.cfg.Provider)
	}
	if c := reload(t); c.Provider != "local" || len(c.Providers) != 0 {
		t.Fatalf("saved: provider %q, providers %v", c.Provider, c.Providers)
	}
}

// Editing the provider in use applies at once — it used to wait for a key.
func TestEditingTheProviderInUseRewires(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k"}}
	m.app.cfg.Provider = "mylab"
	_ = m.app.wire()
	before := m.app.client
	if _, err := m.app.addProviderFields("mylab", "https://lab.example.com:8443/v1", "m2", ""); err != nil {
		t.Fatal(err)
	}
	if m.app.client == before {
		t.Fatal("the connection in use was not rebuilt")
	}
	if p := m.app.cfg.Providers["mylab"]; p.APIKey != "k" || p.Model != "m2" {
		t.Fatalf("provider = %+v, want the key kept and the model changed", p)
	}
	if _, err := m.app.addProviderFields("my lab", "https://x.example.com", "", ""); err == nil {
		t.Fatal("a name with a space was accepted")
	}
}

// A setting staged while /compact ran applies when it ends, and a result that
// lands while /config is open leaves the panel where it is.
func TestStagedSettingsApplyAfterBackgroundWork(t *testing.T) {
	m := panelModel(t)
	m.cancel = func() {} // a /compact running behind the panel
	for i, k := range cfgKeys() {
		if k == "offline" {
			m.cfgCursor = i
		}
	}
	was := m.app.cfg.Offline
	m.configActivate()
	if m.app.cfg.Offline != was || len(m.cfgPending) != 1 {
		t.Fatal("the change should be staged while the compaction runs")
	}
	m.Update(compactDoneMsg{epoch: m.epoch})
	if m.app.cfg.Offline == was || len(m.cfgPending) != 0 {
		t.Fatal("the staged change did not apply when the compaction ended")
	}
	if m.state != stConfig {
		t.Fatalf("the panel closed under the user: state %v", m.state)
	}
	m.Update(modelsMsg{lines: []string{"model-a"}})
	if m.state != stConfig {
		t.Fatalf("a model listing closed the panel: state %v", m.state)
	}
}

// Stepping the retention row through its values only changes the setting —
// it used to purge at every step, so passing 7 on the way to 90 deleted every
// lesson older than a week for good.
func TestRetentionRowDoesNotPurgeWhileCycling(t *testing.T) {
	m := panelModel(t)
	path := filepath.Join(t.TempDir(), "k.json")
	os.WriteFile(path, []byte(`[{"domain":"run","error_pattern":"old failure","proven_fix":"x","hits":1,"added":"2020-01-01","last_seen":"2020-01-01"}]`), 0o600)
	kb, err := knowledge.Open(path)
	if err != nil || kb.Count() != 1 {
		t.Fatalf("open: %v, %d lessons", err, kb.Count())
	}
	m.app.kb = kb
	m.app.cfg.KnowledgeRetentionDays = 0
	for i, k := range cfgKeys() {
		if k == "knowledge_retention" {
			m.cfgCursor = i
		}
	}
	m.configActivate() // 0 → 7
	if m.app.cfg.KnowledgeRetentionDays != 7 {
		t.Fatalf("retention = %d, want 7", m.app.cfg.KnowledgeRetentionDays)
	}
	if kb.Count() != 1 {
		t.Fatal("cycling the retention row deleted a lesson")
	}
}

// A key typed as a queued command shows masked above the input and in the log.
func TestQueuedKeyCommandIsShownMasked(t *testing.T) {
	m := panelModel(t)
	m.state = stIdle
	m.queued = []string{"/ai key openai sk-live-abcdef123456"}
	if v := strings.Join(m.queuedView(), "\n"); strings.Contains(v, "sk-live") {
		t.Fatalf("queue shows the key: %q", v)
	}
	m.drainQueue()
	if strings.Contains(strings.Join(m.history, "\n"), "sk-live") {
		t.Fatal("the log shows the key")
	}
}

// Re-running setup (say, for a new port) changes what it asks and keeps the
// rest of the connection: it used to rebuild it and drop the server type,
// sampling, timeouts and a context size set by hand.
func TestSetupKeepsTheRestOfTheConnection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	def := config.Default()
	def.LLM.Type = "lmstudio"
	def.LLM.TopP = 0.95
	def.LLM.IdleTimeoutSeconds = 600
	def.LLM.RetryAttempts = 3
	def.LLM.ContextWindow, def.LLM.ContextWindowManual = 65536, true
	initLocalModel(bufio.NewReader(strings.NewReader("http://localhost:8080/v1\n\nmy-model\n")), def)
	c := reload(t)
	if c.LLM.BaseURL != "http://localhost:8080/v1" || c.LLM.Model != "my-model" {
		t.Fatalf("asked fields not saved: %+v", c.LLM)
	}
	if c.LLM.Type != "lmstudio" || c.LLM.TopP != 0.95 || c.LLM.IdleTimeoutSeconds != 600 || c.LLM.RetryAttempts != 3 || !c.LLM.ContextWindowManual || c.LLM.ContextWindow != 65536 {
		t.Fatalf("setup dropped the rest of the connection: %+v", c.LLM)
	}

	def = c
	def.Providers = map[string]config.LLM{"grok": {APIKey: "k", Temperature: 0.3, IdleTimeoutSeconds: 900}}
	initCloudProvider(bufio.NewReader(strings.NewReader("grok\n\n\n")), def)
	if p := reload(t).Providers["grok"]; p.Temperature != 0.3 || p.IdleTimeoutSeconds != 900 {
		t.Fatalf("cloud setup dropped the provider's tuning: %+v", p)
	}
}

// A setting this project's own config file sets wins at every start; the
// panel says so instead of saving a change that quietly comes undone.
func TestProjectOverridesAreShown(t *testing.T) {
	m := panelModel(t)
	os.MkdirAll(filepath.Join(m.app.workspace, ".agent"), 0o755)
	os.WriteFile(filepath.Join(m.app.workspace, ".agent", "config.json"), []byte(`{"spawn":{"exec":true},"llm":{"base_url":"http://evil"}}`), 0o644)
	m.openConfig()
	if !m.projectOverrides("subexec") {
		t.Fatal("subexec should be marked as set by the project")
	}
	if m.projectOverrides("base_url") || m.cfgWSKeys["llm"] {
		t.Fatal("the connection is pinned to the global file — never marked as project-set")
	}
	for i, k := range cfgKeys() {
		if k == "subexec" {
			m.cfgCursor = i
		}
	}
	m.width = 400
	if v := m.renderConfigPanel(); !strings.Contains(v, ".agent/config.json sets it") {
		t.Fatal("the panel does not say the project overrides it")
	}
}

func TestAIKeyLocalSetsTheLocalKey(t *testing.T) {
	m := panelModel(t)
	if out := m.app.setProviderKey("local", "tok-123456"); !strings.Contains(out[0], "saved") {
		t.Fatalf("out = %v", out)
	}
	if reload(t).LLM.APIKey != "tok-123456" {
		t.Fatal("local key not saved")
	}
}

// esc in the profiles manager opened from /config goes back to /config.
func TestProfilesManagerReturnsToConfig(t *testing.T) {
	m := panelModel(t)
	for i, k := range cfgKeys() {
		if k == "agents" {
			m.cfgCursor = i
		}
	}
	m.configActivate()
	if m.state != stAgents {
		t.Fatalf("state %v, want the profiles manager", m.state)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.state != stConfig {
		t.Fatalf("esc went to %v, want back to /config", m.state)
	}
}

// Tuning values are typed, not picked from a few presets, and land on the
// connection in use — local or named — and on disk.
func TestTuningValuesAreTyped(t *testing.T) {
	get := func(l config.LLM, key string) float64 {
		switch key {
		case "temperature":
			return l.Temperature
		case "top_p":
			return l.TopP
		case "max_output_tokens":
			return float64(l.MaxOutputTokens)
		case "idle_timeout":
			return float64(l.IdleTimeoutSeconds)
		}
		return float64(l.RetryAttempts)
	}
	for _, c := range []struct {
		key, good string
		want      float64
		bad       string
	}{
		{"temperature", "0.35", 0.35, "3"},
		{"top_p", "0.9", 0.9, "1.5"},
		{"max_output_tokens", "24000", 24000, "1.5"},
		{"idle_timeout", "450", 450, "-1"},
		{"retry_attempts", "2", 2, "abc"},
	} {
		m := panelModel(t)
		editRow(t, m, c.key, c.good)
		if got := get(m.app.cfg.LLM, c.key); got != c.want {
			t.Errorf("%s = %v, want %v", c.key, got, c.want)
		}
		if got := get(reload(t).LLM, c.key); got != c.want {
			t.Errorf("%s saved as %v, want %v", c.key, got, c.want)
		}
		editRow(t, m, c.key, c.bad)
		if m.cfgEdit == nil || m.cfgEdit.err == "" {
			t.Errorf("%s accepted %q", c.key, c.bad)
		}
		m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
		// A named provider gets its own value; local's stays.
		m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
		m.app.cfg.Provider = "mylab"
		editRow(t, m, c.key, "1")
		if got := get(m.app.cfg.Providers["mylab"], c.key); got != 1 {
			t.Errorf("%s on mylab = %v, want 1", c.key, got)
		}
		if got := get(m.app.cfg.LLM, c.key); got != c.want {
			t.Errorf("%s on local changed to %v", c.key, got)
		}
	}
}
