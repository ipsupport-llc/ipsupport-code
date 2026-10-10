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

// Removing is picked from the saved providers, asks twice, and the provider in
// use falls back to local.
func TestRemovingAProviderConfirmsAndFallsBack(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"airllm": {BaseURL: "https://a.example.com/v1"}, "mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.app.cfg.Provider = "mylab"
	cursorOn(m, "removeprovider")
	m.configActivate()
	if m.cfgPick == nil || len(m.cfgPick.p.visible()) != 2 {
		t.Fatalf("no list of the saved providers: %+v", m.cfgPick)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown}) // airllm → mylab
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if _, still := m.app.cfg.Providers["mylab"]; !still || m.cfgPick == nil || m.cfgPick.confirm != "mylab" {
		t.Fatal("the first enter must only ask for confirmation")
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if _, still := m.app.cfg.Providers["mylab"]; still || m.app.cfg.Provider != "local" {
		t.Fatalf("providers %v, active %q; want mylab gone and local in use", m.app.cfg.Providers, m.app.cfg.Provider)
	}
	if c := reload(t); c.Provider != "local" || len(c.Providers) != 1 {
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
	def.LLM.Model = "my-model" // the same model: its hand-set size stays
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

// blockSaves makes every global config write fail: the config directory's
// parent is a file.
func blockSaves(t *testing.T) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(os.Getenv("HOME"), ".config"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
}

func cursorOn(m *tuiModel, key string) {
	for i, k := range cfgKeys() {
		if k == key {
			m.cfgCursor = i
		}
	}
}

func typeKeys(m *tuiModel, s string) {
	m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)})
}

// A key typed for one name, then the name changed: the key belonged to the
// first provider and must not be saved under (and sent to) the second.
func TestAddFormDropsTheKeyWhenTheNameChanges(t *testing.T) {
	m := panelModel(t)
	cursorOn(m, "addprovider")
	m.configActivate()
	enter := tea.KeyMsg{Type: tea.KeyEnter}
	esc := tea.KeyMsg{Type: tea.KeyEsc}
	typeKeys(m, "lab-a")
	m.handleKey(enter)
	typeKeys(m, "https://a.example.com/v1")
	m.handleKey(enter)
	m.handleKey(enter) // no model
	typeKeys(m, "sk-for-lab-a")
	m.handleKey(esc)
	m.handleKey(esc)
	m.handleKey(esc) // back at the name
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlE})
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlU})
	typeKeys(m, "lab-b")
	m.handleKey(enter)
	if m.cfgDraft.url.value() != "" {
		t.Fatalf("lab-a's address carried over: %q", m.cfgDraft.url.value())
	}
	typeKeys(m, "https://b.example.com/v1")
	m.handleKey(enter)
	m.handleKey(enter)
	m.handleKey(enter) // save, key field as left
	if p, ok := m.app.cfg.Providers["lab-b"]; !ok || p.APIKey != "" {
		t.Fatalf("lab-b = %+v (saved %v), want it saved without lab-a's key", p, ok)
	}
}

// Setup re-run on a configured machine: the saved key is neither shown nor lost.
func TestSetupNeverEchoesASavedKey(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("OPENAI_API_KEY", "")
	out := captureStdout(t, func() {
		def := config.Default()
		def.LLM.BaseURL, def.LLM.APIKey = "http://127.0.0.1:1/v1", "sk-secret-local"
		initLocalModel(bufio.NewReader(strings.NewReader("\n\n\n")), def)
		def.Providers = map[string]config.LLM{"openai": {APIKey: "sk-secret-cloud", Model: "gpt-x"}}
		initCloudProvider(bufio.NewReader(strings.NewReader("openai\n\n\n")), def)
	})
	if strings.Contains(out, "sk-secret") {
		t.Fatalf("setup printed a saved key:\n%s", out)
	}
	c := reload(t)
	if c.LLM.APIKey != "sk-secret-local" || c.Providers["openai"].APIKey != "sk-secret-cloud" {
		t.Fatalf("Enter did not keep the keys: local %q, openai %q", c.LLM.APIKey, c.Providers["openai"].APIKey)
	}
}

func captureStdout(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	was := os.Stdout
	os.Stdout = w
	done := make(chan string)
	go func() {
		var b strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := r.Read(buf)
			b.Write(buf[:n])
			if err != nil {
				break
			}
		}
		done <- b.String()
	}()
	f()
	os.Stdout = was
	w.Close()
	return <-done
}

// A context size set by hand belongs to its model: setup answering another
// model detects again; the same model keeps it.
func TestSetupForgetsAHandSetWindowForANewModel(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	def := config.Default()
	def.LLM.BaseURL, def.LLM.Model = "http://127.0.0.1:1/v1", "old-model"
	def.LLM.ContextWindow, def.LLM.ContextWindowManual = 65536, true
	initLocalModel(bufio.NewReader(strings.NewReader("\n\nold-model\n")), def)
	if !reload(t).LLM.ContextWindowManual {
		t.Fatal("the same model lost its hand-set size")
	}
	initLocalModel(bufio.NewReader(strings.NewReader("\n\nnew-model\n")), def)
	if reload(t).LLM.ContextWindowManual {
		t.Fatal("a new model kept the old model's hand-set size")
	}
}

// A provider switch staged during a task probes the new window when it applies.
func TestStagedProviderSwitchProbesTheWindow(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k"}}
	m.cancel = func() {}
	cursorOn(m, "provider")
	m.configActivate() // the list opens even while the task runs
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if len(m.cfgPending) != 1 || m.cfgPending[0] != "provider=mylab" {
		t.Fatalf("pending = %v, want the switch staged", m.cfgPending)
	}
	m.cancel = nil
	if cmd := m.applyPendingConfig(); cmd == nil {
		t.Fatal("the staged switch did not ask for the window to be probed")
	}
	if m.app.cfg.Provider != "mylab" {
		t.Fatal("the staged switch did not apply")
	}
}

// Changing how the window is found, or the provider in use, re-probes it.
func TestConnectionChangesProbeTheWindow(t *testing.T) {
	m := panelModel(t)
	cursorOn(m, "conn_type")
	if _, cmd := m.configActivate(); cmd == nil {
		t.Fatal("toggling the server type did not re-probe the window")
	}
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k"}}
	m.app.cfg.Provider = "mylab"
	_ = m.app.wire()
	cursorOn(m, "addprovider")
	m.configActivate()
	enter := tea.KeyMsg{Type: tea.KeyEnter}
	typeKeys(m, "mylab")
	m.handleKey(enter)
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlE})
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlU})
	typeKeys(m, "https://lab.example.com:8443/v1")
	m.handleKey(enter)
	m.handleKey(enter)
	if _, cmd := m.handleKey(enter); cmd == nil {
		t.Fatal("moving the provider in use did not re-probe the window")
	}
}

func TestAFailedProfileSaveLeavesTheRosterAlone(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Agents = map[string]config.AgentProfile{"codex": {Provider: "local", Model: "a"}}
	blockSaves(t)
	m.agDraft = agentDraft{orig: "codex", provider: "local", model: "b", name: "renamed"}
	m.saveDraft("renamed")
	if p, ok := m.app.cfg.Agents["codex"]; !ok || p.Model != "a" || len(m.app.cfg.Agents) != 1 {
		t.Fatalf("roster = %v, want it unchanged after a failed save", m.app.cfg.Agents)
	}
}

// A refused save leaves every setting as it was.
func TestSettingsRollBackOnAFailedSave(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.LLM.Temperature = 0.2
	blockSaves(t)
	for _, c := range []struct{ key, value string }{
		{"temperature", "0.9"}, {"context_window", "65536"}, {"judge_max_output_tokens", "9000"},
	} {
		editRow(t, m, c.key, c.value)
		if m.cfgEdit == nil || m.cfgEdit.err == "" {
			t.Errorf("%s: a failed save was reported as done", c.key)
		}
		m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	}
	if l := m.app.cfg.LLM; l.Temperature != 0.2 || l.ContextWindowManual {
		t.Fatalf("connection changed in memory: %+v", l)
	}
	if m.app.cfg.JudgeMaxOutputTokens != 0 {
		t.Fatalf("judge cap = %d after a failed save", m.app.cfg.JudgeMaxOutputTokens)
	}
}

func TestNotANumberIsRefused(t *testing.T) {
	m := panelModel(t)
	for _, v := range []string{"NaN", "Inf"} {
		editRow(t, m, "temperature", v)
		if m.cfgEdit == nil || m.cfgEdit.err == "" {
			t.Errorf("temperature accepted %q", v)
		}
		m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	}
}

// Re-saving what is already there changes nothing: the same model keeps a
// hand-set size, the detected size re-entered stays detected.
func TestResavingTheSameValueKeepsHowTheWindowIsFound(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.LLM.Model = "m"
	m.app.cfg.LLM.ContextWindow, m.app.cfg.LLM.ContextWindowManual = 65536, true
	m.app.setModel("m")
	if !m.app.cfg.LLM.ContextWindowManual {
		t.Fatal("re-picking the same model forgot the hand-set size")
	}
	m.app.cfg.LLM.ContextWindow, m.app.cfg.LLM.ContextWindowManual = 32768, false
	editRow(t, m, "context_window", "32768")
	if m.app.cfg.LLM.ContextWindowManual {
		t.Fatal("re-entering the detected size pinned it")
	}
}

// The project's .agent/config.json names this provider: removing it would
// leave the next start pointing at nothing.
func TestAProjectPinnedProviderIsNotRemoved(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.app.cfg.Provider = "mylab"
	dir := filepath.Join(m.app.workspace, ".agent")
	os.MkdirAll(dir, 0o700)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"provider":"mylab"}`), 0o600)
	if _, err := m.app.removeProvider("mylab"); err == nil {
		t.Fatal("removed a provider the project pins")
	}
	if _, ok := m.app.cfg.Providers["mylab"]; !ok {
		t.Fatal("the provider is gone")
	}
}

// Moving off the name awaiting its second enter cancels the removal.
func TestRemoveProviderConfirmIsForTheHighlightedName(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"airllm": {BaseURL: "https://a.example.com/v1"}, "mylab": {BaseURL: "https://lab.example.com/v1"}}
	cursorOn(m, "removeprovider")
	m.configActivate()
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // airllm: asks
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})  // now on mylab
	if m.cfgPick.confirm != "" {
		t.Fatalf("still asking about %q after moving off it", m.cfgPick.confirm)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if len(m.app.cfg.Providers) != 2 || m.cfgPick.confirm != "mylab" {
		t.Fatalf("providers %v, confirm %q; want nothing removed and mylab asked about", m.app.cfg.Providers, m.cfgPick.confirm)
	}
}

// Clearing the saved key while the environment still has one says so.
func TestClearingAKeyNamesTheEnvironment(t *testing.T) {
	m := panelModel(t)
	t.Setenv("OPENAI_API_KEY", "sk-env")
	m.app.cfg.Providers = map[string]config.LLM{"openai": {APIKey: "sk-saved"}}
	m.app.cfg.Provider = "openai"
	msg, err := m.app.setActiveKey("", true)
	if err != nil || !strings.Contains(msg, "environment") {
		t.Fatalf("msg %q, err %v; want the environment named", msg, err)
	}
}

// /ai add takes the form's checks and reconnects the provider in use.
func TestAIAddChecksAndReconnects(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k", Model: "m1"}}
	m.app.cfg.Provider = "mylab"
	_ = m.app.wire()
	before := m.app.client
	m.app.addProvider("mylab https://lab.example.com:9443/v1")
	if m.app.client == before {
		t.Fatal("the provider in use was not reconnected")
	}
	if p := m.app.cfg.Providers["mylab"]; p.Model != "m1" || p.APIKey != "k" {
		t.Fatalf("mylab = %+v, want model and key kept", p)
	}
	for _, bad := range []string{"my/lab https://x.example.com", "other localhost:8080"} {
		m.app.addProvider(bad)
	}
	if len(m.app.cfg.Providers) != 1 {
		t.Fatalf("a bad /ai add was saved: %v", m.app.cfg.Providers)
	}
}

// ↑ on a queued /ai key does not put the token back on screen.
func TestRecallingAQueuedKeyDropsIt(t *testing.T) {
	m := panelModel(t)
	m.state = stRunning
	m.queued = []string{"/ai key openai sk-live-abcdef123456"}
	m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if strings.Contains(m.input.Value(), "sk-live") || strings.Contains(strings.Join(m.history, "\n"), "sk-live") {
		t.Fatalf("recall showed the key: input %q", m.input.Value())
	}
	if len(m.queued) != 0 {
		t.Fatal("the line stayed queued")
	}
}

func TestJudgeOutputCapIsTyped(t *testing.T) {
	m := panelModel(t)
	editRow(t, m, "judge_max_output_tokens", "12000")
	if m.app.cfg.JudgeMaxOutputTokens != 12000 || reload(t).JudgeMaxOutputTokens != 12000 {
		t.Fatalf("judge cap = %d", m.app.cfg.JudgeMaxOutputTokens)
	}
}
