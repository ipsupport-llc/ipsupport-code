package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// The interactive /config panel: a sectioned settings screen navigated with
// ↑↓, where Enter cycles a value in place (mode, color, channel, provider,
// permissions, run timeout) or hands off to the matching flow (model list, key
// entry, rename). esc closes it. Cycling applies and persists immediately.

// cfgRow is one line: a section header (header != "") or a selectable setting
// (key != "").
type cfgRow struct {
	header string
	key    string
}

var configRows = []cfgRow{
	{header: "Model & provider"},
	{key: "provider"},
	{key: "addprovider"},
	{key: "model"},
	{key: "apikey"},
	{key: "context_window"},
	{key: "reasoning"},
	{key: "temperature"},
	{key: "top_p"},
	{key: "max_output_tokens"},
	{key: "loop_detection"},
	{key: "idle_timeout"},
	{header: "Behavior"},
	{key: "mode"},
	{key: "perm_files"},
	{key: "perm_run"},
	{key: "timeout"},
	{key: "budget"},
	{key: "offline"},
	{key: "memory"},
	{key: "compact_threshold"},
	{key: "max_steps"},
	{key: "max_history"},
	{key: "max_stuck_turns"},
	{header: "Sub-agents"},
	{key: "agents"},
	{key: "spawn"},
	{key: "subexec"},
	{header: "Appearance & updates"},
	{key: "color"},
	{key: "channel"},
	{key: "name"},
}

// cfgKeys is the selectable keys in order (cfgCursor indexes into this).
func cfgKeys() []string {
	var ks []string
	for _, r := range configRows {
		if r.key != "" {
			ks = append(ks, r.key)
		}
	}
	return ks
}

// openConfig enters the panel.
func (m *tuiModel) openConfig() {
	m.cfgCursor = 0
	m.cfgPhase = cfgPhaseList
	m.state = stConfig
}

// The add-provider form lives INSIDE the panel (no hand-off that dumps the user
// back at the prompt): name → base URL → model → key, esc steps back.
const (
	cfgPhaseList = iota
	cfgPhaseName
	cfgPhaseURL
	cfgPhaseModel
	cfgPhaseKey
)

// providerDraft is the provider being added via the panel form.
type providerDraft struct {
	name, url, model, key string
}

// cfgAddField returns the form field being edited for the current phase.
func (m *tuiModel) cfgAddField() *string {
	switch m.cfgPhase {
	case cfgPhaseName:
		return &m.cfgDraft.name
	case cfgPhaseURL:
		return &m.cfgDraft.url
	case cfgPhaseModel:
		return &m.cfgDraft.model
	default:
		return &m.cfgDraft.key
	}
}

// configAddKey handles typing in the add-provider form: enter advances (saving on
// the last field), esc steps back (to the list from the first).
func (m *tuiModel) configAddKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	field := m.cfgAddField()
	switch k.String() {
	case "enter":
		switch m.cfgPhase {
		case cfgPhaseName:
			name := strings.TrimSpace(m.cfgDraft.name)
			if name == "" {
				return m, nil // a name is required
			}
			// Editing an already-added provider: prefill its current URL/model so
			// the form shows what's on file instead of forcing a blank retype of
			// the exact original values (the key is deliberately left blank —
			// configAddSave only touches it when something new is typed).
			if config.IsCustomProvider(m.app.cfg, name) && m.cfgDraft.url == "" {
				p := m.app.cfg.Providers[name]
				m.cfgDraft.url = p.BaseURL
				m.cfgDraft.model = p.Model
			}
			m.cfgPhase = cfgPhaseURL
		case cfgPhaseURL:
			if u := strings.TrimSpace(m.cfgDraft.url); !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
				return m, nil // a real URL is required
			}
			m.cfgPhase = cfgPhaseModel
		case cfgPhaseModel: // optional — empty means "pick one later with /model"
			m.cfgPhase = cfgPhaseKey
		default: // key (optional — empty = keyless, e.g. Ollama/vLLM): save
			m.configAddSave()
		}
	case "esc":
		if m.cfgPhase == cfgPhaseName {
			m.cfgPhase = cfgPhaseList
		} else {
			m.cfgPhase--
		}
	case "backspace":
		if r := []rune(*field); len(r) > 0 {
			*field = string(r[:len(r)-1])
		}
	default:
		if s, ok := insertedText(k); ok { // typed or pasted
			*field += s
		}
	}
	return m, nil
}

// configAddSave registers the drafted provider (reusing the /ai add validation)
// and, when a key was given, stores it too — then returns to the panel list.
func (m *tuiModel) configAddSave() {
	d := m.cfgDraft
	name := strings.TrimSpace(d.name)
	arg := strings.TrimSpace(name + " " + strings.TrimSpace(d.url) + " " + strings.TrimSpace(d.model))
	lines := m.app.addProvider(arg)
	// Check the REAL outcome (did addProvider actually create the entry) rather
	// than string-matching its human-readable message — that broke silently
	// once addProvider's wording changed to "added" (it used to say "saved").
	if key := strings.TrimSpace(d.key); key != "" {
		if _, ok := m.app.cfg.Providers[name]; ok {
			lines = append(lines, m.app.setProviderKey(name, key)...)
		}
	}
	m.pushLines(lines)
	m.cfgPhase = cfgPhaseList
	m.cfgDraft = providerDraft{}
}

// renderAddProviderForm draws the in-panel add-provider form.
func (m *tuiModel) renderAddProviderForm(accent lipgloss.Style) []string {
	row := func(phase int, label, val, hint string) string {
		line := fmt.Sprintf("  %-9s %s", label, val)
		if m.cfgPhase == phase {
			return accent.Render(" ▸") + accent.Bold(true).Render(line[2:]) + cDim.Render("▏ "+hint)
		}
		return "  " + line
	}
	header := "add provider — any OpenAI-compatible endpoint"
	keyHint := "optional — empty = keyless (Ollama/vLLM)"
	if name := strings.TrimSpace(m.cfgDraft.name); config.IsCustomProvider(m.app.cfg, name) {
		header = fmt.Sprintf("edit provider %q — current values prefilled", name)
		keyHint = "optional — leave blank to keep the current key"
	}
	return []string{
		accent.Bold(true).Render(header),
		"",
		row(cfgPhaseName, "name", m.cfgDraft.name, "e.g. ollama · enter next"),
		row(cfgPhaseURL, "base URL", m.cfgDraft.url, "e.g. http://localhost:11434/v1"),
		row(cfgPhaseModel, "model", m.cfgDraft.model, "optional — /model lists them later"),
		row(cfgPhaseKey, "api key", m.cfgDraft.key, keyHint),
		"",
		cDim.Render("  enter next/save · esc back"),
	}
}

// finalizeTaskDoneAway drains a task that finished while a modal (a /config-style
// panel, or the approval prompt) was showing over it — idle, then anything queued.
// Reports whether it applied (m.taskDoneAway was set) so a caller with its own
// restore-state (e.g. resolveApproval's m.preApprove) knows to keep that instead.
func (m *tuiModel) finalizeTaskDoneAway() (tea.Model, tea.Cmd, bool) {
	if !m.taskDoneAway {
		return m, nil, false
	}
	m.taskDoneAway = false
	m.state = stIdle
	if len(m.queued) > 0 {
		model, cmd := m.drainQueue()
		return model, cmd, true
	}
	return m, m.input.Focus(), true
}

// closePanel leaves a modal panel: back to the still-running task if one is live,
// otherwise to idle — finalizing (queue drain) a task that finished while the panel
// was open over it.
func (m *tuiModel) closePanel() (tea.Model, tea.Cmd) {
	if m.cancel != nil { // a task is still running behind the panel
		m.state = stRunning
		return m, nil
	}
	if model, cmd, drained := m.finalizeTaskDoneAway(); drained {
		return model, cmd
	}
	m.state = stIdle
	return m, m.input.Focus()
}

// configMove moves the cursor by delta, wrapping.
func (m *tuiModel) configMove(delta int) {
	n := len(cfgKeys())
	m.cfgCursor = (m.cfgCursor + delta + n) % n
}

// configKey returns the currently selected setting key.
func (m *tuiModel) configKey() string { return cfgKeys()[m.cfgCursor] }

// configRowView returns the display label, current value, and a hint for a key.
func (m *tuiModel) configRowView(key string) (label, value, hint string) {
	act := m.app.activeLLM()
	switch key {
	case "provider":
		extra := "enter: cycle"
		if len(m.app.configuredProviderNames()) < 2 {
			extra = "enter: cycle · use “add provider” below"
		}
		return "provider", m.app.providerName(), extra
	case "addprovider":
		return "add provider", "＋", "enter: any OpenAI-compatible endpoint"
	case "model":
		return "model", act.Model, "enter: choose"
	case "apikey":
		v := "— none"
		if act.APIKey != "" {
			v = "● set"
		}
		return "api key", v, "enter: add/set a provider key"
	case "context_window":
		return "context window", ctxLabel(act.ContextWindow), "enter: cycle (0 = auto-detect / provider default)"
	case "reasoning":
		return "reasoning", m.app.reasoningLevel(m.app.providerName(), act.Model), "enter: cycle off→high (trims a thinking model)"
	case "temperature":
		v := "server default"
		if act.Temperature > 0 {
			v = fmt.Sprintf("%g", act.Temperature)
		}
		return "temperature", v, "enter: cycle sampling temperature"
	case "top_p":
		v := "server default"
		if act.TopP > 0 {
			v = fmt.Sprintf("%g", act.TopP)
		}
		return "top_p", v, "enter: cycle nucleus sampling (0.95 · 1.0 = NVIDIA rec pair)"
	case "max_output_tokens":
		v := "server default"
		if act.MaxOutputTokens > 0 {
			v = fmt.Sprintf("%d", act.MaxOutputTokens)
		}
		return "max output", v, "enter: cycle (server's own cap can cut a reasoning model off early)"
	case "loop_detection":
		return "loop detection", onOff(!act.DisableLoopDetection), "enter: toggle (aborts a model stuck repeating itself)"
	case "idle_timeout":
		return "idle timeout", idleTimeoutLabel(act.IdleTimeoutSeconds), "enter: cycle (no response/stream data → retry)"
	case "mode":
		v := "⏵⏵ auto"
		if m.app.planMode {
			v = "⏸ plan"
		}
		return "mode", v, "enter: toggle"
	case "perm_files":
		return "file writes", m.app.cfg.File.Default, "enter: ask/allow/deny"
	case "perm_run":
		return "shell run", m.app.cfg.Run.Default, "enter: ask/allow/deny"
	case "timeout":
		return "run timeout", runTimeoutLabel(m.app.cfg.Run.TimeoutSeconds), "enter: cycle"
	case "budget":
		v := "— none"
		if m.app.cfg.SessionBudgetUSD > 0 {
			v = fmt.Sprintf("$%.2f/run (spent ~$%.2f)", m.app.cfg.SessionBudgetUSD, m.app.sessionCost())
		}
		return "spend cap", v, "enter: set via /budget"
	case "offline":
		return "offline", onOff(m.app.cfg.Offline), "enter: toggle (no internet egress)"
	case "memory":
		v := "summary"
		if m.app.cfg.Memory == "raw" {
			v = "raw (never summarize)"
		}
		return "memory", v, "enter: toggle summary/raw"
	case "compact_threshold":
		v := fmt.Sprintf("%.0f%% of context", compactThreshold(m.app.cfg.CompactThreshold)*100)
		return "compact at", v, "enter: cycle (only used in summary memory)"
	case "max_steps":
		v := fmt.Sprintf("auto (%d)", m.app.goalSteps())
		if m.app.cfg.GoalMaxSteps > 0 {
			v = fmt.Sprintf("%d", m.app.cfg.GoalMaxSteps)
		}
		return "max steps/task", v, "enter: cycle (0 = auto-scale from context window)"
	case "max_history":
		v := fmt.Sprintf("auto (%d)", autoMaxHistory(act.ContextWindow))
		if m.app.cfg.MaxHistory > 0 {
			v = fmt.Sprintf("%d", m.app.cfg.MaxHistory)
		}
		return "max history", v, "enter: cycle (0 = auto-scale from context window)"
	case "max_stuck_turns":
		v := fmt.Sprintf("default (%d)", agent.DefaultMaxStuckTurns)
		if m.app.cfg.MaxStuckTurns > 0 {
			v = fmt.Sprintf("%d", m.app.cfg.MaxStuckTurns)
		}
		return "stuck tolerance", v, "enter: cycle (consecutive unproductive turns before giving up)"
	case "agents":
		return "profiles", fmt.Sprintf("%d configured", len(m.app.cfg.Agents)), "enter: add (provider → model)"
	case "spawn":
		return "spawn approval", m.app.cfg.Spawn.Default, "enter: toggle ask/allow"
	case "subexec":
		v := "off"
		if m.app.cfg.Spawn.Exec {
			v = "on"
		}
		return "sub-agent shell", v, "enter: toggle (give sub-agents run)"
	case "color":
		return "color", colorLabel(m.accent), "enter: cycle"
	case "channel":
		return "channel", channelOf(m.app.cfg), "enter: toggle"
	case "name":
		return "name", m.app.cfg.Name, "enter: rename"
	}
	return key, "", ""
}

// configActivate handles Enter on the selected row.
func (m *tuiModel) configActivate() (tea.Model, tea.Cmd) {
	if m.cancel != nil { // a task is running behind this panel — changing a setting would re-wire it live
		m.push(cDim.Render("  settings are view-only while a task runs — esc to return, then change them"))
		return m, nil
	}
	switch m.configKey() {
	case "provider":
		m.cycleProvider()
	case "mode":
		m.app.setMode(!m.app.planMode)
	case "perm_files":
		m.cyclePerm(&m.app.cfg.File.Default)
	case "perm_run":
		m.cyclePerm(&m.app.cfg.Run.Default)
	case "timeout":
		m.cycleTimeout()
	case "color":
		m.setColor("") // cycle accent
	case "channel":
		m.toggleChannel()
	case "spawn": // toggle ask ⇄ allow
		arg := "on" // → allow (spawn without asking)
		if m.app.cfg.Spawn.Default == "allow" {
			arg = "off" // → ask
		}
		m.app.permissionsSetSpawn(arg)
	case "subexec": // toggle whether sub-agents get the run tool
		arg := "on"
		if m.app.cfg.Spawn.Exec {
			arg = "off"
		}
		m.app.agentsExec(arg)
	case "agents": // open the interactive profile manager (provider → model → name)
		m.openAgents()
	case "budget": // needs a number — hand off to /budget with the flow prefilled
		m.state = stIdle
		m.push(cDim.Render("  /budget <usd> caps estimated spend per run · /budget off disables"))
		m.input.SetValue("/budget ")
		m.input.CursorEnd()
	case "offline": // toggle internet egress
		m.pushLines(m.app.offlineCommand(map[bool]string{true: "off", false: "on"}[m.app.cfg.Offline]))
	case "memory": // toggle summary ⇄ raw
		next := "raw"
		if m.app.cfg.Memory == "raw" {
			next = ""
		}
		m.app.cfg.Memory = next
		if err := config.SaveMemory(next); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		}
		_ = m.app.wire() // raw needs a much higher history cap (see wire) — apply it now
	case "compact_threshold":
		m.cycleCompactThreshold()
	case "max_steps":
		m.cycleMaxSteps()
	case "max_history":
		m.cycleMaxHistory()
	case "max_stuck_turns":
		m.cycleMaxStuckTurns()
	case "reasoning": // cycle the active model's reasoning effort off→high
		provider, model := m.app.providerName(), m.app.activeLLM().Model
		next := nextReasoning(provider, m.app.reasoningLevel(provider, model))
		if _, ok := m.app.applyReasoning(provider+"/"+model, provider, next); !ok {
			m.push(cDim.Render("  " + provider + " reasoning must be set raw in config.json (key " + provider + "/" + model + ")"))
		}
	case "temperature":
		m.cycleTemperature()
	case "top_p":
		m.cycleTopP()
	case "max_output_tokens":
		m.cycleMaxOutputTokens()
	case "loop_detection": // toggle the active connection's repetition detectors
		if err := m.app.toggleLoopDetection(); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else if err := m.app.wire(); err != nil {
			m.push(cErr.Render("  " + err.Error()))
		}
	case "idle_timeout":
		m.cycleIdleTimeout()
	case "model": // needs the live model list — hand off to /model
		m.state = stIdle
		return m.runCommand("/model")
	case "addprovider": // open the in-panel form (name → URL → model → key)
		m.cfgPhase = cfgPhaseName
		m.cfgDraft = providerDraft{}
	case "apikey": // add or set a provider key — prefill; pick provider (Tab) + paste token
		m.state = stIdle
		m.input.SetValue("/ai key ")
		m.input.CursorEnd()
	case "context_window":
		m.cycleContextWindow()
	case "name":
		m.state = stIdle
		m.input.SetValue("/rename ")
		m.input.CursorEnd()
	}
	return m, nil
}

// cycleProvider switches to the next configured provider and re-wires.
func (m *tuiModel) cycleProvider() {
	provs := m.app.configuredProviderNames()
	if len(provs) < 2 {
		return
	}
	cur := m.app.providerName()
	idx := 0
	for i, p := range provs {
		if p == cur {
			idx = i
			break
		}
	}
	_ = m.app.setProvider(provs[(idx+1)%len(provs)]) // saves + re-wires + re-detects
}

var permCycle = []string{"ask", "allow", "deny"}

// cyclePerm advances one policy default (file OR run, independently) through
// ask → allow → deny, persists the workspace policy, and re-wires.
func (m *tuiModel) cyclePerm(field *string) {
	*field = nextStr(*field, permCycle)
	_ = config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File)
	_ = m.app.wire()
}

var timeoutCycle = []int{60, 120, 300, 600}

// cycleTimeout advances the run timeout through the preset values, persists, and
// re-wires so the run tool picks up the new default.
func (m *tuiModel) cycleTimeout() {
	m.app.cfg.Run.TimeoutSeconds = nextInt(m.app.cfg.Run.TimeoutSeconds, timeoutCycle)
	_ = config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File)
	_ = m.app.wire()
}

// nextStr returns the element after cur in cycle (wrapping); cycle[0] if cur
// isn't found or is the last.
func nextStr(cur string, cycle []string) string {
	for i, v := range cycle {
		if v == cur {
			return cycle[(i+1)%len(cycle)]
		}
	}
	return cycle[0]
}

// nextInt is nextStr for ints.
func nextInt(cur int, cycle []int) int {
	for i, v := range cycle {
		if v == cur {
			return cycle[(i+1)%len(cycle)]
		}
	}
	return cycle[0]
}

// nextFloat is nextStr for float64 presets (exact match — the cycle only ever
// contains our own preset values, never a user-typed one).
func nextFloat(cur float64, cycle []float64) float64 {
	for i, v := range cycle {
		if v == cur {
			return cycle[(i+1)%len(cycle)]
		}
	}
	return cycle[0]
}

var compactThresholdCycle = []float64{0.5, 0.65, 0.75, 0.85, 0.95}

// cycleCompactThreshold advances the auto-compact threshold through preset
// fill levels, persists, and re-wires so autoCompactNeeded picks it up.
func (m *tuiModel) cycleCompactThreshold() {
	m.app.cfg.CompactThreshold = nextFloat(compactThreshold(m.app.cfg.CompactThreshold), compactThresholdCycle)
	if err := config.SaveCompactThreshold(m.app.cfg.CompactThreshold); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
	}
}

// maxStepsCycle presets for the /config "max_steps" row. 0 clears the manual
// override, letting the per-goal step budget auto-scale from the active
// connection's context window again (see stepBudget/autoStepBudget).
var maxStepsCycle = []int{0, 40, 80, 120, 200, 400}

// cycleMaxSteps advances the manual GoalMaxSteps override through presets,
// persists it, and re-wires so the rebuilt Agent picks up the new budget.
func (m *tuiModel) cycleMaxSteps() {
	m.app.cfg.GoalMaxSteps = nextInt(m.app.cfg.GoalMaxSteps, maxStepsCycle)
	if err := config.SaveGoalMaxSteps(m.app.cfg.GoalMaxSteps); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// maxHistoryCycle presets for the /config "max_history" row. 0 clears the
// manual override, letting the cross-task memory cap auto-scale from the
// active connection's context window again (see autoMaxHistory).
var maxHistoryCycle = []int{0, 16, 32, 64, 128, 256, rawMemoryMaxHistory}

// cycleMaxHistory advances the manual Config.MaxHistory override through
// presets, persists it, and re-wires so Agent.remember picks up the new cap.
func (m *tuiModel) cycleMaxHistory() {
	m.app.cfg.MaxHistory = nextInt(m.app.cfg.MaxHistory, maxHistoryCycle)
	if err := config.SaveMaxHistory(m.app.cfg.MaxHistory); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// maxStuckTurnsCycle presets for the /config "max_stuck_turns" row. 0 clears
// the override, falling back to internal/agent's own DefaultMaxStuckTurns.
var maxStuckTurnsCycle = []int{0, 3, 5, 8, 15, 25}

// cycleMaxStuckTurns advances the manual Config.MaxStuckTurns override
// through presets, persists it, and re-wires so the rebuilt Agent picks up
// the new tolerance.
func (m *tuiModel) cycleMaxStuckTurns() {
	m.app.cfg.MaxStuckTurns = nextInt(m.app.cfg.MaxStuckTurns, maxStuckTurnsCycle)
	if err := config.SaveMaxStuckTurns(m.app.cfg.MaxStuckTurns); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// temperatureCycle presets for the /config "temperature" row. 0 = server
// default (see Chat's c.temp > 0 gate); 1.0 is NVIDIA's recommended pairing
// with top_p=0.95 below.
var temperatureCycle = []float64{0, 0.2, 0.7, 1.0}

// cycleTemperature advances the active provider's sampling temperature through
// preset values, persists it (same local-vs-named-provider branching as
// setModel), and re-wires so the client picks it up.
func (m *tuiModel) cycleTemperature() {
	next := nextFloat(m.app.activeLLM().Temperature, temperatureCycle)
	if err := m.app.setTemperature(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// topPCycle presets for the /config "top_p" row. 0 = server default; 0.95 is
// NVIDIA's recommended pairing with temperature=1.0 above.
var topPCycle = []float64{0, 0.7, 0.9, 0.95, 1.0}

// cycleTopP advances the active provider's nucleus-sampling top_p through
// preset values, persists it, and re-wires.
func (m *tuiModel) cycleTopP() {
	next := nextFloat(m.app.activeLLM().TopP, topPCycle)
	if err := m.app.setTopP(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// maxOutputTokensCycle presets for the /config "max_output_tokens" row. 0 =
// server default (often too small for a reasoning model's own thinking phase —
// observed live: a reply cut off, finish_reason=length, mid-reasoning, well
// under the context window's own limit).
var maxOutputTokensCycle = []int{0, 2000, 4000, 8000, 16000, 32000}

// cycleMaxOutputTokens advances the active provider's max_tokens override
// through presets, persists it, and re-wires.
func (m *tuiModel) cycleMaxOutputTokens() {
	next := nextInt(m.app.activeLLM().MaxOutputTokens, maxOutputTokensCycle)
	if err := m.app.setMaxOutputTokens(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// idleTimeoutCycle presets (seconds) for the /config "idle_timeout" row. 0 =
// the client's built-in 90s default; the larger values suit a hosted
// reasoning model that can think silently (no streamed deltas) for longer.
var idleTimeoutCycle = []int{0, 60, 120, 180, 300, 600}

// idleTimeoutLabel renders the idle timeout for the panel (0 = the built-in 90s).
func idleTimeoutLabel(sec int) string {
	if sec <= 0 {
		return "90s (default)"
	}
	return (time.Duration(sec) * time.Second).String()
}

// cycleIdleTimeout advances the active provider's idle watchdog through preset
// values, persists it (same local-vs-named-provider branching as
// setTemperature/setTopP), and re-wires so the client picks it up.
func (m *tuiModel) cycleIdleTimeout() {
	next := nextInt(m.app.activeLLM().IdleTimeoutSeconds, idleTimeoutCycle)
	if err := m.app.setIdleTimeout(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// contextWindowCycle presets (tokens) for the /config "context_window" row. 0
// clears the override (falls back to auto-detect / the provider's default);
// the rest cover common local and hosted model window sizes.
var contextWindowCycle = []int{0, 4096, 8192, 16384, 32768, 65536, 131072}

// cycleContextWindow advances the active provider's context-window override
// through preset sizes, persists it (same local-vs-named-provider branching as
// the other per-connection rows), and re-wires so auto-compact/maxRespTk sizing
// picks it up immediately.
func (m *tuiModel) cycleContextWindow() {
	next := nextInt(m.app.activeLLM().ContextWindow, contextWindowCycle)
	if err := m.app.setContextWindow(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// toggleChannel flips stable ⇄ nightly and persists it.
func (m *tuiModel) toggleChannel() {
	next := "nightly"
	if channelOf(m.app.cfg) == "nightly" {
		next = "stable"
	}
	m.app.cfg.Channel = next
	_ = config.SaveChannel(next)
}

// renderConfigPanel draws the boxed, sectioned settings screen.
func (m *tuiModel) renderConfigPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	if m.cfgPhase != cfgPhaseList { // the in-panel add-provider form
		box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
		return box.Render(strings.Join(m.renderAddProviderForm(accent), "\n"))
	}
	cur := m.configKey()

	lines := []string{accent.Bold(true).Render("config")}
	for _, r := range configRows {
		if r.header != "" {
			lines = append(lines, "", accent.Render("  "+r.header))
			continue
		}
		label, value, hint := m.configRowView(r.key)
		// Pad by DISPLAY width (values carry wide/ambiguous glyphs like ⏵⏵ / ● / ▮),
		// so the hint column lines up cleanly instead of ragged.
		labelCol := padVis(label, 15)
		valCol := padVis(value, 22)
		if r.key == cur {
			lines = append(lines, accent.Render(" ▸ ")+accent.Bold(true).Render(labelCol+" "+valCol)+" "+cDim.Render(hint))
		} else {
			lines = append(lines, "   "+cDim.Render(labelCol)+" "+valCol+" "+cDim.Render(hint))
		}
	}
	footer := "  ↑↓ move · enter change · esc close"
	if m.cancel != nil {
		footer = "  ↑↓ move · view-only while a task runs · esc back to it"
	}
	lines = append(lines, "", cDim.Render(footer))
	lines = append(lines, cDim.Render("  saved to ~/.config/ipsupport-code/config.json (kept private)"))

	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// runTimeoutLabel renders the run timeout for the panel (0 = the built-in 60s).
func runTimeoutLabel(sec int) string {
	if sec <= 0 {
		return "60s (default)"
	}
	return (time.Duration(sec) * time.Second).String()
}

// colorLabel maps the current accent color code back to its name (or the raw
// code if it isn't one of the named colors).
// padVis right-pads s to a target DISPLAY width (rune/ANSI-aware via lipgloss.Width),
// so columns whose values carry wide or styled glyphs still align.
func padVis(s string, w int) string {
	if pad := w - lipgloss.Width(s); pad > 0 {
		return s + strings.Repeat(" ", pad)
	}
	return s
}

func colorLabel(c lipgloss.Color) string {
	code := string(c)
	for name, v := range colorNames {
		if v == code {
			return "▮ " + name
		}
	}
	return "▮ " + code
}
