package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

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
	{key: "reasoning"},
	{header: "Behavior"},
	{key: "mode"},
	{key: "perm_files"},
	{key: "perm_run"},
	{key: "timeout"},
	{key: "budget"},
	{key: "offline"},
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
			if strings.TrimSpace(m.cfgDraft.name) == "" {
				return m, nil // a name is required
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
		if len(k.String()) == 1 { // printable
			*field += k.String()
		}
	}
	return m, nil
}

// configAddSave registers the drafted provider (reusing the /ai add validation)
// and, when a key was given, stores it too — then returns to the panel list.
func (m *tuiModel) configAddSave() {
	d := m.cfgDraft
	arg := strings.TrimSpace(strings.TrimSpace(d.name) + " " + strings.TrimSpace(d.url) + " " + strings.TrimSpace(d.model))
	lines := m.app.addProvider(arg)
	if key := strings.TrimSpace(d.key); key != "" && strings.Contains(strings.Join(lines, " "), "saved") {
		lines = append(lines, m.app.setProviderKey(strings.TrimSpace(d.name), key)...)
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
	return []string{
		accent.Bold(true).Render("add provider — any OpenAI-compatible endpoint"),
		"",
		row(cfgPhaseName, "name", m.cfgDraft.name, "e.g. ollama · enter next"),
		row(cfgPhaseURL, "base URL", m.cfgDraft.url, "e.g. http://localhost:11434/v1"),
		row(cfgPhaseModel, "model", m.cfgDraft.model, "optional — /model lists them later"),
		row(cfgPhaseKey, "api key", m.cfgDraft.key, "optional — empty = keyless (Ollama/vLLM)"),
		"",
		cDim.Render("  enter next/save · esc back"),
	}
}

// closePanel leaves a modal panel: back to the still-running task if one is live,
// otherwise to idle — finalizing (queue drain) a task that finished while the panel
// was open over it.
func (m *tuiModel) closePanel() (tea.Model, tea.Cmd) {
	if m.cancel != nil { // a task is still running behind the panel
		m.state = stRunning
		return m, nil
	}
	m.state = stIdle
	if m.taskDoneAway {
		m.taskDoneAway = false
		if len(m.queued) > 0 {
			return m.drainQueue()
		}
	}
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
	case "reasoning":
		return "reasoning", m.app.reasoningLevel(m.app.providerName(), act.Model), "enter: cycle off→high (trims a thinking model)"
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
	case "reasoning": // cycle the active model's reasoning effort off→high
		provider, model := m.app.providerName(), m.app.activeLLM().Model
		next := nextReasoning(m.app.reasoningLevel(provider, model))
		if _, ok := m.app.applyReasoning(provider+"/"+model, provider, next); !ok {
			m.push(cDim.Render("  " + provider + " reasoning must be set raw in config.json (key " + provider + "/" + model + ")"))
		}
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
