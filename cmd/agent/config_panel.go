package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

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
	// The connection in use, every part of it editable here: where it points,
	// which model, its key. The local server's address and key had no row at
	// all, and the hint that sent people here for them led nowhere.
	{header: "Connection"},
	{key: "provider"},
	{key: "base_url"},
	{key: "model"},
	{key: "apikey"},
	{key: "conn_type"},
	{key: "context_window"},
	{header: "Providers"},
	{key: "addprovider"},
	{key: "removeprovider"},
	{header: "Model tuning"},
	{key: "reasoning"},
	{key: "temperature"},
	{key: "top_p"},
	{key: "max_output_tokens"},
	{key: "loop_detection"},
	{key: "idle_timeout"},
	{key: "retry_attempts"},
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
	// Goal pursuit and the learning store used to have no home here at all: the
	// judge's two settings sat under "Model & provider" (they describe the
	// judging STEP, not the connection), and goal TTL, the idle nudge, the
	// judge's reasoning level, reflection and knowledge retention were reachable
	// only as slash commands — discoverable only if you already knew they
	// existed.
	{header: "Goal & judge"},
	{key: "goal_ttl"},
	{key: "goal_nudge"},
	{key: "judge_reasoning"},
	{key: "judge_max_output_tokens"},
	{key: "judge_criteria"},
	{header: "Learning"},
	{key: "reflection"},
	{key: "knowledge_retention"},
	{key: "knowledge"},
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
	m.cfgEdit = nil
	m.cfgPick = nil
	m.cfgNote = nil
	m.cfgWSKeys = config.WorkspaceKeys(m.app.workspace)
	m.state = stConfig
}

// cfgRowSetting is the config key a global-saved row writes, for the rows a
// workspace's own config file can override.
var cfgRowSetting = map[string]string{
	"provider": "provider", "reasoning": "reasoning", "judge_reasoning": "reasoning",
	"budget": "session_budget_usd", "offline": "offline", "memory": "memory",
	"compact_threshold": "compact_threshold", "max_steps": "goal_max_steps",
	"max_history": "max_history", "max_stuck_turns": "max_stuck_turns",
	"goal_ttl": "goal_max_returns", "goal_nudge": "goal_nudge",
	"judge_max_output_tokens": "judge_max_output_tokens", "reflection": "reflect_disabled",
	"knowledge_retention": "knowledge_retention_days", "agents": "agents",
	"spawn": "spawn", "subexec": "spawn", "color": "color", "channel": "channel", "name": "name",
}

// projectOverrides reports whether this project's .agent/config.json sets the
// row's setting — then a change saved here (globally) does not survive a restart.
func (m *tuiModel) projectOverrides(key string) bool {
	k, ok := cfgRowSetting[key]
	return ok && m.cfgWSKeys[k]
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

// providerDraft is the provider being added via the panel form. prefilled is
// the name whose saved values the URL and model fields hold, so a name changed
// after stepping back reloads them instead of saving one provider's address
// under another's name.
type providerDraft struct {
	name, url, model, key textField
	prefilled             string
	err                   string
}

func newProviderDraft() providerDraft {
	return providerDraft{key: newTextField("", true)}
}

// cfgAddField returns the form field being edited for the current phase.
func (m *tuiModel) cfgAddField() *textField {
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
// the last field), esc steps back (to the list from the first). A field that
// cannot be accepted says why, and keeps what was typed.
func (m *tuiModel) configAddKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	d := &m.cfgDraft
	switch k.String() {
	case "enter":
		d.err = ""
		switch m.cfgPhase {
		case cfgPhaseName:
			name := strings.TrimSpace(d.name.value())
			if err := validProviderName(name); err != nil {
				d.err = err.Error()
				return m, nil
			}
			if _, ok := config.ProviderTemplates[name]; ok {
				d.err = fmt.Sprintf("%q is built in — select it as the provider; its key and address are under Connection", name)
				return m, nil
			}
			// Editing a provider already added: its saved address and model,
			// reloaded whenever the name changes (the key is never shown — an
			// empty key field keeps the one on file).
			if name != d.prefilled {
				p, saved := m.app.cfg.Providers[name]
				if saved {
					d.url, d.model = newTextField(p.BaseURL, false), newTextField(p.Model, false)
				} else if d.prefilled != "" {
					d.url, d.model = textField{}, textField{}
				}
				// A key typed for the previous name is that provider's: kept,
				// it would be saved under this one and sent to its server.
				if d.prefilled != "" {
					d.key = newTextField("", true)
				}
				d.prefilled = name
			}
			m.cfgPhase = cfgPhaseURL
		case cfgPhaseURL:
			if _, err := parseBaseURL(d.url.value()); err != nil {
				d.err = err.Error()
				return m, nil
			}
			m.cfgPhase = cfgPhaseModel
		case cfgPhaseModel: // optional — empty means "pick one later with /model"
			m.cfgPhase = cfgPhaseKey
		default: // key (optional — empty = keyless, e.g. Ollama/vLLM): save
			msg, err := m.app.addProviderFields(d.name.value(), d.url.value(), d.model.value(), d.key.value())
			if err != nil {
				d.err = err.Error()
				return m, nil
			}
			m.push(cDim.Render("  " + msg))
			m.cfgPhase = cfgPhaseList
			m.cfgDraft = newProviderDraft()
			return m, m.detectWindowCmd() // the provider in use may have moved
		}
	case "esc":
		d.err = ""
		if m.cfgPhase == cfgPhaseName {
			m.cfgPhase = cfgPhaseList
		} else {
			m.cfgPhase--
		}
	default:
		m.cfgAddField().key(k)
	}
	return m, nil
}

// renderAddProviderForm draws the in-panel add-provider form.
func (m *tuiModel) renderAddProviderForm(accent lipgloss.Style) []string {
	d := m.cfgDraft
	row := func(phase int, label string, f textField, hint string) string {
		if m.cfgPhase == phase {
			return accent.Render(" ▸ ") + accent.Bold(true).Render(padVis(label, 9)) + " " + f.view() + cDim.Render("  "+hint)
		}
		shown := f.value()
		if f.masked {
			shown = strings.Repeat("●", len([]rune(shown)))
		}
		return "   " + padVis(label, 9) + " " + shown
	}
	header := "add provider — any OpenAI-compatible endpoint"
	keyHint := "optional — empty = keyless (Ollama/vLLM)"
	if name := strings.TrimSpace(d.name.value()); config.IsCustomProvider(m.app.cfg, name) {
		header = fmt.Sprintf("edit provider %q — saved values loaded", name)
		keyHint = "optional — leave empty to keep the saved key"
	}
	lines := []string{
		accent.Bold(true).Render(header),
		"",
		row(cfgPhaseName, "name", d.name, "e.g. ollama · letters, digits, - _ ."),
		row(cfgPhaseURL, "base URL", d.url, "e.g. http://localhost:11434/v1"),
		row(cfgPhaseModel, "model", d.model, "optional — /model lists them later"),
		row(cfgPhaseKey, "api key", d.key, keyHint),
	}
	if d.err != "" {
		lines = append(lines, "", cErr.Render("  "+d.err))
	}
	return append(lines, "", cDim.Render("  enter next/save · esc back · ←→ home end ctrl+u edit"))
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
		extra := "enter: pick from the list"
		if len(m.app.configuredProviderNames()) < 2 {
			extra = "enter: pick · use “add provider” below to add one"
			if m.cancel != nil {
				extra = "enter: pick · adding one waits until the task ends"
			}
		}
		return "provider", m.app.providerName(), extra
	case "addprovider":
		return "add provider", "＋", "enter: any OpenAI-compatible endpoint (or edit one you added)"
	case "removeprovider":
		n := len(m.app.savedProviderNames())
		return "remove provider", fmt.Sprintf("%d saved", n), "enter: forget a saved provider (a built-in one: its key and overrides)"
	case "base_url":
		return "address", act.BaseURL, "enter: edit host, port, path"
	case "model":
		return "model", act.Model, "enter: pick from the server's list (or type one it doesn't list)"
	case "apikey":
		v := "— none"
		if act.APIKey != "" {
			v = "● set"
		}
		return "api key", v, "enter: set or replace (masked) · ctrl+d in it removes"
	case "conn_type":
		if !m.app.isLocal() {
			return "type", "OpenAI-compatible", "fixed for " + m.app.providerName()
		}
		return "type", localTypeLabel(m.app.cfg.LLM.Type), "enter: LM Studio ⇄ OpenAI-compatible"
	case "context_window":
		v := ctxLabel(act.ContextWindow)
		if act.ContextWindowManual {
			v += " (set)"
		}
		return "context window", v, "enter: type a size · 0 = auto-detect"
	case "reasoning":
		hint := "enter: cycle off→high (trims a thinking model)"
		if _, ok := reasoningShape(m.app.providerName(), "high"); !ok {
			hint = "not settable here for " + m.app.providerName() + " — enter shows the config.json key"
		}
		return "reasoning", m.app.reasoningLevel(m.app.providerName(), act.Model), hint
	case "temperature":
		v := "server default"
		if act.Temperature > 0 {
			v = fmt.Sprintf("%g", act.Temperature)
		}
		return "temperature", v, "enter: type a value (0 = server default)"
	case "top_p":
		v := "server default"
		if act.TopP > 0 {
			v = fmt.Sprintf("%g", act.TopP)
		}
		return "top_p", v, "enter: type a value (0.95 · 1.0 = NVIDIA rec pair)"
	case "max_output_tokens":
		v := "server default"
		if act.MaxOutputTokens > 0 {
			v = fmt.Sprintf("%d", act.MaxOutputTokens)
		}
		return "max output", v, "enter: type tokens (server's own cap can cut a reasoning model off early)"
	case "judge_max_output_tokens":
		v := "same as the task model"
		if m.app.cfg.JudgeMaxOutputTokens > 0 {
			v = fmt.Sprintf("%d", m.app.cfg.JudgeMaxOutputTokens)
		}
		return "judge max output", v, "enter: type tokens, 0 = the task model's (the goal judge is cut off mid-verdict on a small budget)"
	case "judge_criteria":
		v, hint := "— none", "enter: how to create it"
		if src := m.app.judgeCriteriaSource(); src != "" {
			v, hint = "● "+src, "enter: how to edit it"
		}
		return "judge criteria", v, hint
	case "retry_attempts":
		v := "8 (default)"
		if act.RetryAttempts > 0 {
			v = fmt.Sprintf("%d", act.RetryAttempts)
		}
		return "retry attempts", v, "enter: type a count (transient failures; 1 = fail fast when the server is down)"
	case "goal_ttl":
		v := "off — the model's own finish stands"
		if m.app.cfg.GoalMaxReturns > 0 {
			v = fmt.Sprintf("%d re-feed(s)", m.app.cfg.GoalMaxReturns)
		}
		return "goal TTL", v, "enter: cycle (how many times an unmet goal is re-fed)"
	case "goal_nudge":
		return "idle nudge", onOff(m.app.cfg.GoalNudge), "enter: toggle (push once when a re-fed goal does no work)"
	case "judge_reasoning":
		prov := m.app.providerName()
		v := "same as the task model"
		if m.app.judgeScoped(prov, act.Model) {
			v, _ = m.app.scopedReasoningLevel("judge:", prov, act.Model)
		}
		return "judge reasoning", v, "enter: cycle (a yes/no check shouldn't be a thinking task — minimal suits it)"
	case "reflection":
		v := "on"
		if m.app.cfg.ReflectDisabled {
			v = "off"
		} else if p := strings.TrimSpace(m.app.cfg.ReflectProfile); p != "" {
			v += " · " + p
		}
		return "reflection", v, "enter: toggle (distills facts and lessons after each task)"
	case "knowledge_retention":
		v := "off (kept forever)"
		if m.app.cfg.KnowledgeRetentionDays > 0 {
			v = fmt.Sprintf("%d days", m.app.cfg.KnowledgeRetentionDays)
		}
		return "knowledge retention", v, "enter: cycle (drop lessons not seen for N days)"
	case "knowledge":
		nf, nl := m.app.factsCount(), len(m.app.kb.All())
		return "knowledge", fmt.Sprintf("%d fact(s) · %d lesson(s)", nf, nl), "enter: list them (facts are in every prompt)"
	case "loop_detection":
		return "loop detection", onOff(!act.DisableLoopDetection), "enter: toggle (aborts a model stuck repeating itself)"
	case "idle_timeout":
		return "idle timeout", idleTimeoutLabel(act.IdleTimeoutSeconds), "enter: type seconds (no response/stream data → retry)"
	case "mode":
		v := "⏵⏵ auto"
		if m.app.planMode {
			v = "⏸ plan"
		}
		return "mode", v, "enter: toggle · this session only (shift+tab does the same)"
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

// cfgLiveRows are the rows that do something NOW even while a task runs: they
// only print, so there is nothing to defer and nothing to race.
var cfgLiveRows = map[string]bool{"judge_criteria": true, "knowledge": true}

// handleConfigKey routes a key in the /config panel.
func (m *tuiModel) handleConfigKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.cfgPhase != cfgPhaseList { // typing inside the add-provider form
		return m.configAddKey(k)
	}
	if m.cfgEdit != nil { // typing a value in place on its row
		return m.configEditKey(k)
	}
	if m.cfgPick != nil { // a list open on its row
		return m.configPickKey(k)
	}
	switch k.String() {
	case "up", "k":
		m.configMove(-1)
	case "down", "j":
		m.configMove(1)
	case "enter", "right", "l", " ":
		return m.configActivate()
	case "esc", "q":
		return m.closePanel()
	}
	return m, nil
}

// configActivate handles Enter on the selected row. While a task is running,
// applying the change here would re-wire the agent underneath it, so the
// activation is STAGED instead of refused: the same keystroke is replayed when
// the task ends (see applyPendingConfig). Every stageable row is a deterministic
// toggle or cycle, so pressing enter three times stages three cycles and lands
// where pressing it three times live would have.
func (m *tuiModel) configActivate() (tea.Model, tea.Cmd) {
	key := m.configKey()
	if key == "provider" || key == "model" || key == "removeprovider" { // open a list; what is picked there stages
		return m.activateConfigRow(key)
	}
	if m.cancel == nil || cfgLiveRows[key] {
		return m.activateConfigRow(key)
	}
	// A task is running: see staging.go.
	if liveTuneRows[key] {
		return m.tuneLive(key)
	}
	if _, ok := cycleRows[key]; ok {
		m.stageCycle(key)
		return m, nil
	}
	label, _, _ := m.configRowView(key)
	m.push(cDim.Render("  " + label + " can't change under a running task — set it once the task ends (esc goes back to it)"))
	return m, nil
}

// applyPendingConfig saves what was staged in /config while a task ran (see
// staging.go). Called once the task is over, from the one goroutine allowed
// to re-wire the agent. It returns what the changes asked for — a provider or
// model switched needs its window probed.
func (m *tuiModel) applyPendingConfig() tea.Cmd {
	if len(m.cfgPending) == 0 {
		return nil
	}
	pending := m.cfgPending
	m.cfgPending = nil
	before := len(m.history)
	var cmds []tea.Cmd
	rewire, saved := false, 0
	for _, e := range pending {
		cmd, rw, ok := m.applyStaged(e)
		cmds = append(cmds, cmd)
		rewire = rewire || rw
		if ok {
			saved++
		}
		if e.key == "provider" || e.key == "model" || e.key == "removeprovider" {
			m.cfgPick = nil // a list open over the task was for the connection that just changed
		}
	}
	if rewire { // once, from what was saved: a failed save leaves no client tuned to it
		if err := m.app.wire(); err != nil {
			m.push(cErr.Render("  " + err.Error()))
		}
		cmds = append(cmds, m.detectWindowCmd())
	}
	msg := fmt.Sprintf("  saved %d staged /config change(s)", saved)
	if failed := len(pending) - saved; failed > 0 {
		msg += fmt.Sprintf(" · %d not saved — see above", failed)
	}
	m.push(cDim.Render(msg))
	if m.state == stConfig && len(m.history) >= before { // the panel covers the log
		said := m.history[before:]
		m.cfgNote = said[max(len(said)-4, 0):]
	}
	return tea.Batch(cmds...)
}

// activateConfigRow performs a row's action for real. It assumes no task is
// running: several branches re-wire the agent or leave the panel.
func (m *tuiModel) activateConfigRow(key string) (tea.Model, tea.Cmd) {
	if m.projectOverrides(key) {
		m.push(cErr.Render("  note: this project's .agent/config.json sets " + cfgRowSetting[key] + " — this change lasts until the next start; edit that file to keep it"))
	}
	switch key {
	case "provider":
		m.cfgPick = &cfgPicker{key: key, p: newPicker(m.app.providerPickItems(), m.app.providerName())}
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
		m.agFromConfig = true // esc comes back here, not to the prompt
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
		if err := config.SaveMemory(next); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
			break
		}
		m.app.cfg.Memory = next
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
		if _, ok, err := m.app.applyReasoning(provider+"/"+model, provider, next); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else if !ok {
			m.push(cDim.Render("  " + provider + " reasoning must be set raw in config.json (key " + provider + "/" + model + ")"))
		}
	case "temperature", "top_p", "max_output_tokens", "idle_timeout", "retry_attempts", "judge_max_output_tokens":
		m.cfgEdit = &cfgEditor{key: key, f: newTextField(m.app.numberValue(key), false)}
	case "goal_ttl":
		m.pushLines(m.app.goalTTL("ttl", []string{"ttl", fmt.Sprint(nextInt(m.app.cfg.GoalMaxReturns, goalTTLCycle))}))
	case "goal_nudge":
		if err := m.app.setGoalNudge(!m.app.cfg.GoalNudge); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else {
			m.push(cDim.Render("  idle nudge → " + onOff(m.app.cfg.GoalNudge)))
		}
	case "judge_reasoning":
		cur, set := m.app.scopedReasoningLevel("judge:", m.app.providerName(), m.app.activeLLM().Model)
		if !set {
			cur = "default"
		}
		m.pushLines(m.app.reasoningCommand("judge " + nextLevel(cur)))
	case "reflection":
		m.pushLines(m.app.reflectCommand(map[bool]string{true: "on", false: "off"}[m.app.cfg.ReflectDisabled]))
	case "knowledge_retention":
		// The setting only: stepping 0 → 7 → 30 → 90 must not purge at each
		// stop on the way, as /knowledge retain does — passing 7 deleted every
		// lesson older than a week. The purge happens at the next launch.
		before := m.app.cfg.KnowledgeRetentionDays
		next := nextInt(before, knowledgeRetentionCycle)
		m.app.cfg.KnowledgeRetentionDays = next
		if err := config.SaveKnowledgeRetention(next); err != nil {
			m.app.cfg.KnowledgeRetentionDays = before
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else if next == 0 {
			m.push(cDim.Render("  knowledge retention → off (lessons kept forever)"))
		} else {
			m.push(cDim.Render(fmt.Sprintf("  knowledge retention → %d days — older lessons go at the next launch (/knowledge retain %d drops them now)", next, next)))
		}
	case "knowledge":
		m.pushLines(m.app.knowledgeCommand("list"))
	case "judge_criteria":
		// Prose, not a value to cycle: point at the file. It is re-read on every
		// run, so an edit applies without a restart.
		ws := filepath.Join(m.app.workspace, ".agent", "judge.md")
		m.push(cDim.Render("  judge criteria — extra acceptance rules, applied ON TOP of the goal (never replacing it):"))
		m.push(cDim.Render("    this project:  " + ws))
		m.push(cDim.Render("    every project: " + config.JudgePromptPath()))
		m.push(cDim.Render("  e.g. \"a fix is not done until the tests were actually RUN, not just written\""))
	case "loop_detection": // toggle the active connection's repetition detectors
		if err := m.app.toggleLoopDetection(); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else if err := m.app.wire(); err != nil {
			m.push(cErr.Render("  " + err.Error()))
		}
	case "base_url":
		m.cfgEdit = &cfgEditor{key: key, f: newTextField(m.app.activeLLM().BaseURL, false)}
	case "model":
		m.cfgPick = &cfgPicker{key: key, p: picker{loading: true, free: true}, epoch: m.app.modelEpoch.Load()}
		return m, m.fetchModelsCmd()
	case "apikey":
		m.cfgEdit = &cfgEditor{key: key, f: newTextField("", true)}
	case "context_window":
		m.cfgEdit = &cfgEditor{key: key, f: newTextField(fmt.Sprint(m.app.activeLLM().ContextWindow), false)}
	case "removeprovider":
		if len(m.app.savedProviderNames()) == 0 {
			m.push(cDim.Render("  no saved providers to remove"))
			break
		}
		var items []pickItem
		for _, n := range m.app.savedProviderNames() {
			items = append(items, pickItem{value: n})
		}
		m.cfgPick = &cfgPicker{key: key, p: newPicker(items, "")}
	case "conn_type":
		if msg, err := m.app.cycleLocalType(); err != nil {
			m.push(cErr.Render("  " + err.Error()))
		} else {
			m.push(cDim.Render("  " + msg))
			return m, m.detectWindowCmd() // detected another way now
		}
	case "addprovider": // open the in-panel form (name → URL → model → key)
		m.cfgPhase = cfgPhaseName
		m.cfgDraft = newProviderDraft()
	case "name":
		m.state = stIdle
		m.input.SetValue("/rename ")
		m.input.CursorEnd()
	}
	return m, nil
}

// cfgEditor is a value being typed in place on its row: the active
// connection's address, key or context size, or a tuning number.
type cfgEditor struct {
	key string
	f   textField
	err string
}

// cfgPicker is a list opened on its row: the provider to use, the model, or
// the provider to remove.
type cfgPicker struct {
	key     string
	p       picker
	epoch   int64  // model: the connection the list was fetched for
	confirm string // removeprovider: the name awaiting a second enter
	err     string
}

// configPickKey moves through the open list; enter acts on the pick.
func (m *tuiModel) configPickKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	cp := m.cfgPick
	if k.String() == "enter" && cp.p.loading {
		cp.err = "still loading the list…"
		return m, nil
	}
	act := cp.p.key(k)
	if act == pickNone { // moved or filtered: whatever was being confirmed is off
		cp.confirm, cp.err = "", ""
	}
	switch act {
	case pickCancel:
		m.cfgPick = nil
	case pickChose:
		v, _ := cp.p.choice()
		busy := m.cancel != nil
		switch cp.key {
		case "provider":
			if v == m.app.providerName() {
				m.unstage(stagedEdit{key: "provider"})
				m.cfgPick = nil
				m.push(cDim.Render("  already on " + v))
				return m, nil
			}
			if busy {
				if m.removalStaged(v) {
					cp.err = v + " is staged for removal — it can't also be switched to"
					return m, nil
				}
				m.stage(stagedEdit{key: "provider", value: v})
				m.push(cDim.Render("  staged — provider " + v + " when the task finishes"))
				m.cfgPick = nil
				return m, nil
			}
			out := m.app.setProvider(v) // saves + re-wires; says so, or why not
			if len(out) > 0 && !strings.HasPrefix(out[0], "→") {
				cp.err = out[0] // stays open: pick another, or esc
				return m, nil
			}
			m.cfgPick = nil
			m.pushLines(out)
			return m, m.detectWindowCmd()
		case "model":
			if cp.epoch != m.app.modelEpoch.Load() {
				cp.err = "the connection changed since this list — esc and open it again"
				return m, nil
			}
			prov := m.app.providerName()
			if busy && m.removalStaged(prov) {
				cp.err = prov + " is staged for removal — its model can't change too"
				return m, nil
			}
			if busy {
				m.cfgPick = nil
				if v == m.app.activeLLM().Model {
					m.unstage(stagedEdit{key: "model", prov: prov})
					m.push(cDim.Render("  already on " + v))
					return m, nil
				}
				m.stage(stagedEdit{key: "model", value: v, prov: prov})
				m.push(cDim.Render("  staged — model " + v + " on " + prov + " when the task finishes"))
				return m, nil
			}
			out := m.app.setModel(v)
			if len(out) > 0 && strings.HasPrefix(out[0], "error") {
				cp.err = out[0]
				return m, nil
			}
			m.cfgPick = nil
			m.pushLines(out)
			return m, m.detectWindowCmd()
		case "removeprovider":
			if busy {
				for _, e := range m.cfgPending {
					if (e.key == "provider" && e.value == v) || (e.key != "removeprovider" && e.prov == v) {
						cp.err = "changes to " + v + " are staged — it can't also be removed"
						return m, nil
					}
				}
			}
			if cp.confirm != v { // first enter: ask again
				cp.confirm = v
				return m, nil
			}
			if busy {
				m.stage(stagedEdit{key: "removeprovider", prov: v})
				m.push(cDim.Render("  staged — " + v + " is removed when the task finishes"))
				m.cfgPick = nil
				return m, nil
			}
			msg, err := m.app.removeProvider(v)
			if err != nil {
				cp.confirm, cp.err = "", err.Error()
				return m, nil
			}
			m.cfgPick = nil
			m.push(cDim.Render("  " + msg))
			return m, m.detectWindowCmd()
		}
	}
	return m, nil
}

// removalStaged reports whether removing name waits for the task's end.
func (m *tuiModel) removalStaged(name string) bool {
	for _, e := range m.cfgPending {
		if e.key == "removeprovider" && e.prov == name {
			return true
		}
	}
	return false
}

// pickerHint is what the open list offers, under it.
func (m *tuiModel) pickerHint() string {
	cp := m.cfgPick
	switch {
	case cp.err != "":
		return cErr.Render("     " + cp.err)
	case cp.confirm != "":
		return cErr.Render(fmt.Sprintf("     enter again to remove %q · esc keeps it", cp.confirm))
	}
	h := "↑↓ move · type to filter · enter pick · esc cancel"
	switch cp.key {
	case "model":
		h = "↑↓ move · type to filter (or a model the server doesn't list) · enter pick · esc cancel"
	case "provider":
		if m.cancel != nil {
			h += " · applies when the task ends"
		}
	}
	return cDim.Render("     " + h)
}

// configEditKey types into the open editor: enter saves (or says why not and
// stays open), esc cancels, ctrl+d on the key removes it.
func (m *tuiModel) configEditKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	e := m.cfgEdit
	switch k.String() {
	case "esc":
		m.cfgEdit = nil
		return m, nil
	case "ctrl+d":
		if e.key == "apikey" && m.cancel == nil {
			return m.finishEdit(m.app.setActiveKey("", true))
		}
	case "enter":
		v := e.f.value()
		if m.cancel != nil && !liveTuneRows[e.key] { // only a live tuning row's editor opens mid-task
			e.err = "a task is running — this waits until it ends (esc)"
			return m, nil
		}
		switch e.key {
		case "base_url":
			return m.finishEdit(m.app.setBaseURL(v))
		case "apikey":
			return m.finishEdit(m.app.setActiveKey(v, false))
		case "context_window":
			return m.finishEdit(m.app.setContextWindowValue(v))
		case "temperature", "top_p", "max_output_tokens", "idle_timeout", "retry_attempts", "judge_max_output_tokens":
			if m.cancel != nil && liveTuneRows[e.key] {
				return m.tuneTyped(e.key, v)
			}
			return m.finishEdit(m.app.setNumber(e.key, v))
		}
		return m, nil
	}
	if e.f.key(k) {
		e.err = ""
	}
	return m, nil
}

// finishEdit closes the editor on success, or keeps it open with the reason.
func (m *tuiModel) finishEdit(msg string, err error) (tea.Model, tea.Cmd) {
	if err != nil {
		m.cfgEdit.err = err.Error()
		return m, nil
	}
	m.cfgEdit = nil
	m.push(cDim.Render("  " + msg))
	return m, m.detectWindowCmd()
}

// editorHint is what the open editor offers, under its row.
func (m *tuiModel) editorHint() string {
	e := m.cfgEdit
	if e.err != "" {
		return cErr.Render("     " + e.err)
	}
	h := map[string]string{
		"base_url":       "host, port and path — e.g. http://localhost:8080/v1",
		"apikey":         "type or paste · empty enter keeps the saved key · ctrl+d removes it",
		"context_window": "tokens, e.g. 32768 · 0 = auto-detect",
	}[e.key]
	if spec, ok := numberRows[e.key]; ok {
		h = fmt.Sprintf("%g to %g, e.g. %s · 0 = server default", spec.min, spec.max, spec.example)
	}
	if e.key == "base_url" && !m.app.isLocal() && !config.IsCustomProvider(m.app.cfg, m.app.cfg.Provider) {
		h += " · empty = its default"
	}
	return cDim.Render("     " + h + " · enter save · esc cancel")
}

var permCycle = []string{"ask", "allow", "deny"}

// cyclePerm advances one policy default (file OR run, independently) through
// ask → allow → deny, persists the workspace policy, and re-wires.
func (m *tuiModel) cyclePerm(field *string) {
	before := *field
	*field = nextStr(*field, permCycle)
	if err := config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File); err != nil {
		*field = before // shown only once it is on disk
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

var timeoutCycle = []int{60, 120, 300, 600}

// cycleTimeout advances the run timeout through the preset values, persists, and
// re-wires so the run tool picks up the new default.
func (m *tuiModel) cycleTimeout() {
	before := m.app.cfg.Run.TimeoutSeconds
	m.app.cfg.Run.TimeoutSeconds = nextInt(before, timeoutCycle)
	if err := config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File); err != nil {
		m.app.cfg.Run.TimeoutSeconds = before
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
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
	before := m.app.cfg.CompactThreshold
	m.app.cfg.CompactThreshold = nextFloat(compactThreshold(before), compactThresholdCycle)
	if err := config.SaveCompactThreshold(m.app.cfg.CompactThreshold); err != nil {
		m.app.cfg.CompactThreshold = before // shown only once it is on disk
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
	before := m.app.cfg.GoalMaxSteps
	m.app.cfg.GoalMaxSteps = nextInt(before, maxStepsCycle)
	if err := config.SaveGoalMaxSteps(m.app.cfg.GoalMaxSteps); err != nil {
		m.app.cfg.GoalMaxSteps = before
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
	before := m.app.cfg.MaxHistory
	m.app.cfg.MaxHistory = nextInt(before, maxHistoryCycle)
	if err := config.SaveMaxHistory(m.app.cfg.MaxHistory); err != nil {
		m.app.cfg.MaxHistory = before
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
	before := m.app.cfg.MaxStuckTurns
	m.app.cfg.MaxStuckTurns = nextInt(before, maxStuckTurnsCycle)
	if err := config.SaveMaxStuckTurns(m.app.cfg.MaxStuckTurns); err != nil {
		m.app.cfg.MaxStuckTurns = before
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	_ = m.app.wire()
}

// goalTTLCycle presets for the "goal TTL" row. 0 = pursuit off: the model's own
// finish stands.
var goalTTLCycle = []int{0, 1, 2, 3, 6, 12, 255}

// knowledgeRetentionCycle presets (days) for the "knowledge retention" row.
// 0 = keep forever.
var knowledgeRetentionCycle = []int{0, 7, 30, 90, 180}

// nextLevel cycles a reasoning level for the judge row. "minimal" is first after
// the inherited default because a yes/no acceptance check is not a thinking
// task — a judge that reasons its way through its whole output budget never
// reaches a verdict at all.
func nextLevel(cur string) string {
	order := []string{"minimal", "low", "medium", "high", "off"}
	for i, l := range order {
		if l == cur {
			return order[(i+1)%len(order)]
		}
	}
	return order[0]
}

// idleTimeoutLabel renders the idle timeout for the panel (0 = the built-in 90s).
func idleTimeoutLabel(sec int) string {
	if sec <= 0 {
		return "90s (default)"
	}
	return (time.Duration(sec) * time.Second).String()
}

// toggleChannel flips stable ⇄ nightly and persists it.
func (m *tuiModel) toggleChannel() {
	next := "nightly"
	if channelOf(m.app.cfg) == "nightly" {
		next = "stable"
	}
	if err := config.SaveChannel(next); err != nil {
		m.push(cErr.Render("  could not persist: " + err.Error()))
		return
	}
	m.app.cfg.Channel = next
}

// renderConfigPanel draws the boxed, sectioned settings screen.
func (m *tuiModel) renderConfigPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	if m.cfgPhase != cfgPhaseList { // the in-panel add-provider form
		return m.panelBox(m.renderAddProviderForm(accent))
	}
	cur := m.configKey()

	// Only the rows that fit. Reported live: the panel rendered every row
	// unconditionally, so on a short terminal the top — including the title and
	// the first section — was simply pushed off the screen, with no way to reach
	// it. Growing the panel by two sections is what exposed it, but any terminal
	// short enough would always have hit this.
	rows, first := m.configWindow()

	lines := []string{accent.Bold(true).Render("config")}
	if first > 0 {
		lines = append(lines, cDim.Render("   ↑ more above"))
	}
	for _, r := range rows {
		if r.header != "" {
			lines = append(lines, "", accent.Render("  "+r.header))
			continue
		}
		label, value, hint := m.configRowView(r.key)
		if r.key == cur && m.cfgEdit != nil { // the value being typed, in place
			lines = append(lines, accent.Render(" ▸ ")+accent.Bold(true).Render(padVis(label, 15))+" "+m.cfgEdit.f.view(), m.editorHint())
			continue
		}
		if r.key == cur && m.cfgPick != nil { // the list to pick from, under its row
			lines = append(lines, accent.Render(" ▸ ")+accent.Bold(true).Render(padVis(label, 15))+" "+value)
			lines = append(lines, m.cfgPick.p.view(accent, m.spin.View(), "     ")...)
			lines = append(lines, m.pickerHint())
			continue
		}
		if m.projectOverrides(r.key) {
			hint = "this project's .agent/config.json sets it — that value wins at the next start"
		}
		if v, h := m.stagedRowView(r.key, value); h != "" {
			// a.cfg — what configRowView shows — gets it when the task ends.
			value, hint = v, h
		}
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
	if first+len(rows) < len(configRows) {
		lines = append(lines, cDim.Render("   ↓ more below"))
	}
	footer := "  ↑↓ move · enter change · esc close"
	if m.cancel != nil {
		footer = "  ↑↓ move · enter: tuning applies now, the rest when the task ends · esc back to it"
	}
	for _, l := range m.cfgNote { // what the last key said — the log is behind the panel
		lines = append(lines, l)
	}
	lines = append(lines, "", cDim.Render(footer))
	lines = append(lines, cDim.Render("  saved privately to ~/.config/ipsupport-code/config.json · file writes, shell run, run timeout: this project's .agent/config.json"))
	return m.panelBox(lines)
}

// panelBox draws the panel's lines in its border, each cut to the terminal's
// width: a long model name or hint used to widen the box past the screen and
// wrap, which the height budget never counted.
func (m *tuiModel) panelBox(lines []string) string {
	if w := m.width - 4; w > 20 { // border and padding
		for i, l := range lines {
			lines[i] = ansi.Truncate(l, w, "…")
		}
	}
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// configWindow is the slice of configRows that fits on screen, always including
// the selected row, plus the index it starts at.
//
// Scrolls by whole rows and keeps a row of context past the cursor where there
// is one, so arrowing to the end of a section doesn't leave the next header
// invisible. A terminal too short for even a few rows still gets those few
// rather than a panel drawn off the top of the screen.
func (m *tuiModel) configWindow() ([]cfgRow, int) {
	// The panel is drawn INTO the log area, not onto the whole screen, so the
	// budget is the viewport's — the status line, both rules, the hint line and
	// the input box all sit below it. Sizing against m.height instead let the
	// box grow past the visible region, and the terminal cut it from the TOP:
	// the title and the first section header scrolled away with no way back to
	// them, which is the opposite of what a scrolling panel is for.
	//
	// Inside that: the title, the blank line and two footer lines, the box's two
	// border lines, and BOTH scroll markers — in the middle of a long list both
	// are drawn.
	chrome := 8
	if m.cfgEdit != nil {
		chrome++ // the editor's hint line under its row
	}
	chrome += len(m.cfgNote)
	if m.cfgPick != nil {
		// The list gets what is left after the panel's own lines, its filter,
		// scroll markers and hint, and three rows of the panel — ten items on
		// a 24-line terminal pushed the panel's top off the screen.
		m.cfgPick.p.rows = min(max(m.viewportHeight()-chrome-4-3, 3), pickerRows)
		chrome += m.cfgPick.p.height() + 1 // the list and its hint under its row
	}
	avail := m.viewportHeight() - chrome
	if avail < 3 {
		avail = 3
	}
	if cost(configRows) <= avail {
		return configRows, 0
	}
	// Where the cursor's key actually sits among ALL rows, headers included —
	// cfgCursor indexes the selectable keys only.
	sel := 0
	key := m.configKey()
	for i, r := range configRows {
		if r.key == key {
			sel = i
			break
		}
	}
	// Grow outward from the selection while the window still fits. A section
	// header costs TWO lines (a blank one precedes it), which is why the window
	// is measured in emitted lines rather than in rows — sizing it by row count
	// overflowed by exactly the number of headers on screen.
	first, last := sel, sel
	for {
		grew := false
		if last+1 < len(configRows) && cost(configRows[first:last+2]) <= avail {
			last++
			grew = true
		}
		if first > 0 && cost(configRows[first-1:last+1]) <= avail {
			first--
			grew = true
		}
		if !grew {
			break
		}
	}
	return configRows[first : last+1], first
}

// cost is how many lines a run of rows renders as: one per setting, two per
// section header (it is preceded by a blank line).
func cost(rows []cfgRow) int {
	n := 0
	for _, r := range rows {
		if r.header != "" {
			n += 2
			continue
		}
		n++
	}
	return n
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
