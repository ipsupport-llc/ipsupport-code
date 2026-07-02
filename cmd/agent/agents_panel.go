package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// The interactive sub-agent profile manager (stAgents): a list of profiles with
// add / edit / delete, and a three-step builder — pick provider → pick model
// (from the provider's live model list, type to filter) → name it. esc steps
// back; esc on the list closes the panel. Everything applies and saves on save.

type agentPhase int

const (
	agList agentPhase = iota
	agPickProvider
	agPickModel
	agName
)

// agentDraft is the profile being added or edited.
type agentDraft struct {
	orig     string // the existing name when editing; "" when adding
	provider string
	model    string
	name     string
}

// agentModelsMsg carries the result of the async model-list fetch.
type agentModelsMsg struct {
	models []string
	err    string
}

const agModelWindow = 12 // visible rows in the (possibly long) model list

// openAgents enters the profile manager at the list.
func (m *tuiModel) openAgents() {
	m.state = stAgents
	m.agPhase = agList
	m.agCursor = 0
}

func (m *tuiModel) agProviders() []string { return m.app.configuredProviderNames() }

// agentsKey routes a key to the handler for the current builder phase.
func (m *tuiModel) agentsKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch m.agPhase {
	case agPickProvider:
		return m.agentsProviderKey(k)
	case agPickModel:
		return m.agentsModelKey(k)
	case agName:
		return m.agentsNameKey(k)
	default:
		return m.agentsListKey(k)
	}
}

func (m *tuiModel) agentsListKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	names := agentProfileNames(m.app.cfg)
	n := len(names) + 1 // +1 for the "add new" row
	switch k.String() {
	case "up", "k":
		m.agCursor = (m.agCursor - 1 + n) % n
	case "down", "j":
		m.agCursor = (m.agCursor + 1) % n
	case "d": // delete the highlighted profile
		if m.agCursor < len(names) {
			_ = m.app.agentsRemove(names[m.agCursor]) // persists + re-wires
			if left := len(agentProfileNames(m.app.cfg)); m.agCursor > left {
				m.agCursor = left
			}
		}
	case "enter", "right", "l":
		if m.agCursor >= len(names) { // "add new"
			m.agDraft = agentDraft{}
			m.agCursor = 0
		} else { // edit the selected profile
			name := names[m.agCursor]
			p := m.app.cfg.Agents[name]
			m.agDraft = agentDraft{orig: name, provider: p.Provider, model: p.Model, name: name}
			m.agCursor = indexOf(m.agProviders(), p.Provider)
		}
		m.agPhase = agPickProvider
	case "esc", "q":
		m.state = stIdle
	}
	return m, nil
}

func (m *tuiModel) agentsProviderKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	provs := m.agProviders()
	switch k.String() {
	case "up", "k":
		m.agCursor = (m.agCursor - 1 + len(provs)) % len(provs)
	case "down", "j":
		m.agCursor = (m.agCursor + 1) % len(provs)
	case "enter", "right", "l":
		m.agDraft.provider = provs[m.agCursor]
		m.agPhase = agPickModel
		m.agCursor, m.agFilter, m.agModelsAll, m.agModelsErr = 0, "", nil, ""
		m.agLoading = true
		return m, m.fetchAgentModels(m.agDraft.provider)
	case "esc":
		m.agPhase = agList
		m.agCursor = 0
	}
	return m, nil
}

func (m *tuiModel) agentsModelKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.agLoading {
		if k.String() == "esc" {
			m.agPhase, m.agCursor = agPickProvider, 0
		}
		return m, nil
	}
	models := m.filteredModels()
	switch k.String() {
	case "up":
		if len(models) > 0 {
			m.agCursor = (m.agCursor - 1 + len(models)) % len(models)
		}
	case "down":
		if len(models) > 0 {
			m.agCursor = (m.agCursor + 1) % len(models)
		}
	case "enter", "right":
		if m.agCursor < len(models) {
			m.agDraft.model = models[m.agCursor]
			m.agPhase = agName // name stays as-is: empty for a new profile, the old name when editing
		}
	case "esc":
		m.agPhase, m.agCursor = agPickProvider, 0
	case "backspace":
		if r := []rune(m.agFilter); len(r) > 0 {
			m.agFilter = string(r[:len(r)-1])
			m.agCursor = 0
		}
	default:
		if len(k.String()) == 1 { // type to filter the (often long) model list
			m.agFilter += k.String()
			m.agCursor = 0
		}
	}
	return m, nil
}

func (m *tuiModel) agentsNameKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch k.String() {
	case "enter":
		name := strings.TrimSpace(m.agDraft.name)
		if name == "" { // blank → fall back to a name derived from the model id
			name = defaultProfileName(m.agDraft.model)
		}
		if name == "" {
			return m, nil
		}
		m.saveDraft(name)
		m.agPhase, m.agCursor = agList, 0
	case "esc":
		m.agPhase, m.agCursor = agPickModel, 0
	case "backspace":
		if r := []rune(m.agDraft.name); len(r) > 0 {
			m.agDraft.name = string(r[:len(r)-1])
		}
	default:
		if len(k.String()) == 1 {
			m.agDraft.name += k.String()
		}
	}
	return m, nil
}

// fetchAgentModels lists the chosen provider's models off-thread.
func (m *tuiModel) fetchAgentModels(provider string) tea.Cmd {
	conn, errLine := m.app.providerConn(provider)
	if errLine != "" {
		return func() tea.Msg { return agentModelsMsg{err: errLine} }
	}
	ctx := m.ctx
	return func() tea.Msg {
		c, cancel := context.WithTimeout(ctx, 8*time.Second)
		defer cancel()
		ids := listModelIDs(c, conn)
		if len(ids) == 0 {
			return agentModelsMsg{err: "no models returned (check the provider key / connection)"}
		}
		return agentModelsMsg{models: ids}
	}
}

// filteredModels applies the type-to-filter substring to the fetched models.
func (m *tuiModel) filteredModels() []string {
	if m.agFilter == "" {
		return m.agModelsAll
	}
	q := strings.ToLower(m.agFilter)
	out := make([]string, 0, len(m.agModelsAll))
	for _, id := range m.agModelsAll {
		if strings.Contains(strings.ToLower(id), q) {
			out = append(out, id)
		}
	}
	return out
}

// saveDraft writes the drafted profile (renaming if the name changed) and re-wires.
func (m *tuiModel) saveDraft(name string) {
	if m.app.cfg.Agents == nil {
		m.app.cfg.Agents = map[string]config.AgentProfile{}
	}
	if m.agDraft.orig != "" && m.agDraft.orig != name {
		delete(m.app.cfg.Agents, m.agDraft.orig) // a rename
	}
	m.app.cfg.Agents[name] = config.AgentProfile{Provider: m.agDraft.provider, Model: m.agDraft.model}
	_ = config.SaveAgents(m.app.cfg.Agents)
	_ = m.app.wire() // the agent tool / roster may have changed
}

// agentsHint is the bottom-line help for the current phase.
func (m *tuiModel) agentsHint() string {
	switch m.agPhase {
	case agPickProvider:
		return "  ↑↓ move · enter pick provider · esc back"
	case agPickModel:
		return "  ↑↓ move · type to filter · enter pick · esc back"
	case agName:
		return "  type a name · enter save · esc back"
	default:
		return "  ↑↓ move · enter add/edit · d delete · esc close"
	}
}

// renderAgentsPanel draws the boxed manager for the current phase.
func (m *tuiModel) renderAgentsPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	var lines []string
	switch m.agPhase {
	case agPickProvider:
		lines = append(lines, accent.Bold(true).Render("add profile — pick a provider"))
		for i, p := range m.agProviders() {
			lines = append(lines, agRow(accent, p, i == m.agCursor))
		}
		lines = append(lines, "", cDim.Render("  no provider you want? add a key with /ai key <name> <token>"))
	case agPickModel:
		lines = append(lines, accent.Bold(true).Render("add profile — pick a model "+cDim.Render("("+m.agDraft.provider+")")))
		switch {
		case m.agLoading:
			lines = append(lines, "", "  "+m.spin.View()+cDim.Render(" loading models…"))
		case m.agModelsErr != "":
			lines = append(lines, "", cErr.Render("  "+m.agModelsErr))
		default:
			models := m.filteredModels()
			if m.agFilter != "" {
				lines = append(lines, cDim.Render("  filter: ")+m.agFilter+cDim.Render(fmt.Sprintf("  (%d match)", len(models))))
			}
			if len(models) == 0 {
				lines = append(lines, "", cDim.Render("  no match — backspace to widen"))
			}
			lo, hi := windowBounds(m.agCursor, len(models), agModelWindow)
			for i := lo; i < hi; i++ {
				lines = append(lines, agRow(accent, models[i], i == m.agCursor))
			}
			if hi < len(models) {
				lines = append(lines, cDim.Render(fmt.Sprintf("    …%d more below", len(models)-hi)))
			}
		}
	case agName:
		lines = append(lines, accent.Bold(true).Render("add profile — name it"))
		lines = append(lines, "", "  "+accent.Render("name: ")+m.agDraft.name+accent.Render("▌"))
		if strings.TrimSpace(m.agDraft.name) == "" {
			lines = append(lines, cDim.Render("  (blank → "+defaultProfileName(m.agDraft.model)+")"))
		}
		lines = append(lines, "", cDim.Render(fmt.Sprintf("  %s · %s", m.agDraft.provider, m.agDraft.model)))
	default: // agList
		lines = append(lines, accent.Bold(true).Render("sub-agent profiles"))
		names := agentProfileNames(m.app.cfg)
		if len(names) == 0 {
			lines = append(lines, cDim.Render(`  no profiles yet — select "+ add new" below`))
		}
		for i, n := range names {
			p := m.app.cfg.Agents[n]
			label := fmt.Sprintf("%-14s %s · %s", n, p.Provider, p.Model)
			lines = append(lines, agRow(accent, label, i == m.agCursor))
		}
		lines = append(lines, agRow(accent, "+ add new", m.agCursor >= len(names)))
	}
	lines = append(lines, "", cDim.Render(strings.TrimLeft(m.agentsHint(), " ")))
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// agRow renders a selectable row with a cursor marker.
func agRow(accent lipgloss.Style, text string, selected bool) string {
	if selected {
		return accent.Render(" ▸ ") + accent.Bold(true).Render(text)
	}
	return "   " + text
}

// windowBounds returns a [lo,hi) slice window of size n centered on cursor.
func windowBounds(cursor, total, n int) (int, int) {
	if total <= n {
		return 0, total
	}
	lo := cursor - n/2
	if lo < 0 {
		lo = 0
	}
	hi := lo + n
	if hi > total {
		hi, lo = total, total-n
	}
	return lo, hi
}

// defaultProfileName derives a short profile name from a model id, e.g.
// "x-ai/grok-4.3" → "grok-4.3", "openai/gpt-4o:free" → "gpt-4o".
func defaultProfileName(model string) string {
	name := model
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	return name
}

func indexOf(ss []string, v string) int {
	for i, s := range ss {
		if s == v {
			return i
		}
	}
	return 0
}
