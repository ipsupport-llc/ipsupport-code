package main

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// picker is a list to choose from: ↑↓ move (pgup/pgdn a page, home/end the
// ends), typing narrows it, enter picks. Reported: removing a provider and
// picking a model meant typing a name the panel already knew.
type picker struct {
	items   []pickItem
	cursor  int // index into visible()
	filter  []rune
	loading bool   // the list is still being fetched
	err     string // why there is no list
	free    bool   // enter with nothing matching takes the typed text (a model the server doesn't list)
}

type pickItem struct {
	value string
	note  string // shown dim after the value: "current", "loaded · ctx 32K"
}

type pickAction int

const (
	pickNone pickAction = iota
	pickChose
	pickCancel
)

// pickerRows is how many items a picker shows at once.
const pickerRows = 10

// newPicker starts on current when it is listed.
func newPicker(items []pickItem, current string) picker {
	p := picker{items: items}
	p.seek(current)
	return p
}

func (p *picker) seek(value string) {
	for i, it := range p.visible() {
		if it.value == value {
			p.cursor = i
			return
		}
	}
}

// load fills a picker that was waiting for its list.
func (p *picker) load(items []pickItem, err, current string) {
	p.loading, p.items, p.err = false, items, err
	p.cursor = 0
	p.seek(current)
}

func (p *picker) visible() []pickItem {
	if len(p.filter) == 0 {
		return p.items
	}
	q := strings.ToLower(string(p.filter))
	var out []pickItem
	for _, it := range p.items {
		if strings.Contains(strings.ToLower(it.value), q) {
			out = append(out, it)
		}
	}
	return out
}

// choice is what enter picks: the highlighted item, or — for a free picker
// with nothing matching — the typed text.
func (p *picker) choice() (string, bool) {
	if v := p.visible(); len(v) > 0 {
		return v[p.cursor].value, true
	}
	if t := strings.TrimSpace(string(p.filter)); p.free && t != "" {
		return t, true
	}
	return "", false
}

func (p *picker) key(k tea.KeyMsg) pickAction {
	n := len(p.visible())
	switch k.String() {
	case "esc":
		return pickCancel
	case "enter":
		if _, ok := p.choice(); ok {
			return pickChose
		}
		return pickNone
	case "up", "ctrl+p":
		if n > 0 {
			p.cursor = (p.cursor - 1 + n) % n
		}
	case "down", "ctrl+n", "tab":
		if n > 0 {
			p.cursor = (p.cursor + 1) % n
		}
	case "pgup":
		p.cursor = max(p.cursor-pickerRows, 0)
	case "pgdown":
		p.cursor = max(min(p.cursor+pickerRows, n-1), 0)
	case "home":
		p.cursor = 0
	case "end":
		p.cursor = max(n-1, 0)
	case "backspace":
		if len(p.filter) > 0 {
			p.filter = p.filter[:len(p.filter)-1]
			p.cursor = 0
		}
	case "ctrl+u":
		p.filter, p.cursor = nil, 0
	default:
		if s, ok := insertedText(k); ok {
			p.filter = append(p.filter, []rune(strings.ReplaceAll(s, "\n", ""))...)
			p.cursor = 0
		}
	}
	return pickNone
}

// view draws the filter line and the window of items around the cursor.
func (p *picker) view(accent lipgloss.Style, spin string, indent string) []string {
	var out []string
	if len(p.filter) > 0 {
		out = append(out, indent+cDim.Render("filter: ")+string(p.filter)+accent.Render("▌"))
	} else {
		out = append(out, indent+cDim.Render("type to filter"))
	}
	switch {
	case p.loading:
		return append(out, indent+spin+cDim.Render(" loading…"))
	case p.err != "":
		out = append(out, indent+cErr.Render(p.err))
		if p.free {
			out = append(out, indent+cDim.Render("type a name and press enter"))
		}
		return out
	}
	v := p.visible()
	if len(v) == 0 {
		msg := "nothing matches — backspace to widen"
		if p.free && len(p.filter) > 0 {
			msg = "not listed — enter uses " + fmt.Sprintf("%q", string(p.filter))
		}
		return append(out, indent+cDim.Render(msg))
	}
	lo, hi := windowBounds(p.cursor, len(v), pickerRows)
	if lo > 0 {
		out = append(out, indent+cDim.Render(fmt.Sprintf("  ↑ %d more", lo)))
	}
	for i := lo; i < hi; i++ {
		note := ""
		if v[i].note != "" {
			note = "  " + cDim.Render(v[i].note)
		}
		if i == p.cursor {
			out = append(out, indent+accent.Render("› ")+accent.Bold(true).Render(v[i].value)+note)
		} else {
			out = append(out, indent+"  "+v[i].value+note)
		}
	}
	if hi < len(v) {
		out = append(out, indent+cDim.Render(fmt.Sprintf("  ↓ %d more", len(v)-hi)))
	}
	return out
}

// height is how many lines view draws, for the panels' height budgets.
func (p *picker) height() int {
	return len(p.view(lipgloss.NewStyle(), "", ""))
}

// pickListMsg carries a fetched list to the picker waiting for it.
type pickListMsg struct {
	items []pickItem
	err   string
	epoch int64 // the model epoch it was fetched for; another switch since drops it
}

// fetchModelsCmd lists the active connection's models off the UI thread.
func (m *tuiModel) fetchModelsCmd() tea.Cmd {
	act := m.app.activeLLM()
	ep := m.app.modelEpoch.Load()
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	return func() tea.Msg {
		c, cancel := context.WithTimeout(ctx, 8*time.Second)
		defer cancel()
		items, err := modelPickItems(c, act)
		msg := pickListMsg{items: items, epoch: ep}
		if err != nil {
			msg.err = "couldn't list models: " + err.Error()
		} else if len(items) == 0 {
			msg.err = "the server lists no models"
		}
		return msg
	}
}

// modelPickItems is the connection's models, LM Studio's with what is loaded
// and its context size.
func modelPickItems(ctx context.Context, act config.LLM) ([]pickItem, error) {
	var out []pickItem
	if act.LMStudio() {
		ms, err := llm.ListLMStudioModels(ctx, act.BaseURL, http.DefaultClient)
		if err != nil {
			return nil, err
		}
		for _, m := range ms {
			var note []string
			if m.State == "loaded" {
				note = append(note, "loaded")
			}
			switch {
			case m.LoadedContextLength > 0:
				note = append(note, "ctx "+humanK(m.LoadedContextLength))
			case m.MaxContextLength > 0:
				note = append(note, "max "+humanK(m.MaxContextLength))
			}
			if m.Quantization != "" {
				note = append(note, m.Quantization)
			}
			out = append(out, pickItem{value: m.ID, note: strings.Join(note, " · ")})
		}
		return markCurrent(out, act.Model), nil
	}
	ids, err := llm.ListModels(ctx, act.BaseURL, act.APIKey, http.DefaultClient)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		out = append(out, pickItem{value: id})
	}
	return markCurrent(out, act.Model), nil
}

func markCurrent(items []pickItem, current string) []pickItem {
	for i := range items {
		if items[i].value == current {
			items[i].note = strings.TrimSuffix("current · "+items[i].note, " · ")
		}
	}
	return items
}

// providerPickItems is every provider that can be switched to.
func (a *app) providerPickItems() []pickItem {
	var out []pickItem
	for _, n := range a.configuredProviderNames() {
		note := a.cfg.LLM.BaseURL
		if n != "local" {
			l, _ := config.ResolveProvider(a.cfg, n)
			note = l.BaseURL
		}
		if n == a.providerName() {
			note = "current · " + note
		}
		out = append(out, pickItem{value: n, note: note})
	}
	return out
}

// sessionPickItems is the saved sessions, newest first as listSessions has them.
func (a *app) sessionPickItems() []pickItem {
	var out []pickItem
	for _, s := range a.listSessions() {
		note := fmt.Sprintf("%d exchange(s) · %s", s.count/2, humanizeAgo(s.mod))
		if s.active {
			note = "current · " + note
		}
		out = append(out, pickItem{value: s.name, note: note})
	}
	return out
}

// openPick shows a list in its own panel.
func (m *tuiModel) openPick(kind string, p picker) (tea.Model, tea.Cmd) {
	m.pick, m.pickKind = &p, kind
	m.state = stPick
	return m, nil
}

func (m *tuiModel) pickTitle() string {
	switch m.pickKind {
	case "model":
		return "model on " + m.app.providerName() + " — now " + m.app.activeLLM().Model
	case "provider":
		return "provider — now " + m.app.providerName()
	}
	return "sessions — now " + m.app.cfg.Name
}

// pickKey moves through the open list; enter switches to the pick.
func (m *tuiModel) pickKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch m.pick.key(k) {
	case pickCancel:
		m.pick = nil
		return m.idleDrain()
	case pickChose:
		v, _ := m.pick.choice()
		kind := m.pickKind
		m.pick = nil
		var detect tea.Cmd
		switch kind {
		case "model":
			m.pushLines(m.app.setModel(v))
			detect = m.detectWindowCmd()
		case "provider":
			m.pushLines(m.app.setProvider(v))
			detect = m.detectWindowCmd()
		case "session":
			lines, switched := m.app.sessionsCommand(v)
			m.pushLines(lines)
			if switched {
				m.push(m.sessionRecap()...)
			}
		}
		model, cmd := m.idleDrain()
		return model, tea.Batch(cmd, detect)
	}
	return m, nil
}

// renderPickPanel draws stPick's list in its box.
func (m *tuiModel) renderPickPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	lines := []string{accent.Bold(true).Render(m.pickTitle())}
	lines = append(lines, m.pick.view(accent, m.spin.View(), "  ")...)
	return m.panelBox(lines)
}
