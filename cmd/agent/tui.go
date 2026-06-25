package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

type uiState int

const (
	stIdle uiState = iota
	stRunning
	stApprove
)

type tuiModel struct {
	app    *app
	ctx    context.Context
	bridge *uiBridge

	vp    viewport.Model
	input textinput.Model
	spin  spinner.Model

	state         uiState
	history       []string
	pending       *approvalReq
	cancel        context.CancelFunc
	width, height int
	ready         bool
}

// Bubble Tea messages.
type eventMsg uiEvent
type approvalMsg approvalReq
type taskDoneMsg struct{}

// runTUI installs the UI bridge as the agent's tracer + approver, then runs the
// Bubble Tea program.
func (a *app) runTUI(ctx context.Context) error {
	b := newBridge()
	a.uiTracer = b
	a.approver = b
	if err := a.wire(); err != nil {
		return err
	}

	in := textinput.New()
	in.Placeholder = "type a task, or /help"
	in.Prompt = "› "
	in.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = cToolCall

	m := &tuiModel{app: a, ctx: ctx, bridge: b, input: in, spin: sp, state: stIdle}
	m.pushLine(cTitle.Render("ipsupport-code") + cDim.Render("  — task, or /help · ctrl-c quits"))

	_, err := tea.NewProgram(m, tea.WithAltScreen(), tea.WithContext(ctx)).Run()
	return err
}

func (m *tuiModel) Init() tea.Cmd {
	return tea.Batch(m.spin.Tick, textinput.Blink, m.waitEvent(), m.waitApproval())
}

func (m *tuiModel) waitEvent() tea.Cmd {
	return func() tea.Msg { return eventMsg(<-m.bridge.events) }
}

func (m *tuiModel) waitApproval() tea.Cmd {
	return func() tea.Msg { return approvalMsg(<-m.bridge.approvals) }
}

func (m *tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		if !m.ready {
			m.vp = viewport.New(msg.Width, m.viewportHeight())
			m.ready = true
		} else {
			m.vp.Width = msg.Width
			m.vp.Height = m.viewportHeight()
		}
		m.input.Width = msg.Width - 4
		m.refresh()
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case eventMsg:
		for _, ln := range eventLines(uiEvent(msg)) {
			m.pushLine(ln)
		}
		return m, m.waitEvent()

	case approvalMsg:
		req := approvalReq(msg)
		m.pending = &req
		m.state = stApprove
		m.pushLine(cToolCall.Render("  ⚠ approve "+req.kind+": ") + req.detail + cDim.Render("   [y/N]"))
		return m, m.waitApproval()

	case taskDoneMsg:
		m.state = stIdle
		m.cancel = nil
		return m, m.input.Focus()

	case tea.KeyMsg:
		return m.handleKey(msg)
	}

	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *tuiModel) handleKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch m.state {
	case stApprove:
		switch k.String() {
		case "y", "Y":
			m.resolveApproval(true)
		case "n", "N", "esc", "enter":
			m.resolveApproval(false)
		}
		return m, nil

	case stRunning:
		if s := k.String(); s == "ctrl+c" || s == "esc" {
			if m.cancel != nil {
				m.cancel()
			}
			m.pushLine(cDim.Render("  …cancelling"))
		}
		return m, nil

	default: // idle
		switch k.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "enter":
			line := strings.TrimSpace(m.input.Value())
			m.input.SetValue("")
			if line == "" {
				return m, nil
			}
			return m.submit(line)
		case "up", "down", "pgup", "pgdown", "home", "end":
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(k)
			return m, cmd
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(k)
		return m, cmd
	}
}

func (m *tuiModel) resolveApproval(ok bool) {
	if m.pending != nil {
		m.pending.reply <- ok
		verdict := cErr.Render("denied")
		if ok {
			verdict = cOk.Render("allowed")
		}
		m.pushLine(cDim.Render("  → ") + verdict)
		m.pending = nil
	}
	m.state = stRunning
}

func (m *tuiModel) submit(line string) (tea.Model, tea.Cmd) {
	if strings.HasPrefix(line, "/") {
		return m.runCommand(line)
	}
	return m.startTask(line)
}

func (m *tuiModel) startTask(goal string) (tea.Model, tea.Cmd) {
	m.pushLine(cYou.Render("you › ") + goal)
	return m, m.runGoals(1, goal)
}

func (m *tuiModel) runCommand(line string) (tea.Model, tea.Cmd) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/help", "/?":
		m.pushBlock(helpText())
	case "/status":
		m.pushBlock(m.app.statusText())
	case "/usage":
		m.pushBlock(m.app.usageText())
	case "/login":
		if err := m.app.reconfigure(); err != nil {
			m.pushLine(cErr.Render("reload failed: " + err.Error()))
		} else {
			m.pushLine(cDim.Render("config reloaded from " + config.GlobalPath() + " — edit it or run `ipsupport-code -init` to change the connection"))
		}
	case "/goal":
		if rest == "" {
			m.pushLine(cDim.Render("usage: /goal <task>"))
			return m, nil
		}
		return m.startTask(rest)
	case "/loop":
		n, goal := parseLoop(rest)
		if goal == "" {
			m.pushLine(cDim.Render("usage: /loop [count] <task>"))
			return m, nil
		}
		m.pushLine(cYou.Render(fmt.Sprintf("you › /loop %d ", n)) + goal)
		return m, m.runGoals(n, goal)
	case "/exit", "/quit":
		return m, tea.Quit
	default:
		m.pushLine(cDim.Render("unknown command " + cmd + " — /help"))
	}
	return m, nil
}

// runGoals runs a goal n times in a background goroutine, streaming progress via
// the bridge and returning taskDoneMsg when finished.
func (m *tuiModel) runGoals(n int, goal string) tea.Cmd {
	m.state = stRunning
	m.input.Blur()
	tctx, cancel := context.WithCancel(m.ctx)
	m.cancel = cancel
	return func() tea.Msg {
		defer cancel()
		for i := 0; i < n; i++ {
			if tctx.Err() != nil {
				break
			}
			if n > 1 {
				m.bridge.Emit("loop", map[string]any{"i": i + 1, "n": n})
			}
			m.app.runTaskStreaming(tctx, goal)
		}
		return taskDoneMsg{}
	}
}

func (m *tuiModel) View() string {
	if !m.ready {
		return "loading…"
	}
	var status string
	switch m.state {
	case stRunning:
		status = m.spin.View() + cDim.Render(" working… (esc to cancel)")
	case stApprove:
		status = cToolCall.Render("approve? press y / n")
	default:
		status = cDim.Render("ready")
	}
	p, c := m.app.client.Usage()
	bar := cDim.Render(fmt.Sprintf("%s · %s · %d tok", m.app.cfg.LLM.Model, filepath.Base(m.app.workspace), p+c))
	bottom := m.input.View()
	if m.state != stIdle {
		bottom = cDim.Render("(input disabled while running)")
	}
	return m.vp.View() + "\n" + status + "\n" + bar + "\n" + bottom
}

func (m *tuiModel) viewportHeight() int {
	if h := m.height - 4; h > 0 {
		return h
	}
	return 1
}

func (m *tuiModel) pushLine(s string) {
	m.history = append(m.history, s)
	m.refresh()
}

func (m *tuiModel) pushBlock(s string) {
	for _, ln := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		m.history = append(m.history, cDim.Render(ln))
	}
	m.refresh()
}

func (m *tuiModel) refresh() {
	if !m.ready {
		return
	}
	m.vp.SetContent(strings.Join(m.history, "\n"))
	m.vp.GotoBottom()
}

// eventLines renders one streamed agent event into styled history lines.
func eventLines(e uiEvent) []string {
	switch e.kind {
	case "assistant":
		if c, _ := e.fields["content"].(string); strings.TrimSpace(c) != "" {
			return []string{cBot.Render(c)}
		}
	case "tool_call":
		t, _ := e.fields["tool"].(string)
		a, _ := e.fields["action"].(string)
		return []string{cToolCall.Render("  ⚙ "+t+" "+a) + cDim.Render(" "+compactJSON(e.fields["params"]))}
	case "observation":
		isErr, _ := e.fields["is_error"].(bool)
		c, _ := e.fields["content"].(string)
		if isErr {
			return []string{cErr.Render("  ✖ " + firstLine(c))}
		}
		return []string{cDim.Render("  → ") + cOk.Render(firstLine(c))}
	case "final":
		if c, _ := e.fields["text"].(string); strings.TrimSpace(c) != "" {
			return []string{"", cFinal.Render(c)}
		}
	case "lesson":
		d, _ := e.fields["domain"].(string)
		f, _ := e.fields["proven_fix"].(string)
		return []string{cLesson.Render("  ✦ learned ["+d+"] ") + cDim.Render(f)}
	case "loop":
		return []string{cDim.Render(fmt.Sprintf("— loop %d/%d —", toInt(e.fields["i"]), toInt(e.fields["n"])))}
	case "error":
		c, _ := e.fields["text"].(string)
		return []string{cErr.Render("error: " + c)}
	}
	return nil
}

func compactJSON(v any) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	s := string(b)
	if s == "{}" || s == "null" {
		return ""
	}
	if len(s) > 80 {
		s = s[:80] + "…"
	}
	return s
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 200 {
		s = s[:200] + "…"
	}
	return s
}

func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	return 0
}
