package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

type uiState int

const (
	stIdle uiState = iota
	stRunning
	stApprove
)

// tuiModel runs in INLINE mode (no alt-screen): the transcript is printed into
// the terminal's normal scrollback via tea.Println — so native scrolling and
// Ctrl-L keep working — and only the bottom region (divider, status, input box,
// status bar) is the live, redrawn View.
type tuiModel struct {
	app    *app
	ctx    context.Context
	bridge *uiBridge

	input textinput.Model
	spin  spinner.Model

	state         uiState
	pending       *approvalReq
	cancel        context.CancelFunc
	width, height int
}

// Bubble Tea messages.
type eventMsg uiEvent
type approvalMsg approvalReq
type taskDoneMsg struct{}

// runTUI installs the UI bridge as the agent's tracer + approver, then runs the
// Bubble Tea program in inline mode.
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
	_, err := tea.NewProgram(m, tea.WithContext(ctx)).Run()
	return err
}

func (m *tuiModel) Init() tea.Cmd {
	banner := cTitle.Render("ipsupport-code") + cDim.Render("  — task or /help · ctrl-c quit · ctrl-l clear")
	return tea.Batch(m.spin.Tick, textinput.Blink, m.waitEvent(), m.waitApproval(), tea.Println(banner))
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
		m.input.Width = msg.Width - 6
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case eventMsg:
		return m, tea.Batch(m.print(eventLines(uiEvent(msg))...), m.waitEvent())

	case approvalMsg:
		req := approvalReq(msg)
		m.pending = &req
		m.state = stApprove
		line := cToolCall.Render("  ⚠ approve "+req.kind+": ") + req.detail + cDim.Render("  [y/N]")
		return m, tea.Batch(m.print(line), m.waitApproval())

	case taskDoneMsg:
		m.state = stIdle
		m.cancel = nil
		return m, m.input.Focus()

	case tea.KeyMsg:
		return m.handleKey(msg)
	}
	return m, nil
}

func (m *tuiModel) handleKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	if k.String() == "ctrl+l" {
		return m, tea.ClearScreen
	}
	switch m.state {
	case stApprove:
		switch k.String() {
		case "y", "Y":
			return m, m.resolveApproval(true)
		case "n", "N", "esc", "enter":
			return m, m.resolveApproval(false)
		}
		return m, nil

	case stRunning:
		if s := k.String(); s == "ctrl+c" || s == "esc" {
			if m.cancel != nil {
				m.cancel()
			}
			return m, m.print(cDim.Render("  …cancelling"))
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
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(k)
		return m, cmd
	}
}

func (m *tuiModel) resolveApproval(ok bool) tea.Cmd {
	m.state = stRunning
	if m.pending == nil {
		return nil
	}
	m.pending.reply <- ok
	m.pending = nil
	verdict := cErr.Render("denied")
	if ok {
		verdict = cOk.Render("allowed")
	}
	return m.print(cDim.Render("  → ") + verdict)
}

func (m *tuiModel) submit(line string) (tea.Model, tea.Cmd) {
	if strings.HasPrefix(line, "/") {
		return m.runCommand(line)
	}
	return m, tea.Batch(m.print(cYou.Render("you › ")+line), m.runGoals(1, line))
}

func (m *tuiModel) runCommand(line string) (tea.Model, tea.Cmd) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/help", "/?":
		return m, m.printBlock(helpText())
	case "/status":
		return m, m.printBlock(m.app.statusText())
	case "/usage":
		return m, m.printBlock(m.app.usageText())
	case "/login":
		if err := m.app.reconfigure(); err != nil {
			return m, m.print(cErr.Render("reload failed: " + err.Error()))
		}
		return m, m.print(cDim.Render("config reloaded from " + config.GlobalPath() + " — edit it or run `ipsupport-code -init` to change the connection"))
	case "/new", "/reset":
		m.app.ag.Reset()
		return m, m.print(cDim.Render("session cleared"))
	case "/goal":
		if rest == "" {
			return m, m.print(cDim.Render("usage: /goal <task>"))
		}
		return m, tea.Batch(m.print(cYou.Render("you › ")+rest), m.runGoals(1, rest))
	case "/loop":
		n, goal := parseLoop(rest)
		if goal == "" {
			return m, m.print(cDim.Render("usage: /loop [count] <task>"))
		}
		return m, tea.Batch(m.print(cYou.Render(fmt.Sprintf("you › /loop %d ", n))+goal), m.runGoals(n, goal))
	case "/exit", "/quit":
		return m, tea.Quit
	default:
		return m, m.print(cDim.Render("unknown command " + cmd + " — /help"))
	}
}

// runGoals sets running state and returns a cmd that runs the goal n times in the
// background, streaming via the bridge and ending with taskDoneMsg.
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
	width := m.width
	if width < 1 {
		width = 60
	}
	var status string
	switch m.state {
	case stRunning:
		status = m.spin.View() + cDim.Render(" working… (esc to cancel)")
	case stApprove:
		status = cToolCall.Render("approve? press y / n")
	default:
		status = cDim.Render("ready · /help for commands")
	}

	field := m.input.View()
	if m.state != stIdle {
		field = cDim.Render("(input disabled while a task runs)")
	}
	box := cBox.Width(width - 4).Render(field)

	p, c := m.app.client.Usage()
	bar := cDim.Render(fmt.Sprintf("%s · %s · %d tok", m.app.cfg.LLM.Model, filepath.Base(m.app.workspace), p+c))
	divider := cDim.Render(strings.Repeat("─", width))

	return strings.Join([]string{divider, status, box, bar}, "\n")
}

// print emits transcript lines into the terminal's scrollback (above the live
// region). Multiple lines are joined so they print atomically and in order.
func (m *tuiModel) print(lines ...string) tea.Cmd {
	if len(lines) == 0 {
		return nil
	}
	return tea.Println(strings.Join(lines, "\n"))
}

func (m *tuiModel) printBlock(s string) tea.Cmd {
	var lines []string
	for _, ln := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		lines = append(lines, cDim.Render(ln))
	}
	return m.print(lines...)
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
