package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/alecthomas/chroma/v2/quick"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

type uiState int

const (
	stIdle uiState = iota
	stRunning
	stApprove
)

// chromeRows: status line + top rule + input + bottom rule + hint line.
const chromeRows = 5

// tuiModel is a full-screen coding-agent UI: a scrollable log fills the screen,
// and a bottom region (a titled top rule, the input line, a bottom rule, and a
// hint) holds the chat. The input stays live while a task runs — Enter queues
// type-ahead. Scrolling is in-app (↑↓ / PgUp/PgDn; iTerm maps the wheel to
// arrows in alt-screen); Ctrl-L clears the log.
type tuiModel struct {
	app    *app
	ctx    context.Context
	bridge *uiBridge

	vp    viewport.Model
	input textinput.Model
	spin  spinner.Model

	state         uiState
	history       []string
	queued        []string // type-ahead while a task runs
	pending       *approvalReq
	cancel        context.CancelFunc
	taskStart     time.Time
	startTok      int
	accent        lipgloss.Color
	accentIdx     int
	width, height int
	ready         bool
}

// Bubble Tea messages.
type eventMsg uiEvent
type approvalMsg approvalReq
type taskDoneMsg struct{}

// newTUIModel installs the UI bridge as the agent's tracer + approver, wires the
// stack, and builds the model. Split out from runTUI so tests can drive it.
func (a *app) newTUIModel(ctx context.Context) (*tuiModel, error) {
	b := newBridge()
	a.uiTracer = b
	a.approver = b
	if err := a.wire(); err != nil {
		return nil, err
	}

	in := textinput.New()
	in.Placeholder = "type a task, or /help"
	in.Prompt = "❯ "
	in.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Points
	sp.Style = cToolCall

	m := &tuiModel{app: a, ctx: ctx, bridge: b, input: in, spin: sp, state: stIdle, accent: lipgloss.Color("13")}
	m.history = []string{cDim.Render("ipsupport-code — type a task, or /help. ctrl-l clear · ↑↓ scroll · ctrl-c quit")}
	return m, nil
}

func (a *app) runTUI(ctx context.Context) error {
	m, err := a.newTUIModel(ctx)
	if err != nil {
		return err
	}
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithContext(ctx)).Run()
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
		h := m.viewportHeight()
		if !m.ready {
			m.vp = viewport.New(msg.Width, h)
			m.ready = true
		} else {
			m.vp.Width = msg.Width
			m.vp.Height = h
		}
		m.input.Width = msg.Width - 4
		m.vp.SetContent(strings.Join(m.history, "\n"))
		m.vp.GotoBottom()
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case eventMsg:
		m.push(m.renderEvent(uiEvent(msg))...)
		return m, m.waitEvent()

	case approvalMsg:
		req := approvalReq(msg)
		m.pending = &req
		m.state = stApprove
		m.push(cToolCall.Render("  ⚠ approve "+req.kind+": ") + req.detail + cDim.Render("  [y/N]"))
		return m, m.waitApproval()

	case taskDoneMsg:
		m.state = stIdle
		m.cancel = nil
		if len(m.queued) > 0 { // run the next type-ahead
			next := m.queued[0]
			m.queued = m.queued[1:]
			m.push(cYou.Render("❯ ") + next)
			return m, m.runGoals(1, next)
		}
		return m, m.input.Focus()

	case tea.KeyMsg:
		return m.handleKey(msg)
	}

	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *tuiModel) handleKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch k.String() {
	case "ctrl+c":
		return m, tea.Quit
	case "ctrl+l":
		m.history = m.history[:0]
		if m.ready {
			m.vp.SetContent("")
		}
		return m, nil
	}

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
		switch k.String() {
		case "esc":
			if m.cancel != nil {
				m.cancel()
			}
			m.push(cDim.Render("  …cancelling"))
			return m, nil
		case "enter":
			if line := strings.TrimSpace(m.input.Value()); line != "" {
				m.input.SetValue("")
				m.queued = append(m.queued, line)
				m.push(cDim.Render("queued ❯ ") + line)
			}
			return m, nil
		case "up", "down", "pgup", "pgdown", "home", "end":
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(k)
			return m, cmd
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(k) // live typing while it thinks
		return m, cmd

	default: // idle
		switch k.String() {
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
	m.state = stRunning
	if m.pending == nil {
		return
	}
	m.pending.reply <- ok
	m.pending = nil
	verdict := cErr.Render("denied")
	if ok {
		verdict = cOk.Render("allowed")
	}
	m.push(cDim.Render("  → ") + verdict)
}

func (m *tuiModel) submit(line string) (tea.Model, tea.Cmd) {
	if strings.HasPrefix(line, "/") {
		return m.runCommand(line)
	}
	m.push(cYou.Render("❯ ") + line)
	return m, m.runGoals(1, line)
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
			m.push(cErr.Render("reload failed: " + err.Error()))
		} else {
			m.push(cDim.Render("config reloaded from " + config.GlobalPath() + " — edit it or run `ipsupport-code -init` to change the connection"))
		}
	case "/new", "/reset":
		m.app.ag.Reset()
		m.push(cDim.Render("session cleared"))
	case "/color":
		m.setColor(rest)
	case "/goal":
		if rest == "" {
			m.push(cDim.Render("usage: /goal <task>"))
			return m, nil
		}
		m.push(cYou.Render("❯ ") + rest)
		return m, m.runGoals(1, rest)
	case "/loop":
		n, goal := parseLoop(rest)
		if goal == "" {
			m.push(cDim.Render("usage: /loop [count] <task>"))
			return m, nil
		}
		m.push(cYou.Render(fmt.Sprintf("❯ /loop %d ", n)) + goal)
		return m, m.runGoals(n, goal)
	case "/exit", "/quit":
		return m, tea.Quit
	default:
		m.push(cDim.Render("unknown command " + cmd + " — /help"))
	}
	return m, nil
}

// runGoals sets running state and returns a cmd that runs the goal n times in the
// background, streaming via the bridge and ending with taskDoneMsg.
func (m *tuiModel) runGoals(n int, goal string) tea.Cmd {
	m.state = stRunning
	m.taskStart = time.Now()
	p, c := m.app.client.Usage()
	m.startTok = p + c
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
	p, c := m.app.client.Usage()
	total := p + c

	var status string
	switch m.state {
	case stRunning:
		elapsed := time.Since(m.taskStart).Truncate(time.Second)
		status = m.spin.View() + cToolCall.Render(fmt.Sprintf(" Thinking… (%s · ↓%s tok)", elapsed, humanK(total-m.startTok)))
	case stApprove:
		status = cToolCall.Render("⚠ approve? press y / n")
	default:
		status = cDim.Render(fmt.Sprintf("%s · %s · %d tok · ready", m.app.cfg.LLM.Model, filepath.Base(m.app.workspace), total))
	}

	hint := cDim.Render("/help · ctrl-l clear · ↑↓ scroll · ctrl-c quit")
	if m.state == stRunning {
		hint = cDim.Render("typing is live — Enter queues · esc cancels · ↑↓ scroll")
	}

	frame := lipgloss.NewStyle().Foreground(m.accent)
	return strings.Join([]string{
		m.vp.View(),
		status,
		m.topRule(frame),
		m.input.View(),
		frame.Render(strings.Repeat("─", m.width)),
		hint,
	}, "\n")
}

// topRule draws "────…──── ipsupport-code ──" with the label pinned right, in
// the current accent color.
func (m *tuiModel) topRule(frame lipgloss.Style) string {
	const label = "ipsupport-code"
	right := " " + label + " ──"
	n := m.width - lipgloss.Width(right)
	if n < 0 {
		n = 0
	}
	return frame.Render(strings.Repeat("─", n)+" ") + frame.Bold(true).Render(label) + frame.Render(" ──")
}

// setColor changes the frame accent: a name, a raw 0-255 code, or cycle on empty.
func (m *tuiModel) setColor(arg string) {
	arg = strings.ToLower(strings.TrimSpace(arg))
	switch {
	case arg == "":
		m.accentIdx = (m.accentIdx + 1) % len(colorCycle)
		m.accent = lipgloss.Color(colorCycle[m.accentIdx])
	case colorNames[arg] != "":
		m.accent = lipgloss.Color(colorNames[arg])
	default:
		m.accent = lipgloss.Color(arg) // raw ANSI 256 code
	}
	m.push(lipgloss.NewStyle().Foreground(m.accent).Render("frame color → " + string(m.accent)))
}

func (m *tuiModel) viewportHeight() int {
	if h := m.height - chromeRows - 1; h > 0 { // -1: status row sits above the rule
		return h
	}
	return 1
}

// push appends styled lines to the log, auto-scrolling to the bottom only if the
// user was already there (so scrolling up to read isn't interrupted).
func (m *tuiModel) push(lines ...string) {
	if len(lines) == 0 {
		return
	}
	atBottom := !m.ready || m.vp.AtBottom()
	m.history = append(m.history, lines...)
	if m.ready {
		m.vp.SetContent(strings.Join(m.history, "\n"))
		if atBottom {
			m.vp.GotoBottom()
		}
	}
}

func (m *tuiModel) pushBlock(s string) {
	for _, ln := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		m.push(cDim.Render(ln))
	}
}

// renderEvent renders one streamed agent event into styled history lines.
func (m *tuiModel) renderEvent(e uiEvent) []string {
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
		if tool, _ := e.fields["tool"].(string); tool == "file" {
			if action, _ := e.fields["action"].(string); action == "read" {
				return renderCode(c) // syntax-highlight file reads
			}
		}
		return []string{cDim.Render("  → ") + cOk.Render(firstLine(c))}
	case "diff":
		path, _ := e.fields["path"].(string)
		d, _ := e.fields["diff"].(string)
		return m.renderDiff(path, d)
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

// renderDiff renders a unified diff like a code host: `● Update(path)` + a
// summary, then only the changed hunks (±3 context lines). Added rows get a
// green background WITH chroma syntax colors; removed rows a red background with
// plain white text; both fill the row width so the gutter is highlighted too.
func (m *tuiModel) renderDiff(path, diff string) []string {
	width := m.width
	if width < 1 {
		width = 80
	}
	add, del := 0, 0
	oldNo, newNo := 0, 0
	var body []string
	for _, ln := range strings.Split(strings.TrimRight(diff, "\n"), "\n") {
		switch {
		case strings.HasPrefix(ln, "+++"), strings.HasPrefix(ln, "---"):
			continue
		case strings.HasPrefix(ln, "@@"):
			oldNo, newNo = parseHunk(ln)
		case strings.HasPrefix(ln, "+"):
			add++
			body = append(body, diffAddRow(newNo, ln[1:], width))
			newNo++
		case strings.HasPrefix(ln, "-"):
			del++
			body = append(body, diffDelRow(oldNo, ln[1:], width))
			oldNo++
		default:
			body = append(body, diffCtx.Render(fmt.Sprintf(" %4d    %s", newNo, strings.TrimPrefix(ln, " "))))
			oldNo++
			newNo++
		}
	}
	h1 := lipgloss.NewStyle().Foreground(m.accent).Render("● ") + lipgloss.NewStyle().Bold(true).Render("Update("+path+")")
	h2 := cDim.Render(fmt.Sprintf("  ⎿  Added %d %s, removed %d %s", add, plural(add, "line"), del, plural(del, "line")))
	return append([]string{h1, h2}, body...)
}

// ANSI control sequences for building diff rows by hand (so a full-row
// background coexists with chroma's per-token foreground colors).
const (
	bgGreen = "\x1b[48;5;22m"
	bgRed   = "\x1b[48;5;52m"
	fgWhite = "\x1b[97m"
	fgReset = "\x1b[39m" // reset foreground only — keeps our background
	allOff  = "\x1b[0m"
)

// ansiBG matches background SGR sequences so chroma's own backgrounds can be
// stripped, letting our green/red show through.
var ansiBG = regexp.MustCompile("\x1b\\[(4[0-9]|48;5;[0-9]+|49)m")

func diffAddRow(no int, code string, width int) string {
	hl := ansiBG.ReplaceAllString(highlightCode(code), "")
	hl = strings.ReplaceAll(strings.TrimRight(hl, "\r\n"), allOff, fgReset)
	content := fmt.Sprintf(" %4d +  ", no) + hl
	if pad := width - lipgloss.Width(content); pad > 0 {
		content += strings.Repeat(" ", pad)
	}
	return bgGreen + content + allOff
}

func diffDelRow(no int, code string, width int) string {
	content := fmt.Sprintf(" %4d -  %s", no, code)
	if pad := width - lipgloss.Width(content); pad > 0 {
		content += strings.Repeat(" ", pad)
	}
	return bgRed + fgWhite + content + allOff
}

func plural(n int, w string) string {
	if n == 1 {
		return w
	}
	return w + "s"
}

// renderCode syntax-highlights a file's content (capped) for a file.read result.
func renderCode(content string) []string {
	lines := strings.Split(content, "\n")
	capped := false
	if len(lines) > 40 {
		lines, capped = lines[:40], true
	}
	out := []string{cDim.Render("  → read:")}
	for _, ln := range strings.Split(highlightCode(strings.Join(lines, "\n")), "\n") {
		out = append(out, "    "+ln)
	}
	if capped {
		out = append(out, cDim.Render("    …"))
	}
	return out
}

func highlightCode(code string) string {
	var b strings.Builder
	if err := quick.Highlight(&b, code, "", "terminal256", "github-dark"); err != nil {
		return code
	}
	return b.String()
}

func parseHunk(s string) (oldStart, newStart int) {
	var a, bb, c, d int
	if n, _ := fmt.Sscanf(s, "@@ -%d,%d +%d,%d @@", &a, &bb, &c, &d); n >= 3 {
		return a, c
	}
	a, c = 0, 0
	fmt.Sscanf(s, "@@ -%d +%d @@", &a, &c)
	return a, c
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

func humanK(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
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
