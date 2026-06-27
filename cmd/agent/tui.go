package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
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
	approveChoice bool // selected Yes(true)/No(false) while answering an approval
	cancel        context.CancelFunc
	taskStart     time.Time
	startTok      int
	retry         *retryInfo
	accent        lipgloss.Color
	accentIdx     int
	width, height int
	ready         bool

	// model-proposed, Tab-acceptable next-step suggestion (parsed from NEXT:)
	suggestion string
}

const defaultPlaceholder = "type a task, or /help"

// retryInfo tracks an in-progress LLM backoff so the UI can show it.
type retryInfo struct {
	attempt int
	until   time.Time
}

// Bubble Tea messages.
type eventMsg uiEvent
type approvalMsg approvalReq
type taskDoneMsg struct{}
type compactDoneMsg struct {
	n   int
	err error
}
type skillsMsg struct {
	names []string
	err   error
}

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
	in.Placeholder = defaultPlaceholder
	in.Prompt = "❯ "
	in.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Points
	sp.Style = cToolCall

	name := a.cfg.Name
	if name == "" {
		name = "ipsupport-code"
	}
	m := &tuiModel{app: a, ctx: ctx, bridge: b, input: in, spin: sp, state: stIdle, accent: lipgloss.Color("13")}
	m.history = bannerLines(name, a.cfg.LLM.Model, a.workspace, a.cfg.LLM.ContextWindow, m.accent)
	return m, nil
}

// bannerLines builds the Claude-Code-style startup card: a rounded box with the
// agent name, model, working directory, and detected context window, then a
// one-line key hint.
func bannerLines(name, model, cwd string, window int, accent lipgloss.Color) []string {
	if h, err := os.UserHomeDir(); err == nil && h != "" && strings.HasPrefix(cwd, h) {
		cwd = "~" + cwd[len(h):]
	}
	label := lipgloss.NewStyle().Bold(true).Foreground(accent)
	rows := []string{
		label.Render("✦ " + name),
		cDim.Render("model  ") + cBot.Render(model),
		cDim.Render("cwd    ") + cBot.Render(cwd),
	}
	if window > 0 {
		rows = append(rows, cDim.Render("ctx    ")+cBot.Render(humanK(window)+" tokens"))
	}
	body := strings.Join(rows, "\n")
	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(accent).
		Padding(0, 2).
		Render(body)
	hint := cDim.Render("type a task, or /help · ctrl-l clear · ↑↓ scroll · ctrl-c quit")
	return append(strings.Split(box, "\n"), "", hint)
}

func (a *app) runTUI(ctx context.Context) error {
	m, err := a.newTUIModel(ctx)
	if err != nil {
		return err
	}
	// WithMouseCellMotion lets the wheel scroll the log in the alt-screen (without
	// it the terminal swallows the wheel and nothing moves).
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion(), tea.WithContext(ctx)).Run()
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
			m.vp.MouseWheelEnabled = true
			m.ready = true
		} else {
			m.vp.Width = msg.Width
			m.vp.Height = h
		}
		m.input.Width = msg.Width - 4
		m.vp.SetContent(m.renderContent())
		m.vp.GotoBottom()
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case eventMsg:
		e := uiEvent(msg)
		if e.kind == "retry" {
			wait := time.Duration(toInt(e.fields["wait_ms"])) * time.Millisecond
			m.retry = &retryInfo{attempt: toInt(e.fields["attempt"]), until: time.Now().Add(wait)}
			m.push(cErr.Render(fmt.Sprintf("  ⟳ server hiccup — retry %d, backing off %s", m.retry.attempt, wait)))
		} else {
			m.retry = nil // progress resumed
			m.push(m.renderEvent(e)...)
			if e.kind == "final" {
				if sug, _ := e.fields["suggest"].(string); strings.TrimSpace(sug) != "" {
					m.setSuggestion(strings.TrimSpace(sug))
				}
			}
		}
		return m, m.waitEvent()

	case approvalMsg:
		req := approvalReq(msg)
		m.pending = &req
		// Don't steal the keyboard: stay running so the input remains editable
		// (finish your message). The user presses ↑ to switch to answering. Do NOT
		// fetch the next approval yet — that would overwrite m.pending and orphan
		// this one's reply channel (a hang when the model batches calls).
		m.push(cToolCall.Render("  ⚠ approve "+req.kind+": ") + req.detail + cDim.Render("  — press ↑ to answer"))
		return m, nil

	case taskDoneMsg:
		m.state = stIdle
		m.cancel = nil
		m.retry = nil
		if len(m.queued) > 0 { // run the next type-ahead
			next := m.queued[0]
			m.queued = m.queued[1:]
			m.syncViewport() // it left the pinned queue
			m.push(cYou.Render("❯ ") + next)
			return m, m.runGoals(1, next)
		}
		if m.app.shouldAutoCompact() { // context near the limit — fold it down
			return m, m.startCompact(true)
		}
		return m, m.input.Focus()

	case compactDoneMsg:
		m.state = stIdle
		switch {
		case msg.err != nil:
			m.push(cErr.Render("compact failed: " + msg.err.Error()))
		case msg.n == 0:
			m.push(cDim.Render("nothing to compact"))
		default:
			m.app.saveSession()
			m.push(cDim.Render(fmt.Sprintf("compacted %d messages → summary", msg.n)))
		}
		return m, m.input.Focus()

	case skillsMsg:
		m.state = stIdle
		if msg.err != nil {
			m.push(cErr.Render("install failed: " + msg.err.Error()))
		} else {
			_ = m.app.wire() // register the skill tool + index on the UI thread (no race)
			m.push(cDim.Render("installed & enabled: " + strings.Join(msg.names, ", ")))
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
	case "shift+tab":
		m.app.setMode(!m.app.planMode) // cycle plan/auto; the bottom indicator updates
		return m, nil
	}

	// A pending approval doesn't grab the keyboard — keep typing. Press ↑ to
	// switch to answering it (your input text is preserved).
	if m.pending != nil && m.state != stApprove && k.String() == "up" {
		m.state = stApprove
		m.approveChoice = true
		return m, nil
	}

	switch m.state {
	case stApprove:
		// Answer the prompt, then (and only then) wait for the next queued
		// approval — keeps exactly one reader, so none get overwritten.
		switch k.String() {
		case "left", "right", "up", "down", "tab":
			m.approveChoice = !m.approveChoice // toggle Yes/No
			return m, nil
		case "y", "Y":
			m.resolveApproval(true)
			return m, m.waitApproval()
		case "n", "N":
			m.resolveApproval(false)
			return m, m.waitApproval()
		case "enter":
			m.resolveApproval(m.approveChoice)
			return m, m.waitApproval()
		case "esc":
			m.state = stRunning // back to typing; the approval stays pending
			return m, nil
		}
		return m, nil // ignore other keys; keep showing the prompt

	case stRunning:
		switch k.String() {
		case "esc":
			if m.cancel != nil {
				m.cancel()
			}
			m.bridge.Abort() // deny any in-flight approvals so no tool goroutine hangs
			m.push(cDim.Render("  …cancelling"))
			if m.pending != nil {
				// The shown approval consumed the single reader; re-arm one so the
				// next task's approvals are still read.
				m.pending = nil
				return m, m.waitApproval()
			}
			return m, nil
		case "enter":
			line := strings.TrimSpace(m.input.Value())
			if line == "" {
				return m, nil
			}
			m.input.SetValue("")
			if strings.HasPrefix(line, "/") {
				return m.commandWhileBusy(line)
			}
			m.queued = append(m.queued, line) // type-ahead: run after the task
			m.syncViewport()                  // show it pinned above the input
			return m, nil
		case "up":
			// Changed your mind about a queued message? With an empty input, Up
			// pulls the last one back to edit (Enter re-queues it; clear to drop).
			if strings.TrimSpace(m.input.Value()) == "" && len(m.queued) > 0 {
				last := m.queued[len(m.queued)-1]
				m.queued = m.queued[:len(m.queued)-1]
				m.input.SetValue(last)
				m.input.CursorEnd()
				m.syncViewport() // the pinned queue shrank
				return m, nil
			}
			fallthrough
		case "down", "pgup", "pgdown", "home", "end":
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
		case "tab":
			switch {
			case strings.HasPrefix(m.input.Value(), "/"):
				m.completeCommand()
			case m.input.Value() == "" && m.suggestion != "":
				m.input.SetValue(m.suggestion)
				m.input.CursorEnd()
				m.suggestion = ""
				m.input.Placeholder = defaultPlaceholder
			}
			return m, nil
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
		m.push(m.renderHelp()...)
	case "/status":
		m.push(m.renderStatus()...)
	case "/usage":
		m.push(m.renderUsage()...)
	case "/login":
		if err := m.app.reconfigure(); err != nil {
			m.push(cErr.Render("reload failed: " + err.Error()))
		} else {
			m.push(cDim.Render("config reloaded from " + config.GlobalPath() + " — edit it or run `ipsupport-code -init` to change the connection"))
		}
	case "/new", "/reset":
		m.app.ag.Reset()
		m.app.saveSession()
		m.push(cDim.Render("session cleared"))
	case "/clear":
		m.app.ag.Reset()
		m.app.saveSession()
		m.history = m.history[:0]
		if m.ready {
			m.vp.SetContent("")
		}
		m.push(cDim.Render("cleared — fresh screen and session"))
	case "/compact":
		return m, m.startCompact(false)
	case "/plan":
		m.push(cDim.Render(m.app.setMode(true)))
	case "/auto":
		m.push(cDim.Render(m.app.setMode(false)))
	case "/skills":
		return m.skillsCmd(rest)
	case "/permissions", "/perms":
		m.pushLines(m.app.permissionsCommand(rest))
		return m, nil
	case "/color":
		m.setColor(rest)
	case "/rename":
		m.rename(rest)
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

// skillsCmd runs a /skills subcommand. install hits the network, so it runs off
// the UI thread (like /compact) and re-wires on the result; the rest are instant
// local filesystem ops.
func (m *tuiModel) skillsCmd(rest string) (tea.Model, tea.Cmd) {
	if m.app.skills == nil {
		m.push(cDim.Render("skills unavailable"))
		return m, nil
	}
	if sub, arg := splitCommand(rest); sub == "install" || sub == "add" {
		if strings.TrimSpace(arg) == "" {
			m.push(cDim.Render("usage: /skills install <url|git>"))
			return m, nil
		}
		m.state = stRunning
		m.taskStart = time.Now()
		ctx, src := m.ctx, arg
		m.push(cDim.Render("installing " + src + " …"))
		return m, func() tea.Msg {
			names, err := m.app.skills.Install(ctx, src)
			return skillsMsg{names: names, err: err}
		}
	}
	m.pushLines(m.app.skillsCommand(m.ctx, rest))
	return m, nil
}

// pushLines appends plain command-output lines to the log, dimmed.
func (m *tuiModel) pushLines(lines []string) {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = cDim.Render(l)
	}
	m.push(out...)
}

// commandWhileBusy handles a /command typed while a task is running: exit quits
// now, read-only commands run now, /goal queues the next task, everything else
// waits — so you never have to sit through "thinking" just to leave or peek.
func (m *tuiModel) commandWhileBusy(line string) (tea.Model, tea.Cmd) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/exit", "/quit":
		return m, tea.Quit
	case "/status", "/usage", "/help", "/?", "/color":
		return m.runCommand(line) // read-only — doesn't touch the running task
	case "/goal":
		if rest == "" {
			m.push(cDim.Render("usage: /goal <task>"))
			return m, nil
		}
		m.queued = append(m.queued, rest)
		m.syncViewport()
		return m, nil
	default:
		m.push(cDim.Render("busy — " + cmd + " will run once the current task finishes"))
		return m, nil
	}
}

// rename changes the display name and persists it to the user config.
func (m *tuiModel) rename(name string) {
	if name = strings.TrimSpace(name); name == "" {
		m.push(cDim.Render("usage: /rename <new name>"))
		return
	}
	m.app.cfg.Name = name
	if err := config.SaveGlobal(name, m.app.cfg.LLM); err != nil {
		m.push(cErr.Render("could not save name: " + err.Error()))
		return
	}
	m.push(m.accentBold().Render("renamed → " + name))
}

// setSuggestion shows a model-proposed next step (parsed from the final answer's
// NEXT: line) as Tab-acceptable ghost text plus a visible hint line.
func (m *tuiModel) setSuggestion(text string) {
	m.suggestion = text
	m.input.Placeholder = text + "   (Tab to accept)"
	m.push(cDim.Render("  💡 next: ") + m.accentBold().Render(text) + cDim.Render("   (Tab)"))
}

// startCompact folds the session into a summary in the background (manual via
// /compact, or auto when the context nears the limit), ending with
// compactDoneMsg.
func (m *tuiModel) startCompact(auto bool) tea.Cmd {
	m.state = stRunning
	m.taskStart = time.Now()
	_, c := m.app.client.Usage()
	m.startTok = c
	if auto {
		m.push(cDim.Render("  ⓘ context near limit — auto-compacting to free room"))
	}
	ctx := m.ctx
	return func() tea.Msg {
		n, err := m.app.ag.Compact(ctx)
		return compactDoneMsg{n: n, err: err}
	}
}

// runGoals sets running state and returns a cmd that runs the goal n times in the
// background, streaming via the bridge and ending with taskDoneMsg.
func (m *tuiModel) runGoals(n int, goal string) tea.Cmd {
	m.state = stRunning
	m.taskStart = time.Now()
	_, c := m.app.client.Usage()
	m.startTok = c // count completion (generated) tokens for this task
	m.suggestion = ""
	m.input.Placeholder = defaultPlaceholder
	m.bridge.arm() // fresh abort signal so a previous cancel doesn't deny this task's approvals
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
	_, c := m.app.client.Usage() // c = cumulative completion (generated) tokens

	var status string
	switch {
	case m.state == stApprove:
		status = m.approvePrompt()
	case m.pending != nil:
		// An approval is waiting but you can keep typing; ↑ switches to answering.
		status = cToolCall.Render("⚠ approval needed") + cDim.Render(" — press ↑ to answer")
	case m.state == stRunning:
		if m.retry != nil {
			remain := time.Until(m.retry.until).Truncate(100 * time.Millisecond)
			if remain < 0 {
				remain = 0
			}
			status = cErr.Render(fmt.Sprintf("⟳ retrying (attempt %d) — backing off, %s left", m.retry.attempt, remain))
		} else {
			elapsed := time.Since(m.taskStart).Truncate(time.Second)
			gen := c - m.startTok // completion tokens generated this task
			// Until the first token streams (the model is still reading the
			// prompt) show just the clock — a stuck "↑0 tok" reads as broken.
			detail := elapsed.String()
			if gen > 0 {
				detail = fmt.Sprintf("%s · ↑%s tok", elapsed, humanK(gen))
			}
			status = m.spin.View() + cToolCall.Render(fmt.Sprintf(" Thinking… (%s)", detail))
		}
	default:
		// ctx = size of the last prompt vs the window (auto-compacts as it fills);
		// ↑ = tokens the model generated this whole session.
		ctxStr := humanK(m.app.client.Context())
		if w := m.app.cfg.LLM.ContextWindow; w > 0 {
			ctxStr += "/" + humanK(w)
		}
		status = cDim.Render(fmt.Sprintf("%s · %s · ctx %s · ↑%s · ready",
			m.app.cfg.LLM.Model, filepath.Base(m.app.workspace), ctxStr, humanK(c)))
	}

	bottom := m.modeLine()
	switch {
	case m.state == stApprove:
		bottom = cDim.Render("  ←→ select · enter confirm · y/n shortcut · esc back to typing")
	case m.pending != nil:
		bottom = m.modeLine() + cDim.Render("  · ↑ to answer the approval")
	case m.state == stRunning:
		bottom += cDim.Render("  · esc cancels")
	}

	frame := lipgloss.NewStyle().Foreground(m.accent)
	parts := []string{m.vp.View(), status}
	parts = append(parts, m.queuedView()...) // pinned just above the input
	parts = append(parts, m.topRule(frame), m.input.View(), frame.Render(strings.Repeat("─", m.width)), bottom)
	return strings.Join(parts, "\n")
}

// approvePrompt renders the Yes/No selector shown while answering an approval.
func (m *tuiModel) approvePrompt() string {
	detail := ""
	if m.pending != nil {
		detail = m.pending.kind + " " + m.pending.detail
	}
	yes, no := cDim.Render("  Yes  "), cDim.Render("  No  ")
	if m.approveChoice {
		yes = cOk.Bold(true).Render("  ▸Yes  ")
	} else {
		no = cErr.Bold(true).Render("  ▸No  ")
	}
	return cToolCall.Render("⚠ approve "+detail+"  ") + yes + no
}

// modeLine is the bottom indicator: auto (executes) or plan (proposes only),
// cycled with shift+tab — the same affordance as Claude Code.
func (m *tuiModel) modeLine() string {
	if m.app.planMode {
		return "  " + m.accentBold().Render("⏸ plan mode on") + cDim.Render("  (shift+tab to cycle · read-only, it proposes)")
	}
	return "  " + cOk.Render("⏵⏵ auto mode on") + cDim.Render("  (shift+tab to cycle)")
}

// topRule draws "────…──── <name> ──" with the label pinned right, in the
// current accent color.
func (m *tuiModel) topRule(frame lipgloss.Style) string {
	label := m.app.cfg.Name
	if label == "" {
		label = "ipsupport-code"
	}
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
	// The pinned queue region (if any) eats into the log height so nothing
	// overflows the screen.
	if h := m.height - chromeRows - 1 - len(m.queuedView()); h > 0 {
		return h
	}
	return 1
}

// queuedView renders the type-ahead queue pinned just above the input, so
// waiting messages stay visible (and ↑-editable) instead of scrolling away in
// the log. Empty when nothing is queued.
func (m *tuiModel) queuedView() []string {
	if len(m.queued) == 0 {
		return nil
	}
	const max = 4
	out := []string{cDim.Render("  queued — runs after this · ↑ to edit or drop the last:")}
	for i, q := range m.queued {
		if i == max {
			out = append(out, cDim.Render(fmt.Sprintf("  … +%d more", len(m.queued)-max)))
			break
		}
		out = append(out, cYou.Render("  ⟳ ")+q)
	}
	return out
}

// syncViewport re-fits the log to the current size (the queue region changed how
// much room it has) and re-renders, keeping the bottom pinned if we were there.
func (m *tuiModel) syncViewport() {
	if !m.ready {
		return
	}
	atBottom := m.vp.AtBottom()
	m.vp.Height = m.viewportHeight()
	m.vp.SetContent(m.renderContent())
	if atBottom {
		m.vp.GotoBottom()
	}
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
		m.vp.SetContent(m.renderContent())
		if atBottom {
			m.vp.GotoBottom()
		}
	}
}

// renderContent joins the log, soft-wrapping any line wider than the viewport so
// a long single-line input/answer doesn't run off the edge. Lines that already
// fit (including the width-padded diff rows) pass through untouched.
func (m *tuiModel) renderContent() string {
	if m.width < 1 {
		return strings.Join(m.history, "\n")
	}
	wrap := lipgloss.NewStyle().Width(m.width)
	out := make([]string, len(m.history))
	for i, ln := range m.history {
		if lipgloss.Width(ln) > m.width {
			out[i] = wrap.Render(ln)
		} else {
			out[i] = ln
		}
	}
	return strings.Join(out, "\n")
}

// commandList is the single source for Tab completion and the /help display.
type cmdInfo struct{ name, desc string }

var commandList = []cmdInfo{
	{"/help", "this list"},
	{"/status", "config, knowledge base, trace paths"},
	{"/usage", "session counters + live token usage"},
	{"/login", "(re)configure server URL / model / key, then reload"},
	{"/new", "clear the session conversation memory"},
	{"/clear", "fresh start — clear the screen and the session"},
	{"/compact", "summarize the session so far to free up context"},
	{"/plan", "plan mode — propose a plan, change nothing"},
	{"/auto", "auto mode — execute the task (default)"},
	{"/skills", "list/toggle/install on-demand instruction packs"},
	{"/permissions", "relax approval for non-destructive file/shell actions"},
	{"/color", "change the frame color (cycles if no name)"},
	{"/rename", "rename the agent (saved in settings)"},
	{"/goal", "run a task explicitly"},
	{"/loop", "run a task n times so lessons compound"},
	{"/exit", "leave"},
}

// completeCommand completes a partial /command on Tab.
func (m *tuiModel) completeCommand() {
	val := m.input.Value()
	if !strings.HasPrefix(val, "/") {
		return
	}
	var matches []string
	for _, c := range commandList {
		if strings.HasPrefix(c.name, val) {
			matches = append(matches, c.name)
		}
	}
	switch len(matches) {
	case 0:
		return
	case 1:
		m.input.SetValue(matches[0] + " ")
		m.input.CursorEnd()
	default:
		if lcp := longestCommonPrefix(matches); len(lcp) > len(val) {
			m.input.SetValue(lcp)
			m.input.CursorEnd()
		} else {
			m.push(cDim.Render("  " + strings.Join(matches, "   ")))
		}
	}
}

func longestCommonPrefix(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	p := ss[0]
	for _, s := range ss[1:] {
		for !strings.HasPrefix(s, p) {
			if p = p[:len(p)-1]; p == "" {
				return ""
			}
		}
	}
	return p
}

func (m *tuiModel) accentBold() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(m.accent).Bold(true)
}

func (m *tuiModel) renderHelp() []string {
	cmd := m.accentBold()
	out := []string{cmd.Render("  commands") + cDim.Render("   (Tab completes)")}
	for _, c := range commandList {
		out = append(out, "  "+cmd.Render(fmt.Sprintf("%-9s", c.name))+"  "+cDim.Render(c.desc))
	}
	out = append(out, cDim.Render("  anything else is run as a task"))
	return out
}

func (m *tuiModel) renderKV(title string, rows [][2]string) []string {
	out := []string{m.accentBold().Render("  " + title)}
	for _, r := range rows {
		out = append(out, "    "+cDim.Render(fmt.Sprintf("%-13s", r[0]))+"  "+cBot.Render(r[1]))
	}
	return out
}

func (m *tuiModel) renderStatus() []string {
	c := m.app.cfg
	instr := m.app.instrSrc
	if instr == "" {
		instr = "(none)"
	}
	return m.renderKV("status", [][2]string{
		{"server", c.LLM.BaseURL},
		{"model", c.LLM.Model},
		{"workspace", c.Workspace},
		{"jail", c.File.Jail},
		{"defaults", fmt.Sprintf("run=%s  file=%s", c.Run.Default, c.File.Default)},
		{"instructions", instr},
		{"session", fmt.Sprintf("%d messages", m.app.ag.SessionLen())},
		{"knowledge", fmt.Sprintf("%s (%d lessons)", c.KBPath, len(m.app.kb.All()))},
		{"trace", c.TracePath},
	})
}

func (m *tuiModel) renderUsage() []string {
	p, c := m.app.client.Usage()
	return m.renderKV("usage (this session)", [][2]string{
		{"tasks", fmt.Sprintf("%d", m.app.tasks)},
		{"steps", fmt.Sprintf("%d", m.app.steps)},
		{"tool calls", fmt.Sprintf("%d", m.app.toolCalls)},
		{"tokens", fmt.Sprintf("%d + %d = %d", p, c, p+c)},
		{"lessons", fmt.Sprintf("%d", len(m.app.kb.All()))},
	})
}

// renderEvent renders one streamed agent event into styled history lines.
func (m *tuiModel) renderEvent(e uiEvent) []string {
	switch e.kind {
	case "assistant":
		if c, _ := e.fields["content"].(string); strings.TrimSpace(c) != "" {
			return strings.Split(renderMarkdown(c, m.width), "\n")
		}
	case "tool_call":
		t, _ := e.fields["tool"].(string)
		a, _ := e.fields["action"].(string)
		detail := compactJSON(e.fields["params"])
		if t == "file" { // the content shows below as a diff; just name the path here
			if p := paramStr(e.fields["params"], "path"); p != "" {
				detail = p
			}
		}
		return []string{cToolCall.Render("  ⚙ "+t+" "+a) + cDim.Render(" "+detail)}
	case "observation":
		isErr, _ := e.fields["is_error"].(bool)
		c, _ := e.fields["content"].(string)
		if tool, _ := e.fields["tool"].(string); tool == "file" && !isErr {
			if action, _ := e.fields["action"].(string); action == "read" {
				return renderCode(c) // syntax-highlight file reads
			}
		}
		if isErr {
			return outputLines(c, "✖", cErr)
		}
		return outputLines(c, "→", cOk)
	case "diff":
		path, _ := e.fields["path"].(string)
		d, _ := e.fields["diff"].(string)
		return m.renderDiff(path, d)
	case "final":
		if c, _ := e.fields["text"].(string); strings.TrimSpace(c) != "" {
			return append([]string{""}, strings.Split(renderMarkdown(c, m.width), "\n")...)
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
	const maxBody = 60
	if len(body) > maxBody {
		more := len(body) - maxBody
		body = append(body[:maxBody:maxBody], cDim.Render(fmt.Sprintf("    … %d more %s", more, plural(more, "line"))))
	}
	verb := "Update"
	if strings.Contains(diff, "@@ -0,0 ") { // new file → all additions
		verb = "Create"
	}
	h1 := lipgloss.NewStyle().Foreground(m.accent).Render("● ") + lipgloss.NewStyle().Bold(true).Render(verb+"("+path+")")
	h2 := cDim.Render(fmt.Sprintf("  ⎿  Added %d %s, removed %d %s", add, plural(add, "line"), del, plural(del, "line")))
	return append([]string{h1, h2}, body...)
}

// paramStr extracts a string field from a tool-call's params map.
func paramStr(v any, key string) string {
	if m, ok := v.(map[string]any); ok {
		if s, ok := m[key].(string); ok {
			return s
		}
	}
	return ""
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
	if clipped, truncated := textutil.Clip(s, 80); truncated {
		s = clipped + "…"
	}
	return s
}

// outputLines renders a (possibly multi-line) tool result, capped, with a marker
// on the first line — so command output and multi-line errors are actually
// visible instead of truncated to one line.
func outputLines(content, marker string, style lipgloss.Style) []string {
	lines := strings.Split(strings.TrimRight(content, "\n"), "\n")
	capped := false
	if len(lines) > 25 {
		lines, capped = lines[:25], true
	}
	out := make([]string, 0, len(lines)+1)
	for i, ln := range lines {
		if i == 0 {
			out = append(out, cDim.Render("  "+marker+" ")+style.Render(ln))
		} else {
			out = append(out, cDim.Render("    ")+style.Render(ln))
		}
	}
	if capped {
		out = append(out, cDim.Render("    …"))
	}
	return out
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
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}
