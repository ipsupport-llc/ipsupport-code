package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/selfupdate"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

type uiState int

const (
	stIdle uiState = iota
	stRunning
	stApprove
	stConfig        // interactive /config settings panel
	stChooseSession // startup: pick a saved session to restore / start new / delete
	stAgents        // interactive sub-agent profile manager (add/edit/delete)
	stRewind        // pick a checkpoint to rewind to
	stPlanReview    // a plan-mode task just proposed a plan: accept (→auto+execute) or keep planning
	stHistSearch    // Ctrl+R incremental reverse-search over the prompt history
	stGoalResume    // startup: an unfinished standing goal — offer to resume it
)

// chromeFixed: status line + top rule + bottom rule + hint line + 1 margin. The
// input's own height (1..maxInputLines) is added separately, since it grows with
// multi-line content (pastes, alt+enter).
const chromeFixed = 5

// maxInputLines caps how tall the input box grows before it scrolls internally.
const maxInputLines = 10

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
	input textarea.Model
	spin  spinner.Model

	state         uiState
	history       []string
	queued        []string // pending user messages (tasks + /commands), drained in order
	histIdx       int      // recall cursor into app.promptHist; == len means "not browsing"
	histDraft     string   // in-progress input saved when history browsing begins
	pendingMode   *bool    // plan(true)/auto(false) requested via shift+tab mid-task; applied on task done
	taskDoneAway  bool     // a task finished while a modal panel was open over it; finalize on panel close
	planTask      bool     // the running task started in plan mode → offer accept-plan when it finishes
	taskCancelled bool     // the running task was cancelled by esc → don't offer accept-plan
	searchQuery   string   // Ctrl+R reverse-search query (stHistSearch)
	searchIdx     int      // index into app.promptHist of the current match; -1 = none
	pending       *approvalReq
	approveChoice bool          // selected Yes(true)/No(false) while answering an approval
	cfgCursor     int           // selected row in the /config panel (stConfig)
	cfgPhase      int           // stConfig sub-flow: cfgPhaseList, or an add-provider form field
	cfgDraft      providerDraft // the provider being added in the panel form
	chooseRows    []sessionMeta // saved sessions offered by the startup chooser (stChooseSession)
	chooseCursor  int           // selected row (0..len = the "new session" row)
	cancel        context.CancelFunc
	taskStart     time.Time
	startTok      int
	retry         *retryInfo
	accent        lipgloss.Color
	accentIdx     int
	width, height int
	ready         bool
	inputLines    int        // current input box height in rows (grows with content)
	busyMsg       string     // status label while running non-task work (update/compact/model); "" = a model task ("thinking")
	subs          []*liveSub // sub-agents running right now, one live status line each

	// sub-agent profile manager (stAgents): a list + a provider→model→name builder
	agPhase     agentPhase
	agCursor    int
	agDraft     agentDraft
	agModelsAll []string // models fetched for the chosen provider
	agModelsErr string
	agFilter    string // type-to-filter over the model list
	agLoading   bool
	agInstalled map[string]bool // PATH scan of the CLI catalog, cached on entering agPickTool

	rewindRows   []rewindRow // checkpoints offered by the /rewind picker (stRewind)
	rewindCursor int
	rewindPrev   []string // cached colored preview (diffs) of the selected step

	// model-proposed, Tab-acceptable next-step suggestion (parsed from NEXT:)
	suggestion string
}

// liveSub is a sub-agent currently running, shown as its own status line during a
// parallel fan-out. Its detailed steps stay out of the scrollback (which would
// interleave chaotically across several sub-agents); only spawn/finish markers go
// to the log.
type liveSub struct {
	id       string // matches the "agent" field on the sub-agent's events
	label    string // "profile · dir"
	activity string // what it's doing right now
	steps    int
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
type updateMsg struct{ notice string }   // startup freshness check result
type updateDoneMsg struct{ text string } // /update result
type shellDoneMsg struct{}               // returned from a drop-to-shell
type shellCmdMsg struct{ out string }    // output of a one-off !cmd
type windowMsg struct {                  // re-detected context window for a provider
	provider string
	tokens   int
}
type modelsMsg struct { // /model result: lines to show, or setTo to switch model
	lines []string
	setTo string
}

// newTUIModel installs the UI bridge as the agent's tracer + approver, wires the
// stack, and builds the model. Split out from runTUI so tests can drive it.
func (a *app) newTUIModel(ctx context.Context) (*tuiModel, error) {
	b := newBridge()
	a.uiTracer = b
	a.approver = b
	a.tui = true // detect the context window off-thread, not inline
	if err := a.wire(); err != nil {
		return nil, err
	}

	in := textarea.New()
	in.Placeholder = defaultPlaceholder
	in.ShowLineNumbers = false
	in.CharLimit = 0 // no limit — allow large multi-line pastes (e.g. a YAML block)
	// "❯ " on the first row, aligned spaces on wrapped/continuation rows.
	in.SetPromptFunc(2, func(i int) string {
		if i == 0 {
			return "❯ "
		}
		return "  "
	})
	// Enter submits (handled in handleKey); alt+enter / ctrl+j insert a newline.
	in.KeyMap.InsertNewline = key.NewBinding(key.WithKeys("alt+enter", "ctrl+j"))
	in.FocusedStyle.CursorLine = lipgloss.NewStyle() // no full-width highlight bar
	in.SetHeight(1)
	in.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Points
	sp.Style = cToolCall

	name := a.cfg.Name
	if name == "" {
		name = "ipsupport-code"
	}
	m := &tuiModel{app: a, ctx: ctx, bridge: b, input: in, spin: sp, state: stIdle, accent: lipgloss.Color("13"), inputLines: 1}
	m.histIdx = len(a.promptHist) // start "not browsing": first ↑ recalls the most recent prompt
	act := a.activeLLM()
	m.history = bannerLines(name, version, a.providerName(), act.Model, a.workspace, act.ContextWindow, m.accent)
	if a.goal.Status == "active" && a.goal.Text != "" {
		m.history = append(m.history, cDim.Render("◎ standing goal: "+oneLine(a.goal.Text, 60)+"  — /goal go to resume"))
	}
	switch {
	case a.sessionRestored: // -session restored it already
		m.history = append(m.history, m.sessionRecap()...)
	case !a.startNew: // offer the saved sessions to restore / start fresh / delete
		if rows := a.listSessions(); len(rows) > 0 {
			m.chooseRows = rows
			m.state = stChooseSession
		}
	}
	if m.state == stIdle {
		m.offerGoalResume() // an unfinished goal? offer to pick it back up
	}
	return m, nil
}

// goalPending reports whether there's a standing goal worth resuming.
func (m *tuiModel) goalPending() bool {
	g := m.app.goal
	return g.Text != "" && (g.Status == "active" || g.Status == "incomplete")
}

// offerGoalResume, when a standing goal is unfinished, prompts to resume it instead
// of dropping straight to an empty prompt.
func (m *tuiModel) offerGoalResume() {
	if !m.goalPending() {
		return
	}
	m.state = stGoalResume
	m.push(m.accentBold().Render("  ◎ resume goal: ") + oneLine(m.app.goal.Text, 60) +
		cDim.Render("  — enter: resume · esc: not now"))
}

// sessionRecap renders the tail of a restored session as log lines, so the screen
// shows the recent conversation "as if you never left".
func (m *tuiModel) sessionRecap() []string {
	const maxExchanges, maxFinalLines = 5, 10
	h := m.app.ag.History()
	if len(h) == 0 {
		return nil
	}
	if len(h) > maxExchanges*2 {
		h = h[len(h)-maxExchanges*2:]
	}
	out := []string{"", cDim.Render("  ── restored session ──")}
	for _, msg := range h {
		switch msg.Role {
		case "user":
			goal, _ := textutil.Clip(strings.ReplaceAll(msg.Content, "\n", " "), 200)
			out = append(out, cYou.Render("❯ ")+goal)
		case "assistant":
			if strings.TrimSpace(msg.Content) == "" {
				continue
			}
			lines := strings.Split(strings.TrimRight(msg.Content, "\n"), "\n")
			if len(lines) > maxFinalLines {
				lines = append(lines[:maxFinalLines:maxFinalLines], cDim.Render("  …"))
			}
			out = append(out, lines...)
		}
	}
	return append(out, cDim.Render("  ── end of recap · continuing where you left off ──"), "")
}

// chooseActivate handles Enter/esc on the startup session chooser: open the
// highlighted session (restore its thread), or start a fresh one.
func (m *tuiModel) chooseActivate() (tea.Model, tea.Cmd) {
	m.state = stIdle
	if m.chooseCursor >= len(m.chooseRows) { // the "start new" row
		// Don't restore anything; take a fresh auto-named thread so we don't save
		// the empty startup state over an existing session. Don't persist the name.
		m.app.cfg.Name = m.app.autoSessionName()
		m.app.ag.Reset()
		m.app.ag.SetSystem(m.app.systemPrompt())
		m.push(cDim.Render("  — new session: " + m.app.cfg.Name + " —"))
		m.offerGoalResume()
		return m, nil
	}
	name := m.chooseRows[m.chooseCursor].name
	if name == slugName(m.app.cfg.Name) {
		m.app.loadSession() // the current name — restore, keep its display name
	} else if err := m.app.switchSession(name); err != nil {
		m.push(cErr.Render("couldn't open session: " + err.Error()))
		return m, nil
	}
	m.push(m.sessionRecap()...)
	m.offerGoalResume()
	return m, nil
}

// renderChooser draws the startup "resume a session?" picker.
func (m *tuiModel) renderChooser() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	cur := slugName(m.app.cfg.Name)
	lines := []string{
		accent.Bold(true).Render("resume a session?") + cDim.Render("   ↑↓ move · enter open · d delete · esc = newest"),
		"",
	}
	for i, s := range m.chooseRows {
		label := fmt.Sprintf("%-22s %d exchange(s) · %s", s.name, s.count/2, humanizeAgo(s.mod))
		if s.name == cur {
			label += "  (current)"
		}
		if i == m.chooseCursor {
			lines = append(lines, accent.Render(" ▸ "+label))
		} else {
			lines = append(lines, "   "+cDim.Render(label))
		}
	}
	newRow := "＋ start a new session"
	if m.chooseCursor == len(m.chooseRows) {
		lines = append(lines, accent.Render(" ▸ "+newRow))
	} else {
		lines = append(lines, "   "+cDim.Render(newRow))
	}
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// bannerLines builds the Claude-Code-style startup card: a rounded box with the
// agent name + version, model, working directory, and detected context window,
// then a one-line key hint.
func bannerLines(name, ver, provider, model, cwd string, window int, accent lipgloss.Color) []string {
	if h, err := os.UserHomeDir(); err == nil && h != "" && strings.HasPrefix(cwd, h) {
		cwd = "~" + cwd[len(h):]
	}
	label := lipgloss.NewStyle().Bold(true).Foreground(accent)
	modelRow := cDim.Render("model  ") + cBot.Render(model)
	if provider != "" && provider != "local" { // surface the provider when it isn't the local default
		modelRow += cDim.Render("  (" + provider + ")")
	}
	rows := []string{
		label.Render("✦ "+name) + cDim.Render("  "+ver),
		modelRow,
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
	tip := cDim.Render(`type a task — e.g. "explain what main.go does" — or /help for commands`)
	keys := cDim.Render("Tab complete · ctrl+r history · alt+enter newline · ctrl+u clear · ctrl+c quit · shift+tab plan⇄auto")
	return append(strings.Split(box, "\n"), "", tip, keys)
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
	return tea.Batch(m.spin.Tick, textarea.Blink, m.waitEvent(), m.waitApproval(), m.checkUpdate())
}

// detectWindowCmd re-detects the context window off the UI thread once the model
// is loaded (it's usually unloaded at startup, so the detector returned 0 then).
// Returns nil once we have the real value. It only reads config and returns the
// number via windowMsg — the write happens on the UI thread, so no race.
func (m *tuiModel) detectWindowCmd() tea.Cmd {
	if m.app.windowDetected {
		return nil
	}
	// Capture the target on the UI thread (race-free); probe off-thread; apply via
	// windowMsg on the UI thread. Handles local (LM Studio) and external providers
	// (context_length from /models) so a /ai or /model switch never blocks the UI.
	act, provider, local := m.app.activeLLM(), m.app.providerName(), m.app.isLocal()
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		tok := 0
		if local {
			if act.LMStudio() {
				tok = llm.DetectContextWindow(ctx, act.BaseURL, act.Model, http.DefaultClient)
			}
		} else {
			tok = llm.DetectModelContext(ctx, act.BaseURL, act.APIKey, act.Model, http.DefaultClient)
		}
		return windowMsg{provider: provider, tokens: tok}
	}
}

// checkUpdate runs the startup freshness check off the UI thread.
func (m *tuiModel) checkUpdate() tea.Cmd {
	if m.app.cfg.Offline { // offline: don't reach GitHub at startup
		return nil
	}
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(m.ctx, 3*time.Second)
		defer cancel()
		return updateMsg{notice: m.app.freshnessNotice(ctx)}
	}
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
		m.input.SetWidth(msg.Width - 4)
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
			return m, m.waitEvent()
		}
		m.retry = nil // progress resumed
		switch {
		case e.kind == "subagent": // a sub-agent started — log it + open a live line
			m.subStart(e)
			m.push(m.renderEvent(e)...)
		case e.kind == "subagent_done": // finished — close the line, log the outcome
			m.push(m.subDone(e)...)
		case e.kind == "reflecting": // task is done; now distilling lessons
			m.busyMsg = "distilling lessons from this task"
		case e.fields["agent"] != nil: // a sub-agent's own step — update its line only
			m.subUpdate(e)
		default:
			m.push(m.renderEvent(e)...)
			if e.kind == "final" {
				m.subs = nil // a stray sub-agent line can't outlive the task
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
		// Multi-line: the head on the ⚠ line, extra detail lines (e.g. the FULL
		// task of a sub-agent/external launch) pushed raw so they soft-wrap, and
		// the keys on their own line — nothing is truncated out of sight.
		lines := strings.Split(req.detail, "\n")
		m.push(cToolCall.Render("  ⚠ approve "+req.kind+": ") + lines[0])
		m.pushLines(lines[1:])
		m.push(cDim.Render("    y approve · n deny · a allow all " + categoryLabel(approvalCategory(req.kind)) + " this session · ↑ Yes/No · or keep typing"))
		return m, nil

	case taskDoneMsg:
		m.cancel = nil
		m.retry = nil
		m.applyPendingMode()          // a shift+tab during the task takes effect now, before the next one
		detect := m.detectWindowCmd() // model is loaded now — confirm the real window
		if m.state != stRunning {     // a modal panel is open over the finished task — keep it; finalize on close
			m.taskDoneAway = true
			return m, detect
		}
		if m.planTask && !m.taskCancelled && len(m.queued) == 0 {
			// A plan-mode task just proposed a plan — offer the accept→execute handshake.
			m.planTask = false
			m.state = stPlanReview
			m.push(m.accentBold().Render("  ▸ plan ready — ") + cDim.Render("enter: do it (switch to auto & execute) · esc: keep planning (type to refine)"))
			return m, detect
		}
		m.planTask = false
		m.state = stIdle
		if len(m.queued) > 0 { // drain the next pending message(s): tasks + /commands
			model, cmd := m.drainQueue()
			return model, tea.Batch(detect, cmd)
		}
		if m.app.shouldAutoCompact() { // context near the limit — fold it down
			return m, tea.Batch(detect, m.startCompact(true))
		}
		return m, tea.Batch(detect, m.input.Focus())

	case compactDoneMsg:
		switch {
		case msg.err != nil:
			m.push(cErr.Render("compact failed: " + msg.err.Error()))
		case msg.n == 0:
			m.push(cDim.Render("nothing to compact"))
		default:
			m.app.saveSession()
			m.push(cDim.Render(fmt.Sprintf("compacted %d messages → summary", msg.n)))
		}
		return m.idleDrain()

	case skillsMsg:
		if msg.err != nil {
			m.push(cErr.Render("install failed: " + msg.err.Error()))
		} else {
			_ = m.app.wire() // register the skill tool + index on the UI thread (no race)
			m.push(cDim.Render("installed & enabled: " + strings.Join(msg.names, ", ")))
		}
		return m.idleDrain()

	case updateMsg:
		if strings.TrimSpace(msg.notice) != "" {
			m.push(m.accentBold().Render("  ⬆ " + msg.notice)) // a newer build is out
		}
		return m, nil

	case updateDoneMsg:
		m.push(cDim.Render("  " + msg.text))
		return m.idleDrain()

	case shellDoneMsg:
		m.push(cDim.Render("  ⇱ back in ipsupport-code"))
		return m, m.input.Focus()

	case shellCmdMsg:
		if msg.out != "" {
			m.push(outputLines(msg.out, "→", cOk)...)
		}
		return m, nil

	case windowMsg:
		// applied on the UI thread, so View/auto-compact never race the write
		if msg.tokens > 0 {
			m.app.applyWindow(msg.provider, msg.tokens)
			if msg.provider == m.app.providerName() {
				m.app.windowDetected = true
			}
		}
		return m, nil

	case agentModelsMsg:
		m.agLoading = false
		m.agModelsAll, m.agModelsErr = msg.models, msg.err
		m.agCursor = 0
		return m, nil

	case modelsMsg:
		if msg.setTo != "" { // resolved a /model arg to one model — switch (UI thread)
			m.pushLines(m.app.setModel(msg.setTo))
			model, cmd := m.idleDrain()
			return model, tea.Batch(cmd, m.detectWindowCmd()) // re-detect off-thread
		}
		m.pushLines(msg.lines)
		return m.idleDrain()

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
	case "ctrl+r": // reverse-search the prompt history (from the idle prompt)
		if m.state == stIdle && len(m.app.promptHist) > 0 {
			m.state = stHistSearch
			m.searchQuery = ""
			m.searchIdx = m.searchFrom(len(m.app.promptHist) - 1)
		}
		return m, nil
	case "ctrl+l":
		m.history = m.history[:0]
		if m.ready {
			m.vp.SetContent("")
		}
		return m, nil
	case "ctrl+u":
		// Nuke the whole input — fast recovery from a bad clipboard paste. (In the
		// profile builder's name step it clears that field instead.)
		if m.state == stAgents && m.agPhase == agName {
			m.agDraft.name = ""
		} else {
			m.input.SetValue("")
		}
		return m, nil
	case "shift+tab":
		switch m.state {
		case stIdle:
			m.app.setMode(!m.app.planMode) // cycle plan/auto; the bottom indicator updates
			m.pendingMode = nil
		case stRunning:
			// Can't flip live — the running agent reads the mode (flipping races +
			// splits the run). Remember it and apply when this task finishes.
			cur := m.app.planMode
			if m.pendingMode != nil {
				cur = *m.pendingMode
			}
			next := !cur
			m.pendingMode = &next
			m.push(cDim.Render("  ⇄ " + modeName(next) + " — applies on the next task"))
		}
		// In a modal panel (config/sessions/agents/rewind) shift+tab is inert.
		return m, nil
	}

	// A pending approval doesn't grab the keyboard — keep typing. Press ↑ to
	// switch to answering it (your input text is preserved). Only from the plain
	// idle/running screens: inside a panel (e.g. /config over a task) ↑ must keep
	// moving the cursor — esc the panel first, then answer.
	if m.pending != nil && (m.state == stRunning || m.state == stIdle) && k.String() == "up" {
		m.state = stApprove
		m.approveChoice = true
		return m, nil
	}

	switch m.state {
	case stChooseSession:
		n := len(m.chooseRows) + 1 // +1 for the "start new" row
		switch k.String() {
		case "up", "k":
			m.chooseCursor = (m.chooseCursor - 1 + n) % n
		case "down", "j":
			m.chooseCursor = (m.chooseCursor + 1) % n
		case "d": // delete the highlighted saved session
			if m.chooseCursor < len(m.chooseRows) {
				m.app.deleteSessionNamed(m.chooseRows[m.chooseCursor].name)
				m.chooseRows = m.app.listSessions()
				if len(m.chooseRows) == 0 {
					m.state = stIdle // nothing left — start fresh
				} else if m.chooseCursor >= len(m.chooseRows) {
					m.chooseCursor = len(m.chooseRows) // clamp onto "start new"
				}
			}
		case "enter", "right", "l", " ":
			return m.chooseActivate()
		case "esc", "q": // default = the most recent session (first row)
			m.chooseCursor = 0
			return m.chooseActivate()
		}
		return m, nil
	case stConfig:
		if m.cfgPhase != cfgPhaseList { // typing inside the add-provider form
			return m.configAddKey(k)
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
	case stAgents:
		return m.agentsKey(k)
	case stRewind:
		switch k.String() {
		case "up", "k":
			if len(m.rewindRows) > 0 {
				m.rewindCursor = (m.rewindCursor - 1 + len(m.rewindRows)) % len(m.rewindRows)
				m.refreshRewindPreview()
			}
		case "down", "j":
			if len(m.rewindRows) > 0 {
				m.rewindCursor = (m.rewindCursor + 1) % len(m.rewindRows)
				m.refreshRewindPreview()
			}
		case "enter", "right", "l":
			if m.rewindCursor < len(m.rewindRows) {
				m.pushLines(m.app.applyRewind(m.rewindRows[m.rewindCursor].idx))
			}
			m.state = stIdle
		case "esc", "q":
			m.state = stIdle
		}
		return m, nil

	case stHistSearch:
		switch k.String() {
		case "ctrl+r": // step to an older match
			if m.searchIdx > 0 {
				if i := m.searchFrom(m.searchIdx - 1); i >= 0 {
					m.searchIdx = i
				}
			}
		case "enter": // use the match: drop it into the input
			if m.searchIdx >= 0 {
				m.input.SetValue(m.app.promptHist[m.searchIdx])
				m.input.CursorEnd()
			}
			m.state = stIdle
			m.histIdx = len(m.app.promptHist)
		case "esc", "ctrl+g":
			m.state = stIdle
		case "backspace":
			if r := []rune(m.searchQuery); len(r) > 0 {
				m.searchQuery = string(r[:len(r)-1])
				m.searchIdx = m.searchFrom(len(m.app.promptHist) - 1)
			}
		default:
			if len(k.String()) == 1 { // narrow the search
				m.searchQuery += k.String()
				m.searchIdx = m.searchFrom(len(m.app.promptHist) - 1)
			}
		}
		return m, nil

	case stGoalResume:
		switch k.String() {
		case "enter", "y": // resume the standing goal
			text := m.app.goal.Text
			m.app.setGoal(text) // re-activate + persist
			m.push(cYou.Render("❯ ") + text)
			return m, m.runTask(text)
		case "esc", "n": // not now — drop to the prompt (goal stays; /goal go later)
			m.state = stIdle
			return m, m.input.Focus()
		}
		return m, nil

	case stPlanReview:
		switch k.String() {
		case "enter", "y", "a": // accept: switch to auto and execute the proposed plan
			m.app.setMode(false)
			m.push(cYou.Render("❯ ") + "execute the plan")
			return m, m.runTask("Execute the plan you just proposed above, step by step. If a detail is unclear, make a reasonable call and proceed.")
		case "esc", "n", "q": // keep planning — stay in plan mode; type to refine or shift+tab to auto
			m.state = stIdle
			return m, m.input.Focus()
		}
		return m, nil

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
		case "a", "A": // allow this kind for the rest of the session (in-memory)
			m.approveSession()
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
		case "y", "Y", "n", "N", "a", "A":
			// Answer a pending approval directly — but only with an empty input, so
			// typing a word starting with y/n/a mid-sentence still just types.
			if m.pending != nil && strings.TrimSpace(m.input.Value()) == "" {
				switch k.String() {
				case "a", "A":
					m.approveSession()
				default:
					m.resolveApproval(k.String() == "y" || k.String() == "Y")
				}
				return m, m.waitApproval()
			}
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(k)
			return m, cmd
		case "esc":
			if m.cancel == nil && m.busyMsg != "" {
				// A non-task chore (update / compact / model list) has no cancel
				// hook — saying "cancelling" would be a lie.
				m.push(cDim.Render("  " + m.busyMsg + " can't be cancelled — it finishes on its own"))
				return m, nil
			}
			if m.cancel != nil {
				m.cancel()
			}
			m.taskCancelled = true // a cancelled plan run shouldn't pop the accept-plan prompt
			m.bridge.Abort()       // deny any in-flight approvals so no tool goroutine hangs
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
			m.recordInput(line)
			if isCommandLine(line) {
				return m.commandWhileBusy(line)
			}
			m.queued = append(m.queued, line) // type-ahead: run after the task
			m.syncViewport()                  // show it pinned above the input
			return m, nil
		case "tab":
			if strings.HasPrefix(m.input.Value(), "/") {
				m.completeCommand() // complete /commands while a task runs, too
			}
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
			if line == "" {
				return m, nil // empty Enter is a no-op (shell is `!` or /shell)
			}
			m.input.SetValue("")
			m.recordInput(line)
			return m.submit(line)
		case "tab":
			switch {
			case strings.HasPrefix(m.input.Value(), "/"):
				m.completeCommand()
			case strings.Contains(m.input.Value(), "@"):
				m.completeAtFile() // complete an @path reference against workspace files
			case m.input.Value() == "" && m.suggestion != "":
				m.input.SetValue(m.suggestion)
				m.input.CursorEnd()
				m.suggestion = ""
				m.input.Placeholder = defaultPlaceholder
			}
			return m, nil
		case "up":
			// Recall a previous message when the input is empty or already browsing
			// history; otherwise scroll the log. PgUp/wheel always scroll.
			if m.browsing() || strings.TrimSpace(m.input.Value()) == "" {
				m.historyPrev()
				return m, nil
			}
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(k)
			return m, cmd
		case "down":
			if m.browsing() {
				m.historyNext()
				return m, nil
			}
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(k)
			return m, cmd
		case "pgup", "pgdown", "home", "end":
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(k)
			return m, cmd
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(k)
		return m, cmd
	}
}

// modeName labels a plan/auto boolean for the mid-task mode-switch hint.
func modeName(plan bool) string {
	if plan {
		return "plan mode"
	}
	return "auto mode"
}

// applyPendingMode applies a plan/auto switch requested (via shift+tab) while a
// task was running — deferred so it never flips the mode mid-run. No-op if none
// pending or it already matches the current mode.
func (m *tuiModel) applyPendingMode() {
	if m.pendingMode == nil {
		return
	}
	want := *m.pendingMode
	m.pendingMode = nil
	if want != m.app.planMode {
		m.push(cDim.Render("  " + m.app.setMode(want)))
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

// approveSession approves the pending request AND stops asking about its whole
// category for the rest of the session (in-memory; cleared on /new & /clear).
func (m *tuiModel) approveSession() {
	if m.pending == nil {
		return
	}
	cat := categoryLabel(approvalCategory(m.pending.kind))
	m.app.allowSession(m.pending.kind)
	m.state = stRunning
	m.pending.reply <- true
	m.pending = nil
	m.push(cDim.Render("  → ") + cOk.Render("allowed") + cDim.Render(" · won't ask about "+cat+" again this session"))
}

// hist is the recall ring — the app's per-workspace prompt history, persisted so ↑
// reaches prompts from earlier runs.
func (m *tuiModel) hist() []string { return m.app.promptHist }

// recordInput persists a submitted line to the prompt history and resets the
// browse cursor to "not browsing".
func (m *tuiModel) recordInput(line string) {
	m.app.addPromptHist(line)
	m.histIdx = len(m.app.promptHist)
	m.histDraft = ""
}

// browsing reports whether the input is currently showing a recalled history line.
func (m *tuiModel) browsing() bool { return m.histIdx < len(m.hist()) }

// searchFrom returns the highest index ≤ start whose prompt contains the current
// search query (case-insensitive), or -1 — the reverse-search step.
func (m *tuiModel) searchFrom(start int) int {
	h := m.app.promptHist
	if start >= len(h) {
		start = len(h) - 1
	}
	q := strings.ToLower(m.searchQuery)
	for i := start; i >= 0; i-- {
		if strings.Contains(strings.ToLower(h[i]), q) {
			return i
		}
	}
	return -1
}

// historyPrev recalls an older submitted line into the input (↑).
func (m *tuiModel) historyPrev() {
	h := m.hist()
	if len(h) == 0 {
		return
	}
	if m.histIdx >= len(h) {
		m.histIdx = len(h)
		m.histDraft = m.input.Value() // stash the in-progress line to restore on the way back
	}
	if m.histIdx > 0 {
		m.histIdx--
	}
	m.input.SetValue(h[m.histIdx])
	m.input.CursorEnd()
}

// historyNext walks back toward the newest line, restoring the draft past the end (↓).
func (m *tuiModel) historyNext() {
	h := m.hist()
	if m.histIdx >= len(h) {
		return
	}
	m.histIdx++
	if m.histIdx == len(h) {
		m.input.SetValue(m.histDraft)
	} else {
		m.input.SetValue(h[m.histIdx])
	}
	m.input.CursorEnd()
}

// idleDrain returns to idle after a non-task async op (compact / update / model
// list / skills install) and runs anything the user queued while it worked — the
// same finalization taskDoneMsg does. Without this, a type-ahead submitted during
// such an op sat in the queue until the NEXT task finished (or forever).
func (m *tuiModel) idleDrain() (tea.Model, tea.Cmd) {
	m.state = stIdle
	if len(m.queued) > 0 {
		return m.drainQueue()
	}
	return m, m.input.Focus()
}

// drainQueue runs the next pending messages: it executes queued /commands in place
// (continuing to the next) and stops on the first task (going to running) or when a
// command opens a panel / goes async. Called when a task finishes.
func (m *tuiModel) drainQueue() (tea.Model, tea.Cmd) {
	for len(m.queued) > 0 {
		next := m.queued[0]
		m.queued = m.queued[1:]
		m.syncViewport() // it left the pinned queue
		m.push(cYou.Render("❯ ") + next)
		if isCommandLine(next) {
			model, cmd := m.runCommand(next)
			m = model.(*tuiModel)
			if cmd != nil || m.state != stIdle { // started a task / opened a panel / went async
				return m, cmd
			}
			continue // synchronous command done → drain the next
		}
		return m, m.runTask(next)
	}
	return m, m.input.Focus()
}

// isCommandLine reports whether a queued message is a slash-command or bang-shell,
// not a task to run through the agent.
func isCommandLine(s string) bool {
	return strings.HasPrefix(s, "/") || strings.HasPrefix(s, "!")
}

func (m *tuiModel) submit(line string) (tea.Model, tea.Cmd) {
	switch {
	case line == "!": // bang alias for /shell
		return m.runCommand("/shell")
	case strings.HasPrefix(line, "!"): // !cmd — run one shell command
		return m, m.runShellCmd(strings.TrimPrefix(line, "!"))
	case strings.HasPrefix(line, "/"):
		return m.runCommand(line)
	}
	m.push(cYou.Render("❯ ") + line)
	return m, m.runTask(line)
}

// runShellCmd runs a single shell command (the !cmd shortcut) in the workspace
// and shows its output — the user's own command, ungated by the agent policy.
func (m *tuiModel) runShellCmd(cmdline string) tea.Cmd {
	cmdline = strings.TrimSpace(cmdline)
	if cmdline == "" {
		return nil
	}
	m.push(cYou.Render("! ") + cmdline)
	dir := m.app.workspace
	return func() tea.Msg {
		c := exec.Command(shellPath(), "-c", cmdline)
		c.Dir = dir
		out, _ := c.CombinedOutput()
		return shellCmdMsg{out: strings.TrimRight(string(out), "\n")}
	}
}

func (m *tuiModel) runCommand(line string) (tea.Model, tea.Cmd) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/help", "/?":
		m.push(m.renderHelp()...)
	case "/status":
		m.push(m.renderStatus()...)
	case "/usage":
		if lines, handled := m.app.usageManage(strings.TrimSpace(rest)); handled {
			m.pushLines(lines)
		} else {
			m.push(m.renderUsage()...)
		}
		return m, nil
	case "/sessions":
		lines, switched := m.app.sessionsCommand(rest)
		m.pushLines(lines)
		if switched {
			m.push(m.sessionRecap()...)
		}
		return m, nil
	case "/agents", "/agent":
		m.pushLines(m.app.agentsCommand(m.ctx, rest))
		return m, nil
	case "/login", "/init":
		if err := m.app.reconfigure(); err != nil {
			m.push(cErr.Render("reload failed: " + err.Error()))
			return m, nil
		}
		m.push(cDim.Render("config reloaded from " + config.GlobalPath() + " — edit it or run `ipsupport-code -init` to change the connection"))
		return m, m.detectWindowCmd()
	case "/new": // branch to a NEW session; the current one stays in /sessions
		name, persist := strings.TrimSpace(rest), true
		if name == "" {
			name, persist = m.app.autoSessionName(), false // scratch thread; don't drift the default name
		}
		if err := m.app.newNamedSession(name, persist); err != nil {
			m.push(cErr.Render("could not start session: " + err.Error()))
			return m, nil
		}
		m.history = m.history[:0]
		if m.ready {
			m.vp.SetContent("")
		}
		m.push(cDim.Render("started a new session “" + m.app.cfg.Name + "” — the previous one is in /sessions"))
		return m, m.detectWindowCmd()
	case "/clear", "/reset": // wipe THIS thread + the screen
		m.app.ag.Reset()
		m.app.resetSessionAllow()
		m.app.saveSession()
		m.history = m.history[:0]
		if m.ready {
			m.vp.SetContent("")
		}
		m.push(cDim.Render("cleared — fresh screen, same session"))
	case "/compact":
		return m, m.startCompact(false)
	case "/plan":
		m.push(cDim.Render(m.app.setMode(true)))
	case "/auto":
		m.push(cDim.Render(m.app.setMode(false)))
	case "/update":
		if m.app.cfg.Offline {
			m.push(cDim.Render("offline mode is on — /update needs the internet. Run /offline off first."))
			return m, nil
		}
		return m, m.startUpdate(strings.TrimSpace(rest))
	case "/offline":
		m.pushLines(m.app.offlineCommand(rest))
		return m, nil
	case "/cd":
		m.pushLines(m.app.cdCommand(rest))
		return m, nil
	case "/knowledge", "/kb":
		m.pushLines(m.app.knowledgeCommand(rest))
		return m, nil
	case "/mcp":
		m.pushLines(strings.Split(m.app.mcpList(m.ctx), "\n"))
		return m, nil
	case "/rewind":
		m.openRewind()
		return m, nil
	case "/reflect":
		m.pushLines(m.app.reflectCommand(rest))
		return m, nil
	case "/goal":
		if text, ok := m.app.launchGoalText(rest); ok {
			m.app.setGoal(text)
			m.push(cYou.Render("❯ ") + text)
			return m, m.runTask(text)
		}
		m.pushLines(m.app.goalCommand(rest))
		return m, nil
	case "/history":
		m.pushLines(m.app.historyCommand(rest))
		return m, nil
	case "/budget":
		m.pushLines(m.app.budgetCommand(rest))
		return m, nil
	case "/diff":
		for _, l := range m.app.diffCommand() {
			m.push(colorizeDiff(l))
		}
		return m, nil
	case "/reasoning":
		m.pushLines(m.app.reasoningCommand(rest))
		return m, nil
	case "/ai":
		m.pushLines(m.app.aiCommand(rest))
		return m, m.detectWindowCmd() // re-detect the window off-thread after a switch
	case "/config":
		m.openConfig()
		return m, nil
	case "/model":
		act, name := m.app.activeLLM(), m.app.providerName()
		arg := strings.TrimSpace(rest)
		m.state = stRunning
		m.taskStart = time.Now()
		verb := "listing models on " + name
		if arg != "" {
			verb = "finding \"" + arg + "\" on " + name
		}
		m.busyMsg = verb
		m.push(cDim.Render("  " + verb + "…"))
		ctx := m.ctx
		return m, func() tea.Msg {
			c, cancel := context.WithTimeout(ctx, 8*time.Second)
			defer cancel()
			if arg == "" {
				return modelsMsg{lines: modelLines(c, act, name)}
			}
			setTo, lines := resolveModelArg(listModelIDs(c, act), arg)
			return modelsMsg{setTo: setTo, lines: lines}
		}
	case "/shell", "/sh":
		sh := shellPath()
		c := exec.Command(sh)
		c.Dir = m.app.workspace
		m.push(cDim.Render("  ⇲ dropping to " + sh + " — exit to return"))
		return m, tea.ExecProcess(c, func(error) tea.Msg { return shellDoneMsg{} })
	case "/skills":
		return m.skillsCmd(rest)
	case "/permissions", "/perms":
		m.pushLines(m.app.permissionsCommand(rest))
		return m, nil
	case "/color":
		m.setColor(rest)
	case "/rename":
		m.rename(rest)
	case "/loop":
		interval, max, goal, ok := parseLoop(rest)
		if !ok {
			m.push(cDim.Render(loopUsage))
			return m, nil
		}
		m.push(cYou.Render("❯ /loop "+loopLabel(interval, max)+" ") + goal)
		return m, m.runLoop(interval, max, goal)
	case "/exit", "/quit":
		return m, tea.Quit
	default:
		m.push(cDim.Render("unknown command " + cmd + " — try /help"))
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
// now, read-only commands run now, everything else waits — so you never have to
// sit through "thinking" just to leave or peek. (To queue a follow-up task, just
// type it — plain text is queued as type-ahead.)
func (m *tuiModel) commandWhileBusy(line string) (tea.Model, tea.Cmd) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/exit", "/quit":
		return m, tea.Quit
	case "/status", "/help", "/?", "/color", "/diff", "/history":
		// Always safe: pure info (incl. /history <filter> and the read-only /diff),
		// or /color which only recolors the frame.
		return m.runCommand(line)
	case "/config":
		// Open the settings panel OVER the running task to view it; changing a
		// value stays blocked while the task runs (re-wiring the agent it's using
		// would race), the panel stays put when the task ends, and it finalizes
		// (queue drain) on close.
		m.openConfig()
		return m, nil
	case "/usage", "/sessions", "/agents", "/agent", "/skills", "/permissions",
		"/budget", "/reasoning", "/goal", "/ai", "/knowledge", "/kb", "/mcp",
		"/offline", "/cd", "/reflect":
		// The bare form is a read-only report/listing; a subcommand may mutate or
		// re-wire the running stack, so defer those until the task finishes.
		if strings.TrimSpace(rest) == "" {
			return m.runCommand(cmd)
		}
	}
	// Defer it into the queue (drained in order when the task finishes) — don't drop it.
	m.queued = append(m.queued, line)
	m.syncViewport()
	m.push(cDim.Render("queued — " + cmd + " runs when the current task finishes"))
	return m, nil
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
	m.push(cDim.Render("  ✦ next: ") + m.accentBold().Render(text) + cDim.Render("   (Tab)"))
}

// startUpdate self-updates from GitHub in the background. An optional
// "stable"/"nightly" arg switches and saves the channel. The binary is replaced
// in place; a restart picks it up.
func (m *tuiModel) startUpdate(arg string) tea.Cmd {
	channel := m.app.cfg.Channel
	if channel == "" {
		channel = selfupdate.Stable
	}
	if arg == selfupdate.Stable || arg == selfupdate.Nightly {
		channel = arg
		_ = config.SaveChannel(channel)
		m.app.cfg.Channel = channel
	} else if arg != "" {
		m.push(cDim.Render("usage: /update [stable|nightly]"))
		return nil
	}
	m.state = stRunning
	m.taskStart = time.Now()
	m.busyMsg = "⬆ updating (" + channel + ")"
	m.push(cDim.Render("  ⬆ checking the " + channel + " channel…"))
	ctx, cur := m.ctx, version
	return func() tea.Msg {
		c, cancel := context.WithTimeout(ctx, 90*time.Second)
		defer cancel()
		rel, err := selfupdate.Latest(c, selfupdate.Repo, channel, http.DefaultClient)
		switch {
		case err != nil:
			return updateDoneMsg{"update failed: " + err.Error()}
		case rel.Version == cur:
			return updateDoneMsg{"already up to date — " + cur + " (" + channel + ")"}
		}
		path, err := selfupdate.Apply(c, rel, http.DefaultClient)
		if err != nil {
			return updateDoneMsg{"update failed: " + err.Error()}
		}
		return updateDoneMsg{"updated to " + rel.Version + " — restart to use it (" + path + ")"}
	}
}

// startCompact folds the session into a summary in the background (manual via
// /compact, or auto when the context nears the limit), ending with
// compactDoneMsg.
func (m *tuiModel) startCompact(auto bool) tea.Cmd {
	m.state = stRunning
	m.taskStart = time.Now()
	m.busyMsg = "compacting the session"
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

// startTask flips to running state with a fresh cancelable context and abort
// signal, returning both so the caller's goroutine can defer the cancel.
func (m *tuiModel) startTask() (context.Context, context.CancelFunc) {
	m.state = stRunning
	m.taskStart = time.Now()
	m.busyMsg = "" // a real model task → "thinking", not a labelled chore
	m.subs = nil   // no sub-agents from a previous task linger
	_, c := m.app.client.Usage()
	m.startTok = c // count completion (generated) tokens for this task
	m.suggestion = ""
	m.input.Placeholder = defaultPlaceholder
	m.bridge.arm()              // fresh abort signal so a previous cancel doesn't deny this task's approvals
	m.planTask = m.app.planMode // remember so we can offer accept-plan when it finishes
	m.taskCancelled = false
	tctx, cancel := context.WithCancel(m.ctx)
	m.cancel = cancel
	return tctx, cancel
}

// runTask runs a single goal in the background, streaming via the bridge and
// ending with taskDoneMsg.
func (m *tuiModel) runTask(goal string) tea.Cmd {
	tctx, cancel := m.startTask()
	return func() tea.Msg {
		defer cancel()
		m.app.runTaskStreaming(tctx, goal)
		return taskDoneMsg{}
	}
}

// runLoop re-runs a goal on an interval: it runs once, waits interval, runs
// again, until max iterations (0 = until stopped) or the user cancels (esc).
func (m *tuiModel) runLoop(interval time.Duration, max int, goal string) tea.Cmd {
	tctx, cancel := m.startTask()
	return func() tea.Msg {
		defer cancel()
		for i := 0; max == 0 || i < max; i++ {
			if i > 0 {
				select {
				case <-tctx.Done():
					return taskDoneMsg{}
				case <-time.After(interval):
				}
			}
			if tctx.Err() != nil {
				break
			}
			m.bridge.Emit("loop", map[string]any{"i": i + 1, "max": max, "every": interval.String()})
			m.app.runTaskStreaming(tctx, goal)
		}
		return taskDoneMsg{}
	}
}

func (m *tuiModel) View() string {
	if !m.ready {
		return "loading…"
	}
	m.syncInputHeight()          // grow/shrink the input box to its content before layout
	_, c := m.app.client.Usage() // c = cumulative completion (generated) tokens

	var status string
	switch {
	case m.state == stChooseSession:
		status = cDim.Render("welcome back — pick up a session, or start fresh")
	case m.state == stAgents:
		status = cDim.Render("sub-agent profiles — models the assistant can delegate to")
	case m.state == stRewind:
		status = cDim.Render("rewind — ↑↓ pick a step, enter to return there, esc to cancel")
	case m.state == stConfig:
		switch { // /config is the one panel that can sit over a live task
		case m.pending != nil:
			status = cToolCall.Render("⚠ approval waiting behind this panel") + cDim.Render(" — esc, then answer it")
		case m.cancel != nil:
			status = cDim.Render("settings — view-only while the task runs · esc back to it")
		default:
			status = cDim.Render("settings — changes apply and save as you make them")
		}
	case m.state == stPlanReview:
		status = m.accentBold().Render("▸ plan ready") + cDim.Render(" — enter: do it (switch to auto & execute) · esc: keep planning")
	case m.state == stGoalResume:
		status = m.accentBold().Render("◎ resume goal") + cDim.Render(" — enter: resume · esc: not now")
	case m.state == stHistSearch:
		match := cDim.Render("(no match)")
		if m.searchIdx >= 0 {
			match = cBot.Render("→ " + oneLine(m.app.promptHist[m.searchIdx], 56))
		}
		status = cToolCall.Render("(reverse-search) ") + m.accentBold().Render("`"+m.searchQuery+"`") + "  " + match +
			cDim.Render("  · ctrl+r older · enter use · esc cancel")
	case m.state == stApprove:
		status = m.approvePrompt()
	case m.pending != nil:
		// An approval is waiting but you can keep typing; y/n answers it (↑ opens Yes/No).
		status = cToolCall.Render("⚠ approval needed") + cDim.Render(" — y approve · n deny · a allow-session · ↑ Yes/No · or keep typing")
	case m.state == stRunning:
		if m.retry != nil {
			remain := time.Until(m.retry.until).Truncate(100 * time.Millisecond)
			if remain < 0 {
				remain = 0
			}
			status = cErr.Render(fmt.Sprintf("⟳ retrying (attempt %d) — backing off, %s left", m.retry.attempt, remain))
		} else if m.busyMsg != "" {
			// Non-task work (update / compact / model list) — label it as itself,
			// not "thinking" (which reads as the model running).
			elapsed := time.Since(m.taskStart).Truncate(time.Second)
			status = m.spin.View() + cToolCall.Render(fmt.Sprintf(" %s… (%s)", m.busyMsg, elapsed))
		} else {
			elapsed := time.Since(m.taskStart).Truncate(time.Second)
			gen := c - m.startTok // completion tokens generated this task
			// Until the first token streams (the model is still reading the
			// prompt) show just the clock — a stuck "↑0 tok" reads as broken.
			detail := elapsed.String()
			if gen > 0 {
				detail = fmt.Sprintf("%s · ↑%s tok", elapsed, humanK(gen))
			}
			status = m.spin.View() + cToolCall.Render(fmt.Sprintf(" %s · thinking… (%s)", m.app.providerModel(), detail))
			if meter := m.ctxMeter(); meter != "" { // watch the window fill during a long task
				status += cDim.Render(" · ") + meter
			}
		}
	default:
		// ctx = size of the last prompt vs the window (auto-compacts as it fills);
		// ↑ = tokens the model generated this whole session.
		act := m.app.activeLLM()
		ctxStr := humanK(m.app.client.Context())
		if act.ContextWindow > 0 {
			ctxStr += "/" + humanK(act.ContextWindow)
		}
		meter := ""
		if s := m.ctxMeter(); s != "" {
			meter = cDim.Render(" · ") + s
		}
		status = cDim.Render(fmt.Sprintf("%s · %s · ctx %s", m.app.providerModel(), filepath.Base(m.app.effectiveDir()), ctxStr)) +
			meter + cDim.Render(fmt.Sprintf(" · ↑%s · ready", humanK(c)))
	}

	bottom := m.modeLine()
	switch {
	case m.state == stApprove:
		bottom = cDim.Render("  ←→ select · enter confirm · y/n shortcut · esc back to typing")
	case m.pending != nil:
		bottom = m.modeLine() + cDim.Render("  · ↑ to answer the approval")
	case m.state == stRunning:
		bottom += cDim.Render("  · esc cancels")
	case m.state == stAgents:
		bottom = cDim.Render(m.agentsHint())
	case m.state == stRewind:
		bottom = cDim.Render("  ↑↓ move · enter rewind here · esc cancel")
	}

	content := m.vp.View()
	switch m.state {
	case stConfig:
		content = m.renderConfigPanel()
	case stChooseSession:
		content = m.renderChooser()
	case stAgents:
		content = m.renderAgentsPanel()
	case stRewind:
		content = m.renderRewindPanel()
	}
	frame := lipgloss.NewStyle().Foreground(m.accent)
	parts := []string{content}
	if sub := m.renderSubs(); sub != "" { // live sub-agent lines, above the status
		parts = append(parts, sub)
	}
	parts = append(parts, status)
	parts = append(parts, m.queuedView()...) // pinned just above the input
	parts = append(parts, m.topRule(frame), m.input.View(), frame.Render(strings.Repeat("─", m.width)), bottom)
	return strings.Join(parts, "\n")
}

// approvePrompt renders the Yes/No selector shown while answering an approval.
func (m *tuiModel) approvePrompt() string {
	detail := ""
	if m.pending != nil {
		// One status line — clip; the full request (incl. a long task) is already
		// in the log from the approvalMsg push.
		detail = m.pending.kind + " " + oneLine(m.pending.detail, 70)
	}
	yes, no := cDim.Render("  Yes  "), cDim.Render("  No  ")
	if m.approveChoice {
		yes = cOk.Bold(true).Render("  ▸Yes  ")
	} else {
		no = cErr.Bold(true).Render("  ▸No  ")
	}
	return cToolCall.Render("⚠ approve "+detail+"  ") + yes + no + cDim.Render("  (a = allow all this session)")
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
	// The pinned queue region and the (variable-height) input both eat into the
	// log height so nothing overflows the screen.
	if h := m.height - chromeFixed - m.inputLines - len(m.queuedView()); h > 0 {
		return h
	}
	return 1
}

// syncInputHeight sizes the input box to its content (wrapped rows across all
// lines), capped at maxInputLines — so a multi-line paste or a long wrapped line
// grows the box instead of scrolling on one line.
func (m *tuiModel) syncInputHeight() {
	w := m.width - 4 - 2 // input width minus the 2-col prompt
	if w < 1 {
		w = 1
	}
	rows := 0
	for _, ln := range strings.Split(m.input.Value(), "\n") {
		rows += max(1, (lipgloss.Width(ln)+w-1)/w)
	}
	if rows < 1 {
		rows = 1
	}
	if rows > maxInputLines {
		rows = maxInputLines
	}
	if rows != m.inputLines {
		m.inputLines = rows
		m.input.SetHeight(rows)
		m.syncViewport() // input height changed → re-fit the log
	}
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

// keyHelp documents the keyboard shortcuts for /help.
var keyHelp = [][2]string{
	{"Enter", "send (or queue while a task runs)"},
	{"↑ / ↓", "recall & edit previous messages; ↑ also opens a pending approval / the last queued line"},
	{"ctrl+r", "reverse-search the prompt history (type to narrow · ctrl+r older · enter use)"},
	{"y / n / a", "on an approval: approve · deny · allow all of that kind this session"},
	{"alt+enter", "newline in the input (also ctrl+j)"},
	{"Tab", "complete a /command · an @file path · or accept the NEXT suggestion"},
	{"shift+tab", "toggle plan ⇄ auto (mid-task: applies on the next task)"},
	{"!cmd  ·  !", "run one shell command · bare ! drops to a shell"},
	{"ctrl+u", "clear the input · ctrl+l clear the screen · PgUp/PgDn scroll the log"},
	{"esc", "cancel the running task, or back out of a panel"},
	{"ctrl+c", "quit"},
}

// subagentHelp is the short sub-agent primer for /help.
var subagentHelp = []string{
	"Define profiles in /config → Sub-agents (provider → model → name).",
	"Then ask, e.g. \"review internal/tool across grok and claude, then merge\".",
	"The assistant fans out in parallel; point one at another repo with a dir.",
	"/agents add|rm|exec · /permissions agents on relaxes the per-spawn prompt.",
}

// commandList is the single source for Tab completion and the /help display.
type cmdInfo struct{ name, desc string }

var commandList = []cmdInfo{
	{"/help", "this list"},
	{"/status", "config, knowledge base, trace paths"},
	{"/usage", "token history (day/week/month, by model); clear · purge <days> · retain <days>"},
	{"/budget", "<usd>|off — cap estimated spend per run; refuses new tasks once hit"},
	{"/diff", "show uncommitted changes in the workspace (what the agent changed)"},
	{"/login", "(re)configure server URL / model / key, then reload"},
	{"/new", "start a NEW session (old stays in /sessions); /new <name> to name it"},
	{"/clear", "wipe this session's context + the screen (same session)"},
	{"/compact", "summarize the session so far to free up context"},
	{"/plan", "plan mode — propose a plan, change nothing"},
	{"/auto", "auto mode — execute the task (default)"},
	{"/ai", "switch/add AI provider; /ai key <name> <tok>; /ai add <name> <url> (custom)"},
	{"/model", "list the provider's models, or pick one"},
	{"/config", "settings panel — interactive in the TUI (↑↓ · enter · esc); a static overview in the REPL"},
	{"/update", "self-update from GitHub (stable|nightly)"},
	{"/offline", "on|off — work without internet (disables web + update checks)"},
	{"/cd", "set the working dir (relative paths + sub-agents resolve there)"},
	{"/knowledge", "learned-lessons store: report · clear · purge <days> · retain <days>"},
	{"/mcp", "list configured MCP servers and their tools"},
	{"/rewind", "pick a step to roll back to (restores files + trims the chat)"},
	{"/reflect", "on|off|<profile> — post-task learning; run it on a stronger model"},
	{"/goal", "<text> — set & pursue a multi-turn goal; a judge re-feeds it until met (go · clear · ttl <n>)"},
	{"/reasoning", "off|minimal|low|medium|high (or reflect:) — trim a thinking model's reasoning"},
	{"/shell", "drop to a shell (or !cmd for one command); exit to return"},
	{"/skills", "list/toggle/install on-demand instruction packs"},
	{"/permissions", "relax approval for file / shell / sub-agent-spawn actions"},
	{"/color", "change the frame color (cycles if no name)"},
	{"/rename", "rename the agent (saved in settings)"},
	{"/sessions", "list / switch / delete saved sessions (per agent name)"},
	{"/history", "recent prompts (↑/↓ recalls them into the input); /history <text> to filter"},
	{"/agents", "sub-agent profiles: add (LLM) · add-tool (external CLI: codex/claude/…) · rm · exec"},
	{"/loop", "re-run a task on an interval: /loop 5m <task> (esc to stop)"},
	{"/exit", "leave"},
}

// completeCommand completes a partial /command on Tab. Before the first space it
// completes the command name; after it, the first argument for commands with a
// fixed candidate set (e.g. /ai <provider>, /color <name>).
// completeAtFile Tab-completes an @path token in the input against workspace files.
func (m *tuiModel) completeAtFile() {
	val := m.input.Value()
	at := strings.LastIndexByte(val, '@')
	if at < 0 || (at > 0 && val[at-1] != ' ' && val[at-1] != '\n') {
		return // no @, or @ isn't the start of the current token
	}
	prefix := val[at+1:]
	if strings.ContainsAny(prefix, " \n") {
		return // the @token isn't at the cursor
	}
	lcp, matches := m.app.completePath(prefix)
	switch len(matches) {
	case 0:
		return
	case 1:
		m.input.SetValue(val[:at] + "@" + matches[0] + " ")
		m.input.CursorEnd()
	default:
		if len(lcp) > len(prefix) {
			m.input.SetValue(val[:at] + "@" + lcp)
			m.input.CursorEnd()
		} else {
			if len(matches) > 12 {
				matches = matches[:12]
			}
			m.push(cDim.Render("  " + strings.Join(matches, "   ")))
		}
	}
}

func (m *tuiModel) completeCommand() {
	val := m.input.Value()
	if !strings.HasPrefix(val, "/") {
		return
	}
	if i := strings.IndexByte(val, ' '); i >= 0 {
		m.completeArg(val[:i], strings.TrimLeft(val[i+1:], " "))
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

// completeArg completes a command's argument against its candidates.
func (m *tuiModel) completeArg(name, arg string) {
	// "/ai key <provider>" — complete the provider (ANY known one, since you're
	// adding a key) as the second token; the third token is the secret.
	if name == "/ai" && strings.HasPrefix(arg, "key ") {
		sub := strings.TrimPrefix(arg, "key ")
		if !strings.ContainsRune(sub, ' ') {
			m.applyCompletion("/ai key", sub, config.KnownProviders())
		}
		return
	}
	if strings.ContainsRune(arg, ' ') { // only the first token completes
		return
	}
	m.applyCompletion(name, arg, m.argCandidates(name))
}

// applyCompletion sets the input to "prefix <match>" for a unique/common-prefix
// match of partial against cands, or lists the matches when ambiguous.
func (m *tuiModel) applyCompletion(prefix, partial string, cands []string) {
	if len(cands) == 0 {
		return
	}
	var matches []string
	for _, c := range cands {
		if strings.HasPrefix(c, partial) {
			matches = append(matches, c)
		}
	}
	switch len(matches) {
	case 0:
		return
	case 1:
		m.input.SetValue(prefix + " " + matches[0])
		m.input.CursorEnd()
	default:
		if lcp := longestCommonPrefix(matches); len(lcp) > len(partial) {
			m.input.SetValue(prefix + " " + lcp)
			m.input.CursorEnd()
		} else {
			m.push(cDim.Render("  " + strings.Join(matches, "   ")))
		}
	}
}

// argCandidates returns the completion candidates for a command's first argument
// (sorted), or nil for commands without a fixed set (e.g. /model is dynamic).
func (m *tuiModel) argCandidates(name string) []string {
	switch name {
	case "/ai":
		// Only offer providers you can actually switch to: local plus the external
		// ones (built-in or custom) with a key. Suggesting a keyless provider is a
		// dead end — /ai would just reject it.
		return m.app.configuredProviderNames()
	case "/offline":
		return []string{"on", "off"}
	case "/reflect":
		return append([]string{"on", "off", "self"}, agentProfileNames(m.app.cfg)...)
	case "/goal":
		return []string{"go", "resume", "clear", "ttl", "off", "on"}
	case "/reasoning":
		return []string{"off", "minimal", "low", "medium", "high", "reflect"}
	case "/agents", "/agent":
		return append([]string{"add", "add-tool", "rm", "exec"}, agentProfileNames(m.app.cfg)...)
	case "/skills":
		return []string{"on", "off", "install", "remove"}
	case "/permissions":
		return []string{"files", "run", "agents", "reset"}
	case "/update":
		return []string{"stable", "nightly"}
	case "/budget":
		return []string{"off"}
	case "/knowledge", "/kb":
		return []string{"clear", "purge", "retain"}
	case "/usage":
		return []string{"clear", "purge", "retain"}
	case "/sessions":
		names := []string{"delete"}
		for _, s := range m.app.listSessions() {
			names = append(names, s.name)
		}
		return names
	case "/color":
		names := make([]string, 0, len(colorNames))
		for n := range colorNames {
			names = append(names, n)
		}
		sort.Strings(names)
		return names
	}
	return nil
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
	out := []string{cmd.Render("  commands") + cDim.Render("   (Tab completes, even while busy)")}
	for _, c := range commandList {
		out = append(out, "  "+cmd.Render(fmt.Sprintf("%-13s", c.name))+"  "+cDim.Render(c.desc))
	}
	out = append(out, cDim.Render("  anything else is run as a task"))
	out = append(out, "", cmd.Render("  keys"))
	for _, k := range keyHelp {
		out = append(out, "  "+cmd.Render(fmt.Sprintf("%-10s", k[0]))+"  "+cDim.Render(k[1]))
	}
	out = append(out, "", cmd.Render("  sub-agents"))
	for _, l := range subagentHelp {
		out = append(out, "  "+cDim.Render(l))
	}
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
	act := m.app.activeLLM()
	rows := [][2]string{
		{"version", fmt.Sprintf("%s (%s channel)", version, channelOf(c))},
		{"provider", m.app.providerName()},
		{"server", act.BaseURL},
		{"model", act.Model},
		{"reasoning", m.app.reasoningLevel(m.app.providerName(), act.Model)},
		{"workspace", c.Workspace},
		{"jail", c.File.Jail},
		{"defaults", fmt.Sprintf("run=%s  file=%s", c.Run.Default, c.File.Default)},
		{"prompt", promptOrDefault(m.app.promptSrc)},
		{"instructions", instr},
		{"session", fmt.Sprintf("%d messages", m.app.ag.SessionLen())},
		{"goal", m.app.goalStatusLine()},
		{"budget", m.app.budgetStatusLine()},
		{"knowledge", fmt.Sprintf("%s (%d lessons)", c.KBPath, len(m.app.kb.All()))},
		{"facts", fmt.Sprintf("%s (%d learned)", m.app.factsPath(), len(m.app.facts))},
		{"trace", c.TracePath},
	}
	if c.Offline {
		rows = append(rows, [2]string{"offline", "ON — no internet egress"})
	}
	return m.renderKV("status", rows)
}

func (m *tuiModel) renderUsage() []string {
	p, c := m.app.client.Usage()
	out := m.renderKV("usage (this session)", [][2]string{
		{"tasks", fmt.Sprintf("%d", m.app.tasks)},
		{"steps", fmt.Sprintf("%d", m.app.steps)},
		{"tool calls", fmt.Sprintf("%d", m.app.toolCalls)},
		{"tokens", fmt.Sprintf("%d + %d = %d", p, c, p+c)},
		{"lessons", fmt.Sprintf("%d", len(m.app.kb.All()))},
	})
	if roll := m.app.usageRollups(); len(roll) > 0 {
		out = append(out, "")
		out = append(out, m.renderKV("tokens (cumulative, saved · $ estimated)", roll)...)
	}
	days, models := m.app.usageLedger()
	if len(days) > 0 {
		out = append(out, "")
		out = append(out, m.renderKV("tokens by day", days)...)
	}
	if len(models) > 0 {
		out = append(out, "")
		out = append(out, m.renderKV("tokens by provider/model", models)...)
	}
	out = append(out, "", cDim.Render("  manage · /usage clear · /usage purge <days> · /usage retain <days>"))
	return out
}

// openRewind enters the checkpoint picker (or says there's nothing to rewind).
func (m *tuiModel) openRewind() {
	m.rewindRows = m.app.rewindRows()
	if len(m.rewindRows) == 0 {
		m.push(cDim.Render("nothing to rewind — no turns yet this session"))
		return
	}
	m.rewindCursor = 0
	m.refreshRewindPreview()
	m.state = stRewind
}

// refreshRewindPreview rebuilds the cached, colored diff preview for the selected
// step (computed on cursor move, not every frame). Total height is capped.
func (m *tuiModel) refreshRewindPreview() {
	m.rewindPrev = nil
	if m.rewindCursor >= len(m.rewindRows) {
		return
	}
	items, trimmed := m.app.rewindPreview(m.rewindRows[m.rewindCursor].idx)
	budget := 22
	for _, it := range items {
		if budget <= 0 {
			m.rewindPrev = append(m.rewindPrev, cDim.Render("    …(more files)"))
			break
		}
		switch it.Kind {
		case "delete":
			m.rewindPrev = append(m.rewindPrev, cErr.Render("  ✖ delete "+it.Rel)+cDim.Render(" (created in this step)"))
			budget--
		case "toobig":
			m.rewindPrev = append(m.rewindPrev, cDim.Render("  ~ "+it.Rel+" (too large — left as-is)"))
			budget--
		case "restore":
			d := m.renderDiff(it.Rel, it.Diff)
			if len(d) > budget {
				d = append(d[:budget], cDim.Render("    …"))
			}
			m.rewindPrev = append(m.rewindPrev, d...)
			budget -= len(d)
		}
	}
	if trimmed > 0 {
		m.rewindPrev = append(m.rewindPrev, cDim.Render(fmt.Sprintf("  conversation: trims %d message(s)", trimmed)))
	}
	if len(m.rewindPrev) == 0 {
		m.rewindPrev = []string{cDim.Render("  (no changes to undo for this step)")}
	}
}

// renderRewindPanel draws the boxed, navigable checkpoint list.
func (m *tuiModel) renderRewindPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	lines := []string{accent.Bold(true).Render("rewind — pick a step to return to (before it ran)")}
	for i, r := range m.rewindRows {
		label := fmt.Sprintf("%-46s %d file(s)", oneLine(r.goal, 46), r.files)
		lines = append(lines, agRow(accent, label, i == m.rewindCursor))
	}
	if len(m.rewindPrev) > 0 {
		lines = append(lines, "", accent.Render("  ── what rewinding here changes ──"))
		lines = append(lines, m.rewindPrev...)
	}
	lines = append(lines, "", cDim.Render("  shell/git/network are NOT undone"))
	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// subStart opens a live status line for a freshly spawned sub-agent.
func (m *tuiModel) subStart(e uiEvent) {
	id, _ := e.fields["agent"].(string)
	label, _ := e.fields["profile"].(string)
	if dir, _ := e.fields["dir"].(string); dir != "" {
		label += " · " + filepath.Base(dir)
	}
	m.subs = append(m.subs, &liveSub{id: id, label: label, activity: "starting…"})
}

// subUpdate advances a sub-agent's live line from one of its own events.
func (m *tuiModel) subUpdate(e uiEvent) {
	id, _ := e.fields["agent"].(string)
	s := m.findSub(id)
	if s == nil {
		return
	}
	switch e.kind {
	case "tool_call":
		t, _ := e.fields["tool"].(string)
		act, _ := e.fields["action"].(string)
		s.steps++
		s.activity = strings.TrimSpace(t + " " + act)
	case "goal":
		s.activity = "reading the task…"
	case "nudge":
		s.activity = "rethinking…"
	case "final":
		s.activity = "wrapping up…"
	}
}

// subDone closes a sub-agent's live line and returns its outcome marker for the log.
func (m *tuiModel) subDone(e uiEvent) []string {
	id, _ := e.fields["agent"].(string)
	prof, _ := e.fields["profile"].(string)
	m.dropSub(id)
	if ok, _ := e.fields["ok"].(bool); !ok {
		errs, _ := e.fields["error"].(string)
		return []string{cErr.Render("  ✖ sub-agent "+prof+" failed") + cDim.Render(" "+errs)}
	}
	return []string{cOk.Render("  ✓ sub-agent " + prof + " finished")}
}

func (m *tuiModel) findSub(id string) *liveSub {
	for _, s := range m.subs {
		if s.id == id {
			return s
		}
	}
	return nil
}

func (m *tuiModel) dropSub(id string) {
	for i, s := range m.subs {
		if s.id == id {
			m.subs = append(m.subs[:i], m.subs[i+1:]...)
			return
		}
	}
}

// renderSubs is the live block of running sub-agents, one line each, shown just
// above the status line during a fan-out.
func (m *tuiModel) renderSubs() string {
	if len(m.subs) == 0 {
		return ""
	}
	lines := make([]string, 0, len(m.subs))
	for _, s := range m.subs {
		lines = append(lines, cToolCall.Render("  ● "+s.label)+cDim.Render(fmt.Sprintf("  %s (step %d)", s.activity, s.steps)))
	}
	return strings.Join(lines, "\n")
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
				path, _ := e.fields["path"].(string)
				return renderCode(c, path) // syntax-highlight file reads
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
	case "nudge":
		return []string{cToolCall.Render("  ↻ that's not working — asking it to rethink")}
	case "continue":
		line := fmt.Sprintf("  ↻ goal not met yet — pushing on (%d/%d)", toInt(e.fields["return"]), toInt(e.fields["of"]))
		if miss, _ := e.fields["missing"].(string); strings.TrimSpace(miss) != "" {
			line += ": " + oneLine(miss, 60)
		}
		return []string{cToolCall.Render(line)}
	case "judge":
		if done, _ := e.fields["done"].(bool); done {
			return []string{cOk.Render("  ✓ judge: goal met")}
		}
	case "lesson":
		d, _ := e.fields["domain"].(string)
		f, _ := e.fields["proven_fix"].(string)
		return []string{cLesson.Render("  ✦ learned ["+d+"] ") + cDim.Render(f)}
	case "fact":
		f, _ := e.fields["text"].(string)
		return []string{cLesson.Render("  ✦ noted ") + cDim.Render(f)}
	case "subagent":
		prof, _ := e.fields["profile"].(string)
		model, _ := e.fields["model"].(string)
		dir, _ := e.fields["dir"].(string)
		task, _ := e.fields["task"].(string)
		head := prof
		if model != "" {
			head += " · " + model
		}
		if dir != "" {
			head += " · " + filepath.Base(dir)
		}
		return []string{cToolCall.Render("  ⇉ spawned "+head) + cDim.Render("  "+task)}
	case "loop":
		label := fmt.Sprintf("↻ loop %d", toInt(e.fields["i"]))
		if total := toInt(e.fields["max"]); total > 0 {
			label += fmt.Sprintf("/%d", total)
		}
		if every, _ := e.fields["every"].(string); every != "" {
			label += " · every " + every
		}
		return []string{cDim.Render("— " + label + " —")}
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
			body = append(body, diffAddRow(newNo, ln[1:], path, width))
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
	h2 := cDim.Render(fmt.Sprintf("  ⎿  %d %s added, %d %s removed", add, plural(add, "line"), del, plural(del, "line")))
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

func diffAddRow(no int, code, path string, width int) string {
	hl := ansiBG.ReplaceAllString(highlightCode(code, path), "")
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
func renderCode(content, path string) []string {
	lines := strings.Split(content, "\n")
	capped := false
	if len(lines) > 40 {
		lines, capped = lines[:40], true
	}
	out := []string{cDim.Render("  → read:")}
	for _, ln := range strings.Split(highlightCode(strings.Join(lines, "\n"), path), "\n") {
		out = append(out, "    "+ln)
	}
	if capped {
		out = append(out, cDim.Render("    …"))
	}
	return out
}

// highlightCode colours code for the terminal, choosing the lexer by file path
// when known (reliable) and only falling back to content analysis otherwise —
// guessing alone often misses Python on short or single-line snippets.
func highlightCode(code, path string) string {
	lexer := lexers.Fallback
	if path != "" {
		if l := lexers.Match(path); l != nil {
			lexer = l
		}
	}
	if lexer == lexers.Fallback {
		if l := lexers.Analyse(code); l != nil {
			lexer = l
		}
	}
	style := styles.Get("github-dark")
	if style == nil {
		style = styles.Fallback
	}
	formatter := formatters.Get("terminal256")
	if formatter == nil {
		formatter = formatters.Fallback
	}
	it, err := lexer.Tokenise(nil, code)
	if err != nil {
		return code
	}
	var b strings.Builder
	if err := formatter.Format(&b, style, it); err != nil {
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

// ctxMeter renders the current context fill as a colored percentage — dim under
// half, warn as it approaches the auto-compact threshold, red once it will compact.
// Empty when there's no window or nothing sent yet.
func (m *tuiModel) ctxMeter() string {
	return ctxMeterFor(m.app.client.Context(), m.app.activeLLM().ContextWindow)
}

// ctxMeterFor is the pure part of the meter, split out for tests.
func ctxMeterFor(used, win int) string {
	if win <= 0 || used <= 0 {
		return ""
	}
	pct := used * 100 / win
	s := fmt.Sprintf("ctx %d%%", pct)
	switch {
	case float64(used) >= float64(win)*autoCompactRatio:
		return cErr.Render(s + "⚠") // at/over the auto-compact threshold
	case pct >= ctxWarnPercent:
		return cToolCall.Render(s)
	default:
		return cDim.Render(s)
	}
}

// ctxWarnPercent is the fill level at which the ctx meter turns to the warn color
// (the red ⚠ comes later, at the autoCompactRatio threshold).
const ctxWarnPercent = 50

// colorizeDiff tints one unified-diff line: added green, removed red, hunk headers
// accented, file headers dim — leaving stat/plain lines as-is.
func colorizeDiff(line string) string {
	switch {
	case strings.HasPrefix(line, "diff --git"), strings.HasPrefix(line, "index "),
		strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
		return cDim.Render(line)
	case strings.HasPrefix(line, "@@"):
		return cToolCall.Render(line)
	case strings.HasPrefix(line, "+"):
		return cOk.Render(line)
	case strings.HasPrefix(line, "-"):
		return cErr.Render(line)
	default:
		return line
	}
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
