package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
	"github.com/ipsupport-llc/ipsupport-code/internal/usage"
)

func TestSplitCommand(t *testing.T) {
	cmd, rest := splitCommand("/loop 3 do the thing")
	if cmd != "/loop" || rest != "3 do the thing" {
		t.Errorf("splitCommand = %q,%q", cmd, rest)
	}
	// Empty/whitespace must not panic (regression: /usage and /ai key crashed).
	for _, in := range []string{"", "   ", "\t"} {
		if c, r := splitCommand(in); c != "" || r != "" {
			t.Errorf("splitCommand(%q) = %q,%q, want empty", in, c, r)
		}
	}
	// usageManage("") must show the report, not crash.
	a := &app{cfg: config.Default()}
	if _, handled := a.usageManage(""); handled {
		t.Error("usageManage(\"\") should not be handled as a subcommand (no panic, shows report)")
	}
}

func TestFreshnessNoticeSkippedWhenUpdateCheckOff(t *testing.T) {
	// With the check disabled, freshnessNotice must return "" before any network
	// call — independent of Offline. (An enabled dev build also returns "", so we
	// assert the disabled path via the guard, not the network.)
	a := &app{cfg: config.Config{UpdateCheck: false}}
	if got := a.freshnessNotice(context.Background()); got != "" {
		t.Errorf("freshnessNotice with update_check off = %q, want empty", got)
	}
}

// A background job that finishes MID-task reaches the model on its next step via
// the beforeTurn hook (not only at the next task boundary), lands in durable
// history, and isn't re-delivered by the task-boundary path.
func TestBeforeTurnDeliversFinishedJob(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	a.jobs = append(a.jobs, &job{id: 1, profile: "codex", task: "fix the repo", done: true, ok: true, result: "codex report: patched"})
	a.addBtw("also check the tests") // both channels fold through beforeTurn

	before := a.ag.SessionLen()
	msgs := a.beforeTurn()
	if len(msgs) != 2 {
		t.Fatalf("beforeTurn returned %d messages, want 2 (btw + job)", len(msgs))
	}
	joined := msgs[0].Content + "\n" + msgs[1].Content
	if !strings.Contains(joined, "also check the tests") || !strings.Contains(joined, "codex report: patched") {
		t.Fatalf("beforeTurn missing btw or job note: %q", joined)
	}

	// The job note is durable (folded into history); the btw aside is not.
	h := a.ag.History()
	if len(h) != before+1 || !strings.Contains(h[len(h)-1].Content, "codex report: patched") {
		t.Fatalf("job note not persisted to history")
	}

	// The task-boundary path must not re-deliver an already-delivered job.
	a.injectJobResults()
	if a.ag.SessionLen() != before+1 {
		t.Error("injectJobResults re-delivered a job already drained by beforeTurn")
	}
	if got := a.drainJobResults(); got != nil {
		t.Errorf("drainJobResults re-delivered: %v", got)
	}
}

func TestLineTap(t *testing.T) {
	var got []string
	tap := &lineTap{onLine: func(s string) { got = append(got, s) }}
	// A partial line spanning two writes joins; blank lines are skipped; each
	// complete line is trimmed.
	tap.Write([]byte("hello\nwor"))
	tap.Write([]byte("ld\n\n  spaced  \n"))
	tap.Write([]byte("no newline yet")) // buffered, not emitted
	want := []string{"hello", "world", "spaced"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("lineTap emitted %v, want %v", got, want)
	}
}

func TestJobsShowsLiveProgress(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.jobs = append(a.jobs, &job{id: 1, profile: "codex", task: "fix the repo", started: time.Now()})

	a.setJobProgress(1, "applying patch to foo.go")
	list := strings.Join(a.jobsCommand(""), "\n")
	if !strings.Contains(list, "↳ applying patch to foo.go") || !strings.Contains(list, "running") {
		t.Fatalf("/jobs missing live progress line:\n%s", list)
	}
}

func TestSubagentRenderShowsFullTask(t *testing.T) {
	m := &tuiModel{width: 60}
	task := "Fix ALL issues identified in the code review of the notecli repository and then open a pull request"
	lines := m.renderEvent(uiEvent{kind: "subagent", fields: map[string]any{
		"profile": "codex", "model": "codex", "dir": "/x/notecli", "task": task}})
	if len(lines) < 2 {
		t.Fatalf("expected the head line + wrapped task, got %v", lines)
	}
	norm := strings.Join(strings.Fields(stripAnsi(strings.Join(lines, " "))), " ")
	if !strings.Contains(norm, task) { // the whole task survives, wrapped not clipped
		t.Errorf("full task not shown:\n%s", norm)
	}
}

func TestBackgroundJobStreamsProgress(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.Agents = map[string]config.AgentProfile{
		"streamer": {Kind: "external", Command: "sh", Args: []string{"-c", "echo working on it; sleep 0.1"}, Timeout: 30},
	}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.spawnAgentBackground(context.Background(), "streamer", "do stuff", ""); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200 && a.jobsPending() > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	a.jobMu.Lock()
	last := a.jobs[0].lastLine
	a.jobMu.Unlock()
	if !strings.Contains(last, "working on it") { // the tapped line reached the job
		t.Errorf("job lastLine = %q, want the streamed output", last)
	}
}

func TestParseLoop(t *testing.T) {
	// interval + task, no count cap
	if iv, max, g, ok := parseLoop("5m build it"); !ok || iv != 5*time.Minute || max != 0 || g != "build it" {
		t.Errorf("parseLoop(5m build it) = %v,%d,%q,%v", iv, max, g, ok)
	}
	// interval + xN count cap
	if iv, max, g, ok := parseLoop("30s x10 tail the log"); !ok || iv != 30*time.Second || max != 10 || g != "tail the log" {
		t.Errorf("parseLoop(30s x10 …) = %v,%d,%q,%v", iv, max, g, ok)
	}
	// a bare number is NOT a valid interval anymore → not ok
	if _, _, _, ok := parseLoop("5 build it"); ok {
		t.Error("parseLoop should reject a bare count without a duration interval")
	}
	// missing task or interval → not ok
	if _, _, _, ok := parseLoop("5m"); ok {
		t.Error("parseLoop with no task should be !ok")
	}
	if _, _, _, ok := parseLoop(""); ok {
		t.Error("parseLoop empty should be !ok")
	}
}

func TestLongestCommonPrefix(t *testing.T) {
	if got := longestCommonPrefix([]string{"/color", "/compact"}); got != "/co" {
		t.Errorf("lcp = %q, want /co", got)
	}
	if got := longestCommonPrefix([]string{"/help"}); got != "/help" {
		t.Errorf("lcp single = %q", got)
	}
}

func TestTabCompletesSingleMatch(t *testing.T) {
	m := &tuiModel{input: textarea.New()}
	m.input.SetValue("/comp")
	m.completeCommand()
	if m.input.Value() != "/compact " {
		t.Errorf("completed to %q, want '/compact '", m.input.Value())
	}
}

func TestTabCompletesArgument(t *testing.T) {
	// /ai completes only providers that are actually configured (key set) + local.
	// Clear provider env keys so the test doesn't pick up the host's, then
	// configure only grok via a preset.
	for _, env := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY"} {
		t.Setenv(env, "")
	}
	m := &tuiModel{input: textarea.New(), app: &app{
		cfg: config.Config{Providers: map[string]config.LLM{"grok": {APIKey: "x"}}},
	}}

	m.input.SetValue("/ai l") // only "local" matches
	m.completeCommand()
	if m.input.Value() != "/ai local" {
		t.Errorf("completed to %q, want '/ai local'", m.input.Value())
	}
	// Only grok is configured, so "/ai g" completes straight to grok — groq is
	// NOT offered (no key), so it isn't ambiguous.
	m.input.SetValue("/ai g")
	m.completeCommand()
	if m.input.Value() != "/ai grok" {
		t.Errorf("completed to %q, want '/ai grok' (only configured provider)", m.input.Value())
	}
}

func TestConfigPanelNav(t *testing.T) {
	m := &tuiModel{app: &app{cfg: config.Default()}}
	m.openConfig()
	if m.state != stConfig || m.cfgCursor != 0 {
		t.Fatalf("openConfig: state=%v cursor=%d", m.state, m.cfgCursor)
	}
	n := len(cfgKeys())
	m.configMove(-1) // wrap to last
	if m.cfgCursor != n-1 {
		t.Errorf("configMove(-1) = %d, want %d (wrap)", m.cfgCursor, n-1)
	}
	m.configMove(1) // wrap back to first
	if m.cfgCursor != 0 {
		t.Errorf("configMove(1) = %d, want 0 (wrap)", m.cfgCursor)
	}
	// row views are non-empty for every selectable key
	for _, k := range cfgKeys() {
		if l, _, h := m.configRowView(k); l == "" || h == "" {
			t.Errorf("configRowView(%q) label/hint empty: %q/%q", k, l, h)
		}
	}
}

func TestResolveModelArg(t *testing.T) {
	ids := []string{"openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "x-ai/grok-4.3"}
	// exact id → switch
	if set, _ := resolveModelArg(ids, "x-ai/grok-4.3"); set != "x-ai/grok-4.3" {
		t.Errorf("exact: set=%q", set)
	}
	// unique substring → switch
	if set, _ := resolveModelArg(ids, "sonnet"); set != "anthropic/claude-3.5-sonnet" {
		t.Errorf("unique substring: set=%q", set)
	}
	// ambiguous substring → list, no switch
	if set, lines := resolveModelArg(ids, "gpt-4o"); set != "" || len(lines) != 3 {
		t.Errorf("ambiguous: set=%q lines=%v (want no set, header+2 matches)", set, lines)
	}
	// no match → trust the user verbatim
	if set, _ := resolveModelArg(ids, "brand-new-model"); set != "brand-new-model" {
		t.Errorf("no match: set=%q, want verbatim", set)
	}
	// no list (offline) → verbatim
	if set, _ := resolveModelArg(nil, "whatever"); set != "whatever" {
		t.Errorf("no list: set=%q", set)
	}
}

func TestConfigCycles(t *testing.T) {
	if got := nextStr("ask", permCycle); got != "allow" {
		t.Errorf("nextStr(ask) = %q, want allow", got)
	}
	if got := nextStr("deny", permCycle); got != "ask" {
		t.Errorf("nextStr(deny) = %q, want ask (wrap)", got)
	}
	if got := nextInt(60, timeoutCycle); got != 120 {
		t.Errorf("nextInt(60) = %d, want 120", got)
	}
	if got := nextInt(600, timeoutCycle); got != 60 {
		t.Errorf("nextInt(600) = %d, want 60 (wrap)", got)
	}
	if got := nextInt(0, timeoutCycle); got != 60 {
		t.Errorf("nextInt(0=unset) = %d, want 60", got)
	}
}

func TestHighlightByExtension(t *testing.T) {
	// A .py path must drive the Python lexer (so keywords get coloured) rather
	// than relying on content guessing, which misses Python on short snippets.
	out := highlightCode("def foo():\n    return 42", "organizer/main.py")
	if !strings.Contains(out, "\x1b[") {
		t.Errorf("Python (by .py path) should be colourised, got %q", out)
	}
}

func TestRenderDiff(t *testing.T) {
	m := &tuiModel{width: 80, accent: lipgloss.Color("13")}
	diff := "--- a.txt\n+++ a.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+LINE2\n line3\n"
	out := strings.Join(m.renderDiff("a.txt", diff), "\n")

	if !strings.Contains(out, "Update(a.txt)") {
		t.Error("missing header Update(a.txt)")
	}
	if !strings.Contains(out, "1 line added, 1 line removed") {
		t.Errorf("summary not as expected in:\n%s", out)
	}
	if !strings.Contains(out, bgGreen) || !strings.Contains(out, bgRed) {
		t.Error("missing green/red row backgrounds")
	}
}

// A restored session must survive opening the TUI. newTUIModel re-wires the
// stack to install its bridge, which rebuilds the agent; if that rebuild drops
// the loaded history, every launch starts from a clean slate (the reported bug).
func TestSessionSurvivesTUILaunch(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, ".agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	hist := []llm.Message{llm.User("build a thing"), {Role: "assistant", Content: "built it"}}
	data, _ := json.Marshal(hist)
	if err := os.WriteFile(filepath.Join(ws, ".agent", "session.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.Default()
	cfg.Workspace = ws
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.loadSession()
	if a.ag.SessionLen() != 2 {
		t.Fatalf("after loadSession SessionLen=%d, want 2", a.ag.SessionLen())
	}

	if _, err := a.newTUIModel(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := a.ag.SessionLen(); got != 2 {
		t.Errorf("session dropped when TUI launched: SessionLen=%d, want 2", got)
	}
}

func TestGoalSetPersistAndLoad(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, ".agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	a := &app{workspace: ws, cfg: config.Default()}
	a.setGoal("  ship the feature  ")
	if a.goal.Text != "ship the feature" || a.goal.Status != "active" {
		t.Fatalf("setGoal stored %+v", a.goal)
	}

	// A fresh app over the same workspace recovers the standing goal.
	b := &app{workspace: ws, cfg: config.Default()}
	b.loadGoal()
	if b.goal.Text != "ship the feature" || b.goal.Status != "active" {
		t.Fatalf("loadGoal recovered %+v", b.goal)
	}
}

func TestGoalTTLForOnlyAppliesToActiveGoal(t *testing.T) {
	a := &app{workspace: t.TempDir(), cfg: config.Default()} // GoalMaxReturns=6
	// No goal set → a plain task gets no judge loop.
	if got := a.goalTTLFor("anything"); got != 0 {
		t.Errorf("plain task TTL = %d, want 0", got)
	}
	a.goal = goalState{Text: "the goal", Status: "active"}
	if got := a.goalTTLFor("the goal"); got != 6 {
		t.Errorf("active goal TTL = %d, want 6", got)
	}
	if got := a.goalTTLFor("a different task"); got != 0 {
		t.Errorf("off-goal task TTL = %d, want 0", got)
	}
	a.goal.Status = "done"
	if got := a.goalTTLFor("the goal"); got != 0 {
		t.Errorf("finished goal TTL = %d, want 0", got)
	}
}

func TestLaunchGoalTextDistinguishesSubcommands(t *testing.T) {
	a := &app{workspace: t.TempDir(), cfg: config.Default()}
	for _, sub := range []string{"", "clear", "ttl 5", "off", "on", "go"} {
		if _, ok := a.launchGoalText(sub); ok {
			t.Errorf("%q wrongly treated as a goal to launch", sub)
		}
	}
	if text, ok := a.launchGoalText("build the thing"); !ok || text != "build the thing" {
		t.Errorf("launchGoalText(goal) = %q,%v", text, ok)
	}
	// /goal go resumes the standing goal's text.
	a.goal = goalState{Text: "standing goal", Status: "incomplete"}
	if text, ok := a.launchGoalText("go"); !ok || text != "standing goal" {
		t.Errorf("/goal go = %q,%v, want the standing goal", text, ok)
	}
}

func TestFinishGoalStatusFromTranscript(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, ".agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	a := &app{workspace: ws, cfg: config.Default()}

	a.setGoal("do it")
	a.finishGoal("do it", agent.Transcript{GoalMet: true})
	if a.goal.Status != "done" {
		t.Errorf("met goal status = %q, want done", a.goal.Status)
	}

	a.setGoal("do it")
	a.finishGoal("do it", agent.Transcript{GoalMet: false})
	if a.goal.Status != "incomplete" {
		t.Errorf("unmet goal status = %q, want incomplete", a.goal.Status)
	}

	// A run that isn't the standing goal must not touch its status.
	a.setGoal("do it")
	a.finishGoal("some other task", agent.Transcript{GoalMet: true})
	if a.goal.Status != "active" {
		t.Errorf("off-goal run changed status to %q, want active", a.goal.Status)
	}
}

// A provider defined purely in config with NO api_key (a local Ollama/vLLM/etc.)
// must be usable: listed for switching and connectable. Only a custom provider
// missing base_url is rejected — with a clear message.
func TestKeylessCustomProviderIsUsable(t *testing.T) {
	cfg := config.Default()
	cfg.Providers = map[string]config.LLM{
		"ollama": {BaseURL: "http://localhost:11434/v1", Model: "qwen2.5-coder:7b"}, // no api_key
	}
	a := &app{cfg: cfg, workspace: t.TempDir()}

	listed := false
	for _, n := range a.configuredProviderNames() {
		if n == "ollama" {
			listed = true
		}
	}
	if !listed {
		t.Errorf("keyless custom provider not listed: %v", a.configuredProviderNames())
	}
	if l, msg := a.providerConn("ollama"); msg != "" || l.BaseURL == "" {
		t.Errorf("providerConn(ollama) = %+v, %q; want a keyless connection, no error", l, msg)
	}

	cfg.Providers["broken"] = config.LLM{Model: "x"} // custom but no base_url
	if _, msg := a.providerConn("broken"); !strings.Contains(msg, "base_url") {
		t.Errorf("providerConn(broken) msg = %q, want a base_url complaint", msg)
	}
}

func TestInputHistoryRecallAndPersist(t *testing.T) {
	ws := t.TempDir()
	m := &tuiModel{input: textarea.New(), app: &app{cfg: config.Default(), workspace: ws}}
	m.recordInput("first")
	m.recordInput("second")
	m.recordInput("second") // a consecutive duplicate is not stored twice
	if len(m.app.promptHist) != 2 {
		t.Fatalf("promptHist = %v, want [first second]", m.app.promptHist)
	}

	m.input.SetValue("draft-in-progress")
	m.histIdx = len(m.app.promptHist) // not browsing
	m.historyPrev()
	if m.input.Value() != "second" {
		t.Errorf("↑ once = %q, want second", m.input.Value())
	}
	m.historyPrev()
	if m.input.Value() != "first" {
		t.Errorf("↑ twice = %q, want first", m.input.Value())
	}
	m.historyPrev() // clamp at the oldest
	if m.input.Value() != "first" {
		t.Errorf("↑ past oldest = %q, want first", m.input.Value())
	}
	m.historyNext()
	if m.input.Value() != "second" {
		t.Errorf("↓ once = %q, want second", m.input.Value())
	}
	m.historyNext() // back past the newest restores the stashed draft
	if m.input.Value() != "draft-in-progress" {
		t.Errorf("↓ past newest = %q, want the stashed draft", m.input.Value())
	}

	// A fresh app over the same workspace recovers the history (survives restart).
	b := &app{cfg: config.Default(), workspace: ws}
	b.loadPromptHist()
	if len(b.promptHist) != 2 || b.promptHist[1] != "second" {
		t.Errorf("persisted history = %v, want [first second]", b.promptHist)
	}
}

func TestIsCommandLine(t *testing.T) {
	for _, c := range []struct {
		in   string
		want bool
	}{{"/plan", true}, {"!ls", true}, {"!", true}, {"fix the bug", false}, {"", false}} {
		if got := isCommandLine(c.in); got != c.want {
			t.Errorf("isCommandLine(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// shift+tab during a task doesn't flip the mode live (that races the running
// agent) — it queues the switch and applies it when the task finishes.
func TestDeferredModeSwitchWhileRunning(t *testing.T) {
	m := &tuiModel{state: stRunning, input: textarea.New(), app: &app{cfg: config.Default()}}
	m.app.planMode = false // start in auto

	m.handleKey(tea.KeyMsg{Type: tea.KeyShiftTab})
	if m.pendingMode == nil || !*m.pendingMode {
		t.Fatalf("shift+tab while running should queue plan mode, pending=%v", m.pendingMode)
	}
	if m.app.planMode {
		t.Error("mode must NOT flip live while a task runs")
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyShiftTab}) // toggles the queued target back
	if *m.pendingMode {
		t.Error("second shift+tab should flip the queued target back to auto")
	}

	plan := true
	m.pendingMode = &plan
	m.applyPendingMode()
	if !m.app.planMode {
		t.Error("applyPendingMode should have switched to plan")
	}
	if m.pendingMode != nil {
		t.Error("pending mode should be cleared after applying")
	}
}

// countingApprover records how many times the real prompt was hit.
type countingApprover struct {
	calls int
	reply bool
}

func (c *countingApprover) Approve(_, _ string) bool { c.calls++; return c.reply }

// An "allow this kind for the session" grant short-circuits the real prompt for the
// whole category, leaves other categories asking, and clears on reset.
func TestSessionAllowGate(t *testing.T) {
	inner := &countingApprover{reply: false}
	a := &app{cfg: config.Default(), approver: inner}

	if a.approveGated("write", "x") { // not allowed yet → hits inner (denies)
		t.Error("write should be denied by the inner approver before any grant")
	}
	if inner.calls != 1 {
		t.Fatalf("inner calls = %d, want 1", inner.calls)
	}

	a.allowSession("edit") // edit → "file" category
	if !a.approveGated("write", "y") {
		t.Error("write should be auto-allowed after a file-category session grant")
	}
	if inner.calls != 1 {
		t.Errorf("inner was called again (%d) despite the session grant", inner.calls)
	}
	if a.approveGated("run", "z"); inner.calls != 2 {
		t.Errorf("a different category (run) must still hit the prompt; calls=%d", inner.calls)
	}

	a.resetSessionAllow()
	a.approveGated("write", "w")
	if inner.calls != 3 {
		t.Errorf("after reset, write must hit the prompt again; calls=%d", inner.calls)
	}
}

// closePanel returns to the running task if one is still live, else to idle —
// finalizing (draining the queue) a task that finished while the panel was open.
func TestClosePanelStates(t *testing.T) {
	// Task still running behind the panel → back to stRunning.
	m := &tuiModel{state: stConfig, input: textarea.New(), app: &app{cfg: config.Default()}}
	m.cancel = func() {}
	if model, _ := m.closePanel(); model.(*tuiModel).state != stRunning {
		t.Errorf("with a live task, closePanel → %v, want stRunning", m.state)
	}

	// Task finished while the panel was open → idle, and the queued command drains.
	m = &tuiModel{state: stConfig, input: textarea.New(), app: &app{cfg: config.Default()}}
	m.taskDoneAway = true
	m.queued = []string{"/help"}
	if _, _ = m.closePanel(); m.state != stIdle {
		t.Errorf("after task-done-away, closePanel → %v, want stIdle", m.state)
	}
	if len(m.queued) != 0 {
		t.Errorf("queued command should have drained on close, left %v", m.queued)
	}
	if m.taskDoneAway {
		t.Error("taskDoneAway should be cleared after finalizing")
	}
}

// A plan-mode task that finishes opens the accept→execute handshake: enter switches
// to auto and launches execution; esc keeps planning; a cancelled run skips it.
func TestPlanReviewHandshake(t *testing.T) {
	newM := func() *tuiModel {
		m := &tuiModel{state: stRunning, accent: lipgloss.Color("13"), input: textarea.New(),
			bridge: newBridge(), ctx: context.Background(), app: &app{cfg: config.Default()}}
		m.app.windowDetected = true // detectWindowCmd → nil (no probe)
		m.app.client = llm.NewOpenAIClient(config.LLM{})
		m.app.ag = agent.New(m.app.client, tool.NewRegistry(tool.NewCalc()), nil, nil, "", 5)
		m.app.planMode = true
		m.planTask = true
		return m
	}

	// plan task finishes → plan-review prompt
	m := newM()
	m.Update(taskDoneMsg{})
	if m.state != stPlanReview {
		t.Fatalf("after a plan task, state = %v, want stPlanReview", m.state)
	}
	// enter = accept → switch to auto, a run starts (the cmd's goroutine isn't executed here)
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.app.planMode || m.state != stRunning {
		t.Errorf("accept → planMode=%v state=%v; want auto + running", m.app.planMode, m.state)
	}

	// esc = keep planning → idle, plan mode preserved
	m = newM()
	m.Update(taskDoneMsg{})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.state != stIdle || !m.app.planMode {
		t.Errorf("keep planning → state=%v planMode=%v; want stIdle + plan kept", m.state, m.app.planMode)
	}

	// a cancelled plan run must NOT pop the handshake
	m = newM()
	m.taskCancelled = true
	m.Update(taskDoneMsg{})
	if m.state == stPlanReview {
		t.Error("a cancelled plan task should skip plan-review")
	}
}

func TestReverseSearch(t *testing.T) {
	app := &app{promptHist: []string{"fix the parser", "add tests", "fix the ci build", "write docs"}}
	m := &tuiModel{input: textarea.New(), app: app}

	m.searchQuery = "fix"
	if got := m.searchFrom(len(app.promptHist) - 1); got != 2 {
		t.Errorf("newest 'fix' = idx %d, want 2", got)
	}
	if got := m.searchFrom(1); got != 0 {
		t.Errorf("older 'fix' from idx 1 = %d, want 0", got)
	}
	m.searchQuery = "zzz"
	if got := m.searchFrom(3); got != -1 {
		t.Errorf("no match = %d, want -1", got)
	}

	// full flow: ctrl+r → type → enter drops the match into the input
	m2 := &tuiModel{state: stIdle, input: textarea.New(), app: app}
	m2.handleKey(tea.KeyMsg{Type: tea.KeyCtrlR})
	if m2.state != stHistSearch {
		t.Fatalf("ctrl+r → %v, want stHistSearch", m2.state)
	}
	for _, r := range "docs" {
		m2.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	if m2.searchIdx != 3 {
		t.Errorf("search 'docs' → idx %d, want 3", m2.searchIdx)
	}
	m2.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m2.state != stIdle || m2.input.Value() != "write docs" {
		t.Errorf("enter → state=%v input=%q, want idle + 'write docs'", m2.state, m2.input.Value())
	}
}

func TestGoalResumeOffer(t *testing.T) {
	m := &tuiModel{state: stIdle, input: textarea.New(), accent: lipgloss.Color("13"),
		bridge: newBridge(), ctx: context.Background(), app: &app{cfg: config.Default(), workspace: t.TempDir()}}
	m.app.windowDetected = true
	m.app.client = llm.NewOpenAIClient(config.LLM{})
	m.app.ag = agent.New(m.app.client, tool.NewRegistry(tool.NewCalc()), nil, nil, "", 5)

	m.offerGoalResume() // no goal → nothing offered
	if m.state != stIdle {
		t.Errorf("no goal → %v, want stIdle", m.state)
	}

	m.app.goal = goalState{Text: "ship it", Status: "incomplete"}
	m.offerGoalResume()
	if m.state != stGoalResume {
		t.Fatalf("unfinished goal → %v, want stGoalResume", m.state)
	}

	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc}) // not now
	if m.state != stIdle || m.app.goal.Text != "ship it" {
		t.Errorf("esc → state=%v goal=%q; want idle + goal kept", m.state, m.app.goal.Text)
	}

	m.offerGoalResume()
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // resume
	if m.state != stRunning || m.app.goal.Status != "active" {
		t.Errorf("resume → state=%v status=%q; want running + active", m.state, m.app.goal.Status)
	}
}

func TestAtFileCompletion(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"src/main.go", "src/model.go", "README.md"} {
		os.WriteFile(filepath.Join(dir, filepath.FromSlash(p)), []byte("x"), 0o644)
	}
	c := config.Default()
	c.Workspace = dir
	c.File = config.FilePolicy{Default: "allow", Jail: "."}
	e, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	a := &app{cfg: c, pol: e}

	if _, matches := a.completePath("README"); len(matches) != 1 || matches[0] != "README.md" {
		t.Errorf("README → %v, want [README.md]", matches)
	}
	if lcp, matches := a.completePath("src/m"); len(matches) != 2 || lcp != "src/m" {
		t.Errorf("src/m → lcp %q matches %v, want src/m + 2", lcp, matches)
	}

	// Tab on "@src/mai" completes to the single file.
	m := &tuiModel{input: textarea.New(), app: a}
	m.input.SetValue("look at @src/mai")
	m.completeAtFile()
	if m.input.Value() != "look at @src/main.go " {
		t.Errorf("completeAtFile → %q, want 'look at @src/main.go '", m.input.Value())
	}
}

func TestZaiProvider(t *testing.T) {
	t.Setenv("ZAI_API_KEY", "zk-test")
	l, ok := config.ResolveProvider(config.Config{}, "zai")
	if !ok || l.BaseURL != "https://api.z.ai/api/paas/v4" || l.Model != "glm-5.2" || l.APIKey != "zk-test" {
		t.Errorf("ResolveProvider(zai) = %+v,%v — want the template + env key", l, ok)
	}
	// GLM thinking is binary: off → disabled, any level → enabled.
	if s, ok := reasoningShape("zai", "off"); !ok || !strings.Contains(string(s), "disabled") {
		t.Errorf("zai off = %s,%v", s, ok)
	}
	if s, ok := reasoningShape("zai", "high"); !ok || !strings.Contains(string(s), "enabled") {
		t.Errorf("zai high = %s,%v", s, ok)
	}
}

// The /config "add provider" row opens an IN-PANEL form (no dump back to the
// prompt): name → URL → model → key, esc steps back, save registers the provider.
func TestConfigAddProviderFlow(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // SaveProviders writes the global config
	m := &tuiModel{state: stConfig, width: 100, input: textarea.New(),
		app: &app{cfg: config.Default(), workspace: t.TempDir()}}
	for i, k := range cfgKeys() { // put the cursor on the add-provider row
		if k == "addprovider" {
			m.cfgCursor = i
		}
	}
	typeIn := func(s string) {
		for _, r := range s {
			m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		}
	}
	enter := func() { m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) }

	m.configActivate()
	if m.cfgPhase != cfgPhaseName {
		t.Fatalf("activate → phase %d, want name form", m.cfgPhase)
	}
	enter() // empty name is refused
	if m.cfgPhase != cfgPhaseName {
		t.Fatal("empty name must not advance")
	}
	typeIn("ollama")
	enter()
	typeIn("localhost:11434/v1") // no scheme — must be refused
	enter()
	if m.cfgPhase != cfgPhaseURL {
		t.Fatal("URL without a scheme must not advance")
	}
	m.cfgDraft.url = "http://localhost:11434/v1"
	enter()
	enter() // model: optional, skip
	enter() // key: optional (keyless), save
	if m.cfgPhase != cfgPhaseList {
		t.Errorf("after save → phase %d, want back at the list", m.cfgPhase)
	}
	if m.state != stConfig {
		t.Errorf("must stay IN the panel, state=%v", m.state)
	}
	p, ok := m.app.cfg.Providers["ollama"]
	if !ok || p.BaseURL != "http://localhost:11434/v1" || p.APIKey != "" {
		t.Errorf("provider not saved keyless: %+v ok=%v", p, ok)
	}

	// esc from the first field returns to the list, not out of the panel
	m.configActivate()
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.cfgPhase != cfgPhaseList || m.state != stConfig {
		t.Errorf("esc → phase %d state %v, want list inside the panel", m.cfgPhase, m.state)
	}
}

// ctxMeter thresholds: empty without data, dim under 50%, warn at ≥50%, red⚠ at
// the auto-compact line (the styles degrade to plain text in tests).
func TestCtxMeterFor(t *testing.T) {
	if got := ctxMeterFor(0, 1000); got != "" {
		t.Errorf("no usage → %q, want empty", got)
	}
	if got := ctxMeterFor(500, 0); got != "" {
		t.Errorf("no window → %q, want empty", got)
	}
	for _, c := range []struct {
		used int
		want string
	}{{300, "ctx 30%"}, {600, "ctx 60%"}, {800, "ctx 80%⚠"}} {
		if got := ctxMeterFor(c.used, 1000); !strings.Contains(got, c.want) {
			t.Errorf("used=%d → %q, want it to contain %q", c.used, got, c.want)
		}
	}
}

// externalResult keeps the parent's context lean: stdout tail always, stderr only
// on failure, diff summary with the /diff pointer.
func TestExternalResult(t *testing.T) {
	ok := externalResult("codex", "did the thing", "warn noise", "1 file changed", nil)
	if !strings.Contains(ok, "did the thing") || strings.Contains(ok, "warn noise") {
		t.Errorf("success must include stdout, not stderr:\n%s", ok)
	}
	if !strings.Contains(ok, "1 file changed") || !strings.Contains(ok, "/diff") {
		t.Errorf("diff summary + /diff pointer missing:\n%s", ok)
	}
	bad := externalResult("codex", "partial", "boom", "", fmt.Errorf("exit 1"))
	if !strings.Contains(bad, "boom") {
		t.Errorf("failure must include the stderr tail:\n%s", bad)
	}
	long := externalResult("codex", strings.Repeat("x", maxExternalStdout+500), "", "", nil)
	if !strings.Contains(long, "…") {
		t.Error("oversized stdout should be tail-clipped with a marker")
	}
}

// While a task runs behind the /config panel, changing a value must be refused
// (applying would re-wire the agent the task is using).
func TestConfigViewOnlyWhileBusy(t *testing.T) {
	m := &tuiModel{state: stConfig, width: 80, input: textarea.New(),
		app: &app{cfg: config.Default(), workspace: t.TempDir()}}
	m.cancel = func() {} // a live task
	m.cfgCursor = 1      // any actionable row
	before := m.app.cfg
	m.configActivate()
	if !strings.Contains(strings.Join(m.history, "\n"), "view-only") {
		t.Error("expected the view-only notice")
	}
	if m.app.cfg.Provider != before.Provider || m.app.cfg.File.Default != before.File.Default {
		t.Error("a setting changed while a task was running")
	}
}

// A type-ahead queued during a NON-task async op (compact/update/model/skills)
// must run when the op finishes — it used to sit in the queue forever.
func TestAsyncOpsDrainQueue(t *testing.T) {
	m := &tuiModel{state: stRunning, width: 80, input: textarea.New(),
		app: &app{cfg: config.Default(), workspace: t.TempDir()}}
	m.app.client = llm.NewOpenAIClient(config.LLM{})
	m.queued = []string{"/help"}
	m.Update(updateDoneMsg{text: "done"})
	if len(m.queued) != 0 {
		t.Errorf("queue not drained after an async op: %v", m.queued)
	}
	if m.state != stIdle {
		t.Errorf("state = %v, want stIdle", m.state)
	}
	if !strings.Contains(strings.Join(m.history, "\n"), "commands") {
		t.Error("the queued /help should have executed on drain")
	}
}

// With the judge loop off (/goal off) a clean finish must mark the goal done —
// otherwise the startup resume prompt nags forever. Cancelled runs stay incomplete.
func TestGoalDoneWithJudgeOff(t *testing.T) {
	a := &app{cfg: config.Default(), workspace: t.TempDir()}
	a.cfg.GoalMaxReturns = 0 // judge loop off
	a.goal = goalState{Text: "ship it", Status: "active"}
	a.finishGoal("ship it", agent.Transcript{}) // clean finish, GoalMet never set
	if a.goal.Status != "done" {
		t.Errorf("clean TTL-off finish → %q, want done", a.goal.Status)
	}
	a.goal = goalState{Text: "ship it", Status: "active"}
	a.finishGoal("ship it", agent.Transcript{Cancelled: true})
	if a.goal.Status != "incomplete" {
		t.Errorf("cancelled run → %q, want incomplete", a.goal.Status)
	}
}

// Fire-and-forget: agent.run(background=true) returns an ack at once, the job
// runs detached, /jobs tracks it, and injectJobResults folds the result into the
// conversation at the next turn boundary.
func TestBackgroundJobLifecycle(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.Agents = map[string]config.AgentProfile{
		"echo":  {Kind: "external", Command: "echo", Args: []string{"ok:", "{task}"}},
		"sleep": {Kind: "external", Command: "sleep", Timeout: 30},
	}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	ack, err := a.spawnAgentBackground(context.Background(), "echo", "ping the thing", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ack, "job #1") {
		t.Errorf("ack = %q, want an immediate job #1 acknowledgement", ack)
	}
	for i := 0; i < 100 && a.jobsPending() > 0; i++ { // wait for the detached run
		time.Sleep(20 * time.Millisecond)
	}
	list := strings.Join(a.jobsCommand(""), "\n")
	if !strings.Contains(list, "✓ #1") {
		t.Fatalf("job not finished ok:\n%s", list)
	}
	if full := strings.Join(a.jobsCommand("result 1"), "\n"); !strings.Contains(full, "ok: ping the thing") {
		t.Errorf("/jobs result = %q", full)
	}

	// next turn boundary: the result lands in the conversation exactly once
	before := a.ag.SessionLen()
	a.injectJobResults()
	h := a.ag.History()
	if len(h) != before+1 || !strings.Contains(h[len(h)-1].Content, "ok: ping the thing") {
		t.Fatalf("result not injected into history")
	}
	a.injectJobResults()
	if a.ag.SessionLen() != before+1 {
		t.Error("re-injection must not duplicate the note")
	}

	// unknown profile fails NOW, not silently inside a job
	if _, err := a.spawnAgentBackground(context.Background(), "zzz-nope-xyz", "x", ""); err == nil {
		t.Error("unknown profile must fail immediately")
	}

	// kill: a long job is cancelled and records a failure outcome
	if _, err := a.spawnAgentBackground(context.Background(), "sleep", "25", ""); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond) // let it start
	if out := strings.Join(a.jobsCommand("kill 2"), " "); !strings.Contains(out, "cancelling") {
		t.Fatalf("kill = %q", out)
	}
	for i := 0; i < 100 && a.jobsPending() > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}
	if list := strings.Join(a.jobsCommand(""), "\n"); !strings.Contains(list, "✖ #2") {
		t.Errorf("killed job should record failure:\n%s", list)
	}
}

// The agent tool routes background=true to the fire-and-forget spawn func.
func TestAgentToolBackgroundRouting(t *testing.T) {
	var fg, bg bool
	tl := tool.NewAgent(
		func(_ context.Context, _, _, _ string) (string, error) { fg = true; return "fg", nil },
		func(_ context.Context, _, _, _ string) (string, error) { bg = true; return "bg", nil },
	)
	tl.Call(context.Background(), "run", map[string]any{"profile": "p", "task": "t", "background": true})
	if !bg || fg {
		t.Errorf("background=true → bg func, got fg=%v bg=%v", fg, bg)
	}
	fg, bg = false, false
	tl.Call(context.Background(), "run", map[string]any{"profile": "p", "task": "t"})
	if !fg || bg {
		t.Errorf("default → foreground func, got fg=%v bg=%v", fg, bg)
	}
}

// The stAgents panel must not funnel an external profile into the LLM builder
// (that would silently convert it) — enter hands off to its add-tool line; the
// "+ add CLI tool" row hands off to the scan + prefill.
func TestAgentsPanelExternalRows(t *testing.T) {
	cfg := config.Default()
	cfg.Agents = map[string]config.AgentProfile{"codex": {Kind: "external", Command: "codex", Args: []string{"exec", "{task}"}}}
	m := &tuiModel{state: stAgents, input: textarea.New(), app: &app{cfg: cfg, workspace: t.TempDir()}}

	// enter on the external profile → prefilled add-tool line, panel closed
	m.agCursor = 0
	m.agentsKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.state != stIdle || m.input.Value() != "/agents add-tool codex codex exec {task}" {
		t.Errorf("external edit → state=%v input=%q", m.state, m.input.Value())
	}

	// the "+ add CLI tool" row (after the profile and "+ add new") → in-panel picker
	binDir := t.TempDir() // a fake `codex` on PATH so the picker sees ✓ installed
	os.WriteFile(filepath.Join(binDir, "codex"), []byte("#!/bin/sh\nexit 0\n"), 0o755)
	t.Setenv("PATH", binDir)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // add-tool persists via SaveAgents

	m.state, m.agPhase = stAgents, agList
	m.input.SetValue("")
	m.agCursor = 2 // 1 profile, then "add new", then "add CLI tool"
	m.agentsKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.state != stAgents || m.agPhase != agPickTool {
		t.Fatalf("add-CLI-tool row → state=%v phase=%v, want the in-panel picker", m.state, m.agPhase)
	}
	if !m.agInstalled["codex"] || m.agInstalled["gemini"] {
		t.Errorf("PATH scan wrong: %v (fake codex installed, gemini not)", m.agInstalled)
	}

	// enter on a NOT-installed CLI refuses and stays in the picker
	m.agCursor = 2 // gemini
	m.agentsKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.agPhase != agPickTool {
		t.Error("picking a missing CLI must stay in the picker")
	}

	// enter on the installed one registers it and returns to the list
	delete(m.app.cfg.Agents, "codex") // drop the pre-seeded profile; re-add via picker
	m.agCursor = 0                    // codex
	m.agentsKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.agPhase != agList {
		t.Errorf("after adding, phase=%v want back at the list", m.agPhase)
	}
	if p := m.app.cfg.Agents["codex"]; p.Kind != "external" || p.Command != "codex" {
		t.Errorf("picker didn't register codex: %+v", p)
	}

	// the custom… row hands off to the freeform add-tool line
	m.agPhase, m.agCursor = agPickTool, len(externalCatalog)
	m.agentsKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.state != stIdle || m.input.Value() != "/agents add-tool " {
		t.Errorf("custom row → state=%v input=%q", m.state, m.input.Value())
	}
}

func TestAddToolCatalog(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // SaveAgents writes the global config
	a := &app{cfg: config.Default(), workspace: t.TempDir()}

	// bare add-tool lists the catalog with install markers
	list := strings.Join(a.agentsAddExternal(""), "\n")
	for _, want := range []string{"codex", "claude", "aider", "add-tool <name> <command>"} {
		if !strings.Contains(list, want) {
			t.Errorf("catalog listing missing %q:\n%s", want, list)
		}
	}

	// one-word add of a known CLI: catalog flags; PATH check still applies
	out := strings.Join(a.agentsAddExternal("codex"), " ")
	if _, err := exec.LookPath("codex"); err != nil {
		if !strings.Contains(out, "not found in PATH") {
			t.Errorf("codex absent → want the PATH hint, got %q", out)
		}
	} else if _, ok := a.cfg.Agents["codex"]; !ok {
		t.Errorf("codex present → profile should be saved, got %q", out)
	}

	// one-word add of an unknown tool: asks for the full launch, lists the catalog
	out = strings.Join(a.agentsAddExternal("mystery"), " ")
	if !strings.Contains(out, "full launch") || !strings.Contains(out, "codex") {
		t.Errorf("unknown one-word add = %q", out)
	}

	// full form still works with any binary (echo is everywhere)
	out = strings.Join(a.agentsAddExternal("ek echo {task}"), " ")
	if !strings.Contains(out, "saved") {
		t.Fatalf("full form = %q", out)
	}
	if p := a.cfg.Agents["ek"]; p.Kind != "external" || p.Command != "echo" || len(p.Args) != 1 {
		t.Errorf("saved profile = %+v", p)
	}

	// an existing LLM profile is never silently clobbered by add-tool
	a.cfg.Agents["grok"] = config.AgentProfile{Provider: "openrouter", Model: "x-ai/grok"}
	out = strings.Join(a.agentsAddExternal("grok echo {task}"), " ")
	if !strings.Contains(out, "LLM profile") {
		t.Errorf("clobbering an LLM profile should be refused, got %q", out)
	}
	if a.cfg.Agents["grok"].Kind == "external" {
		t.Error("the LLM profile was overwritten")
	}
}

func TestExpandTaskArgs(t *testing.T) {
	// {{task}} must be replaced before {task} (it contains it) — no stray braces.
	got := expandTaskArgs([]string{"exec", "{{task}}", "--x={task}"}, "fix it")
	want := []string{"exec", "fix it", "--x=fix it"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("arg %d = %q, want %q", i, got[i], want[i])
		}
	}
	// no placeholder anywhere → the task is appended
	got = expandTaskArgs([]string{"-p"}, "do x")
	if len(got) != 2 || got[1] != "do x" {
		t.Errorf("append fallback = %v, want [-p, do x]", got)
	}
	if got = expandTaskArgs(nil, "solo"); len(got) != 1 || got[0] != "solo" {
		t.Errorf("nil args = %v, want [solo]", got)
	}
}

// The external-agent approval must be its OWN category: an allow-session granted
// for ordinary sub-agent spawns must not silently cover unsandboxed CLI agents.
func TestExternalApprovalCategorySeparate(t *testing.T) {
	if approvalCategory("external agent") == approvalCategory("spawn agent") {
		t.Fatal("external agents must not share the spawn approval bucket")
	}
	a := &app{}
	a.allowSession("spawn agent") // user pressed 'a' on a normal spawn
	if a.sessionAllowed(approvalCategory("external agent")) {
		t.Error("allow-session for spawns leaked to external agents")
	}
}

func TestTailClip(t *testing.T) {
	if got := textutil.Tail("short", 100); got != "short" {
		t.Errorf("under cap = %q", got)
	}
	long := strings.Repeat("щ", 100) // 2-byte runes — the cut lands mid-rune
	got := textutil.Tail(long, 51)
	if !utf8.ValidString(got) {
		t.Errorf("tailClip split a rune: %q", got)
	}
	if !strings.HasPrefix(got, "…") {
		t.Errorf("clipped tail should mark truncation: %q", got)
	}
}

func TestSpawnExternalAgent(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Default()
	cfg.Workspace = ws
	cfg.Agents = map[string]config.AgentProfile{
		"echo":  {Kind: "external", Command: "echo", Args: []string{"ok:", "{task}"}},
		"sleep": {Kind: "external", Command: "sleep", Timeout: 1},
	}
	a := &app{cfg: cfg, workspace: ws, approver: fixedApprover(true)}

	// dispatch goes through spawnAgent's Kind branch (no LLM provider needed)
	out, err := a.spawnAgent(context.Background(), "echo", "do the thing", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "ok: do the thing") {
		t.Errorf("output = %q, want the echoed task", out)
	}

	// denial refuses the launch — external agents ask even when spawns are relaxed
	a.cfg.Spawn.Default = "allow"
	a.approver = fixedApprover(false)
	if _, err := a.spawnAgent(context.Background(), "echo", "x", ""); err == nil || !strings.Contains(err.Error(), "denied") {
		t.Errorf("denied external agent = %v, want an error", err)
	}

	// a hung (interactive) CLI is killed at its timeout
	a.approver = fixedApprover(true)
	start := time.Now()
	_, err = a.spawnAgent(context.Background(), "sleep", "5", "")
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Errorf("timeout = %v, want timed out", err)
	}
	if time.Since(start) > 4*time.Second {
		t.Error("timeout did not kill the process promptly")
	}
}

func TestDiffCommand(t *testing.T) {
	dir := t.TempDir()
	git := func(args ...string) {
		if err := exec.Command("git", append([]string{"-C", dir}, args...)...).Run(); err != nil {
			t.Skipf("git unavailable: %v", err)
		}
	}
	git("init")
	git("config", "user.email", "a@b.c")
	git("config", "user.name", "t")
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("one\n"), 0o644)
	git("add", "-A")
	git("commit", "-m", "init")
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("two\n"), 0o644) // modified
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("new\n"), 0o644) // untracked

	a := &app{cfg: config.Default(), workspace: dir}
	out := strings.Join(a.diffCommand(), "\n")
	for _, want := range []string{"a.txt", "+two", "b.txt"} {
		if !strings.Contains(out, want) {
			t.Errorf("/diff missing %q in:\n%s", want, out)
		}
	}
	// a non-repo dir → a clear message, not a crash
	a2 := &app{cfg: config.Default(), workspace: t.TempDir()}
	if !strings.Contains(strings.Join(a2.diffCommand(), " "), "git repo") {
		t.Errorf("/diff in a non-repo should say it needs a git repo")
	}
}

func TestColorizeDiff(t *testing.T) {
	if colorizeDiff("context line") != "context line" {
		t.Error("plain lines should pass through unchanged")
	}
	for _, s := range []string{"+added", "-removed", "@@ hunk @@"} {
		if !strings.Contains(colorizeDiff(s), strings.TrimLeft(s, "+-@ ")) {
			t.Errorf("colorizeDiff(%q) dropped its text", s)
		}
	}
}

func TestBudgetGuard(t *testing.T) {
	a := &app{cfg: config.Default()}
	a.sessionCostUSD = 100
	if a.budgetExceeded() {
		t.Error("budget off (0) → never exceeded")
	}
	a.cfg.SessionBudgetUSD = 5
	a.sessionCostUSD = 4.99
	if a.budgetExceeded() {
		t.Error("under the cap → not exceeded")
	}
	a.sessionCostUSD = 5.0
	if !a.budgetExceeded() {
		t.Error("at the cap → exceeded")
	}
}

func TestReasoningLevelAndCycle(t *testing.T) {
	a := &app{cfg: config.Default()}
	a.cfg.Reasoning = map[string]json.RawMessage{}

	if got := a.reasoningLevel("openai", "gpt-4o"); got != "default" {
		t.Errorf("no override → %q, want default", got)
	}
	a.cfg.Reasoning["openai/gpt-4o"] = json.RawMessage(`{"reasoning_effort":"low"}`)
	if got := a.reasoningLevel("openai", "gpt-4o"); got != "low" {
		t.Errorf("low shape → %q, want low", got)
	}
	a.cfg.Reasoning["openai/gpt-4o"] = json.RawMessage(`{"weird":1}`)
	if got := a.reasoningLevel("openai", "gpt-4o"); got != "custom" {
		t.Errorf("unknown shape → %q, want custom", got)
	}

	if got := nextReasoning("off"); got != "minimal" {
		t.Errorf("off → %q, want minimal", got)
	}
	if got := nextReasoning("high"); got != "off" {
		t.Errorf("high → %q, want off (wrap)", got)
	}
	if got := nextReasoning("default"); got != "off" {
		t.Errorf("default → %q, want off (cycle head)", got)
	}
}

func TestApprovalCategory(t *testing.T) {
	for kind, want := range map[string]string{
		"write": "file", "edit": "file", "append": "file", "mkdir": "file",
		"run": "run", "git": "git", "spawn agent": "spawn", "mcp call": "mcp",
	} {
		if got := approvalCategory(kind); got != want {
			t.Errorf("approvalCategory(%q) = %q, want %q", kind, got, want)
		}
	}
}

func TestSlugName(t *testing.T) {
	cases := map[string]string{"Alice Bot": "alice-bot", "  ": "ipsupport-code", "C++/Helper!": "c-helper", "": "ipsupport-code"}
	for in, want := range cases {
		if got := slugName(in); got != want {
			t.Errorf("slugName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSessionsKeyedByName(t *testing.T) {
	ws := t.TempDir()
	a := &app{workspace: ws, cfg: config.Config{Name: "bob"}}
	if got := len(a.listSessions()); got != 0 {
		t.Fatalf("listSessions = %d, want 0 (none yet)", got)
	}
	// write a session for "bob"
	if err := os.MkdirAll(filepath.Dir(a.sessionPath()), 0o755); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal([]llm.Message{llm.User("g1"), {Role: "assistant", Content: "a1"}})
	if err := os.WriteFile(a.sessionPath(), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(a.sessionPath(), filepath.Join(".agent", "sessions", "bob.json")) {
		t.Errorf("sessionPath = %q, want …/sessions/bob.json", a.sessionPath())
	}
	if got := a.listSessions(); len(got) != 1 || got[0].name != "bob" || got[0].count != 2 {
		t.Errorf("listSessions = %+v, want one 'bob' with 2 messages", got)
	}
	a.deleteSessionNamed("bob")
	if got := len(a.listSessions()); got != 0 {
		t.Errorf("after delete = %d, want 0", got)
	}
}

func TestNewSessionPreservesOld(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // newNamedSession(persist) writes the global config
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.ag.SetHistory([]llm.Message{llm.User("g0"), {Role: "assistant", Content: "a0"}})
	a.saveSession() // ipsupport-code.json

	auto := a.autoSessionName()
	if auto != "ipsupport-code-2" {
		t.Fatalf("autoSessionName = %q, want ipsupport-code-2", auto)
	}
	if err := a.newNamedSession(auto, false); err != nil { // bare /new (scratch, no persist)
		t.Fatal(err)
	}
	if a.ag.SessionLen() != 0 {
		t.Errorf("new session should start empty, got %d", a.ag.SessionLen())
	}
	a.ag.SetHistory([]llm.Message{llm.User("g1")})
	a.saveSession()

	names := map[string]bool{}
	for _, s := range a.listSessions() {
		names[s.name] = true
	}
	if !names["ipsupport-code"] || !names["ipsupport-code-2"] {
		t.Errorf("both sessions should exist, got %v", names)
	}
	if loaded, _ := config.Load(cfg.Workspace); loaded.Name == "ipsupport-code-2" {
		t.Errorf("a bare /new must not persist the auto name as the default identity")
	}
}

func TestSessionsListSwitchDelete(t *testing.T) {
	t.Setenv("HOME", t.TempDir())            // isolate SaveGlobal from the real ~/.config
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // (belt-and-suspenders)
	cfg := config.Default()                  // default name "ipsupport-code"
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	// save a thread under the default name
	a.ag.SetHistory([]llm.Message{llm.User("g0"), {Role: "assistant", Content: "a0"}})
	a.saveSession()

	// switch to a new name "alice" — adopts the name, fresh (empty) thread
	lines, switched := a.sessionsCommand("alice")
	if !switched || a.cfg.Name != "alice" {
		t.Fatalf("switch: switched=%v name=%q lines=%v", switched, a.cfg.Name, lines)
	}
	if a.ag.SessionLen() != 0 {
		t.Errorf("alice should start empty, got %d", a.ag.SessionLen())
	}
	a.ag.SetHistory([]llm.Message{llm.User("g1"), {Role: "assistant", Content: "a1"}})
	a.saveSession()

	// list shows both, alice active
	list, _ := a.sessionsCommand("")
	joined := strings.Join(list, "\n")
	if !strings.Contains(joined, "alice") || !strings.Contains(joined, "ipsupport-code") {
		t.Errorf("list missing a session:\n%s", joined)
	}
	if !strings.Contains(joined, "● alice") {
		t.Errorf("alice should be marked active:\n%s", joined)
	}

	// switch back to the default — its thread restores
	a.sessionsCommand("ipsupport-code")
	if a.ag.SessionLen() != 2 || a.cfg.Name != "ipsupport-code" {
		t.Errorf("switch-back: len=%d name=%q", a.ag.SessionLen(), a.cfg.Name)
	}

	// delete alice
	if out := a.deleteSessionNamed("alice"); !strings.Contains(strings.Join(out, " "), "deleted") {
		t.Errorf("delete alice: %v", out)
	}
	list2, _ := a.sessionsCommand("")
	if strings.Contains(strings.Join(list2, "\n"), "alice") {
		t.Errorf("alice should be gone:\n%s", strings.Join(list2, "\n"))
	}
}

type fixedApprover bool

func (f fixedApprover) Approve(_, _ string) bool { return bool(f) }

func TestSubagentTargetsAndDepthCap(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	for _, env := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY"} {
		t.Setenv(env, "")
	}
	kb, _ := knowledge.Open("")
	// no profiles → no agent tool, even with a keyed provider (a profile is the
	// only way to delegate, so it's also the gate)
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	a.cfg.Providers = map[string]config.LLM{"openrouter": {APIKey: "x"}}
	if a.hasSubagentTargets() {
		t.Error("no profiles → no sub-agent targets (a keyed provider alone is not enough)")
	}
	// add a profile → targets exist
	a.cfg.Agents = map[string]config.AgentProfile{"rev": {Provider: "openrouter", Model: "m"}}
	if !a.hasSubagentTargets() {
		t.Error("a configured profile should be a sub-agent target")
	}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	// depth cap: sub-agents must NOT get the agent tool
	if subRegHasTool(a.subReg, "agent") {
		t.Error("sub-agent registry must not contain the agent tool (recursion)")
	}
	// spawn.exec is off by default → no run tool for sub-agents
	if subRegHasTool(a.subReg, "run") {
		t.Error("spawn.exec off (default) → sub-agents must not have the run tool")
	}
	a.cfg.Spawn.Exec = true
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if !subRegHasTool(a.subReg, "run") {
		t.Error("spawn.exec on → sub-agents should have the run tool")
	}
}

func subRegHasTool(reg *tool.Registry, name string) bool {
	for _, fn := range reg.OpenAITools() {
		if f, _ := fn["function"].(map[string]any); f["name"] == name {
			return true
		}
	}
	return false
}

func TestSpawnAgentPaidNeedsApproval(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.Providers = map[string]config.LLM{"openrouter": {APIKey: "x", Model: "m"}}
	// two profiles so a totally-unmatched name is genuinely unknown (with one
	// profile, tolerant matching resolves anything to it — tested separately)
	cfg.Agents = map[string]config.AgentProfile{
		"rev":   {Provider: "openrouter", Model: "m"},
		"other": {Provider: "openrouter", Model: "m2"},
	}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(false)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	// spawn.Default is "ask" by default, so the denying approver blocks it
	if _, err := a.spawnAgent(context.Background(), "rev", "do x", ""); err == nil || !strings.Contains(err.Error(), "denied") {
		t.Errorf("spawn should be denied by the approver, got %v", err)
	}
	// a name matching no profile errors before any spawn
	if _, err := a.spawnAgent(context.Background(), "zzz-nothing-like-it", "do x", ""); err == nil || !strings.Contains(err.Error(), "unknown profile") {
		t.Errorf("unknown profile = %v, want an error", err)
	}
}

func TestSpawnAgentLocalRuns(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"reviewed: looks good"}}]}`)
	}))
	defer srv.Close()
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.LLM.BaseURL = srv.URL + "/v1"
	cfg.LLM.Type = "" // plain OpenAI-compat (skip LM Studio detection)
	cfg.Agents = map[string]config.AgentProfile{"loc": {Provider: "local"}}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	// even a local spawn asks by default; the allowing approver lets it through
	out, err := a.spawnAgent(context.Background(), "loc", "review the code", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "looks good") {
		t.Errorf("sub-agent answer = %q", out)
	}
}

func TestAgentsPanelBuild(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir()) // SaveAgents writes the global config
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	m := &tuiModel{app: a}
	key := func(t tea.KeyType) tea.KeyMsg { return tea.KeyMsg{Type: t} }
	rune1 := func(r rune) tea.KeyMsg { return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}} }

	m.openAgents()
	if m.state != stAgents {
		t.Fatalf("openAgents → state %v, want stAgents", m.state)
	}
	m.agentsKey(key(tea.KeyEnter)) // on "add new" → pick provider
	if m.agPhase != agPickProvider {
		t.Fatalf("phase %v, want agPickProvider", m.agPhase)
	}
	m.agentsKey(key(tea.KeyEnter)) // pick provider (local) → pick model (async)
	if m.agPhase != agPickModel {
		t.Fatalf("phase %v, want agPickModel", m.agPhase)
	}
	// simulate the async fetch landing
	m.agLoading, m.agModelsAll = false, []string{"qwen2.5", "llama3"}
	m.agentsKey(key(tea.KeyEnter)) // pick the first model → name
	if m.agPhase != agName || m.agDraft.model != "qwen2.5" {
		t.Fatalf("phase %v model %q, want agName/qwen2.5", m.agPhase, m.agDraft.model)
	}
	for _, r := range "rev" { // type a name
		m.agentsKey(rune1(r))
	}
	m.agentsKey(key(tea.KeyEnter)) // save
	if p, ok := a.cfg.Agents["rev"]; !ok || p.Model != "qwen2.5" {
		t.Errorf("profile not saved correctly: %+v", a.cfg.Agents)
	}
	if m.agPhase != agList {
		t.Errorf("after save, phase %v, want agList", m.agPhase)
	}
}

func TestResolveProfileName(t *testing.T) {
	a := &app{cfg: config.Default()}
	a.cfg.Agents = map[string]config.AgentProfile{
		"openrouter-nemotron-3-ultra": {Provider: "openrouter"},
		"grok":                        {Provider: "openrouter"},
	}
	cases := []struct {
		in, want string
		ok       bool
	}{
		{"grok", "grok", true}, // exact
		{"GROK", "grok", true}, // case-insensitive
		{"openrouter-nemoton-3-ultra", "openrouter-nemotron-3-ultra", true}, // typo (edit distance 1)
		{"nemotron", "openrouter-nemotron-3-ultra", true},                   // unique substring
		{"totally-different-xyz", "", false},                                // no near match
	}
	for _, c := range cases {
		got, ok := a.resolveProfileName(c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("resolveProfileName(%q) = %q,%v want %q,%v", c.in, got, ok, c.want, c.ok)
		}
	}
	// with a single profile, any name resolves to it
	a.cfg.Agents = map[string]config.AgentProfile{"only": {Provider: "local"}}
	if got, ok := a.resolveProfileName("whatever-the-model-typed"); !ok || got != "only" {
		t.Errorf("single-profile fallback = %q,%v want only,true", got, ok)
	}
}

func TestReflectDisabledIsNoop(t *testing.T) {
	a := &app{cfg: config.Default()}
	a.cfg.ReflectDisabled = true
	// must short-circuit before touching the (nil) reflector
	if n := a.reflectAndStore(context.Background(), agent.Transcript{}); n != 0 {
		t.Errorf("disabled reflection should be a no-op, got %d", n)
	}
}

func TestRewindRestoresFiles(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Default()
	cfg.Workspace = ws
	cfg.File = config.FilePolicy{Default: "allow", Jail: "."}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	abs := filepath.Join(ws, "a.txt")
	newAbs := filepath.Join(ws, "b.txt")
	os.WriteFile(abs, []byte("original"), 0o644)

	a.beginCheckpoint("edit stuff")
	a.snapFile(abs) // the file tool calls this before each change; mimic it here
	os.WriteFile(abs, []byte("changed"), 0o644)
	a.snapFile(newAbs)
	os.WriteFile(newAbs, []byte("brand new"), 0o644)
	a.endCheckpoint()

	rows := a.rewindRows()
	if len(rows) != 1 {
		t.Fatalf("rewindRows = %d, want 1", len(rows))
	}
	a.applyRewind(rows[0].idx)

	if d, _ := os.ReadFile(abs); string(d) != "original" {
		t.Errorf("a.txt = %q, want restored to 'original'", d)
	}
	if _, err := os.Stat(newAbs); !os.IsNotExist(err) {
		t.Error("a file created in the rewound turn should be removed")
	}
	if len(a.checkpoints) != 0 {
		t.Errorf("checkpoints should be trimmed to before the target, got %d", len(a.checkpoints))
	}
}

func TestRewindPreview(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Default()
	cfg.Workspace = ws
	cfg.File = config.FilePolicy{Default: "allow", Jail: "."}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	abs, newAbs := filepath.Join(ws, "a.txt"), filepath.Join(ws, "b.txt")
	os.WriteFile(abs, []byte("old\n"), 0o644)
	a.beginCheckpoint("edit")
	a.snapFile(abs)
	os.WriteFile(abs, []byte("new\n"), 0o644)
	a.snapFile(newAbs)
	os.WriteFile(newAbs, []byte("created\n"), 0o644)
	a.endCheckpoint()

	items, _ := a.rewindPreview(0)
	var restore, del bool
	for _, it := range items {
		if it.Rel == "a.txt" && it.Kind == "restore" && strings.Contains(it.Diff, "old") {
			restore = true
		}
		if it.Rel == "b.txt" && it.Kind == "delete" {
			del = true
		}
	}
	if !restore || !del {
		t.Errorf("preview = %+v; want a restore (a.txt) + a delete (b.txt)", items)
	}
}

func TestCdCommand(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, "proj", "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := config.Default()
	cfg.Workspace = ws
	cfg.File = config.FilePolicy{Default: "allow", Jail: "."}
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if out := strings.Join(a.cdCommand("proj"), " "); !strings.Contains(out, "proj") {
		t.Fatalf("cd proj = %q", out)
	}
	if filepath.Base(a.effectiveDir()) != "proj" {
		t.Errorf("effectiveDir = %q, want …/proj", a.effectiveDir())
	}
	if abs, err := a.pol.Resolve("note.txt"); err != nil || filepath.Base(filepath.Dir(abs)) != "proj" {
		t.Errorf("Resolve(note.txt) = %q,%v, want under proj", abs, err)
	}
	// the workdir survives a re-wire (e.g. a /permissions or /offline toggle)
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if filepath.Base(a.pol.Workdir()) != "proj" {
		t.Errorf("workdir lost after re-wire: %q", a.pol.Workdir())
	}
	// a missing dir and an out-of-jail target both error
	if out := strings.Join(a.cdCommand("proj/nope"), " "); !strings.Contains(out, "cd:") {
		t.Errorf("cd to a missing dir should error: %q", out)
	}
	if out := strings.Join(a.cdCommand("../.."), " "); !strings.Contains(out, "cd:") {
		t.Errorf("cd outside the jail should error: %q", out)
	}
}

func TestCustomProvider(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}

	a.addProvider("mylab http://localhost:8080/v1 llama-3")
	if a.cfg.Providers["mylab"].BaseURL != "http://localhost:8080/v1" {
		t.Fatalf("custom provider not stored: %+v", a.cfg.Providers["mylab"])
	}
	// a key can be set for a custom provider (was rejected before)
	if out := strings.Join(a.setProviderKey("mylab", "sk-x"), " "); !strings.Contains(out, "saved") {
		t.Errorf("setProviderKey(custom) = %q, want saved", out)
	}
	if l, ok := config.ResolveProvider(a.cfg, "mylab"); !ok || l.APIKey != "sk-x" || l.Model != "llama-3" {
		t.Errorf("ResolveProvider(mylab) = %+v,%v", l, ok)
	}
	found := false
	for _, n := range a.configuredProviderNames() {
		if n == "mylab" {
			found = true
		}
	}
	if !found {
		t.Errorf("custom provider missing from configuredProviderNames: %v", a.configuredProviderNames())
	}
	// a name that was never added is still rejected, with the add hint
	if out := strings.Join(a.setProviderKey("ghost", "x"), " "); !strings.Contains(out, "/ai add") {
		t.Errorf("unknown provider key = %q, want the add hint", out)
	}
}

func TestCtrlUClearsInput(t *testing.T) {
	m := &tuiModel{input: textarea.New()}
	m.input.SetValue("oops, pasted clipboard garbage")
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlU})
	if m.input.Value() != "" {
		t.Errorf("ctrl+u should clear the input, got %q", m.input.Value())
	}
}

func TestCommandWhileBusy(t *testing.T) {
	m := &tuiModel{width: 80, input: textarea.New(),
		app: &app{cfg: config.Default(), workspace: t.TempDir()}}
	// a bare, read-only command runs immediately while busy (not queued)
	m.commandWhileBusy("/help")
	if len(m.queued) != 0 {
		t.Errorf("/help should run while busy, not queue: %v", m.queued)
	}
	// a mutating subcommand is queued (not dropped) until the task finishes
	m.history = nil
	m.commandWhileBusy("/sessions delete foo")
	if len(m.queued) != 1 || m.queued[0] != "/sessions delete foo" {
		t.Errorf("/sessions <arg> should be queued while busy, got %v", m.queued)
	}
	if !strings.Contains(strings.Join(m.history, "\n"), "queued") {
		t.Errorf("expected a 'queued' notice, got %q", m.history)
	}
}

func TestSpawnAgentConcurrent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`)
	}))
	defer srv.Close()
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.LLM.BaseURL = srv.URL + "/v1"
	cfg.LLM.Type = ""
	cfg.Spawn.Default = "allow" // relaxed: no prompts, so the fan-out runs unattended
	cfg.Agents = map[string]config.AgentProfile{
		"a": {Provider: "local"}, "b": {Provider: "local"}, "c": {Provider: "local"},
	}
	usageStore, _ := usage.Open(filepath.Join(t.TempDir(), "u.json"))
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, usage: usageStore,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	// spawn three sub-agents at once — the parallel fan-out path. -race guards the
	// shared usage ledger and spawn counter.
	var wg sync.WaitGroup
	for _, p := range []string{"a", "b", "c"} {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			if out, err := a.spawnAgent(context.Background(), p, "do x", ""); err != nil || !strings.Contains(out, "done") {
				t.Errorf("profile %s: out=%q err=%v", p, out, err)
			}
		}(p)
	}
	wg.Wait()
}

func TestRestoredSessionRendersRecap(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.ag.SetHistory([]llm.Message{llm.User("build a thing"), {Role: "assistant", Content: "built it"}})
	a.sessionRestored = true

	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	joined := stripAnsi(strings.Join(m.history, "\n"))
	for _, want := range []string{"restored session", "build a thing", "built it"} {
		if !strings.Contains(joined, want) {
			t.Errorf("recap missing %q in:\n%s", want, joined)
		}
	}
}

// stripAnsi removes color/style escape sequences so tests can assert on the text
// of styled (e.g. markdown-rendered) output.
func stripAnsi(s string) string {
	return regexp.MustCompile("\\x1b\\[[0-9;]*m").ReplaceAllString(s, "")
}

// Changed-your-mind: while a task runs, Up with an empty input pulls the most
// recent queued (type-ahead) message back into the input to edit or drop.
func TestUpRecallsQueuedMessage(t *testing.T) {
	m := &tuiModel{input: textarea.New(), state: stRunning, queued: []string{"first", "second"}}
	m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if m.input.Value() != "second" {
		t.Errorf("recalled %q, want 'second'", m.input.Value())
	}
	if len(m.queued) != 1 || m.queued[0] != "first" {
		t.Errorf("queue after recall = %v, want [first]", m.queued)
	}
}

func TestQueuedViewPinsMessages(t *testing.T) {
	m := &tuiModel{queued: []string{"fix the bug", "add a test"}}
	out := strings.Join(m.queuedView(), "\n")
	if !strings.Contains(out, "fix the bug") || !strings.Contains(out, "add a test") {
		t.Errorf("pinned queue missing messages:\n%s", out)
	}
	if got := (&tuiModel{}).queuedView(); len(got) != 0 {
		t.Errorf("empty queue should pin nothing, got %v", got)
	}
}

// A sub-agent's observation (LLM final / external CLI report) renders as
// markdown — no raw ** markers — and is NOT capped at 25 lines like ordinary
// tool output.
func TestSubagentResultRendersMarkdown(t *testing.T) {
	m := &tuiModel{width: 80, app: &app{cfg: config.Default()}}
	long := strings.Repeat("- finding line\n", 40) // > the 25-line outputLines cap
	got := strings.Join(m.renderEvent(uiEvent{kind: "observation", fields: map[string]any{
		"tool": "agent", "content": "**Critical** issue found\n\n" + long}}), "\n")
	if strings.Contains(got, "**") {
		t.Error("markdown markers leaked into the sub-agent result render")
	}
	if !strings.Contains(got, "Critical") {
		t.Error("content lost in render")
	}
	if n := strings.Count(got, "finding"); n < 40 { // glamour restyles the bullets
		t.Errorf("sub-agent result must not be capped: %d/40 lines", n)
	}
	// ordinary tool output stays capped + raw
	raw := m.renderEvent(uiEvent{kind: "observation", fields: map[string]any{
		"tool": "run", "content": long}})
	if len(raw) > 27 {
		t.Errorf("plain output should stay capped, got %d lines", len(raw))
	}
}

func TestRenderMarkdownKeepsContent(t *testing.T) {
	out := renderMarkdown("Wrote **hello.sh** and ran it.", 80)
	if !strings.Contains(out, "hello.sh") {
		t.Errorf("markdown render dropped content: %q", out)
	}
	if renderMarkdown("", 80) != "" {
		t.Error("empty input must stay empty")
	}
}

func TestModeLine(t *testing.T) {
	m := &tuiModel{app: &app{}, accent: lipgloss.Color("13")}
	if !strings.Contains(m.modeLine(), "auto mode on") {
		t.Errorf("auto: %q", m.modeLine())
	}
	m.app.planMode = true
	if !strings.Contains(m.modeLine(), "plan mode on") {
		t.Errorf("plan: %q", m.modeLine())
	}
}

func TestRenderMarkdownConvertsLatex(t *testing.T) {
	out := renderMarkdown(`.jpg $\rightarrow$ Images`, 80)
	if !strings.Contains(out, "→") || strings.Contains(out, "rightarrow") {
		t.Errorf("LaTeX not converted to unicode arrow: %q", out)
	}
}

// An approval's FULL detail (incl. a long sub-agent/external task) must land in
// the log unabridged — the status line clips, the log must not.
func TestApprovalDetailFullyVisible(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning, width: 80}
	long := strings.Repeat("review everything carefully ", 10) // ~280 chars
	req := approvalReq{kind: "external agent", detail: "codex · /some/dir\n  task: " + long, reply: make(chan bool, 1)}
	m.Update(approvalMsg(req))
	if !strings.Contains(strings.Join(m.history, "\n"), long) {
		t.Error("the full task must be pushed to the log, not truncated")
	}
	// the one-line status stays single-line and clipped
	m.state = stApprove
	if p := m.approvePrompt(); strings.Contains(p, "\n") || strings.Contains(p, long) {
		t.Error("approvePrompt must be a single clipped line")
	}
}

// Batched tool calls each request approval concurrently. Showing one approval
// must NOT immediately wait for the next (that pre-fetch overwrote m.pending and
// orphaned the first reply channel — a forever "Thinking" hang). The next is
// fetched only after the current one is answered.
func TestApprovalSerializedNoPrefetch(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning}

	req := approvalReq{kind: "write", detail: "a.txt", reply: make(chan bool, 1)}
	_, cmd := m.Update(approvalMsg(req))
	if cmd != nil {
		t.Error("showing an approval must not issue a follow-up command (no pre-fetch)")
	}
	if m.pending == nil {
		t.Fatal("approval not recorded")
	}
	if m.state == stApprove {
		t.Error("approval must not steal focus — stays running until ↑")
	}

	// ↑ switches to answering (the input text is left untouched).
	if _, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp}); m.state != stApprove {
		t.Fatalf("↑ did not enter answer mode: state=%v", m.state)
	}
	_, cmd = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	if !<-req.reply {
		t.Error("approval not granted on 'y'")
	}
	if m.pending != nil {
		t.Error("pending should be cleared after answering")
	}
	if cmd == nil {
		t.Error("answering should fetch the next queued approval")
	}
}

// The pending approval must not capture keystrokes — typing keeps editing the
// input until the user deliberately presses ↑ to answer.
func TestApprovalKeepsInputEditable(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning}
	m.input.Focus()
	req := approvalReq{kind: "write", detail: "a.txt", reply: make(chan bool, 1)}
	m.Update(approvalMsg(req))

	// 'h' 'i' should land in the input, not answer the approval.
	m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'i'}})
	if m.input.Value() != "hi" {
		t.Errorf("input = %q, want 'hi' (approval shouldn't eat keystrokes)", m.input.Value())
	}
	if m.pending == nil {
		t.Error("approval should still be pending, unanswered")
	}
}

// A system.md override replaces the built-in base prompt; without one the base
// is used.
func TestSystemPromptOverride(t *testing.T) {
	ws := t.TempDir()
	cfg := config.Default()
	cfg.Workspace = ws
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}

	if p := a.systemPrompt(); !strings.Contains(p, "You are the engine inside") || a.promptSrc != "built-in" {
		t.Fatalf("default should be the built-in base; promptSrc=%q", a.promptSrc)
	}

	if err := os.MkdirAll(filepath.Join(ws, ".agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, ".agent", "system.md"), []byte("MY CUSTOM ENGINE PROMPT"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := a.systemPrompt()
	if !strings.Contains(p, "MY CUSTOM ENGINE PROMPT") || strings.Contains(p, "You are the engine inside") {
		t.Errorf("override not applied:\n%s", p)
	}
	if !strings.HasSuffix(a.promptSrc, "system.md") {
		t.Errorf("promptSrc = %q, want the override path", a.promptSrc)
	}
}

// A /skills or /permissions toggle re-wires the stack (new client); the running
// token total must carry over, not reset to zero.
func TestTokenTotalSurvivesRewire(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.client.SeedUsage(120, 45) // pretend some calls happened
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if p, c := a.client.Usage(); p != 120 || c != 45 {
		t.Errorf("token total reset on re-wire: %d/%d, want 120/45", p, c)
	}
}

// Cancelling a task must free a tool goroutine waiting on approval, whether it's
// blocked offering the request or waiting for the reply — otherwise esc leaves
// the run hung (the bug the watchdog can't see).
func TestBridgeAbortDeniesBlockedApproval(t *testing.T) {
	for _, readFirst := range []bool{false, true} {
		b := newBridge()
		done := make(chan bool, 1)
		go func() { done <- b.Approve("write", "x") }()
		if readFirst {
			<-b.approvals // Approve is now blocked waiting for the reply
		}
		b.Abort()
		select {
		case ok := <-done:
			if ok {
				t.Errorf("readFirst=%v: aborted approval should be denied", readFirst)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("readFirst=%v: Approve never returned after Abort (hang)", readFirst)
		}
	}
}

func TestActiveLLM(t *testing.T) {
	a := &app{cfg: config.Config{LLM: config.LLM{BaseURL: "http://localhost:1234/v1", Model: "local-m"}}}
	if a.activeLLM().BaseURL != "http://localhost:1234/v1" || !a.isLocal() {
		t.Error("default → local connection")
	}
	a.cfg.Provider = "openai"
	a.cfg.Providers = map[string]config.LLM{"openai": {APIKey: "k", Model: "gpt-4o"}}
	if got := a.activeLLM(); got.Model != "gpt-4o" || got.BaseURL != "https://api.openai.com/v1" || a.isLocal() {
		t.Errorf("active = %+v, want the openai provider", got)
	}
	a.cfg.Provider = "local"
	if a.activeLLM().BaseURL != "http://localhost:1234/v1" || !a.isLocal() {
		t.Error("/ai local → cfg.LLM")
	}
}

func TestAutoCompactNeeded(t *testing.T) {
	if !autoCompactNeeded(6200, 8192, 4) {
		t.Error("76% of the window with history should trigger compaction")
	}
	if autoCompactNeeded(4000, 8192, 4) {
		t.Error("49% is well under the threshold")
	}
	if autoCompactNeeded(7000, 8192, 2) {
		t.Error("too little history to bother compacting")
	}
	if autoCompactNeeded(7000, 0, 4) {
		t.Error("a zero window disables auto-compact")
	}
}

// The pipe-through-script smoke test can't reliably confirm quit semantics, so
// verify the exit path directly: /exit must yield tea.Quit.
func TestExitCommandQuits(t *testing.T) {
	m := &tuiModel{input: textarea.New()}
	_, cmd := m.runCommand("/exit")
	if cmd == nil {
		t.Fatal("/exit returned a nil command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("/exit did not produce tea.Quit")
	}
}

// TestBtwSteeringDrainsOnce covers the /btw side-channel: a note is buffered
// (safe from the UI goroutine), the agent's beforeTurn hook folds it in as a
// prefixed user aside exactly once, and the buffer clears so it can't repeat.
func TestBtwSteeringDrainsOnce(t *testing.T) {
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	if a.addBtw("   ") {
		t.Error("blank /btw note must be ignored")
	}
	a.addBtw("look in internal/tool, not cmd")
	a.addBtw("keep the diff small")
	if n := a.btwPending(); n != 2 {
		t.Fatalf("btwPending = %d, want 2", n)
	}

	msgs := a.drainBtw() // this is the agent's beforeTurn hook
	if len(msgs) != 2 {
		t.Fatalf("drainBtw returned %d messages, want 2", len(msgs))
	}
	if msgs[0].Role != "user" || !strings.HasPrefix(msgs[0].Content, "[by the way] ") {
		t.Errorf("note not delivered as a prefixed user aside: %+v", msgs[0])
	}
	if !strings.Contains(msgs[0].Content, "internal/tool") {
		t.Errorf("note content lost: %q", msgs[0].Content)
	}
	if got := a.drainBtw(); got != nil { // drained once — never repeats
		t.Errorf("second drain must be empty, got %v", got)
	}
	if a.btwPending() != 0 {
		t.Error("buffer not cleared after drain")
	}
}

// TestCompleteDirSegments covers /cd Tab-completion: it lists sub-dirs of the
// already-typed parent one segment at a time, preserves that parent in each
// candidate (so completion can descend), appends "/", and hides dot-dirs unless
// the user is explicitly typing one.
func TestCompleteDirSegments(t *testing.T) {
	has := func(ss []string, s string) bool {
		for _, x := range ss {
			if x == s {
				return true
			}
		}
		return false
	}
	ws := t.TempDir()
	for _, d := range []string{"internal/agent", "internal/config", "cmd/agent", ".git/hooks"} {
		if err := os.MkdirAll(filepath.Join(ws, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	cfg := config.Default()
	cfg.Workspace = ws
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: ws, kb: kb,
		reader: bufio.NewReader(strings.NewReader("")), approver: fixedApprover(true)}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}

	_, m := a.completeDir("") // top level: real dirs, dot-dir hidden
	if !has(m, "cmd/") || !has(m, "internal/") {
		t.Fatalf("top-level = %v, want cmd/ and internal/", m)
	}
	if has(m, ".git/") {
		t.Errorf("dot-dir must be hidden by default: %v", m)
	}

	if _, m = a.completeDir("inter"); len(m) != 1 || m[0] != "internal/" {
		t.Fatalf("partial segment = %v, want [internal/]", m)
	}

	_, m = a.completeDir("internal/") // descend: parent preserved in each candidate
	if !has(m, "internal/agent/") || !has(m, "internal/config/") {
		t.Fatalf("descend = %v, want internal/agent/ + internal/config/", m)
	}

	if _, m = a.completeDir("internal/ag"); len(m) != 1 || m[0] != "internal/agent/" {
		t.Fatalf("deeper partial = %v, want [internal/agent/]", m)
	}

	if _, m = a.completeDir("../../etc"); len(m) != 0 { // jail: no escaping the workspace
		t.Errorf("path outside the jail must yield nothing, got %v", m)
	}
}
