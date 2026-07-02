package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
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
	for _, env := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"} {
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
	if !strings.Contains(out, "Added 1 line, removed 1 line") {
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

func TestInputHistoryRecall(t *testing.T) {
	m := &tuiModel{input: textarea.New()}
	m.recordInput("first")
	m.recordInput("second")
	m.recordInput("second") // a consecutive duplicate is not stored twice
	if len(m.inputHist) != 2 {
		t.Fatalf("inputHist = %v, want [first second]", m.inputHist)
	}

	m.input.SetValue("draft-in-progress")
	m.histIdx = len(m.inputHist) // not browsing
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
	for _, env := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"} {
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
	joined := strings.Join(m.history, "\n")
	for _, want := range []string{"restored session", "build a thing", "built it"} {
		if !strings.Contains(joined, want) {
			t.Errorf("recap missing %q in:\n%s", want, joined)
		}
	}
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
