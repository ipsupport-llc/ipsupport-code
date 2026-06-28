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
	cfg.Agents = map[string]config.AgentProfile{"rev": {Provider: "openrouter", Model: "m"}}
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
	// unknown profile errors before any spawn
	if _, err := a.spawnAgent(context.Background(), "nope", "do x", ""); err == nil || !strings.Contains(err.Error(), "unknown profile") {
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
	// a bare, read-only command runs immediately while busy (no deferral notice)
	m.commandWhileBusy("/help")
	if strings.Contains(strings.Join(m.history, "\n"), "will run once") {
		t.Error("/help should run while busy, not defer")
	}
	// a mutating subcommand is deferred until the task finishes
	m.history = nil
	m.commandWhileBusy("/sessions delete foo")
	if !strings.Contains(strings.Join(m.history, "\n"), "will run once") {
		t.Errorf("/sessions <arg> should defer while busy, got %q", m.history)
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
