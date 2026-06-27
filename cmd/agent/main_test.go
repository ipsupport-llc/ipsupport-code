package main

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

func TestSplitCommand(t *testing.T) {
	cmd, rest := splitCommand("/loop 3 do the thing")
	if cmd != "/loop" || rest != "3 do the thing" {
		t.Errorf("splitCommand = %q,%q", cmd, rest)
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
	m := &tuiModel{input: textinput.New()}
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
	m := &tuiModel{input: textinput.New(), app: &app{
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

// Changed-your-mind: while a task runs, Up with an empty input pulls the most
// recent queued (type-ahead) message back into the input to edit or drop.
func TestUpRecallsQueuedMessage(t *testing.T) {
	m := &tuiModel{input: textinput.New(), state: stRunning, queued: []string{"first", "second"}}
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
	m := &tuiModel{bridge: newBridge(), input: textinput.New(), state: stRunning}

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
	m := &tuiModel{bridge: newBridge(), input: textinput.New(), state: stRunning}
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
	m := &tuiModel{input: textinput.New()}
	_, cmd := m.runCommand("/exit")
	if cmd == nil {
		t.Fatal("/exit returned a nil command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("/exit did not produce tea.Quit")
	}
}
