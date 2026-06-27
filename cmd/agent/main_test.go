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
	if n, g := parseLoop("5 build it"); n != 5 || g != "build it" {
		t.Errorf("parseLoop(5 build it) = %d,%q", n, g)
	}
	if n, g := parseLoop("just a task"); n != 3 || g != "just a task" {
		t.Errorf("parseLoop default count = %d,%q", n, g)
	}
	if n, g := parseLoop(""); n != 0 || g != "" {
		t.Errorf("parseLoop empty = %d,%q", n, g)
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
