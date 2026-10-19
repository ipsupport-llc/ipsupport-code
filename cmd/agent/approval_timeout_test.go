package main

import (
	"bufio"
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// askThroughTheUI starts an approval on the bridge and shows it on m, the way
// the TUI does, returning where the asking tool's answer will arrive.
func askThroughTheUI(t *testing.T, m *tuiModel) <-chan [2]bool {
	t.Helper()
	got := make(chan [2]bool, 1)
	go func() {
		ok, answered := m.bridge.ApproveAnswered(context.Background(), "run", "make deploy")
		got <- [2]bool{ok, answered}
	}()
	m.Update(approvalMsg(<-m.bridge.approvals))
	if m.pending == nil {
		t.Fatal("the approval was not shown")
	}
	return got
}

// A prompt nobody answers used to hold the task until someone came back. Now,
// past approvalTimeout, it is a no: the tool is told so and the run goes on —
// and since nobody judged the call, the risk model is not told anything.
func TestAnUnansweredApprovalIsANoAndTheTaskGoesOn(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning, spin: spinner.New()}
	got := askThroughTheUI(t, m)

	m.Update(spinner.TickMsg{}) // still in time
	if m.pending == nil {
		t.Fatal("the approval was denied before its time ran out")
	}

	m.approveBy = time.Now().Add(-time.Second)
	_, cmd := m.Update(spinner.TickMsg{})
	select {
	case r := <-got:
		if r[0] || r[1] {
			t.Fatalf("ApproveAnswered = (ok %v, answered %v), want (false, false) — a timeout is neither a yes nor anyone's answer", r[0], r[1])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the tool was still waiting after the approval ran out")
	}
	if m.pending != nil || m.state != stRunning {
		t.Errorf("pending = %v, state = %v; want none pending, back to running", m.pending, m.state)
	}
	// The next approval still reaches the screen: running out re-arms the reader.
	go m.bridge.ApproveAnswered(context.Background(), "run", "make test")
	msgs := make(chan tea.Msg, 4)
	if batch, ok := cmd().(tea.BatchMsg); ok {
		for _, c := range batch {
			if c != nil {
				go func() { msgs <- c() }()
			}
		}
	}
	for deadline := time.After(3 * time.Second); ; {
		var msg tea.Msg
		select {
		case msg = <-msgs:
		case <-deadline:
			t.Fatal("the next approval never reached the UI — the expired one left nobody reading")
		}
		if _, ok := msg.(approvalMsg); ok {
			break
		}
	}
	if log := strings.Join(m.history, "\n"); !strings.Contains(log, "no answer in "+approvalWait()) {
		t.Errorf("the log does not say why it was denied:\n%s", log)
	}
}

// Running out is not an answer typed into a panel: a panel open over the prompt
// stays open — even when the task finished behind it, which is finalized when
// the panel closes, not when the prompt runs out.
func TestAnExpiredApprovalLeavesAPanelOpen(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRewind, spin: spinner.New(), taskDoneAway: true}
	got := askThroughTheUI(t, m)
	m.approveBy = time.Now().Add(-time.Second)
	m.Update(spinner.TickMsg{})
	if r := <-got; r[0] {
		t.Fatal("an expired approval ran the call")
	}
	if m.state != stRewind || !m.taskDoneAway {
		t.Errorf("state = %v, taskDoneAway = %v; want the rewind panel still open and the finished task still waiting on it", m.state, m.taskDoneAway)
	}
}

// A task that finished behind the prompt is finalized when the prompt runs out,
// even after esc dropped back to typing: nothing else would, and the view sat
// on "running" with no task behind it.
func TestAnExpiredApprovalFinalizesTheTaskBehindIt(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning, spin: spinner.New(), taskDoneAway: true}
	got := askThroughTheUI(t, m)
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc}) // back to typing; still pending
	m.approveBy = time.Now().Add(-time.Second)
	m.Update(spinner.TickMsg{})
	<-got
	if m.state != stIdle || m.taskDoneAway {
		t.Errorf("state = %v, taskDoneAway = %v; want the finished task finalized to idle", m.state, m.taskDoneAway)
	}
}

// esc denies the prompt it drops, whichever task asked it. One asked before the
// current task began holds an abort signal esc never closes.
func TestEscDeniesAPromptFromBeforeTheTask(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stIdle, spin: spinner.New()}
	got := askThroughTheUI(t, m) // a background job asks while idle
	m.bridge.arm()               // the next task starts
	m.state, m.cancel = stRunning, func() {}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	select {
	case r := <-got:
		if r[0] {
			t.Fatal("esc approved the call")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("esc dropped the prompt, but its tool is still waiting for an answer")
	}
}

// A prompt asked after esc was denied tool-side the moment it was asked: it is
// not shown, and the reader moves on to the next one.
func TestAPromptAskedAfterEscIsNotShown(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning, spin: spinner.New()}
	aborted := make(chan struct{})
	close(aborted)
	_, cmd := m.Update(approvalMsg(approvalReq{kind: "run", detail: "make deploy", reply: make(chan bool, 1), aborted: aborted}))
	if m.pending != nil || m.state != stRunning {
		t.Errorf("pending = %v, state = %v; want the dead prompt not shown", m.pending, m.state)
	}
	if cmd == nil {
		t.Error("skipping a dead prompt must keep reading the next one")
	}

	// The same through the bridge: asked, then esc, then read by the UI.
	got := make(chan bool, 1)
	go func() { ok, _ := m.bridge.ApproveAnswered(context.Background(), "run", "make deploy"); got <- ok }()
	req := <-m.bridge.approvals
	m.bridge.Abort()
	if <-got {
		t.Fatal("an aborted approval ran the call")
	}
	m.Update(approvalMsg(req))
	if m.pending != nil {
		t.Error("the bridge's prompt was shown after esc had already denied it")
	}
}

// A read that gave up returns, rather than waiting for a reply its abandoned
// request will never get.
func TestAReadThatGaveUpReturns(t *testing.T) {
	pr, pw := io.Pipe()
	t.Cleanup(func() { pr.Close(); pw.Close() })
	o := newStdinOwner(bufio.NewReader(pr))
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _, _ = o.readApproveLineCtx(ctx); close(done) }()
	time.Sleep(20 * time.Millisecond) // queued, waiting for a line
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the read is still waiting after its reader gave up — a goroutine leaked for good")
	}
}

// The plain prompt runs out the same way, and its read gives up with it: the
// next line typed belongs to the REPL, not to the prompt that already expired.
func TestAnUnansweredPlainPromptIsANo(t *testing.T) {
	defer func(d time.Duration) { approvalTimeout = d }(approvalTimeout)
	approvalTimeout = 50 * time.Millisecond

	pr, pw := io.Pipe()
	t.Cleanup(func() { pr.Close(); pw.Close() })
	s := &stdinApprover{stdin: newStdinOwner(bufio.NewReader(pr))}

	done := make(chan [2]bool, 1)
	go func() {
		ok, answered := s.ApproveAnswered(context.Background(), "run", "make deploy")
		done <- [2]bool{ok, answered}
	}()
	select {
	case r := <-done:
		if r[0] || r[1] {
			t.Fatalf("ApproveAnswered = (ok %v, answered %v), want (false, false)", r[0], r[1])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the plain prompt did not run out")
	}

	go func() { _, _ = io.WriteString(pw, "y\n") }()
	line := make(chan string, 1)
	go func() { l, _ := s.stdin.readCmdLine(); line <- l }()
	select {
	case l := <-line:
		if strings.TrimSpace(l) != "y" {
			t.Errorf("the REPL read %q, want the line typed after the prompt expired", l)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the line typed after the expired prompt never reached the REPL — the dead prompt took it")
	}
}

// silentApprover is a prompt that ran out: a no that nobody gave.
type silentApprover struct{}

func (silentApprover) Approve(ctx context.Context, _, _ string) bool {
	tool.NoAnswer(ctx)
	return false
}

// The model is told a prompt ran out, not that the user said no — through the
// TUI and the plain prompt alike — and a real no is still reported as one.
func TestTheModelIsToldNobodyAnswered(t *testing.T) {
	m := &tuiModel{bridge: newBridge(), input: textarea.New(), state: stRunning, spin: spinner.New()}
	ask := func() <-chan string {
		got := make(chan string, 1)
		go func() { _, gone := tool.Ask(context.Background(), m.bridge, "run", "make deploy"); got <- gone }()
		m.Update(approvalMsg(<-m.bridge.approvals))
		return got
	}
	got := ask()
	m.approveBy = time.Now().Add(-time.Second)
	m.Update(spinner.TickMsg{})
	if gone := <-got; !strings.Contains(gone, "nobody answered") {
		t.Errorf("TUI timeout told the model %q, want that nobody answered", gone)
	}
	got = ask()
	m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	if gone := <-got; gone != "" {
		t.Errorf("an explicit no came back as %q, want the plain refusal", gone)
	}
	// esc is someone there, not someone away — even for a prompt asked before
	// the task, which esc denies by closing its own channel, not the abort.
	got = ask()
	m.bridge.arm()
	m.state, m.cancel = stRunning, func() {}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if gone := <-got; gone != "" {
		t.Errorf("a prompt esc dropped told the model %q, want no claim the user is away", gone)
	}
	m.cancel, m.taskCancelled = nil, false

	defer func(d time.Duration) { approvalTimeout = d }(approvalTimeout)
	approvalTimeout = 50 * time.Millisecond
	pr, pw := io.Pipe()
	t.Cleanup(func() { pr.Close(); pw.Close() })
	s := &stdinApprover{stdin: newStdinOwner(bufio.NewReader(pr))}
	if _, gone := tool.Ask(context.Background(), s, "run", "make deploy"); !strings.Contains(gone, "nobody answered") {
		t.Errorf("plain timeout told the model %q, want that nobody answered", gone)
	}

	// A closed stdin: nobody can answer at all.
	closed := &stdinApprover{stdin: newStdinOwner(bufio.NewReader(strings.NewReader("")))}
	if _, gone := tool.Ask(context.Background(), closed, "run", "make deploy"); !strings.Contains(gone, "nobody answered") {
		t.Errorf("closed stdin told the model %q, want that nobody answered", gone)
	}
	// A cancelled task is not told anyone was away.
	cctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, gone := tool.Ask(cctx, closed, "run", "make deploy"); gone != "" {
		t.Errorf("a cancelled task was told %q, want nothing", gone)
	}
}

// The calls asked about outside the tools say it too: the risk check's own stop,
// a sub-agent spawn and an external agent.
func TestEveryAskerSaysNobodyAnswered(t *testing.T) {
	a, pol, _ := gateApp(t, "run.default=allow")
	a.approver = silentApprover{}
	ctx := scored(t, a, pol, flaggedCommand)
	mustFlag(t, ctx)
	if msg := a.riskGate(pol)(ctx, "run", "shell", map[string]any{"command": flaggedCommand}); !strings.Contains(msg, "nobody answered") || !strings.Contains(msg, "risk check had flagged it") {
		t.Errorf("risk check: %q", msg)
	}

	b := &app{cfg: config.Default(), workspace: t.TempDir(), approver: silentApprover{}}
	plan := spawnPlan{profile: "p", llmCfg: config.LLM{Model: "m"}, subWorkspace: b.workspace, spawnDefault: "ask"}
	if _, err := b.runSpawnPlan(context.Background(), plan, "a task", nil); err == nil || !strings.Contains(err.Error(), "nobody answered") {
		t.Errorf("spawn: %v", err)
	}
	p := config.AgentProfile{Kind: "external", Command: "codex", Args: []string{"exec", "{task}"}}
	if _, err := b.spawnExternalAgent(context.Background(), "codex", p, "a task", b.workspace, nil, nil); err == nil || !strings.Contains(err.Error(), "nobody answered") {
		t.Errorf("external agent: %v", err)
	}
}
