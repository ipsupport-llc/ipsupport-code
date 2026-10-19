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
