package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/exp/teatest"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/mcp"
)

func tuiFakeServer(t *testing.T, responses ...string) string {
	t.Helper()
	i := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		resp := tuiContent("done")
		if i < len(responses) {
			resp = responses[i]
			i++
		}
		io.WriteString(w, resp)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func tuiToolCall(name, argsJSON string) string {
	b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"message": map[string]any{
		"role": "assistant", "content": "",
		"tool_calls": []map[string]any{{"id": "c1", "type": "function",
			"function": map[string]any{"name": name, "arguments": argsJSON}}},
	}}}})
	return string(b)
}

func tuiContent(text string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": text}}},
	})
	return string(b)
}

func tuiTestApp(t *testing.T, url string) *app {
	t.Helper()
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "kb.json"))
	c := config.Default()
	c.Workspace = t.TempDir()
	c.LLM.BaseURL = url
	c.LLM.Model = "fake"
	c.Run.Default = "allow"
	c.File.Default = "allow"
	c.File.Jail = "."
	return &app{cfg: c, workspace: c.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
}

func TestSessionPersistsAcrossRestarts(t *testing.T) {
	url := tuiFakeServer(t, tuiContent("first answer"))
	ws := t.TempDir()

	mk := func() *app {
		kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "kb.json"))
		c := config.Default()
		c.Workspace, c.LLM.BaseURL, c.LLM.Model = ws, url, "fake"
		c.Run.Default, c.File.Default, c.File.Jail = "allow", "allow", "."
		a := &app{cfg: c, workspace: ws, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
		a.approver = &stdinApprover{stdin: newStdinOwner(a.reader)}
		if err := a.wire(); err != nil {
			t.Fatal(err)
		}
		a.loadSession()
		return a
	}

	first := mk()
	first.runOne(context.Background(), "remember this")
	if first.ag.SessionLen() == 0 {
		t.Fatal("no session memory after a run")
	}

	// A brand-new process for the same workspace must recover the conversation.
	second := mk()
	if second.ag.SessionLen() != first.ag.SessionLen() {
		t.Errorf("session not persisted: reloaded %d messages, want %d",
			second.ag.SessionLen(), first.ag.SessionLen())
	}
}

// Drive the real TUI model end to end: type a task, press Enter, and confirm the
// streamed final answer reaches the screen.
func TestTUI_E2E_StreamsAnswer(t *testing.T) {
	url := tuiFakeServer(t, tuiContent("hello from the model"), tuiContent("[]"))
	a := tuiTestApp(t, url)
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("say hi")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "hello from the model")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}

// After a file-write task, the TUI offers "run it" as a next-step suggestion.
func TestTUI_E2E_SuggestsNextStep(t *testing.T) {
	url := tuiFakeServer(t,
		tuiToolCall("file", `{"action":"write","params":{"path":"hello.sh","content":"echo hi"}}`),
		tuiContent("created hello.sh\nNEXT: run it"),
	)
	a := tuiTestApp(t, url)
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("make a script")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "run it")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}

// /help renders into the work area.
func TestTUI_E2E_HelpCommand(t *testing.T) {
	a := tuiTestApp(t, tuiFakeServer(t))
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("/help")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "/loop")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}

// A second Ctrl+R while already reverse-searching must step to an OLDER match,
// not re-find the same one — the standard reverse-incremental-search
// convention (bash/readline). Regression test: the top-level switch's
// unconditional return used to shadow stHistSearch's own "step to an older
// match" case, so repeated Ctrl+R just re-found the same first match forever.
func TestCtrlR_SecondPress_StepsToOlderMatch(t *testing.T) {
	a := tuiTestApp(t, tuiFakeServer(t))
	a.addPromptHist("deploy backend")
	a.addPromptHist("deploy frontend")
	a.addPromptHist("deploy platform")

	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlR})
	if m.state != stHistSearch {
		t.Fatalf("first ctrl+r: state = %v, want stHistSearch", m.state)
	}
	first := m.searchIdx
	if first != 2 {
		t.Fatalf("first ctrl+r: searchIdx = %d, want 2 (most recent entry)", first)
	}

	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlR})
	if m.searchIdx == first {
		t.Fatalf("second ctrl+r: searchIdx stayed at %d, want an older match", m.searchIdx)
	}
	if m.searchIdx != 1 {
		t.Fatalf("second ctrl+r: searchIdx = %d, want 1 (next older entry)", m.searchIdx)
	}
}

// TestMCPCommand_ReturnsCmdWithoutBlocking is a regression test for the /mcp
// deadlock: a not-yet-connected MCP server's launch is approval-gated
// (mcpClient → approveGated → the UI bridge), which blocks until Update
// delivers an approvalMsg and the user answers it. The old code called
// mcpList directly inside Update — the single goroutine that alone can drain
// the bridge — so that call could never return.
//
// This proves both halves of the fix: (1) the exact old call pattern really
// does hang when nothing drains the bridge (reproduced directly, without a
// running tea.Program), and (2) runCommand("/mcp") no longer makes that call
// inline — it returns immediately with a non-nil tea.Cmd, deferring the
// blocking connect+approval to that Cmd's own goroutine.
func TestMCPCommand_ReturnsCmdWithoutBlocking(t *testing.T) {
	a := tuiTestApp(t, tuiFakeServer(t))
	a.cfg.McpServers = map[string]mcp.Server{"test": {URL: "http://mcp.invalid"}} // never dialed: approval blocks first
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// Release the goroutine below once the test is done asserting, instead of
	// leaking a permanently-blocked goroutine past the test.
	t.Cleanup(m.bridge.Abort)

	// (1) The old pattern, reproduced directly: with nobody draining
	// m.bridge.approvals (no tea.Program/waitApproval running), the exact call
	// the old code made inline in Update never returns.
	oldCallReturned := make(chan struct{})
	go func() {
		m.app.mcpList(m.ctx)
		close(oldCallReturned)
	}()
	select {
	case <-oldCallReturned:
		t.Fatal("mcpList returned with no approval answered — test setup didn't reproduce the blocking approval wait")
	case <-time.After(200 * time.Millisecond):
		// Expected: still blocked waiting on the approval bridge, exactly as it
		// would have been if Update had called this inline.
	}

	// (2) The fix: runCommand itself must not block, and must hand back a
	// non-nil tea.Cmd so the connect+approval happens off Update's goroutine.
	done := make(chan struct{})
	var cmd tea.Cmd
	go func() {
		_, cmd = m.runCommand("/mcp")
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(1 * time.Second):
		t.Fatal(`runCommand("/mcp") blocked — the first-connect approval wait must happen inside the returned tea.Cmd, not inline`)
	}
	if cmd == nil {
		t.Fatal(`runCommand("/mcp") returned a nil tea.Cmd — the connect+approval must be dispatched asynchronously`)
	}
}

// TestTUI_E2E_MCPConnectDoesNotBlockUI drives the REAL tea.Program loop
// (teatest) through the exact scenario that used to deadlock the TUI: /mcp
// against a server that has never been approved. Before the fix, the approval
// prompt could never even render — Update itself was wedged inside the
// connect call, unable to process the approvalMsg its own waitApproval Cmd
// had already delivered. If this regresses, both WaitFor calls below time out.
func TestTUI_E2E_MCPConnectDoesNotBlockUI(t *testing.T) {
	mcpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID     *int   `json:"id"`
			Method string `json:"method"`
		}
		_ = json.Unmarshal(body, &req)
		if req.ID == nil { // a notification (initialized) → just accept it
			w.WriteHeader(http.StatusAccepted)
			return
		}
		result := `{}`
		if req.Method == "tools/list" {
			result = `{"tools":[{"name":"echo","description":"echoes input"}]}`
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":%s}`, *req.ID, result)
	}))
	defer mcpSrv.Close()

	a := tuiTestApp(t, tuiFakeServer(t))
	a.cfg.McpServers = map[string]mcp.Server{"test": {URL: mcpSrv.URL}}
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("/mcp")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	// The approval prompt must render — proving Update is still alive and
	// draining waitApproval/waitEvent while the connect blocks on its own Cmd
	// goroutine, not on Update's.
	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "approve mcp launch")
	}, teatest.WithDuration(5*time.Second))

	// Answer it (↑ switches a pending approval into the answerable stApprove
	// state from idle, then y approves) — proving keys still reach Update.
	tm.Send(tea.KeyMsg{Type: tea.KeyUp})
	tm.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")})

	// The catalog comes back via mcpMsg on a later Update call.
	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "echo")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}
