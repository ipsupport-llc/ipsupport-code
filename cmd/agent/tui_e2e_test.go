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
	"sync"
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
		m.app.mcpList(m.ctx, m.app.cfg.McpServers)
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

// TestBtwAsideAgentTracerNoRace reproduces the idle /btw data race one level up
// from the history/system snapshot (that race is covered separately by
// TestAnswerAsideSnapshotDoesNotRaceWithReset in internal/agent). Even with that
// snapshot in place, the /btw handler's goroutine still called
// m.app.ag.AnswerAside(...) and m.app.emit(...) (which reads a.tracer) — both
// live app-field reads happening at whatever later moment the goroutine
// actually runs. wire() (invoked by /model, /permissions, /new, and 30+ other
// places) reassigns a.ag and a.tracer with no lock on EVERY call. Before the
// fix, the goroutine racing a concurrent wire() was a data race under -race,
// not just a stale value. The fix captures a.ag and a.tracer synchronously
// (same goroutine as this test's dispatch loop) before launching the
// goroutine — same pattern as resolveSpawn's tracer capture (spawnPlan.tracer)
// — so the goroutine never touches a.ag/a.tracer again. This test fires many
// /btw asides against a slow fake LLM, calling wire() right after each dispatch
// (a foreground settings change firing while an earlier aside's own goroutine
// is still in flight), and must be -race clean.
func TestBtwAsideAgentTracerNoRace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(2 * time.Millisecond) // keep the aside in flight long enough to overlap wire()
		io.WriteString(w, tuiContent("answer"))
	}))
	defer srv.Close()

	a := tuiTestApp(t, srv.URL)
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	const n = 50
	for i := 0; i < n; i++ {
		m.runCommand("/btw are we there yet")
		// Simulates a foreground settings change (/model, /permissions, /new,
		// ...) firing right after dispatch — while earlier asides' own
		// goroutines may still be running.
		if err := a.wire(); err != nil {
			t.Fatal(err)
		}
	}

	// Wait for every dispatched aside to actually finish (and emit) before this
	// test returns — a goroutine still racing when the process exits would never
	// get the chance to trip -race.
	for i := 0; i < n; i++ {
		select {
		case ev := <-m.bridge.events:
			if ev.kind != "aside" {
				t.Fatalf("unexpected event kind %q, want %q", ev.kind, "aside")
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d of %d asides completed", i, n)
		}
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

// TestSecondTaskRewireNoRaceWithConcurrentUIReads reproduces the
// maybeRewireHistoryTool data race: a session that starts with an empty
// archive gets its first entry archived once the first task's own remember()
// runs; the SECOND task then finds hasArchivedHistory() true and rewires via
// wire(), which reassigns a.client and a.ag with no lock (main.go's
// a.client=.../a.ag=... assignments in wire()). Before the fix, that rewire
// check (maybeRewireHistoryTool) ran from inside runTaskStreaming — i.e. on
// the SAME goroutine Bubble Tea runs a tea.Cmd's closure on — concurrently
// with the UI's Update/View goroutine reading m.app.client/m.app.ag while the
// task is in flight (confirmed with -race: hits at both assignments). The fix
// moves the call into runTask/runLoop, which run synchronously on the Update
// goroutine BEFORE the tea.Cmd closure is even built, so the rewire
// happens-before any concurrent read.
//
// This drives runTask exactly as Update does: get the Cmd synchronously
// (which is where the fix now does the rewire), then run it on the calling
// goroutine while a second, UI-simulating goroutine spins reading
// m.app.client/m.app.ag concurrently — exactly how the TUI behaves while a
// task is in flight. Must stay -race clean.
func TestSecondTaskRewireNoRaceWithConcurrentUIReads(t *testing.T) {
	url := tuiFakeServer(t, tuiContent("first answer"), tuiContent("second answer"))
	a := tuiTestApp(t, url)
	a.cfg.ReflectDisabled = true // keep the fake server's response order to just the two tasks' own calls
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// First task: archive starts empty (wire() above set historyToolOn=false);
	// remember() archives this task's own entry right before Run() returns.
	m.runTask("first task")()
	if !a.hasArchivedHistory() {
		t.Fatal("first task should have archived an entry via remember()")
	}
	if a.historyToolOn {
		t.Fatal("historyToolOn should still be false — the rewire check only runs at the START of a task")
	}

	// Second task: this is where the rewire path triggers. Race the task's
	// own execution against a UI-simulating reader goroutine.
	cmd := m.runTask("second task")
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				m.app.client.Usage() // mirrors View()'s live read
				_ = m.app.ag         // mirrors the UI's other live field read
			}
		}
	}()
	cmd()
	close(stop)
	wg.Wait()

	if !a.historyToolOn {
		t.Error("historyToolOn should be true after the second task rewired in the history tool")
	}
}

// TestMCPCommandDoesNotRaceConfigReload is a regression test for a data race:
// the /mcp command's tea.Cmd goroutine (built in runCommand) used to read
// a.cfg.McpServers live via mcpList → mcpServerNames/mcpClient, which races
// reconfigure() (/login, /init) reassigning a.cfg wholesale with no lock.
// /mcp's bare form never sets m.state busy (see commandWhileBusy), so nothing
// stops the user from running /login the instant after typing /mcp, while an
// earlier /mcp's tea.Cmd goroutine is still in flight.
//
// The fix captures a.cfg.McpServers synchronously in runCommand, before the
// tea.Cmd's goroutine starts, so that goroutine never touches a.cfg again.
// This drives runCommand("/mcp") on the calling goroutine (as Update would)
// and runs the returned tea.Cmd on its own goroutine — exactly how bubbletea
// dispatches it — while concurrently reassigning a.cfg, the same way
// reconfigure() does for /login. Must be -race clean.
func TestMCPCommandDoesNotRaceConfigReload(t *testing.T) {
	a := tuiTestApp(t, tuiFakeServer(t))
	a.cfg.McpServers = map[string]mcp.Server{"a": {URL: "http://mcp.invalid/a"}, "b": {URL: "http://mcp.invalid/b"}}
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.bridge.Abort)
	// Pre-cache a client for every server name used below (including "c",
	// swapped in by the simulated reconfigure) so mcpClient's cache hit always
	// short-circuits before reaching approveGated — which routes through the
	// UI bridge and would otherwise block forever waiting for a real
	// tea.Program's Update to answer it, something this test never drives.
	// This test targets the config-snapshot race, not approval flow (already
	// covered by TestMCPClientGatesLaunchApproval and the e2e tests above).
	m.app.mcpClients = map[string]*mcp.Client{"a": {}, "b": {}, "c": {}}

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		_, cmd := m.runCommand("/mcp") // captures a.cfg.McpServers HERE, synchronously
		if cmd == nil {
			t.Fatal(`runCommand("/mcp") returned a nil tea.Cmd`)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			cmd() // bubbletea would run this on its own goroutine
		}()
		// Simulates reconfigure() (/login, /init): a.cfg reassigned wholesale, no
		// lock, while an earlier /mcp's tea.Cmd goroutine may still be running.
		newCfg := config.Default()
		newCfg.McpServers = map[string]mcp.Server{"c": {URL: "http://mcp.invalid/c"}}
		m.app.cfg = newCfg
	}
	wg.Wait()
}
