package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/feedback"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/telemetry"
)

// An update never turns usage statistics on: an install that existed before
// this build, and never chose, is written off. A first run is left for setup.
func TestSettleTelemetryDefault(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	settleTelemetryDefault(false) // a first run with no setup (piped, CI): nothing decided
	if cfg, _ := config.Load(t.TempDir()); cfg.Telemetry != nil {
		t.Fatalf("a first run without setup decided %v", *cfg.Telemetry)
	}
	if err := config.SaveChannel("stable"); err != nil { // an install from before this build
		t.Fatal(err)
	}
	settleTelemetryDefault(true)
	cfg, _ := config.Load(t.TempDir())
	if cfg.Telemetry == nil || *cfg.Telemetry {
		t.Fatalf("an existing install must be settled OFF, got %v", cfg.Telemetry)
	}
	config.SaveTelemetry(true) // the user's own choice is never overwritten
	settleTelemetryDefault(true)
	if cfg, _ := config.Load(t.TempDir()); !*cfg.Telemetry {
		t.Error("settle overwrote a choice")
	}
}

// First-run setup on a brand-new machine turns it on; re-running setup on an
// existing install doesn't.
func TestFirstRunSetupTurnsTelemetryOn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	maybeInit(bufio.NewReader(strings.NewReader("\n\n\nqwen-test\n")), true)
	if cfg, _ := config.Load(t.TempDir()); cfg.Telemetry == nil || !*cfg.Telemetry {
		t.Fatalf("a first run's setup did not turn it on: %v", cfg.Telemetry)
	}

	t.Setenv("HOME", t.TempDir())
	config.SaveChannel("stable") // existing install, never chose
	maybeInit(bufio.NewReader(strings.NewReader("\n\n\nqwen-test\n")), true)
	if cfg, _ := config.Load(t.TempDir()); cfg.Telemetry != nil {
		t.Errorf("re-running setup decided for an existing install: %v", *cfg.Telemetry)
	}
}

// A checkout's own .agent/config.json can't turn telemetry on for the user.
func TestWorkspaceCannotTurnTelemetryOn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ws := t.TempDir()
	os.MkdirAll(filepath.Join(ws, ".agent"), 0o755)
	os.WriteFile(filepath.Join(ws, ".agent", "config.json"), []byte(`{"telemetry": true}`), 0o644)
	if cfg, _ := config.Load(ws); cfg.Telemetry != nil {
		t.Errorf("workspace config set telemetry to %v", *cfg.Telemetry)
	}
}

func telemetryApp(t *testing.T) *app {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("IPS_API_BASE", "http://127.0.0.1:1") // a dev build records only when pointed at a server
	t.Setenv("DO_NOT_TRACK", "")
	on := true
	a := &app{cfg: config.Default(), workspace: t.TempDir()}
	a.cfg.Telemetry = &on
	if err := telemetry.Enable(telemetryPath()); err != nil {
		t.Fatal(err)
	}
	a.refreshTelemetry() // what wire() does: sets the gate from a.cfg
	return a
}

// A finished task counts what it did — and nothing about what it was.
func TestATaskIsCounted(t *testing.T) {
	a := telemetryApp(t)
	a.cfg.LLM.Model = "mlx-community/Qwen3-Coder-30B"
	a.countTask(agent.Transcript{
		Messages: []llm.Message{
			{Role: "tool", Name: "run", Content: "an EARLIER task's call, in the history"},
			llm.User("refactor internal/secret/plan.go"),
			{Role: "tool", Name: "file", Content: "ok"},
			{Role: "tool", Name: "mcp", Content: "ok"},
		},
		ToolUses: map[string]int{"file": 1, "mcp": 1}, // what this run itself did
	})
	a.countSpawn(true, "")
	s, _ := telemetry.Load(telemetryPath())
	d := s.Days[time.Now().Format(time.DateOnly)]
	if d == nil {
		t.Fatal("nothing recorded")
	}
	for k, want := range map[string]int{"tasks": 1, "tool_calls": 2, "mcp": 1, "local_model": 1, "external_agents": 1} {
		if d.Features[k] != want {
			t.Errorf("%s = %d, want %d (%v)", k, d.Features[k], want, d.Features)
		}
	}
	if strings.Join(d.Families, ",") != "qwen" {
		t.Errorf("families = %v", d.Families)
	}
	raw, _ := os.ReadFile(telemetryPath())
	for _, leak := range []string{"secret", "plan.go", "Qwen3-Coder", "refactor"} {
		if strings.Contains(string(raw), leak) {
			t.Errorf("the state file holds %q:\n%s", leak, raw)
		}
	}
}

// DO_NOT_TRACK pauses it: nothing is recorded.
func TestDoNotTrackPauses(t *testing.T) {
	a := telemetryApp(t)
	t.Setenv("DO_NOT_TRACK", "1")
	a.refreshTelemetry()
	a.countTask(agent.Transcript{})
	if s, _ := telemetry.Load(telemetryPath()); len(s.Days) != 0 {
		t.Errorf("recorded under DO_NOT_TRACK: %v", s.Days)
	}
}

// /telemetry off forgets everything; /telemetry shows exactly what would go.
func TestTelemetryCommand(t *testing.T) {
	a := telemetryApp(t)
	a.countTask(agent.Transcript{})
	shown := strings.Join(a.telemetryCommand(""), "\n")
	if !strings.Contains(shown, `"product":"ipsupport-code"`) || !strings.Contains(shown, `"tasks":1`) {
		t.Errorf("/telemetry does not show the report:\n%s", shown)
	}
	a.telemetryCommand("off")
	if _, err := os.Stat(telemetryPath()); !os.IsNotExist(err) {
		t.Error("off kept the state file")
	}
	if cfg, _ := config.Load(t.TempDir()); cfg.Telemetry == nil || *cfg.Telemetry {
		t.Error("off was not saved")
	}
}

// /rate parses the stars, the words and an optional signature, sends them,
// and stops suggesting itself once a rating went through.
func TestRateCommandSends(t *testing.T) {
	a := telemetryApp(t)
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/reviews" {
			t.Errorf("posted to %s", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&got)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer ts.Close()
	t.Setenv("IPS_API_BASE", ts.URL)

	_, send := a.rateCommand("5 works great with my local model --name Ana")
	if send == nil {
		t.Fatal("nothing to send")
	}
	out := strings.Join(send(context.Background()), " ")
	if !strings.Contains(out, "thank you") {
		t.Fatalf("send said %q", out)
	}
	if got["rating"] != float64(5) || got["text"] != "works great with my local model" || got["author"] != "Ana" || got["product"] != "ipsupport-code" {
		t.Errorf("sent %v", got)
	}
	s, _ := feedback.UpdatePrompt(ratePath(), func(*feedback.PromptState) {})
	if !s.Reviewed {
		t.Error("a sent rating should stop the suggestion")
	}
	if lines, send := a.rateCommand("9 too many stars"); send != nil || !strings.Contains(strings.Join(lines, ""), "1 to 5") {
		t.Errorf("an out-of-range rating was not refused: %v", lines)
	}
}

// The suggestion shows on a release build after the first day, then not
// again for two weeks — showing it is the snooze.
func TestRateHintOncePerTwoWeeks(t *testing.T) {
	a := telemetryApp(t)
	old := version
	version = "v0.61.0"
	defer func() { version = old }()
	feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) { s.FirstLaunch = time.Now().Add(-48 * time.Hour) })
	if a.rateHint() == "" {
		t.Fatal("no suggestion after the first day")
	}
	if h := a.rateHint(); h != "" {
		t.Errorf("suggested twice in a row: %q", h)
	}
}

// The user's explicit choice in the global config is theirs, in both
// directions: a checkout's config can't flip it. (json.Unmarshal writes
// through a non-nil *bool — saving the pointer was not saving the value.)
func TestWorkspaceCannotOverrideAnExplicitChoice(t *testing.T) {
	for _, global := range []bool{false, true} {
		t.Setenv("HOME", t.TempDir())
		config.SaveTelemetry(global)
		ws := t.TempDir()
		os.MkdirAll(filepath.Join(ws, ".agent"), 0o755)
		os.WriteFile(filepath.Join(ws, ".agent", "config.json"),
			[]byte(`{"telemetry": `+map[bool]string{true: "false", false: "true"}[global]+`}`), 0o644)
		cfg, _ := config.Load(ws)
		if cfg.Telemetry == nil || *cfg.Telemetry != global {
			t.Errorf("global %v, workspace said the opposite: got %v", global, cfg.Telemetry)
		}
	}
}

// Going offline mid-session stops recording and sending at once.
func TestOfflineClosesTheGate(t *testing.T) {
	a := telemetryApp(t)
	a.cfg.Offline = true
	a.refreshTelemetry() // what /offline's wire() does
	if a.telemetryOn.Load() {
		t.Fatal("still live offline")
	}
	a.countTask(agent.Transcript{})
	if s, _ := telemetry.Load(telemetryPath()); len(s.Days) != 0 {
		t.Errorf("recorded while offline: %v", s.Days)
	}
}

// A session launched with telemetry off that turns it on starts sending —
// not only after the next restart.
func TestTurningOnMidSessionStartsTheSender(t *testing.T) {
	a := telemetryApp(t)
	got := make(chan string, 4)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()
	t.Setenv("IPS_API_BASE", ts.URL)

	off := false
	a.cfg.Telemetry = &off
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.startTelemetry(ctx) // launched off: no sender, state forgotten
	yesterday := time.Now().AddDate(0, 0, -1).Format(time.DateOnly)
	telemetry.Enable(telemetryPath())
	telemetry.Record(telemetryPath(), yesterday, telemetry.Counts{Features: map[string]int{"tasks": 1}})

	a.telemetryCommand("on")
	select {
	case p := <-got:
		if p != "/telemetry" {
			t.Errorf("sent to %s", p)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("turning it on mid-session never started the sender")
	}
	cancel()
	a.telemetryWG.Wait() // the sender finishes its write before the temp dir goes
}

// Another session's /telemetry off deletes the shared state and saves the
// opt-out. This session still has telemetry on in memory; a re-wire here (a
// model switch) must not recreate the install ID and resume reporting.
func TestRewireDoesNotUndoAnotherSessionsOptOut(t *testing.T) {
	a := telemetryApp(t)
	ctx, cancel := context.WithCancel(context.Background())
	a.startTelemetry(ctx)
	defer func() { cancel(); a.telemetryWG.Wait() }()

	telemetry.Disable(telemetryPath()) // what the OTHER session's /telemetry off did
	config.SaveTelemetry(false)
	a.refreshTelemetry() // this session's wire()
	if s, _ := telemetry.Load(telemetryPath()); s.InstallID != "" {
		t.Fatal("a re-wire recreated the install ID the user deleted")
	}
	if a.telemetryOn.Load() {
		t.Error("the gate stayed open with no state")
	}
}

// Turning it on while offline is still consent: when the session comes back
// online, reporting starts — it is not mistaken for an opt-out made elsewhere.
func TestOptInWhileOfflineSurvivesGoingOnline(t *testing.T) {
	a := telemetryApp(t)
	telemetry.Disable(telemetryPath())
	off := false
	a.cfg.Telemetry, a.cfg.Offline = &off, true
	ctx, cancel := context.WithCancel(context.Background())
	a.startTelemetry(ctx)
	defer func() { cancel(); a.telemetryWG.Wait() }()

	a.telemetryCommand("on") // offline: paused, but chosen
	a.cfg.Offline = false
	a.refreshTelemetry() // what /offline off's wire() does
	if !a.telemetryOn.Load() {
		t.Error("an opt-in made offline was lost on going online")
	}
}

// The documented way to turn it off from a script works like /telemetry off:
// the install ID and unsent counters go at once, which also stops sessions
// already running (they find no state).
func TestConfigSetTelemetryFalseForgetsAtOnce(t *testing.T) {
	a := telemetryApp(t)
	a.countTask(agent.Transcript{})
	config.SaveTelemetry(false) // what `config set telemetry false` writes
	applyTelemetryKey(t.TempDir(), "telemetry")
	if _, err := os.Stat(telemetryPath()); !os.IsNotExist(err) {
		t.Error("config set telemetry false kept the install ID and counters")
	}
}
