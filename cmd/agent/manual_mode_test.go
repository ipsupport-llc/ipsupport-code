package main

import (
	"context"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// Manual mode asks before every change the policy would have let through
// unasked — a shell command, a file write, edit or mkdir — and leaves reads
// alone. An 'a' at a prompt still covers its kind for the session, and auto
// mode is untouched.
func TestManualModeAsksBeforeEveryChange(t *testing.T) {
	a, pol, ap := gateApp(t, "run.default=allow", "file.default=allow")
	gate := a.riskGate(pol)
	ctx := context.Background()
	ls := map[string]any{"command": "ls"}
	note := map[string]any{"path": "note.txt"}

	if msg := gate(ctx, "run", "shell", ls); msg != "" || ap.calls != 0 {
		t.Fatalf("auto mode asked about an allowed command: %q, %d call(s)", msg, ap.calls)
	}

	a.setMode(modeManual)
	for _, c := range []struct{ name, action string }{{"run", "shell"}, {"file", "write"}, {"file", "append"}, {"file", "edit"}, {"file", "mkdir"}} {
		before := ap.calls
		params := note
		if c.name == "run" {
			params = ls
		}
		msg := gate(ctx, c.name, c.action, params)
		if ap.calls != before+1 || !strings.Contains(msg, "denied by user") || strings.Contains(msg, "risk check") {
			t.Errorf("%s %s in manual: asked %d time(s), result %q — want one prompt and a plain refusal", c.name, c.action, ap.calls-before, msg)
		}
	}
	for _, action := range []string{"read", "list", "find", "search"} {
		before := ap.calls
		if msg := gate(ctx, "file", action, note); msg != "" || ap.calls != before {
			t.Errorf("file %s in manual was stopped: %q", action, msg)
		}
	}

	a.allowSession("run") // 'a' at a shell prompt
	before := ap.calls
	if msg := gate(ctx, "run", "shell", ls); msg != "" || ap.calls != before {
		t.Errorf("after 'a', manual still asked about a shell command: %q", msg)
	}
}

// A prompt manual mode raised that nobody answered says so, without a risk
// flag it never had.
func TestManualModeTimeoutSaysNobodyAnswered(t *testing.T) {
	a, pol, _ := gateApp(t, "run.default=allow")
	a.approver = silentApprover{}
	a.setMode(modeManual)
	msg := a.riskGate(pol)(context.Background(), "run", "shell", map[string]any{"command": "ls"})
	if !strings.Contains(msg, "nobody answered") || strings.Contains(msg, "risk check") {
		t.Errorf("result %q, want that nobody answered and no risk flag", msg)
	}
}

// A sub-agent spawn asks in manual even with spawns set to allow.
func TestManualModeAsksBeforeASpawn(t *testing.T) {
	ap := &countingApprover{}
	a := &app{cfg: config.Default(), workspace: t.TempDir(), approver: ap}
	a.setMode(modeManual)
	plan := spawnPlan{profile: "p", llmCfg: config.LLM{Model: "m"}, subWorkspace: a.workspace, spawnDefault: "allow"}
	if _, err := a.runSpawnPlan(context.Background(), plan, "a task", nil); err == nil || ap.calls != 1 {
		t.Errorf("spawn in manual: err %v, asked %d time(s); want one prompt and a refusal", err, ap.calls)
	}
}

// shift+tab and the /config row step auto → plan → manual → auto, and each
// lands where the agent and the indicator agree.
func TestModesCycle(t *testing.T) {
	a := &app{cfg: config.Default()}
	m := &tuiModel{app: a}
	want := []mode{modePlan, modeManual, modeAuto}
	for _, w := range want {
		a.setMode(a.mode().next())
		if a.mode() != w || a.planMode != (w == modePlan) || a.manual.Load() != (w == modeManual) {
			t.Fatalf("mode = %v (plan %v, manual %v), want %v", a.mode(), a.planMode, a.manual.Load(), w)
		}
		if line := m.modeLine(); !strings.Contains(line, w.String()+" mode on") {
			t.Errorf("indicator %q, want %s", line, w)
		}
	}
	a.setMode(modeManual)
	if _, v, _ := m.configRowView("mode"); !strings.Contains(v, "manual") {
		t.Errorf("/config shows %q for manual", v)
	}
	if got := m.shadowLabel("mode", m.liveShadow()); !strings.Contains(got, "manual") {
		t.Errorf("/config preview shows %q for manual", got)
	}
}

// Accepting a plan runs it in manual when manual was chosen while it was being
// written, or when accepted with m — it used to switch to auto regardless, and
// every change then ran without the yes the user had asked to give.
func TestAnAcceptedPlanKeepsManual(t *testing.T) {
	for _, c := range []struct {
		name string
		from mode
		key  tea.KeyMsg
		want mode
	}{
		{"enter from plan", modePlan, tea.KeyMsg{Type: tea.KeyEnter}, modeAuto},
		{"m from plan", modePlan, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'m'}}, modeManual},
		{"enter after shift+tab to manual", modeManual, tea.KeyMsg{Type: tea.KeyEnter}, modeManual},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := &tuiModel{app: &app{cfg: config.Default()}, input: textarea.New(), state: stPlanReview}
			m.app.setMode(c.from)
			if c.from == modeManual && !strings.Contains(m.planReviewKeys(), "in manual") {
				t.Errorf("keys %q do not say the plan runs in manual", m.planReviewKeys())
			}
			mode := m.planTarget(c.key.String() == "m")
			if mode != c.want {
				t.Errorf("accepting with %q from %v runs in %v, want %v", c.key.String(), c.from, mode, c.want)
			}
		})
	}
}

// A run call whose command arrived in the action field is judged as the
// command it runs as: manual mode asks about it. It used to pass the gate as
// nothing, then run once Dispatch recovered the command.
func TestManualModeJudgesARecoveredCall(t *testing.T) {
	a, pol, ap := gateApp(t, "run.default=allow", "file.default=allow")
	a.setMode(modeManual)
	reg := tool.NewRegistry(tool.NewRun(pol, ap, 0, 0, tool.Shell("")))
	action, params := reg.Normalize("run", "touch example.txt", map[string]any{})
	if msg := a.riskGate(pol)(context.Background(), "run", action, params); ap.calls != 1 || !strings.Contains(msg, "touch example.txt") {
		t.Errorf("recovered call: asked %d time(s), result %q; want it asked about as the command it runs", ap.calls, msg)
	}
	// A number is a path to the tool, so it is one to the gate.
	before := ap.calls
	if msg := a.riskGate(pol)(context.Background(), "file", "write", map[string]any{"path": 123}); ap.calls != before+1 || !strings.Contains(msg, "write 123") {
		t.Errorf("numeric path: asked %d time(s), result %q; want a prompt", ap.calls-before, msg)
	}
}
