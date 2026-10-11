package main

import (
	"bufio"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/risk"
)

// gateApp builds a wired app with these overrides and a refusing approver that
// counts how often a person was asked.
func gateApp(t *testing.T, overrides ...string) (*app, *policy.Engine, *countingApprover) {
	t.Helper()
	setHome(t, t.TempDir())
	a, cleanup, err := build(t.TempDir(), "", overrides, false, bufio.NewReader(strings.NewReader("")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cleanup)
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	if a.shadow == nil {
		t.Skip("no scorer")
	}
	pol, err := policy.New(a.cfg)
	if err != nil {
		t.Fatal(err)
	}
	ap := &countingApprover{}
	a.approver = ap
	return a, pol, ap
}

// scored is the context a call runs under once the observer scored it.
func scored(t *testing.T, a *app, pol *policy.Engine, command string) context.Context {
	t.Helper()
	ctx, _ := a.riskObserver(pol, "")(context.Background(), "run", "shell", map[string]any{"command": command})
	return ctx
}

const flaggedCommand = "cat ~/.ssh/id_rsa" // not on the deny floor: the policy alone would run it

func mustFlag(t *testing.T, ctx context.Context) {
	t.Helper()
	if as, _ := risk.AssessmentFrom(ctx); !as.Flagged() {
		t.Skipf("the shipped model no longer flags %q", flaggedCommand)
	}
}

// The policy allows every command, and the scorer flags this one: it stops for
// a person, and a refusal is what the model is told. An unremarkable command
// still runs without anyone being asked.
func TestRiskCheckAsksAboutAFlaggedCallThePolicyAllows(t *testing.T) {
	a, pol, ap := gateApp(t, "run.default=allow")
	gate := a.riskGate(pol)
	params := map[string]any{"command": flaggedCommand}
	ctx := scored(t, a, pol, flaggedCommand)
	mustFlag(t, ctx)

	msg := gate(ctx, "run", "shell", params)
	if ap.calls != 1 || !strings.Contains(msg, "denied by user") {
		t.Fatalf("asked %d time(s), result %q — a flagged call the policy allows must be asked about", ap.calls, msg)
	}
	ls := map[string]any{"command": "ls"}
	if msg := gate(scored(t, a, pol, "ls"), "run", "shell", ls); msg != "" || ap.calls != 1 {
		t.Errorf("an unremarkable call was stopped: asked %d time(s), result %q", ap.calls, msg)
	}
}

// Where the policy asks anyway, the tool's own prompt is the one — the gate
// must not ask a second time.
func TestRiskCheckLeavesAskingToThePolicyWhenItAsks(t *testing.T) {
	a, pol, ap := gateApp(t)
	ctx := scored(t, a, pol, flaggedCommand)
	mustFlag(t, ctx)
	if msg := a.riskGate(pol)(ctx, "run", "shell", map[string]any{"command": flaggedCommand}); msg != "" || ap.calls != 0 {
		t.Errorf("gate asked %d time(s) about a call the policy asks about itself (%q)", ap.calls, msg)
	}
}

// risk_check off — the /config row, -skip-risk-check, and -skip-permissions,
// which expands to it — leaves only the scoring.
func TestRiskCheckOffAsksNothing(t *testing.T) {
	for _, overrides := range [][]string{
		{"run.default=allow", "risk_check=false"},
		{"run.default=allow", "file.default=allow", "risk_check=false"},
	} {
		a, pol, ap := gateApp(t, overrides...)
		ctx := scored(t, a, pol, flaggedCommand)
		mustFlag(t, ctx)
		if msg := a.riskGate(pol)(ctx, "run", "shell", map[string]any{"command": flaggedCommand}); msg != "" || ap.calls != 0 {
			t.Errorf("%v: asked %d time(s), result %q", overrides, ap.calls, msg)
		}
	}
}

// 'a' at a plain prompt lets the category's unremarkable calls through, not
// its flagged ones; 'a' at a flagged prompt lets both through.
func TestSessionAllowCoversFlaggedCallsOnlyFromAFlaggedPrompt(t *testing.T) {
	a, pol, ap := gateApp(t)
	ctx := scored(t, a, pol, flaggedCommand)
	mustFlag(t, ctx)

	a.allowSession("run")
	if a.approveGated(scored(t, a, pol, "ls"), "run", "ls"); ap.calls != 0 {
		t.Fatalf("a plain 'a' did not cover an unremarkable command")
	}
	if a.approveGated(ctx, "run", flaggedCommand); ap.calls != 1 {
		t.Fatalf("a plain 'a' let a flagged command through without asking")
	}
	a.allowFlagged("run")
	if !a.approveGated(ctx, "run", flaggedCommand) || ap.calls != 1 {
		t.Errorf("'a' at a flagged prompt still asks (%d)", ap.calls)
	}
	a.resetSessionAllow()
	if a.approveGated(ctx, "run", flaggedCommand); ap.calls != 2 {
		t.Errorf("/new must drop the flagged allow too (%d)", ap.calls)
	}
}

// The TUI's 'a' on a prompt carrying a score is the flagged allow.
func TestTUIAllowOnAFlaggedPromptCoversFlaggedCalls(t *testing.T) {
	for _, tc := range []struct {
		risk    string
		flagged bool
	}{{"0.98 destructive", true}, {"", false}} {
		a := &app{}
		m := &tuiModel{app: a, bridge: newBridge(), state: stRunning, input: textarea.New()}
		m.Update(approvalMsg(approvalReq{kind: "run", detail: flaggedCommand, risk: tc.risk, reply: make(chan bool, 1)}))
		m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'a'}})
		if !a.sessionAllowed("run") {
			t.Errorf("risk %q: 'a' did not allow the category", tc.risk)
		}
		if a.flaggedAllow["run"] != tc.flagged {
			t.Errorf("risk %q: flagged allow = %v, want %v", tc.risk, a.flaggedAllow["run"], tc.flagged)
		}
	}
}

// A checkout's own .agent/config.json may turn the check on, never off: it is
// what stops a flagged command in a checkout whose file allows everything.
func TestWorkspaceConfigCannotTurnTheRiskCheckOff(t *testing.T) {
	home := t.TempDir()
	setHome(t, home)
	ws := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(ws, ".agent", "config.json"), `{"risk_check": false}`)
	if cfg, err := config.Load(ws); err != nil || !cfg.RiskCheck {
		t.Fatalf("a workspace file turned the risk check off (%v)", err)
	}
	write(config.GlobalPath(), `{"risk_check": false}`)
	if cfg, _ := config.Load(ws); cfg.RiskCheck {
		t.Error("the user's own config could not turn it off")
	}
	write(filepath.Join(ws, ".agent", "config.json"), `{"risk_check": true}`)
	if cfg, _ := config.Load(ws); !cfg.RiskCheck {
		t.Error("a workspace file could not turn it on")
	}
}
