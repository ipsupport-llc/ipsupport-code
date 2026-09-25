package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/risk"
)

// EnvRiskOff turns shadow scoring off for a run. Shadow mode blocks nothing, so
// it is on by default — the point of the first mode is to collect the evidence
// that decides whether the signal is worth anything, and a signal nobody
// enables collects none. This is the escape hatch, not a feature flag: when the
// scorer is ready to mean something it gets a real /config row.
const EnvRiskOff = "IPS_RISK"

// riskObserver builds the shadow-mode hook for the agent, or nil when scoring
// is off or the model is unusable. Everything the classifier needs to be
// compared against lives here rather than in internal/risk or internal/agent:
// this is the one place that holds the model, the permission policy and the
// tool registry at once.
// ensureShadow creates the process-wide scorer, once, on the goroutine that
// calls wire(). Separated from riskObserver because sub-agents ask for an
// observer of their own from their own goroutine, and that must never be the
// call that constructs the shared scorer.
func (a *app) ensureShadow() {
	if a.shadow != nil || strings.EqualFold(os.Getenv(EnvRiskOff), "off") {
		return
	}
	base := risk.DefaultOrNil()
	if base == nil {
		return
	}
	// The local corrections live beside the rest of this workspace's state, not
	// in the binary: they are what THIS project's approvals taught, and a
	// checkout with different norms should not inherit them. A delta that no
	// longer matches the base model's labels is dropped with a line in the log —
	// its rows are positional, so applying it to a retrained model would adjust
	// the wrong things.
	var d *risk.Delta
	if loaded, err := risk.LoadDelta(a.riskDeltaPath(), base); err == nil {
		d = loaded
	} else if !os.IsNotExist(err) {
		slog.Warn("local risk corrections unusable — starting from the base model", "err", err)
	}
	// Created ONCE and never reassigned. wire() runs again on every /config
	// change and every model switch, and a background sub-agent answers its own
	// approvals on its own goroutine — reassigning a.shadow there is a data race
	// on the pointer, which is what CI caught. Reusing it also keeps what the
	// session has learned: rebuilding would drop the in-memory corrections on the
	// floor every time a setting changed.
	a.shadow = risk.NewShadow(risk.NewTuned(base, d))
}

// riskObserver returns the shadow-mode hook for an agent, scoring against the
// policy engine THAT agent runs under. Every agent gets its own: a sub-agent
// has its own workspace and its own policy, and — the part that was actually
// broken — without one of these its tool calls inherit whatever assessment is
// already on the context. That is the PARENT's `agent.spawn`, so refusing a
// sub-agent's file write recorded a correction about the spawn, with the
// spawn's parameters. Wrong call, wrong label, written to the feedback log that
// later fine-tunes the base.
func (a *app) riskObserver(pol *policy.Engine) func(ctx context.Context, tool, action string, params map[string]any) (context.Context, string) {
	sh := a.shadow
	if sh == nil {
		return nil
	}
	return func(ctx context.Context, tool, action string, params map[string]any) (context.Context, string) {
		as := sh.Observe(tool, action, params, policyVerdict(pol, tool, action, params))
		// Hand the score down to the approval prompt, which is where a human
		// answers for this call and so the only place ground truth appears — and
		// back up as a note, so the call's own line can carry it.
		return risk.WithAssessment(ctx, tool, action, params, as), as.Note()
	}
}

// riskDeltaPath is where this workspace's learned corrections live, and
// riskFeedbackPath is the append-only record of what taught them — in the same
// JSONL shape scripts/risk_dataset.jsonl uses, so the examples a run collects
// concatenate straight onto the synthetic dataset and retrain the BASE model
// offline, starting from the existing weights rather than from zero.
func (a *app) riskDeltaPath() string    { return a.statePath("risk-delta.bin") }
func (a *app) riskFeedbackPath() string { return a.statePath("risk-feedback.jsonl") }

// learnFromApproval is called with a human's answer to an approval prompt. It
// teaches the model only on a DISAGREEMENT — a flagged call approved, or an
// unflagged one refused — because anything else would be learning from the
// model's own output. See risk.CorrectionFrom.
//
// A refusal can mean "not now" as easily as "that is dangerous", so one
// correction deliberately moves the score only a little (see risk.learnRate) and
// the delta's total influence is capped. A pattern moves the model; a one-off
// does not.
func (a *app) learnFromApproval(ctx context.Context, approved bool) {
	if a.shadow == nil {
		return
	}
	c, ok := risk.CorrectionFrom(ctx, approved)
	if !ok {
		return
	}
	// In memory, here: it is microseconds, it must be visible to the very next
	// call, and Learn takes the model's own lock.
	n := a.shadow.Model().Learn(c)
	a.shadow.NoteLearned()
	slog.Debug("risk learned", "tool", c.Tool, "action", c.Action, "risky", c.Risky,
		"labels", c.Labels, "adjustments", n)

	// On disk, NOT here. This runs between the human pressing y and the tool
	// actually running, and persisting means two blocking file locks and an
	// fsync — a second session on the same workspace holding either one would
	// hang the approval. Shadow mode is not allowed to affect execution, and
	// that includes making the user wait for it.
	a.riskSaves.Add(1)
	go a.persistRiskLearning(c)
}

// waitRiskSaves blocks until every deferred write has finished. Called on the
// way out: a correction that was learned but never reached disk because the
// process exited a moment later is the one case where moving the write off the
// approval path would have cost something.
func (a *app) waitRiskSaves() { a.riskSaves.Wait() }

// persistRiskLearning writes the correction and the delta off the approval path.
// Losing a correction to a crash between the answer and the write costs one
// training example; blocking the answer on a disk costs the user.
func (a *app) persistRiskLearning(c risk.Correction) {
	defer a.riskSaves.Done()
	a.riskSaveMu.Lock() // one writer at a time: several sub-agents can be answering at once
	defer a.riskSaveMu.Unlock()
	if err := risk.AppendFeedback(a.riskFeedbackPath(), c); err != nil {
		slog.Warn("risk feedback not recorded", "err", err)
	}
	if err := a.shadow.Model().SaveDelta(a.riskDeltaPath()); err != nil {
		slog.Warn("local risk corrections not saved", "err", err)
	}
}

// policyVerdict reports what the permission policy would say about a call — and
// says "unknown" rather than guessing when it cannot know.
//
// The tool itself is the authority on its own gating, so this can only mirror
// the two domains where the policy engine genuinely decides: run's command
// check and file's write check. Everything else returns VerdictUnknown, which
// the shadow log records as "no comparison available" instead of inventing a
// verdict for the disagreement column to be wrong about. A disagreement count
// built on a guess is worse than a smaller one built on facts.
func policyVerdict(pol *policy.Engine, tool, action string, params map[string]any) risk.PolicyVerdict {
	if pol == nil {
		return risk.VerdictUnknown
	}
	switch tool {
	case "run":
		cmd, _ := params["command"].(string)
		if cmd == "" {
			return risk.VerdictUnknown
		}
		return fromDecision(pol.Run(cmd))
	case "file":
		path, _ := params["path"].(string)
		if path == "" {
			return risk.VerdictUnknown
		}
		switch action {
		case "write", "append", "edit", "mkdir":
			d, err := pol.Write(path)
			if err != nil { // outside the jail, or unresolvable — the tool refuses
				return risk.VerdictDeny
			}
			return fromDecision(d)
		case "read", "list", "find", "search":
			if err := pol.Read(path); err != nil {
				return risk.VerdictDeny
			}
			return risk.VerdictAllow
		}
	}
	return risk.VerdictUnknown
}

func fromDecision(d policy.Decision) risk.PolicyVerdict {
	switch d {
	case policy.Allow:
		return risk.VerdictAllow
	case policy.Ask:
		return risk.VerdictAsk
	case policy.Deny:
		return risk.VerdictDeny
	}
	return risk.VerdictUnknown
}

// logRiskSummary writes the run's shadow tally, if anything was scored. One
// line at the end, so the numbers that decide whether this is worth promoting
// out of shadow mode don't have to be reconstructed by counting log lines.
func (a *app) logRiskSummary() {
	if a.shadow == nil {
		return
	}
	if s := a.shadow.Summary(); s != "" {
		slog.Debug(s)
	}
}

// riskCommand backs /risk: what the scorer has done this run, and what it has
// learned locally. Read-only except for "reset", which drops the corrections —
// the escape hatch for a model that learned the wrong lesson from a refusal
// that meant "not now".
func (a *app) riskCommand(rest string) []string {
	if a.shadow == nil {
		return []string{"risk scoring is off (" + EnvRiskOff + "=off, or the model could not load)"}
	}
	switch strings.ToLower(strings.TrimSpace(rest)) {
	case "reset":
		if err := os.Remove(a.riskDeltaPath()); err != nil && !os.IsNotExist(err) {
			return []string{"error: " + err.Error()}
		}
		// Cleared in place rather than by rebuilding through wire(): the scorer is
		// created once and shared with whatever goroutines are mid-call.
		a.shadow.Model().ResetDelta()
		return []string{"local risk corrections cleared — back to the shipped model",
			"  the record of what taught them is kept: " + a.riskFeedbackPath()}
	case "", "status":
		m := a.shadow.Model()
		calls, flagged, disagreed := a.shadow.Stats()
		out := []string{
			fmt.Sprintf("risk scoring: shadow mode — logs, blocks nothing (threshold %.2f)", risk.Threshold),
			fmt.Sprintf("  this run    %d call(s) scored · %d over the threshold · %d disagreed with the policy", calls, flagged, disagreed),
			fmt.Sprintf("  learned     %d correction(s) this run · %d local adjustment(s) in total", a.shadow.Learned(), m.Adjustments()),
			"  labels      " + strings.Join(labelSummary(m.Base()), " · "),
		}
		if n := feedbackCount(a.riskFeedbackPath()); n > 0 {
			out = append(out, fmt.Sprintf("  collected   %d example(s) in %s", n, a.riskFeedbackPath()),
				"              retrain the base on them: python3 scripts/train_risk.py --from internal/risk/model.bin "+a.riskFeedbackPath())
		}
		return append(out, "  /risk reset  drops what this workspace learned")
	}
	return []string{"usage: /risk [status] · /risk reset"}
}

// labelSummary names the model's labels, marking the informational ones — the
// ones reported beside the score without counting toward it.
func labelSummary(m *risk.Model) []string {
	out := make([]string, 0, len(m.Labels))
	for _, l := range m.Labels {
		if m.IsInformational(l) {
			l += " (informational)"
		}
		out = append(out, l)
	}
	return out
}

// feedbackCount counts the collected corrections; 0 on any problem, since this
// is a status line and not a place to fail.
func feedbackCount(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n := 0
	for _, l := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(l) != "" {
			n++
		}
	}
	return n
}
