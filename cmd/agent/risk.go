package main

import (
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
func (a *app) riskObserver(pol *policy.Engine) func(tool, action string, params map[string]any) {
	if strings.EqualFold(os.Getenv(EnvRiskOff), "off") {
		return nil
	}
	sh := risk.NewShadow(risk.DefaultOrNil())
	if sh == nil {
		return nil
	}
	a.shadow = sh
	return func(tool, action string, params map[string]any) {
		sh.Observe(tool, action, params, policyVerdict(pol, tool, action, params))
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
