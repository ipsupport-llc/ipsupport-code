package risk

import (
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
)

// PolicyVerdict is what the permission policy decided for a call, as the risk
// side sees it. Deliberately a small enum of its own rather than
// policy.Decision: internal/risk must not depend on the policy engine, so that
// the classifier can be tested, retrained and reasoned about on its own.
type PolicyVerdict int

const (
	VerdictUnknown PolicyVerdict = iota // the caller could not say (no probe wired)
	VerdictAllow                        // ran without asking
	VerdictAsk                          // prompted the user
	VerdictDeny                         // refused
)

func (v PolicyVerdict) String() string {
	switch v {
	case VerdictAllow:
		return "allow"
	case VerdictAsk:
		return "ask"
	case VerdictDeny:
		return "deny"
	}
	return "unknown"
}

// Threshold is where a score starts counting as "the model thinks this is
// risky" in the shadow log. It is a reporting threshold only — nothing is
// blocked at any value — and it exists so the log can say WHICH labels fired
// instead of printing six numbers on every call.
const Threshold float32 = 0.5

// Shadow scores tool calls and logs what it thought, next to what the policy
// actually did. It blocks nothing, changes nothing, and returns nothing the
// caller is expected to act on: the point of the first mode is to find out
// whether the model's opinion is worth anything before it is allowed to matter.
type Shadow struct {
	model *Model
	// calls/flagged/disagreed are the run's counters, read by Stats for a single
	// end-of-session line. Atomic because tool calls can be dispatched from a
	// sub-agent's goroutine while the main run is dispatching its own.
	calls, flagged, disagreed atomic.Int64
}

// NewShadow returns a shadow scorer, or nil if there is no usable model — a nil
// *Shadow is safe to call, so the hot path never needs a check.
func NewShadow(m *Model) *Shadow {
	if m == nil {
		return nil
	}
	return &Shadow{model: m}
}

// Observe scores one tool call and logs it beside the policy's own verdict.
//
// The interesting line is the disagreement: a call the policy waved through
// that the model thinks is destructive, or one the policy stopped that the
// model finds unremarkable. Those two sets are the entire product of shadow
// mode — the first says what a risk signal could add, the second says how much
// noise it would add.
func (s *Shadow) Observe(tool, action string, params map[string]any, verdict PolicyVerdict) Assessment {
	if s == nil {
		return Assessment{}
	}
	a := s.model.Assess(tool, action, params)
	s.calls.Add(1)

	flagged := a.Risk >= Threshold
	if flagged {
		s.flagged.Add(1)
	}
	// Disagreement, both directions. "Allowed but flagged" is the one worth
	// acting on later; "asked/denied but unremarkable" measures the friction a
	// risk gate would have to justify.
	dis := ""
	switch {
	case flagged && verdict == VerdictAllow:
		dis = "allowed-but-flagged"
	case !flagged && (verdict == VerdictDeny || verdict == VerdictAsk):
		dis = "gated-but-unremarkable"
	}
	if dis != "" {
		s.disagreed.Add(1)
	}

	slog.Debug("risk shadow",
		"tool", tool, "action", action,
		"risk", round2(a.Risk), "top", a.Top,
		"labels", strings.Join(a.Above(Threshold), " "),
		"policy", verdict.String(),
		"disagreement", dis,
		"call", clip(CallText(tool, action, params), 160))
	return a
}

// Stats is the run's tally, for one line at the end rather than a number the
// user has to derive by counting log lines.
func (s *Shadow) Stats() (calls, flagged, disagreed int64) {
	if s == nil {
		return 0, 0, 0
	}
	return s.calls.Load(), s.flagged.Load(), s.disagreed.Load()
}

// Summary is that line, or "" when nothing was scored.
func (s *Shadow) Summary() string {
	c, f, d := s.Stats()
	if c == 0 {
		return ""
	}
	return fmt.Sprintf("risk shadow: scored %d call(s), %d over %.2f, %d disagreed with the policy", c, f, Threshold, d)
}

func round2(f float32) float32 { return float32(int(f*100+0.5)) / 100 }

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
