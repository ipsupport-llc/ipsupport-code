package risk

import (
	"context"
	"fmt"
)

// The assessment travels on the CONTEXT from the point a call is scored to the
// point a human answers for it.
//
// Those are two different places: the score happens in the agent before the
// tool is dispatched, and the approval prompt fires inside the tool, further
// down the same call. A field on the app would be wrong — a sub-agent dispatches
// its own calls on its own goroutine while the main run dispatches its, and a
// single "the call being approved" slot would hand one run's answer to the
// other's call. The context is already threaded through exactly that path
// (execOne -> Dispatch -> the tool -> Approve), so it is the one carrier that is
// correct under concurrency by construction.
type assessKey struct{}

// scored is the assessment plus what it was about, so a correction can be
// rebuilt from the approval alone.
type scored struct {
	tool, action string
	params       map[string]any
	a            Assessment
}

// WithAssessment attaches one call's score to the context handed to the tool.
func WithAssessment(ctx context.Context, tool, action string, params map[string]any, a Assessment) context.Context {
	return context.WithValue(ctx, assessKey{}, &scored{tool: tool, action: action, params: params, a: a})
}

// AssessmentFrom reads back the score attached to this call, for showing it to
// the person about to answer for it.
func AssessmentFrom(ctx context.Context) (Assessment, bool) {
	s, _ := ctx.Value(assessKey{}).(*scored)
	if s == nil {
		return Assessment{}, false
	}
	return s.a, true
}

// Note renders the assessment as the one short line a prompt or a tool-call line
// can carry: "0.98 credential_access". Empty when nothing fired, so a caller can
// use it directly as "show this if non-empty" — a score on every routine call
// would be noise, and noise is what makes a risk signal get ignored.
func (a Assessment) Note() string {
	if a.Risk < Threshold || a.Top == "" {
		return ""
	}
	return fmt.Sprintf("%.2f %s", a.Risk, a.Top)
}

// CorrectionFrom reads back what was scored and reports the correction a human's
// answer implies — ok is false when there is nothing to learn.
//
// Only the two disagreements teach anything:
//
//	flagged and approved -> a false alarm; the labels that fired are exactly known
//	quiet and refused    -> a miss; the human supplies THAT it is risky, and the
//	                        model's own strongest label supplies the guess at which
//
// An approval of a call the model also thought was fine confirms only that the
// two agreed, and a refusal of one it flagged confirms the same. Learning from
// those would be learning from its own output.
func CorrectionFrom(ctx context.Context, approved bool) (Correction, bool) {
	s, _ := ctx.Value(assessKey{}).(*scored)
	if s == nil {
		return Correction{}, false
	}
	flagged := s.a.Risk >= Threshold
	switch {
	case flagged && approved:
		var fired []string
		for l, v := range s.a.Scores {
			if l != LabelSafe && v >= Threshold {
				fired = append(fired, l)
			}
		}
		if len(fired) == 0 {
			return Correction{}, false
		}
		return Correction{Tool: s.tool, Action: s.action, Params: s.params, Risky: false, Labels: fired}, true
	case !flagged && !approved:
		if s.a.Top == "" {
			return Correction{}, false
		}
		return Correction{Tool: s.tool, Action: s.action, Params: s.params, Risky: true, Labels: []string{s.a.Top}}, true
	}
	return Correction{}, false
}
