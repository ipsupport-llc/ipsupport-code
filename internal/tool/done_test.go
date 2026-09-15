package tool

import (
	"context"
	"testing"
)

// The agent loop normally intercepts a done-only turn before dispatch (see
// Agent.Run's isDoneOnly) — this only exercises the tool's own shape: no
// required params, not a mutation (must stay callable in plan mode), and a
// harmless answer for the rare case a call reaches it directly.
func TestDoneToolShape(t *testing.T) {
	d := NewDone()
	if d.Name() != "done" {
		t.Errorf("Name() = %q, want %q", d.Name(), "done")
	}
	actions := d.Actions()
	if len(actions) != 1 || actions[0] != "done" {
		t.Errorf("Actions() = %v, want exactly [\"done\"]", actions)
	}
	if d.Mutates("done") {
		t.Error("done must not be a mutation — plan mode should never block it")
	}
	r := d.Call(context.Background(), "done", map[string]any{})
	if r.IsError {
		t.Errorf("Call(done, {}) = %+v, want a plain success", r)
	}
}

// Extra/unexpected params must be silently ignored, not rejected — done has no
// required params, so a model stuffing a "summary" or "reason" field in (or
// anything else) alongside the bare call must not error.
func TestDoneToolIgnoresExtraParams(t *testing.T) {
	d := NewDone()
	r := d.Call(context.Background(), "done", map[string]any{"summary": "did the thing", "reason": "all set", "whatever": 42})
	if r.IsError {
		t.Errorf("Call with extra params = %+v, want it silently ignored, not an error", r)
	}
}
