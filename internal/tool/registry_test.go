package tool

import (
	"context"
	"strings"
	"testing"
)

type fakeTool struct {
	name    string
	actions []string
	mutates []string
	last    string
	infer   func(map[string]any) string
}

func (f *fakeTool) Name() string          { return f.name }
func (f *fakeTool) Description() string   { return f.name + " tool" }
func (f *fakeTool) Actions() []string     { return f.actions }
func (f *fakeTool) Mutates(a string) bool { return contains(f.mutates, a) }
func (f *fakeTool) Call(_ context.Context, action string, _ map[string]any) Result {
	f.last = action
	return Ok("did " + action)
}
func (f *fakeTool) InferAction(params map[string]any) string {
	if f.infer == nil {
		return ""
	}
	return f.infer(params)
}

func TestDispatchRoutes(t *testing.T) {
	ft := &fakeTool{name: "file", actions: []string{"read", "write"}}
	r := NewRegistry(ft)
	res := r.Dispatch(context.Background(), "file", "read", nil)
	if res.IsError || res.Content != "did read" {
		t.Errorf("res = %+v", res)
	}
	if ft.last != "read" {
		t.Errorf("tool was not called with the action")
	}
}

func TestDispatchWrongToolHint(t *testing.T) {
	file := &fakeTool{name: "file", actions: []string{"read", "write"}}
	web := &fakeTool{name: "web", actions: []string{"search"}}
	r := NewRegistry(file, web)
	res := r.Dispatch(context.Background(), "file", "search", nil)
	if !res.IsError {
		t.Fatal("expected an error result")
	}
	if !strings.Contains(res.Content, `belongs to tool "web"`) {
		t.Errorf("missing belongs-to hint: %q", res.Content)
	}
}

// A model sometimes calls an action name directly as if it were its own tool
// (reported live: "fetch" called as a top-level tool instead of "web" with
// action="fetch" — a common confusion for models trained on frameworks where
// each of these IS its own standalone tool). Dispatch already has the data to
// recognize this (actionToTool, built for the sibling "wrong tool" hint) —
// reuse it instead of just saying "unknown tool".
func TestDispatchActionCalledAsTopLevelToolHint(t *testing.T) {
	web := &fakeTool{name: "web", actions: []string{"search", "fetch"}}
	r := NewRegistry(web)
	res := r.Dispatch(context.Background(), "fetch", "", nil)
	if !res.IsError {
		t.Fatal("expected an error result")
	}
	if !strings.Contains(res.Content, `action of web`) || !strings.Contains(res.Content, `call "web" with action="fetch"`) {
		t.Errorf("missing action-called-as-tool hint: %q", res.Content)
	}
}

// The same hint, but for an action shared by more than one tool (e.g.
// "search" — file, web, and history all have one): naming just the
// first-registered owner would be a confident-sounding but often-wrong guess
// (reported live: it pointed a model looking for a live web API at file's
// local-content search instead). All owners should be named, not just one.
func TestDispatchActionCalledAsTopLevelToolHintAmbiguousOwner(t *testing.T) {
	file := &fakeTool{name: "file", actions: []string{"read", "search"}}
	web := &fakeTool{name: "web", actions: []string{"search", "fetch"}}
	r := NewRegistry(file, web)
	res := r.Dispatch(context.Background(), "search", "", nil)
	if !res.IsError {
		t.Fatal("expected an error result")
	}
	if !strings.Contains(res.Content, `whichever of "file"/"web" actually fits`) {
		t.Errorf("expected an ambiguous-owner hint naming both tools, got: %q", res.Content)
	}
}

// An unknown (not just empty) action string is a model's own tool-call
// convention leaking through instead of ours — the params can still clearly
// imply a real, read-only action, so Dispatch should run the inferred one
// instead of just erroring on the garbled action string.
func TestDispatchInfersActionFromGarbledAction(t *testing.T) {
	web := &fakeTool{
		name:    "web",
		actions: []string{"search", "fetch"},
		infer: func(params map[string]any) string {
			if _, ok := params["url"]; ok {
				return "fetch"
			}
			return ""
		},
	}
	r := NewRegistry(web)
	res := r.Dispatch(context.Background(), "web", "<parameter=params>", map[string]any{"url": "http://example.com"})
	if res.IsError || res.Content != "did fetch" {
		t.Errorf("res = %+v, want a successful inferred fetch", res)
	}
	if web.last != "fetch" {
		t.Errorf("tool was called with %q, want fetch", web.last)
	}
}

// Reported live: a local model repeatedly put the ENTIRE shell command into
// the top-level "action" field instead of action="shell",
// params={"command":...}. run has exactly one action ("shell") with exactly
// one required param ("command"), so there's nowhere else the garbled
// string could have meant to go — Dispatch should recover it instead of
// just erroring.
func TestDispatchRecoversGarbledActionAsSoleParamRun(t *testing.T) {
	rt := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := NewRegistry(rt).Dispatch(context.Background(), "run", "echo hi", map[string]any{})
	if r.IsError || !strings.Contains(r.Content, "hi") {
		t.Errorf("res = %+v, want a successful shell run of the garbled action as the command", r)
	}
}

// Same recovery, for calc: exactly one action ("calculate") with exactly one
// required param ("expression").
func TestDispatchRecoversGarbledActionAsSoleParamCalc(t *testing.T) {
	r := NewRegistry(NewCalc()).Dispatch(context.Background(), "calc", "2+2", map[string]any{})
	if r.IsError || r.Content != "4" {
		t.Errorf("res = %+v, want 4 (as if action=calculate, params={expression:2+2})", r)
	}
}

// A domain with exactly one action is unambiguous even with "action" missing
// entirely — reported live: a model's own "<parameter=params>...</parameter>"
// tag convention leaked into the "action" field's value, and parseArgs (see
// agent.go's recoverActionTag) correctly recovers the embedded params but,
// having no "action" key to recover from inside them, leaves action="".
// Dispatch must still resolve that to run's sole action ("shell") rather than
// erroring, since there's nowhere else it could have meant to go.
func TestDispatchResolvesEmptyActionForSingleActionDomain(t *testing.T) {
	rt := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := NewRegistry(rt).Dispatch(context.Background(), "run", "", map[string]any{"command": "echo hi"})
	if r.IsError || !strings.Contains(r.Content, "hi") {
		t.Errorf("res = %+v, want a successful shell run resolved from the sole action", r)
	}
}

// A multi-action tool (git has 9 actions) is genuinely ambiguous about which
// action a garbled string was meant for — the recovery must NOT fire, and the
// normal "unknown action" error must still surface unchanged.
func TestDispatchDoesNotRecoverGarbledActionForMultiActionTool(t *testing.T) {
	gt := gitToolFor(t, t.TempDir(), yes())
	r := NewRegistry(gt).Dispatch(context.Background(), "git", "do the thing", map[string]any{})
	if !r.IsError || !strings.Contains(r.Content, `unknown action "do the thing"`) {
		t.Errorf("res = %+v, want the unchanged unknown-action error", r)
	}
}

// Same guard as above, but for an entirely empty action: a multi-action tool
// (git has 9) must NOT guess which one was meant — still the normal
// "no action given" error listing every valid action.
func TestDispatchDoesNotResolveEmptyActionForMultiActionTool(t *testing.T) {
	gt := gitToolFor(t, t.TempDir(), yes())
	r := NewRegistry(gt).Dispatch(context.Background(), "git", "", map[string]any{})
	if !r.IsError || !strings.Contains(r.Content, "no action given") {
		t.Errorf("res = %+v, want the unchanged no-action-given error", r)
	}
}

// If params already carries a non-empty value for the sole required param,
// the model DID use params correctly — the action name is wrong for some
// other reason. Don't guess/overwrite; fall through to the normal error.
func TestDispatchDoesNotOverwriteAlreadyGivenSoleParam(t *testing.T) {
	rt := runToolFor(t, t.TempDir(), "allow", yes(), nil)
	r := NewRegistry(rt).Dispatch(context.Background(), "run", "foo", map[string]any{"command": "ls -la /tmp"})
	if !r.IsError || !strings.Contains(r.Content, `unknown action "foo"`) {
		t.Errorf("res = %+v, want the unchanged unknown-action error, not a silent overwrite", r)
	}
}

func TestOpenAIToolsSchema(t *testing.T) {
	r := NewRegistry(&fakeTool{name: "calc", actions: []string{"calculate"}})
	tools := r.OpenAITools()
	if len(tools) != 1 {
		t.Fatalf("len = %d, want 1", len(tools))
	}
	fn := tools[0]["function"].(map[string]any)
	if fn["name"] != "calc" {
		t.Errorf("name = %v, want calc", fn["name"])
	}
	props := fn["parameters"].(map[string]any)["properties"].(map[string]any)
	enum := props["action"].(map[string]any)["enum"].([]string)
	if len(enum) != 1 || enum[0] != "calculate" {
		t.Errorf("action enum = %v, want [calculate]", enum)
	}
}

func TestRegistryMutates(t *testing.T) {
	ft := &fakeTool{name: "file", actions: []string{"read", "write"}, mutates: []string{"write"}}
	r := NewRegistry(ft)
	if r.Mutates("file", "read") {
		t.Error("read should be read-only")
	}
	if !r.Mutates("file", "write") {
		t.Error("write should report as mutating")
	}
	if r.Mutates("nope", "x") {
		t.Error("unknown tool should be non-mutating")
	}
}

func TestRegistryActions(t *testing.T) {
	ft := &fakeTool{name: "file", actions: []string{"read", "write", "edit"}}
	r := NewRegistry(ft)
	if got := r.Actions("file"); len(got) != 3 {
		t.Errorf("Actions(file) = %v, want 3 entries", got)
	}
	if got := r.Actions("nope"); got != nil {
		t.Errorf("Actions(unknown tool) = %v, want nil", got)
	}
}
