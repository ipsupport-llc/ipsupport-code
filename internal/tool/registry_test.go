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
	if !strings.Contains(res.Content, `action of "web"`) || !strings.Contains(res.Content, `action="fetch"`) {
		t.Errorf("missing action-called-as-tool hint: %q", res.Content)
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
