package tool

import (
	"context"
	"strings"
	"testing"
)

type fakeTool struct {
	name    string
	actions []string
	last    string
}

func (f *fakeTool) Name() string        { return f.name }
func (f *fakeTool) Description() string  { return f.name + " tool" }
func (f *fakeTool) Actions() []string    { return f.actions }
func (f *fakeTool) Call(_ context.Context, action string, _ map[string]any) Result {
	f.last = action
	return Ok("did " + action)
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

func TestRequire(t *testing.T) {
	if err := Require(map[string]any{"a": "x"}, "a", "b"); err == nil || !strings.Contains(err.Error(), "b") {
		t.Errorf("err = %v, want to mention missing 'b'", err)
	}
	if err := Require(map[string]any{"a": "x"}, "a"); err != nil {
		t.Errorf("present param should pass, got %v", err)
	}
}
