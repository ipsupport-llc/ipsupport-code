package tool

import (
	"context"
	"strings"
	"testing"
)

func TestDomainGeneratesSchemaAndValidates(t *testing.T) {
	d := NewDomain(DomainSpec{
		Name:    "demo",
		Summary: "A demo tool.",
		NotHere: "NOT here — nothing.",
		Actions: []Action{{
			Name:   "do",
			Note:   "(does it)",
			Params: []Param{Req("a", "str"), Opt("b", "int", "5")},
			Run:    func(_ context.Context, ar Args) Result { return Ok("a=" + ar.Str("a")) },
		}},
	})

	// The help/schema is generated from the declaration — no hand-sync.
	if desc := d.Description(); !strings.Contains(desc, `- do: {"a": str, "b"?: int=5}   (does it)`) {
		t.Errorf("generated description = %q", desc)
	}
	if got := d.Actions(); len(got) != 1 || got[0] != "do" {
		t.Errorf("actions = %v", got)
	}

	// Required params are validated automatically before the handler runs.
	if r := d.Call(context.Background(), "do", map[string]any{}); !r.IsError || !strings.Contains(r.Content, "missing required param(s): a") {
		t.Errorf("missing-required not caught: %+v", r)
	}
	// The handler runs with valid args.
	if r := d.Call(context.Background(), "do", map[string]any{"a": "hi"}); r.IsError || r.Content != "a=hi" {
		t.Errorf("handler result = %+v", r)
	}
}

// A model that called an action wrongly gets a call it can copy: its own values
// for declared params, a placeholder for each required one it left out.
func TestDomainExampleIsACopyableCall(t *testing.T) {
	d := NewDomain(DomainSpec{
		Name: "file",
		Actions: []Action{
			{Name: "read", Params: []Param{Req("path", "str"), Opt("limit", "int", "200")}},
			{Name: "write", Params: []Param{Req("path", "str"), Req("content", "str"), Opt("append", "bool", "false")}},
			{Name: "list", Params: []Param{Opt("path", "str", ".")}},
		},
	})
	cases := []struct {
		action string
		given  map[string]any
		want   string
	}{
		// content left out: a placeholder; the undeclared "text" key is dropped
		{"write", map[string]any{"path": "notes.txt", "text": "hi"}, `{"action":"write","params":{"path":"notes.txt","content":"<content>"}}`},
		// an optional param the model did give is kept
		{"read", map[string]any{"limit": 5}, `{"action":"read","params":{"path":"<path>","limit":5}}`},
		// a long value is never carried truncated
		{"write", map[string]any{"content": strings.Repeat("x", 200)}, `{"action":"write","params":{"path":"<path>","content":"<the content you sent>"}}`},
		// a garbled action names the one it contains
		{"read_file", map[string]any{"path": "a.go"}, `{"action":"read","params":{"path":"a.go"}}`},
		{"WRITE", nil, `{"action":"write","params":{"path":"<path>","content":"<content>"}}`},
		// nothing to match, or more than one: no example
		{"delete", nil, ""},
		{"read_then_write", nil, ""}, // names two actions: no guess
		{"", nil, ""},
	}
	for _, c := range cases {
		if got := d.Example(c.action, c.given); got != c.want {
			t.Errorf("Example(%q, %v)\n got  %s\n want %s", c.action, c.given, got, c.want)
		}
	}
}

// Weak models emit booleans as strings/numbers; Args.Bool must read them as the
// model meant, not silently fall back to false (which e.g. dropped `git diff
// --staged`).
func TestArgsBoolCoercesWeakModelForms(t *testing.T) {
	for _, v := range []any{true, "true", "True", "yes", "1", float64(1)} {
		if !(Args{m: map[string]any{"k": v}}).Bool("k") {
			t.Errorf("Bool(%#v) = false, want true", v)
		}
	}
	for _, v := range []any{false, "false", "no", "0", float64(0), "", nil} {
		if (Args{m: map[string]any{"k": v}}).Bool("k") {
			t.Errorf("Bool(%#v) = true, want false", v)
		}
	}
}
