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
