package tool

import (
	"context"
	"strings"
)

// Param declares one action parameter. It is the single source for BOTH the
// schema shown to the model and the required-arg validation, so help and
// behaviour can never drift apart.
type Param struct {
	Name     string
	Type     string // str | int | bool | number | list
	Required bool
	Default  string // shown in the generated help when set (e.g. "." or "15")
}

// Req and Opt are concise constructors for declaring params.
func Req(name, typ string) Param      { return Param{Name: name, Type: typ, Required: true} }
func Opt(name, typ, def string) Param { return Param{Name: name, Type: typ, Default: def} }

// Args is a typed accessor over the (already validated) params handed to a
// handler.
type Args struct{ m map[string]any }

func (a Args) Str(k string) string       { return Str(a.m, k) }
func (a Args) Int(k string, def int) int { return Int(a.m, k, def) }
func (a Args) Bool(k string) bool        { return Bool(a.m, k) }
func (a Args) Has(k string) bool         { _, ok := a.m[k]; return ok }

// Action is one operation: its declared params and its handler. The handler runs
// only after the required params have been validated.
type Action struct {
	Name   string
	Note   string // optional one-line note appended to the generated help
	Params []Param
	Run    func(ctx context.Context, a Args) Result
}

// DomainSpec declares a fat tool: a name, a one-line summary, optional extra
// details, a "NOT here" routing note, and its actions.
type DomainSpec struct {
	Name    string
	Summary string
	Details string // optional extra lines after the action list
	NotHere string
	Actions []Action
}

// Domain is the universal tool object (MCP-style): standard methods —
// schema/help (Description), action list, and dispatch (Call) — generated from a
// declarative spec. Adding a tool or action is data, not a switch statement, and
// the schema can't fall out of sync with the handlers.
type Domain struct {
	spec  DomainSpec
	index map[string]*Action
}

// NewDomain builds a Domain from its spec. *Domain implements Tool.
func NewDomain(spec DomainSpec) *Domain {
	d := &Domain{spec: spec, index: make(map[string]*Action, len(spec.Actions))}
	for i := range d.spec.Actions {
		d.index[d.spec.Actions[i].Name] = &d.spec.Actions[i]
	}
	return d
}

func (d *Domain) Name() string { return d.spec.Name }

func (d *Domain) Actions() []string {
	out := make([]string, len(d.spec.Actions))
	for i, a := range d.spec.Actions {
		out[i] = a.Name
	}
	return out
}

// Description is generated from the action declarations.
func (d *Domain) Description() string {
	var b strings.Builder
	b.WriteString(d.spec.Summary)
	b.WriteString("\nActions:\n")
	for _, a := range d.spec.Actions {
		b.WriteString("  - " + a.Name + ": " + renderParams(a.Params))
		if a.Note != "" {
			b.WriteString("   " + a.Note)
		}
		b.WriteByte('\n')
	}
	if d.spec.Details != "" {
		b.WriteString(d.spec.Details + "\n")
	}
	if d.spec.NotHere != "" {
		b.WriteString(d.spec.NotHere)
	}
	return strings.TrimRight(b.String(), "\n")
}

// Call validates the action's required params (from its declaration) and then
// runs the handler. The registry already guarantees the action exists.
func (d *Domain) Call(ctx context.Context, action string, params map[string]any) Result {
	a, ok := d.index[action]
	if !ok {
		return Err(d.spec.Name + ": unknown action " + action)
	}
	var missing []string
	for _, p := range a.Params {
		if p.Required {
			if v, ok := params[p.Name]; !ok || isEmpty(v) {
				missing = append(missing, p.Name)
			}
		}
	}
	if len(missing) > 0 {
		return Err("missing required param(s): " + strings.Join(missing, ", "))
	}
	return a.Run(ctx, Args{m: params})
}

// renderParams turns the declared params into the contract shown to the model,
// e.g. {"path": str, "content": str} or {"path"?: str="."}.
func renderParams(ps []Param) string {
	if len(ps) == 0 {
		return "{}"
	}
	parts := make([]string, len(ps))
	for i, p := range ps {
		key := `"` + p.Name + `"`
		if !p.Required {
			key += "?"
		}
		s := key + ": " + p.Type
		if p.Default != "" {
			s += "=" + p.Default
		}
		parts[i] = s
	}
	return "{" + strings.Join(parts, ", ") + "}"
}
