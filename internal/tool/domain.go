package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
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
	Name    string
	Note    string // optional one-line note appended to the generated help
	Mutates bool   // changes state (write/run/git) — blocked in plan mode
	Params  []Param
	Run     func(ctx context.Context, a Args) Result
}

// DomainSpec declares a fat tool: a name, a one-line summary, optional extra
// details, a "NOT here" routing note, and its actions.
type DomainSpec struct {
	Name    string
	Summary string
	Details string // optional extra lines after the action list
	NotHere string
	Actions []Action
	// Infer recovers a likely action from the params when a model omits "action"
	// (some models violate the required+enum schema). It MUST only ever return a
	// read-only action — never guess a mutation — so a wrong guess is harmless and
	// self-correcting. "" = can't tell, surface the normal "no action" error.
	Infer func(params map[string]any) string
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

// InferAction guesses a read-only action from the params when the model omitted
// "action". Returns "" when the spec has no inference or nothing fits.
func (d *Domain) InferAction(params map[string]any) string {
	if d.spec.Infer == nil {
		return ""
	}
	return d.spec.Infer(params)
}

// Mutates reports whether the named action is declared as state-changing.
func (d *Domain) Mutates(action string) bool {
	if a, ok := d.index[action]; ok {
		return a.Mutates
	}
	// A missing or garbled action is not "no mutation": Dispatch infers the
	// action from the params and runs it (run with only "command" runs shell).
	// So an action this domain doesn't recognize mutates if any of its actions
	// can — plan mode's gate and the sequential-batch check both rely on this.
	for _, a := range d.spec.Actions {
		if a.Mutates {
			return true
		}
	}
	return false
}

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
	var missing, required []string
	for _, p := range a.Params {
		if p.Required {
			required = append(required, p.Name)
			if v, ok := params[p.Name]; !ok || isEmpty(v) {
				missing = append(missing, p.Name)
			}
		}
	}
	if len(missing) > 0 {
		given := make([]string, 0, len(params))
		for k := range params {
			given = append(given, k)
		}
		sort.Strings(given)
		// Point at exactly what's wrong: which are missing, what the action needs,
		// and what was actually passed — a weak model self-corrects from this fast.
		return Err(fmt.Sprintf("missing required param(s): %s — %s.%s needs {%s}; you gave {%s}",
			strings.Join(missing, ", "), d.spec.Name, a.Name,
			strings.Join(required, ", "), strings.Join(given, ", ")))
	}
	return a.Run(ctx, Args{m: params})
}

// Example is a ready-to-copy arguments object for one of the domain's actions,
// for a model that just called it wrongly: a weak model copies an example far
// more reliably than it assembles a call from a typed schema. It is built from
// what the model sent — its values for params the action declares, a
// placeholder for each required one it left out, anything undeclared dropped.
// An action name that isn't the domain's own is matched to one it contains or
// is contained in ("read_file" → read); "" when nothing matches.
//
// "" too when the example would correct nothing — every required param given
// and the action already right. Then the call failed for some other reason (a
// remote MCP tool's own validation, say), and labelling the same call "correct"
// would only send the model back to it.
func (d *Domain) Example(action string, given map[string]any) string {
	return d.example(action, given, false)
}

// example is Example; corrected says the call is already being corrected
// elsewhere (sent to the wrong tool), so even a complete one is worth showing.
func (d *Domain) example(action string, given map[string]any, corrected bool) string {
	a, ok := d.index[action]
	corrected = corrected || !ok
	if !ok {
		a, ok = d.closestAction(action)
		if !ok {
			return ""
		}
	}
	var b strings.Builder
	b.WriteString(`{"action":` + jsonString(a.Name) + `,"params":{`)
	first := true
	for _, p := range a.Params {
		v, has := given[p.Name]
		var val string
		switch {
		case p.Required && (!has || isEmpty(v)):
			// Validation counts a blank required value as missing.
			val, corrected = placeholder(p), true
		case has && v != nil:
			// An optional value is kept as sent, a lone space included: replace=" "
			// dropped from the example would turn a replace into a delete.
			val = exampleValue(p, v)
		default:
			continue
		}
		if !first {
			b.WriteByte(',')
		}
		first = false
		b.WriteString(jsonString(p.Name) + ":" + val)
	}
	b.WriteString("}}")
	if !corrected {
		return ""
	}
	return b.String()
}

// declaresAny reports whether action declares any of the params given.
func (d *Domain) declaresAny(action string, given map[string]any) bool {
	a, ok := d.index[action]
	if !ok {
		return false
	}
	for _, p := range a.Params {
		if _, has := given[p.Name]; has {
			return true
		}
	}
	return false
}

// closestAction finds the one action named by a word of the given name
// ("read_file", "readFile", "file-read" → read), ignoring case. Whole words
// only: "catalog" is not log, "check" is not checkout. Ambiguous or none: false.
// Never a mutating one:
// "list_branches" must not come back as git's branch, which creates a branch —
// a guess may only point at something that changes nothing.
func (d *Domain) closestAction(given string) (*Action, bool) {
	words := nameWords(given)
	if len(words) == 0 {
		return nil, false
	}
	var found []*Action
	for i := range d.spec.Actions {
		n := strings.ToLower(d.spec.Actions[i].Name)
		if d.spec.Actions[i].Mutates {
			continue
		}
		if words[n] || strings.EqualFold(strings.TrimSpace(given), n) {
			found = append(found, &d.spec.Actions[i])
		}
	}
	if len(found) != 1 {
		return nil, false
	}
	return found[0], true
}

// nameWords splits an identifier into lower-case words: on anything not a letter
// or digit, and at a lower-to-upper case change ("readFile" → read, file).
func nameWords(s string) map[string]bool {
	words := map[string]bool{}
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			words[strings.ToLower(string(cur))] = true
			cur = cur[:0]
		}
	}
	var prev rune
	for _, r := range s {
		switch {
		case !unicode.IsLetter(r) && !unicode.IsDigit(r):
			flush()
		case unicode.IsUpper(r) && unicode.IsLower(prev):
			flush()
			cur = append(cur, r)
		default:
			cur = append(cur, r)
		}
		prev = r
	}
	flush()
	return words
}

// maxExampleValue caps one value's JSON in an example.
const maxExampleValue = 120

// exampleValue is the model's own value for p, as JSON. A long one — at any
// depth, a list of edits included — is shown as a placeholder instead: a copied
// example must never carry a truncated copy of, say, a file's content, nor
// repeat a large one into the prompt.
//
// The placeholder is the ordinary one, "<content>": a weak model copying the
// example must see a slot to fill, not a sentence it could send as the content.
// A value of the wrong JSON type for its param is a slot too — it is not part of
// a correct call.
func exampleValue(p Param, v any) string {
	out, err := marshalPlain(v)
	if err != nil || len(out) > maxExampleValue || !typeMatches(p.Type, v) {
		return placeholder(p)
	}
	return out
}

// typeMatches reports whether a decoded JSON value fits a declared param type.
func typeMatches(typ string, v any) bool {
	switch v.(type) {
	case string:
		return typ == "str" || typ == ""
	case float64, json.Number, int, int64:
		return typ == "int" || typ == "number"
	case bool:
		return typ == "bool"
	case []any:
		return typ == "list"
	}
	return false
}

// placeholder stands in for a value the model has to supply. Always a marked
// slot, whatever the type: a bare 1 or true would be copied as a real value.
func placeholder(p Param) string {
	if p.Type == "list" {
		return "[" + jsonString("<"+p.Name+">") + "]"
	}
	return jsonString("<" + p.Name + ">")
}

func jsonString(s string) string {
	out, _ := marshalPlain(s)
	return out
}

// marshalPlain is json.Marshal without HTML escaping: the example is read by a
// model, and "\u003cpath\u003e" is not "<path>" to it.
func marshalPlain(v any) (string, error) {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

// soleRequiredParam returns the sole action's name and its ONE required
// param's name, if this domain has EXACTLY one action with EXACTLY one
// required param — the case where a garbled top-level "action" string (a
// model dumping its intended value there instead of into params, e.g.
// run's action="ls -la /tmp" instead of action="shell",
// params={"command":"ls -la /tmp"}) can be recovered UNAMBIGUOUSLY: there's
// nowhere else the garbled string could have meant to go.
func (d *Domain) soleRequiredParam() (action, param string, ok bool) {
	if len(d.spec.Actions) != 1 {
		return "", "", false
	}
	a := d.spec.Actions[0]
	var required []Param
	for _, p := range a.Params {
		if p.Required {
			required = append(required, p)
		}
	}
	if len(required) != 1 {
		return "", "", false
	}
	return a.Name, required[0].Name, true
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
