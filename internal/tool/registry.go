package tool

import (
	"context"
	"fmt"
	"strings"
)

// Registry holds the active tools and routes calls, applying cross-tool action
// correction so a misrouted action returns a hint naming the right tool.
type Registry struct {
	order        []string
	tools        map[string]Tool
	actionToTool map[string][]string // an action can be owned by >1 tool (e.g. "search")
}

// NewRegistry indexes tools by name and their actions by owning tool(s).
func NewRegistry(ts ...Tool) *Registry {
	r := &Registry{tools: map[string]Tool{}, actionToTool: map[string][]string{}}
	for _, t := range ts {
		r.order = append(r.order, t.Name())
		r.tools[t.Name()] = t
		for _, a := range t.Actions() {
			r.actionToTool[a] = append(r.actionToTool[a], t.Name())
		}
	}
	return r
}

// OpenAITools renders the catalog as OpenAI function definitions — one function
// per tool, with the action enum and a freeform params object. Kept tiny on
// purpose so small models prefill it fast.
func (r *Registry) OpenAITools() []map[string]any {
	out := make([]map[string]any, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name(),
				"description": t.Description(),
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"action": map[string]any{
							"type": "string",
							"enum": t.Actions(),
						},
						"params": map[string]any{"type": "object"},
					},
					"required": []string{"action"},
				},
			},
		})
	}
	return out
}

// Dispatch routes (name, action) to its tool. Unknown tool, or an action owned
// by a different tool, returns a self-correcting Result the model can act on.
func (r *Registry) Dispatch(ctx context.Context, name, action string, params map[string]any) Result {
	t, ok := r.tools[name]
	if !ok {
		// A model sometimes calls an action as if it were its own tool (e.g. "fetch"
		// instead of "web" with action="fetch") — actionToTool already indexes every
		// action's real owner(s) for the sibling hint below, so reuse it here too.
		if owners := r.actionToTool[name]; len(owners) > 0 {
			return Err(fmt.Sprintf("no tool named %q — %q is an action of %s, not its own tool; call %s with action=%q instead",
				name, name, strings.Join(owners, "/"), toolChoice(owners), name))
		}
		return Err(fmt.Sprintf("unknown tool %q; available tools: %s", name, strings.Join(r.order, ", ")))
	}
	// Empty action is a common small-model slip — it often means the model didn't
	// format the call. Show the exact JSON shape with a concrete action (a weak
	// model copies the example), not the whole schema dump.
	if action == "" {
		acts := t.Actions()
		if len(acts) == 0 {
			return Err(name + ": this tool has no actions")
		}
		// Some models violate the required+enum schema and omit "action". If the
		// params clearly imply a READ-ONLY action, run it — a wrong guess can't
		// mutate and self-corrects. Otherwise show the shape and the action list.
		if a := inferAction(t, acts, params); a != "" {
			return t.Call(ctx, a, params)
		}
		// A domain with exactly one action and exactly one required param (run,
		// calc, help) is unambiguous even with "action" missing entirely — but
		// only once that required param actually has a value. An empty action
		// AND empty params means nothing useful was given at all, and the terse
		// "no action given" message below (not a verbose required-param error)
		// is the more useful nudge — same as it's always been.
		if d, isDomain := t.(*Domain); isDomain {
			if a, p, ok := d.soleRequiredParam(); ok && !isEmpty(params[p]) {
				return t.Call(ctx, a, params)
			}
		}
		// Lead with the full action list (so a model that meant "edit" isn't nudged
		// toward the first action), then a shape example.
		return Err(fmt.Sprintf(`%s: no action given — set "action" to one of: %s. Shape: {"action":"<one of those>","params":{...}}`,
			name, strings.Join(acts, ", ")))
	}
	if !contains(t.Actions(), action) {
		if owners := otherOwners(r.actionToTool[action], name); len(owners) > 0 {
			return Err(fmt.Sprintf("action %q belongs to tool %q, not %q; call %s with that action instead",
				action, strings.Join(owners, "/"), name, toolChoice(owners)))
		}
		// The action string itself may be garbled rather than just missing (e.g.
		// a model's own tool-call convention leaking through instead of ours) —
		// params can still clearly imply a real action; same inference as above.
		if a := inferAction(t, t.Actions(), params); a != "" {
			return t.Call(ctx, a, params)
		}
		// The domain may have exactly one action with exactly one required
		// param — then there's nowhere else the garbled action string could
		// have meant to go but that param (see Domain.soleRequiredParam).
		if fixedAction, fixedParams, ok := garbledActionAsParam(t, action, params); ok {
			return t.Call(ctx, fixedAction, fixedParams)
		}
		return Err(fmt.Sprintf("%s: unknown action %q; valid actions: %s", name, action, strings.Join(t.Actions(), ", ")))
	}
	return t.Call(ctx, action, params)
}

// inferAction asks t (if it supports inference) to guess a read-only action
// from params, returning "" if it can't or doesn't support it.
func inferAction(t Tool, acts []string, params map[string]any) string {
	inf, ok := t.(interface{ InferAction(map[string]any) string })
	if !ok {
		return ""
	}
	if a := inf.InferAction(params); a != "" && contains(acts, a) {
		return a
	}
	return ""
}

// garbledActionAsParam recovers a call where the model dumped its intended
// value directly into "action" instead of params (see Domain.
// soleRequiredParam's doc). Safe specifically because it only fires for a
// domain with exactly one action and exactly one required param — there's
// nowhere else the garbled string could have meant to go — and dispatch
// still goes through the tool's own normal approval/policy gate afterward
// via the ordinary t.Call path: this only fixes ROUTING, never bypasses
// authorization.
func garbledActionAsParam(t Tool, action string, params map[string]any) (fixedAction string, fixedParams map[string]any, ok bool) {
	d, isDomain := t.(*Domain)
	if !isDomain {
		return "", nil, false
	}
	soleAction, soleParam, hasOne := d.soleRequiredParam()
	if !hasOne || !isEmpty(params[soleParam]) {
		return "", nil, false // the param was already given explicitly — don't clobber a real value
	}
	out := make(map[string]any, len(params)+1)
	for k, v := range params {
		out[k] = v
	}
	out[soleParam] = action
	return soleAction, out, true
}

// toolChoice phrases which tool to call for a hint. An action owned by
// exactly one tool names it with confidence; an action shared by several
// (e.g. "search" — file, web, and history all have one) doesn't pick the
// first alphabetically/registration-order owner as if it were obviously
// right — reported live: that guess sent a model looking for a live web API
// to file's local-content search instead. Naming all of them and asking the
// model to pick lets it use context we don't have, instead of a confident
// but often-wrong single suggestion.
func toolChoice(owners []string) string {
	if len(owners) == 1 {
		return fmt.Sprintf("%q", owners[0])
	}
	quoted := make([]string, len(owners))
	for i, o := range owners {
		quoted[i] = fmt.Sprintf("%q", o)
	}
	return "whichever of " + strings.Join(quoted, "/") + " actually fits"
}

// otherOwners returns the action's owner tools excluding self.
func otherOwners(owners []string, self string) []string {
	var out []string
	for _, o := range owners {
		if o != self {
			out = append(out, o)
		}
	}
	return out
}

// Usage returns a tool's self-describing contract (its Description), or "" if no
// such tool — used to lead the model back to correct usage after an error.
func (r *Registry) Usage(name string) string {
	if t, ok := r.tools[name]; ok {
		return t.Description()
	}
	return ""
}

// Mutates reports whether (name, action) changes state — the gate plan mode uses.
func (r *Registry) Mutates(name, action string) bool {
	if t, ok := r.tools[name]; ok {
		return t.Mutates(action)
	}
	return false
}

// Actions returns a tool's action names, or nil if no such tool — used to
// scope a past-run hint to the action it was actually learned for.
func (r *Registry) Actions(name string) []string {
	if t, ok := r.tools[name]; ok {
		return t.Actions()
	}
	return nil
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
