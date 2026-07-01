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
		// Lead with the full action list (so a model that meant "edit" isn't nudged
		// toward the first action), then a shape example.
		return Err(fmt.Sprintf(`%s: no action given — set "action" to one of: %s. Shape: {"action":"<one of those>","params":{...}}`,
			name, strings.Join(acts, ", ")))
	}
	if !contains(t.Actions(), action) {
		if owners := otherOwners(r.actionToTool[action], name); len(owners) > 0 {
			return Err(fmt.Sprintf("action %q belongs to tool %q, not %q; call %q with that action instead",
				action, strings.Join(owners, "/"), name, owners[0]))
		}
		return Err(fmt.Sprintf("%s: unknown action %q; valid actions: %s", name, action, strings.Join(t.Actions(), ", ")))
	}
	return t.Call(ctx, action, params)
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

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
