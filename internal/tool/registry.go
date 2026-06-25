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
	actionToTool map[string]string
}

// NewRegistry indexes tools by name and their actions by owning tool. Action
// names are expected to be globally unique across tools.
func NewRegistry(ts ...Tool) *Registry {
	r := &Registry{tools: map[string]Tool{}, actionToTool: map[string]string{}}
	for _, t := range ts {
		r.order = append(r.order, t.Name())
		r.tools[t.Name()] = t
		for _, a := range t.Actions() {
			r.actionToTool[a] = t.Name()
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
	if !contains(t.Actions(), action) {
		if owner, ok := r.actionToTool[action]; ok && owner != name {
			return Err(fmt.Sprintf("action %q belongs to tool %q, not %q; call %q with that action instead", action, owner, name, owner))
		}
		return Err(fmt.Sprintf("%s: unknown action %q; valid actions: %s", name, action, strings.Join(t.Actions(), ", ")))
	}
	return t.Call(ctx, action, params)
}

// Usage returns a tool's self-describing contract (its Description), or "" if no
// such tool — used to lead the model back to correct usage after an error.
func (r *Registry) Usage(name string) string {
	if t, ok := r.tools[name]; ok {
		return t.Description()
	}
	return ""
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
