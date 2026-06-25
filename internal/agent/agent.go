// Package agent runs the reason → act → observe loop against an llm.Chatter,
// dispatching fat tools, injecting learned pitfalls into tool errors, and
// tracing every step.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
	"github.com/ipsupport-llc/ipsupport-code/internal/trace"
)

// Transcript is the full record of one task run.
type Transcript struct {
	Messages []llm.Message
	Final    string
	Steps    int
}

// Agent holds the wiring for a run. The knowledge base and tracer may be nil.
type Agent struct {
	llm      llm.Chatter
	reg      *tool.Registry
	kb       *knowledge.KB
	tr       trace.Tracer
	system   string
	maxSteps int
}

// New builds an Agent. maxSteps <= 0 defaults to 12.
func New(l llm.Chatter, reg *tool.Registry, kb *knowledge.KB, tr trace.Tracer, system string, maxSteps int) *Agent {
	if maxSteps <= 0 {
		maxSteps = 12
	}
	if strings.TrimSpace(system) == "" {
		system = DefaultSystemPrompt()
	}
	return &Agent{llm: l, reg: reg, kb: kb, tr: tr, system: system, maxSteps: maxSteps}
}

// DefaultSystemPrompt is the baseline instruction given to the model.
func DefaultSystemPrompt() string {
	return strings.TrimSpace(`You are ipsupport-code, a local command-line agent that completes small, concrete tasks by calling tools.
Rules:
- Prefer tools over guessing. Use calc for ANY arithmetic.
- Each tool takes {"action": <name>, "params": {...}} — put per-action fields inside params.
- When a tool returns an error, read it carefully: it often names the fix or the correct tool. Then retry.
- When the task is done, reply with a short final answer and make NO tool call.`)
}

// Run executes the loop until the model produces a final answer (a reply with no
// tool calls), maxSteps is reached, or the context is cancelled.
func (a *Agent) Run(ctx context.Context, goal string) (Transcript, error) {
	a.emit("goal", map[string]any{"text": goal})
	msgs := []llm.Message{llm.System(a.system), llm.User(goal)}
	tools := a.reg.OpenAITools()

	var tr Transcript
	for step := 0; step < a.maxSteps; step++ {
		tr.Steps = step + 1

		assistant, err := a.llm.Chat(ctx, msgs, tools)
		if err != nil {
			tr.Messages = msgs
			return tr, fmt.Errorf("llm chat (step %d): %w", step+1, err)
		}
		msgs = append(msgs, assistant)
		a.emit("assistant", map[string]any{"content": assistant.Content, "tool_calls": len(assistant.ToolCalls)})

		if len(assistant.ToolCalls) == 0 {
			tr.Final = assistant.Content
			tr.Messages = msgs
			a.emit("final", map[string]any{"text": tr.Final})
			return tr, nil
		}

		msgs = append(msgs, a.runToolCalls(ctx, assistant.ToolCalls)...)
	}

	tr.Messages = msgs
	tr.Final = lastAssistantContent(msgs)
	a.emit("final", map[string]any{"text": tr.Final, "exhausted": true})
	return tr, nil
}

// runToolCalls executes every call from one assistant turn, concurrently when
// the model batched more than one, keeping results in the emitted order.
func (a *Agent) runToolCalls(ctx context.Context, calls []llm.ToolCall) []llm.Message {
	out := make([]llm.Message, len(calls))
	if len(calls) == 1 {
		out[0] = a.execOne(ctx, calls[0])
		return out
	}
	var wg sync.WaitGroup
	for i, c := range calls {
		wg.Add(1)
		go func(i int, c llm.ToolCall) {
			defer wg.Done()
			out[i] = a.execOne(ctx, c)
		}(i, c)
	}
	wg.Wait()
	return out
}

func (a *Agent) execOne(ctx context.Context, c llm.ToolCall) llm.Message {
	action, params := parseArgs(c.Arguments)
	a.emit("tool_call", map[string]any{"tool": c.Name, "action": action, "params": params})

	res := a.reg.Dispatch(ctx, c.Name, action, params)
	content := res.Content
	if res.IsError {
		if hints := a.hints(c.Name, res.Content); hints != "" {
			content = res.Content + "\n" + hints
		}
	}
	a.emit("observation", map[string]any{
		"tool": c.Name, "action": action, "is_error": res.IsError, "content": content,
	})
	return llm.ToolResult(c.ID, c.Name, content)
}

// hints pulls matching learned pitfalls for a failed tool call.
func (a *Agent) hints(domain, errText string) string {
	if a.kb == nil {
		return ""
	}
	ps := a.kb.Query(domain, errText, 3)
	if len(ps) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Hints from past runs:")
	for _, p := range ps {
		fmt.Fprintf(&b, "\n- when you saw %q while %s, this worked: %s", p.ErrorPattern, p.Context, p.ProvenFix)
	}
	return b.String()
}

func (a *Agent) emit(kind string, fields map[string]any) {
	if a.tr != nil {
		a.tr.Emit(kind, fields)
	}
}

// parseArgs decodes a tool-call argument string into (action, params). It
// tolerates models that put per-action fields at the top level instead of inside
// "params".
func parseArgs(raw string) (string, map[string]any) {
	var m map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &m); err != nil || m == nil {
		return "", map[string]any{}
	}
	action, _ := m["action"].(string)
	if p, ok := m["params"].(map[string]any); ok {
		return action, p
	}
	params := map[string]any{}
	for k, v := range m {
		if k != "action" {
			params[k] = v
		}
	}
	return action, params
}

func lastAssistantContent(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" && strings.TrimSpace(msgs[i].Content) != "" {
			return msgs[i].Content
		}
	}
	return ""
}
