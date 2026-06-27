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
// history carries the conversation across Run calls (the session memory); only
// the user goals and final answers are kept, not the tool back-and-forth, so a
// small model's context isn't swamped.
type Agent struct {
	llm      llm.Chatter
	reg      *tool.Registry
	kb       *knowledge.KB
	tr       trace.Tracer
	system   string
	maxSteps int

	history    []llm.Message
	maxHistory int
	planMode   bool
}

// New builds an Agent. maxSteps <= 0 defaults to 12.
func New(l llm.Chatter, reg *tool.Registry, kb *knowledge.KB, tr trace.Tracer, system string, maxSteps int) *Agent {
	if maxSteps <= 0 {
		maxSteps = 12
	}
	if strings.TrimSpace(system) == "" {
		system = DefaultSystemPrompt()
	}
	return &Agent{llm: l, reg: reg, kb: kb, tr: tr, system: system, maxSteps: maxSteps, maxHistory: 16}
}

// Reset clears the session conversation memory.
func (a *Agent) Reset() { a.history = nil }

// SetPlanMode toggles plan mode. In plan mode the agent investigates with
// read-only tools and proposes a plan; mutating tool calls are refused, so it
// can't change anything until switched back to auto.
func (a *Agent) SetPlanMode(on bool) { a.planMode = on }

// PlanMode reports whether plan mode is on.
func (a *Agent) PlanMode() bool { return a.planMode }

// SessionLen reports how many remembered messages are in the current session.
func (a *Agent) SessionLen() int { return len(a.history) }

// History returns a copy of the session conversation (for persistence).
func (a *Agent) History() []llm.Message { return append([]llm.Message(nil), a.history...) }

// SetHistory restores a session conversation (e.g. loaded from disk).
func (a *Agent) SetHistory(h []llm.Message) { a.history = append([]llm.Message(nil), h...) }

// Compact summarizes the session so far into a short recap and replaces the
// history with it, freeing context while keeping continuity. Returns how many
// messages were compacted (0 if there was nothing worth compacting).
func (a *Agent) Compact(ctx context.Context) (int, error) {
	if len(a.history) < 2 {
		return 0, nil
	}
	var b strings.Builder
	for _, m := range a.history {
		switch m.Role {
		case "user":
			b.WriteString("User: " + m.Content + "\n")
		case "assistant":
			if strings.TrimSpace(m.Content) != "" {
				b.WriteString("Assistant: " + m.Content + "\n")
			}
		}
	}
	reply, err := a.llm.Chat(ctx, []llm.Message{
		llm.System("Summarize the conversation so far into a compact recap that preserves the key facts, decisions, files touched, and context needed to keep going. A few sentences, no preamble."),
		llm.User(b.String()),
	}, nil)
	if err != nil {
		return 0, err
	}
	n := len(a.history)
	a.history = []llm.Message{
		{Role: "user", Content: "[Summary of earlier conversation]\n" + reply.Content},
		{Role: "assistant", Content: "Got it — I have that context."},
	}
	return n, nil
}

// remember appends the goal and its final answer to the session, trimming to the
// most recent maxHistory messages.
func (a *Agent) remember(goal, final string) {
	a.history = append(a.history, llm.User(goal), llm.Message{Role: "assistant", Content: final})
	if a.maxHistory > 0 && len(a.history) > a.maxHistory {
		a.history = append([]llm.Message(nil), a.history[len(a.history)-a.maxHistory:]...)
	}
}

// DefaultSystemPrompt is the baseline instruction given to the model. Kept tight
// on purpose — it ships in every request to a small local model.
func DefaultSystemPrompt() string {
	return strings.TrimSpace(`You are the engine inside ipsupport-code, a local terminal coding agent. You run in a loop and act ONLY through tools; the user sees your tool calls and results.

- DO the task with tools: write/edit files with file, run commands with run, use git/web/calc. NEVER just tell the user how to do it ("create a file", "chmod +x", "here's how…") — describing steps instead of doing them is a failure. (e.g. "make a hello script and run it" → file.write then run.shell, then report the output.)
- If what you built is runnable, RUN it yourself with run and report the real output. Don't hand back a "how to test it" recipe — that's the user doing your job.
- Each call is {"action": <name>, "params": {...}}. On an error, read it — it names the fix or the right tool — and retry.
- Small local model in a terminal: be brief. Finish with a one-line summary of what you did — not a tutorial, and not a menu of optional features to add — and no tool call.
- After that summary, add ONE last line exactly: "NEXT: <one short next step the user might want>" (≤6 words; skip the line if nothing fits).`)
}

// planDirective is added (only in plan mode) on top of the system prompt. Kept
// short — it ships in every plan-mode request to a small local model.
const planDirective = `PLAN MODE is ON. Do NOT change anything. You may investigate with read-only tools (file.read, file.list, web, calc), then present a concise, numbered plan of what you WOULD do, and stop with no tool call. Any tool that writes files, runs commands, or changes git is blocked right now.`

// Run executes the loop until the model produces a final answer (a reply with no
// tool calls), maxSteps is reached, or the context is cancelled.
func (a *Agent) Run(ctx context.Context, goal string) (Transcript, error) {
	a.emit("goal", map[string]any{"text": goal})
	msgs := make([]llm.Message, 0, len(a.history)+3)
	msgs = append(msgs, llm.System(a.system))
	if a.planMode {
		msgs = append(msgs, llm.System(planDirective))
	}
	msgs = append(msgs, a.history...) // session memory
	msgs = append(msgs, llm.User(goal))
	tools := a.reg.OpenAITools()

	var tr Transcript
	stuck := 0
	for step := 0; step < a.maxSteps; step++ {
		tr.Steps = step + 1

		assistant, err := a.llm.Chat(ctx, msgs, tools)
		if err != nil {
			tr.Messages = msgs
			return tr, fmt.Errorf("llm chat (step %d): %w", step+1, err)
		}
		msgs = append(msgs, assistant)

		// A reply with no tool calls IS the final answer — emit only "final"
		// (emitting "assistant" too would render the same text twice).
		if len(assistant.ToolCalls) == 0 {
			clean, suggest := splitSuggestion(assistant.Content)
			tr.Final = clean
			tr.Messages = msgs
			a.emit("final", map[string]any{"text": clean, "suggest": suggest})
			a.remember(goal, clean)
			return tr, nil
		}

		// Intermediate turn: show the model's reasoning text (if any) alongside
		// the tool calls it's about to make.
		a.emit("assistant", map[string]any{"content": assistant.Content, "tool_calls": len(assistant.ToolCalls)})
		results, nErr := a.runToolCalls(ctx, assistant.ToolCalls)
		msgs = append(msgs, results...)

		// Stop a model that's stuck emitting invalid calls (e.g. empty action)
		// from burning every step and the user's time. If a whole turn's calls
		// all error, several times running, bail with a clear message.
		if nErr == len(assistant.ToolCalls) {
			if stuck++; stuck >= maxStuckTurns {
				const msg = "Stopped — the model made several invalid tool calls in a row (it isn't making progress). Try rephrasing the task, or use a stronger model."
				tr.Final = msg
				tr.Messages = msgs
				a.emit("final", map[string]any{"text": msg, "suggest": "", "exhausted": true})
				a.remember(goal, msg)
				return tr, nil
			}
		} else {
			stuck = 0
		}
	}

	tr.Messages = msgs
	clean, suggest := splitSuggestion(lastAssistantContent(msgs))
	tr.Final = clean
	a.emit("final", map[string]any{"text": clean, "suggest": suggest, "exhausted": true})
	a.remember(goal, clean)
	return tr, nil
}

// splitSuggestion peels a trailing "NEXT: <step>" line off the final answer,
// returning the answer without it plus the suggested next step ("" if none).
// Only the LAST non-empty line is considered, so a "NEXT:" appearing mid-answer
// (in a code block or a sentence) stays part of the answer and isn't mistaken
// for the suggestion.
func splitSuggestion(text string) (clean, suggestion string) {
	trimmed := strings.TrimRight(text, " \n")
	nl := strings.LastIndexByte(trimmed, '\n') // -1 when single line
	last := strings.TrimSpace(trimmed[nl+1:])
	if !strings.HasPrefix(strings.ToUpper(last), "NEXT:") {
		return text, ""
	}
	suggestion = strings.TrimSpace(strings.Trim(last[len("NEXT:"):], " \"'`"))
	// Models sometimes echo the placeholder shape "NEXT: <do the thing>"; unwrap a
	// fully-bracketed suggestion so it doesn't read as an unfilled template.
	if strings.HasPrefix(suggestion, "<") && strings.HasSuffix(suggestion, ">") {
		suggestion = strings.TrimSpace(suggestion[1 : len(suggestion)-1])
	}
	if nl < 0 {
		return "", suggestion
	}
	return strings.TrimRight(trimmed[:nl], " \n"), suggestion
}

// maxStuckTurns is how many consecutive all-error tool turns end the run early.
const maxStuckTurns = 3

// runToolCalls executes every call from one assistant turn, concurrently when
// the model batched more than one, keeping results in the emitted order. It also
// reports how many of the calls errored (for stuck-loop detection).
func (a *Agent) runToolCalls(ctx context.Context, calls []llm.ToolCall) ([]llm.Message, int) {
	out := make([]llm.Message, len(calls))
	errs := make([]bool, len(calls))
	if len(calls) == 1 {
		out[0], errs[0] = a.execOne(ctx, calls[0])
	} else {
		var wg sync.WaitGroup
		for i, c := range calls {
			wg.Add(1)
			go func(i int, c llm.ToolCall) {
				defer wg.Done()
				out[i], errs[i] = a.execOne(ctx, c)
			}(i, c)
		}
		wg.Wait()
	}
	n := 0
	for _, e := range errs {
		if e {
			n++
		}
	}
	return out, n
}

func (a *Agent) execOne(ctx context.Context, c llm.ToolCall) (llm.Message, bool) {
	action, params := parseArgs(c.Arguments)
	a.emit("tool_call", map[string]any{"tool": c.Name, "action": action, "params": params})

	// Plan mode backstop: refuse mutating calls even if the model ignores the
	// directive, so a weak model can't change anything while planning.
	if a.planMode && a.reg.Mutates(c.Name, action) {
		msg := fmt.Sprintf("plan mode is ON — %s.%s was NOT run. Don't retry it; list it as a step in your plan, then finish.", c.Name, action)
		a.emit("observation", map[string]any{"tool": c.Name, "action": action, "is_error": true, "content": msg})
		return llm.ToolResult(c.ID, c.Name, msg), true
	}

	res := a.reg.Dispatch(ctx, c.Name, action, params)
	content := res.Content
	if res.IsError {
		var extra []string
		// On a misuse error, put the tool's schema right at the error — a weak
		// model corrects far more reliably from that than from a pointer it has
		// to chase. The descriptions are kept lean, so this is cheap and only
		// fires on errors.
		if usageError(res.Content) {
			if u := strings.TrimSpace(a.reg.Usage(c.Name)); u != "" {
				extra = append(extra, c.Name+" usage:\n"+u)
			}
		}
		if hints := a.hints(c.Name, res.Content); hints != "" {
			extra = append(extra, hints)
		}
		if len(extra) > 0 {
			content = res.Content + "\n" + strings.Join(extra, "\n")
		}
	}
	a.emit("observation", map[string]any{
		"tool": c.Name, "action": action, "is_error": res.IsError, "content": content,
		"path": params["path"], // lets the UI pick a syntax lexer by extension
	})
	if res.Diff != "" {
		a.emit("diff", map[string]any{"path": params["path"], "diff": res.Diff})
	}
	return llm.ToolResult(c.ID, c.Name, content), res.IsError
}

// hints pulls matching learned pitfalls for a failed tool call. A pitfall is only
// surfaced when its error pattern actually occurs in THIS error — otherwise a
// loosely keyword-matched lesson (e.g. a "missing path" fix shown on a "no
// action" error) just misleads a weak model.
func (a *Agent) hints(domain, errText string) string {
	if a.kb == nil {
		return ""
	}
	low := strings.ToLower(errText)
	var b strings.Builder
	for _, p := range a.kb.Query(domain, errText, 3) {
		if p.ErrorPattern == "" || !strings.Contains(low, strings.ToLower(p.ErrorPattern)) {
			continue
		}
		if b.Len() == 0 {
			b.WriteString("Hints from past runs:")
		}
		fmt.Fprintf(&b, "\n- when you saw %q while %s, this worked: %s", p.ErrorPattern, p.Context, p.ProvenFix)
	}
	return b.String()
}

// usageError reports whether an error message indicates the model misused a
// tool (wrong/missing params or action) — the cases where showing the real
// schema helps it self-correct.
func usageError(s string) bool {
	l := strings.ToLower(s)
	return strings.Contains(l, "missing required") ||
		strings.Contains(l, "unknown action") ||
		strings.Contains(l, "belongs to tool") ||
		strings.Contains(l, "param")
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
