// Package agent runs the reason → act → observe loop against an llm.Chatter,
// dispatching fat tools, injecting learned pitfalls into tool errors, and
// tracing every step.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
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

// SetSystem swaps the base system prompt (e.g. after learning new project facts),
// so the next run uses it without a full re-wire.
func (a *Agent) SetSystem(s string) { a.system = s }

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

- You CAN edit files. The file tool's write/edit/append actions modify REAL files on disk in this workspace, and the user has already authorized you to use them. NEVER claim you "only have read/run", "can't modify files", or that the user must "enable file editing" / open a different mode — that is false. To change code, just call file (action edit, or write); for a multi-file change, do one file at a time.
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
	slog.Debug("run start", "goal", clip(goal, 120), "tools", toolNames(tools), "plan_mode", a.planMode)

	var tr Transcript
	stuck, nudged := 0, false
	acted := false // did the model call any tool this run?
	for step := 0; step < a.maxSteps; step++ {
		tr.Steps = step + 1

		assistant, err := a.llm.Chat(ctx, msgs, tools)
		if err != nil {
			tr.Messages = msgs
			return tr, fmt.Errorf("llm chat (step %d): %w", step+1, err)
		}
		// At IPS_LOG=debug this shows exactly what the model returned each turn —
		// the actual tool calls, or text with NO tool calls (e.g. a chat model
		// refusing to edit instead of calling file.edit).
		slog.Debug("model turn", "step", step+1,
			"tool_calls", toolCallNames(assistant.ToolCalls),
			"content", clip(strings.TrimSpace(assistant.Content), 240))
		msgs = append(msgs, assistant)

		// A reply with no tool calls IS the final answer — emit only "final"
		// (emitting "assistant" too would render the same text twice).
		if len(assistant.ToolCalls) == 0 {
			clean, suggest := splitSuggestion(assistant.Content)
			if strings.TrimSpace(clean) == "" {
				// Blank final turn. If the model already did work via tools, say it
				// finished (the changes/output are above); otherwise it produced
				// nothing — flag that instead of ending on silence.
				if acted {
					clean = "(done — finished without a written summary; see the changes/output above.)"
				} else {
					clean = "(the model returned an empty reply — no answer and no tool call. Try rephrasing, or pick a stronger model with /model.)"
				}
				suggest = ""
			}
			tr.Final = clean
			tr.Messages = msgs
			a.emit("final", map[string]any{"text": clean, "suggest": suggest})
			a.remember(goal, clean)
			return tr, nil
		}

		// Intermediate turn: show the model's reasoning text (if any) alongside
		// the tool calls it's about to make.
		a.emit("assistant", map[string]any{"content": assistant.Content, "tool_calls": len(assistant.ToolCalls)})
		acted = true
		results, nErr := a.runToolCalls(ctx, assistant.ToolCalls)
		msgs = append(msgs, results...)

		// A model stuck repeating failing calls (e.g. empty action) burns steps and
		// the user's time. After several all-error turns, give it ONE forceful
		// "this isn't working — rethink" nudge with an escape hatch to answer in
		// words. Only if it's STILL stuck after that do we stop, handing the user a
		// steering suggestion they can send in one tap.
		if nErr == len(assistant.ToolCalls) {
			if stuck++; stuck >= maxStuckTurns {
				if !nudged {
					msgs = append(msgs, llm.User(stuckNudge))
					a.emit("nudge", map[string]any{})
					stuck, nudged = 0, true
				} else {
					const msg = "Stopped — it kept making invalid tool calls even after a nudge to rethink. Steer it (a different approach), or use a stronger model."
					tr.Final = msg
					tr.Messages = msgs
					a.emit("final", map[string]any{"text": msg, "suggest": stuckSuggest, "exhausted": true})
					a.remember(goal, msg)
					return tr, nil
				}
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

// clip shortens s for a debug log line (rune-safe).
func clip(s string, n int) string { out, _ := textutil.Clip(s, n); return out }

// toolNames extracts the function names from the OpenAI tool catalog (debug).
func toolNames(tools []map[string]any) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		if fn, ok := t["function"].(map[string]any); ok {
			if n, ok := fn["name"].(string); ok {
				names = append(names, n)
			}
		}
	}
	return names
}

// toolCallNames lists the action names the model called this turn (debug). nil
// when the model returned no tool calls — the tell for a chat-only reply.
func toolCallNames(calls []llm.ToolCall) []string {
	if len(calls) == 0 {
		return nil
	}
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names
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

// maxStuckTurns is how many consecutive all-error tool turns trigger the rethink
// nudge (and, if it doesn't help, the stop).
const maxStuckTurns = 3

// stuckNudge is the one self-correction injected before giving up — with an
// escape hatch so the model answers in words instead of looping.
const stuckNudge = `Those tool calls keep failing the same way — stop repeating them. Re-read the last error: it says exactly what is wrong (the call format, or a missing param). Either fix the call, or take a completely different approach to the goal. If you genuinely cannot proceed, reply in ONE sentence explaining what is blocking you, and do NOT call a tool.`

// stuckSuggest is offered to the user (one tap) when even the nudge didn't help.
const stuckSuggest = "take a different approach — outline the steps first"

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
	// An empty-action error already carries a concrete example; piling the full
	// schema and learned hints on top just buries it for a weak model — keep it to
	// the one clean line.
	if res.IsError && !strings.Contains(res.Content, "no action given") {
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
	return strings.Contains(l, "missing required") || // covers "missing required param(s)"
		strings.Contains(l, "unknown action") ||
		strings.Contains(l, "belongs to tool")
}

func (a *Agent) emit(kind string, fields map[string]any) {
	if a.tr != nil {
		a.tr.Emit(kind, fields)
	}
}

// parseArgs decodes a tool-call argument string into (action, params), tolerating
// the three shapes weak models emit: params as a nested object, params as a
// JSON-encoded STRING (double-encoded — a very common mistake), or the per-action
// fields flattened at the top level. Action may live at the top level or inside
// a stringified params blob.
func parseArgs(raw string) (string, map[string]any) {
	m := decodeObj(raw)
	if m == nil {
		return "", map[string]any{}
	}
	action, _ := m["action"].(string)

	switch p := m["params"].(type) {
	case map[string]any:
		if a, ok := p["action"].(string); ok && action == "" {
			action = a
		}
		delete(p, "action")
		return action, p
	case string: // params double-encoded as a JSON string — decode it
		if inner := decodeObj(p); inner != nil {
			if a, ok := inner["action"].(string); ok && action == "" {
				action = a
			}
			delete(inner, "action")
			return action, inner
		}
	}

	// Flattened: everything except action/params is a param.
	params := map[string]any{}
	for k, v := range m {
		if k != "action" && k != "params" {
			params[k] = v
		}
	}
	return action, params
}

// decodeObj unmarshals s into a JSON object, or nil if it isn't one.
func decodeObj(s string) map[string]any {
	var m map[string]any
	if json.Unmarshal([]byte(strings.TrimSpace(s)), &m) != nil {
		return nil
	}
	return m
}

func lastAssistantContent(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" && strings.TrimSpace(msgs[i].Content) != "" {
			return msgs[i].Content
		}
	}
	return ""
}
