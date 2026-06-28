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
	Messages  []llm.Message
	Final     string
	Steps     int
	Cancelled bool // the user cancelled (esc) specifically
	Stopped   bool // ended before a clean answer (cancel / runaway / stuck / maxSteps / mid-run error) → no reflection
	Returns   int  // goal re-feeds the judge triggered this run
	GoalMet   bool // a goal-loop ran and the judge confirmed the goal was met
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
	label      string // non-empty for a sub-agent; tags its events so the UI can group them

	// Goal pursuit: when the model finalizes but a judge (judgeGoal) decides the
	// goal isn't met, it's re-fed the goal and continues, up to maxReturns (a TTL).
	maxReturns int
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

// SetGoalLoop configures goal pursuit: when the model finalizes, a judge decides
// whether the goal is met; if not, re-feed the goal and keep going, up to
// maxReturns times (a TTL). 0 disables it — one run, the model's finish stands.
func (a *Agent) SetGoalLoop(maxReturns int) {
	a.maxReturns = maxReturns
}

// SetLabel tags this agent as a sub-agent: the label is attached to every event
// it emits (as the "agent" field) so the UI can group a sub-agent's progress on
// its own status line during a parallel fan-out.
func (a *Agent) SetLabel(s string) { a.label = s }

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
// stopNote describes a run that stopped before a clean answer, so the next turn
// ("continue") has context — the edits/output are already on disk/screen. Covers
// the user's cancel and any mid-run failure (runaway cap, transport error).
func stopNote(msgs []llm.Message, cancelled bool, err error) string {
	var did []string
	for _, m := range msgs {
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				did = append(did, tc.Name)
			}
		}
	}
	reason := "cancelled by you mid-task"
	if !cancelled && err != nil {
		reason = "stopped early — " + clip(err.Error(), 140)
	}
	if len(did) == 0 {
		return "(" + reason + ".)"
	}
	if len(did) > 8 {
		did = did[len(did)-8:]
	}
	return "(" + reason + ". Work so far used: " + strings.Join(did, ", ") +
		". Those changes/outputs are kept — say 'continue' or what to do next.)"
}

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

// SubAgentSystemPrompt is the baseline for a sub-agent — a fresh LLM session the
// main assistant spawns to carry out ONE delegated task. Deliberately different
// from the interactive prompt: a sub-agent has no user to talk to and no chat to
// see, so it works autonomously and returns a single, complete answer instead of
// a terse line plus a NEXT suggestion.
func SubAgentSystemPrompt() string {
	return strings.TrimSpace(`You are a sub-agent: a separate LLM session that a coding assistant spawned to carry out ONE delegated task on its own. You act ONLY through tools, and your final output goes back to that assistant, not to a human.

- You CANNOT see the main conversation. Everything you need is in the task. If something is ambiguous, make a reasonable assumption and proceed — you cannot ask back.
- You CAN edit files. The file tool's write/edit/append actions modify REAL files in your working directory, and you are authorized to use them. NEVER claim you "only have read access" or must "enable editing" — that is false.
- DO the task with tools (file/run/git/web/calc); never just describe how. If what you build or change is runnable, RUN it and report the real result.
- Stay inside your working directory — all your paths resolve there.
- Be thorough and complete: this is one shot. End with a single, self-contained final answer — your findings, the result, or exactly what you changed — written for the main assistant to use directly. Don't ask questions back.`)
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
	lastSig := ""             // signature of the previous turn's tool calls (loop detection)
	acted := false            // did the model call any tool this run?
	actedSinceReturn := false // acted since the last goal re-feed (don't burn a return on no progress)
	returns := 0              // goal re-feeds so far (TTL = a.maxReturns)
	goalMet := false          // the judge confirmed the goal was met
	refusalNudged := false    // already pushed back on a "can't edit / here are the files" dodge?
	for step := 0; step < a.maxSteps; step++ {
		tr.Steps = step + 1

		assistant, err := a.llm.Chat(ctx, msgs, tools)
		if err != nil {
			tr.Messages = msgs
			cancelled := ctx.Err() != nil
			// Any stop that leaves work behind — the user's esc, the runaway cap, a
			// mid-run transport error — keeps the partial work and remembers it so a
			// follow-up can continue, ending cleanly instead of throwing the chain
			// away. Only a stop with nothing done yet surfaces as a hard error.
			if cancelled || acted {
				tr.Cancelled, tr.Stopped = cancelled, true
				tr.Final = stopNote(msgs, cancelled, err)
				a.emit("final", map[string]any{"text": tr.Final})
				if acted {
					a.remember(goal, tr.Final)
				}
				return tr, nil
			}
			return tr, fmt.Errorf("llm chat (step %d): %w", step+1, err)
		}
		// At IPS_LOG=debug this shows exactly what the model returned each turn —
		// the actual tool calls, or text with NO tool calls (e.g. a chat model
		// refusing to edit instead of calling file.edit).
		slog.Debug("model turn", "step", step+1,
			"tool_calls", toolCallNames(assistant.ToolCalls),
			"content", clip(strings.TrimSpace(assistant.Content), 240))
		assistant.Content = unwrapEnvelope(assistant.Content) // salvage envelope-as-content leaks
		msgs = append(msgs, assistant)

		// A reply with no tool calls IS the final answer — emit only "final"
		// (emitting "assistant" too would render the same text twice).
		if len(assistant.ToolCalls) == 0 {
			clean, suggest := splitSuggestion(assistant.Content)
			// Refusal guard: a chat model answering an action task by pasting file
			// contents or claiming it "can't access your files" — and doing nothing
			// via tools this run. Push back once, hard, before accepting it.
			if !a.planMode && !refusalNudged && !acted && looksLikeRefusal(clean) {
				msgs = append(msgs, llm.User(refusalNudge))
				a.emit("nudge", map[string]any{})
				refusalNudged = true
				continue
			}
			// Goal pursuit: the model thinks it's finished — but is the GOAL actually
			// met? A judge (a separate model call) decides. If it isn't, re-feed the
			// goal plus what's missing (keeping the objective in focus, not buried) and
			// keep going — up to maxReturns (a TTL). Only after real progress, so a
			// model that just re-finalizes can't burn the budget.
			if !a.planMode && a.maxReturns > 0 && returns < a.maxReturns && actedSinceReturn && acted {
				done, missing := a.judgeGoal(ctx, goal, clean)
				if !done {
					returns++
					actedSinceReturn = false
					msgs = append(msgs, llm.User(goalReturn(goal, missing)))
					a.emit("continue", map[string]any{"return": returns, "of": a.maxReturns, "missing": missing})
					continue
				}
				goalMet = true
				a.emit("judge", map[string]any{"done": true})
			}
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
			tr.Returns, tr.GoalMet = returns, goalMet
			a.emit("final", map[string]any{"text": clean, "suggest": suggest})
			a.remember(goal, clean)
			return tr, nil
		}

		// Intermediate turn: show the model's reasoning text (if any) alongside
		// the tool calls it's about to make.
		a.emit("assistant", map[string]any{"content": assistant.Content, "tool_calls": len(assistant.ToolCalls)})
		acted = true
		actedSinceReturn = true
		results, nErr := a.runToolCalls(ctx, assistant.ToolCalls)
		msgs = append(msgs, results...)

		// A model burns steps either by repeating calls that all fail (e.g. empty
		// action) OR by repeating the exact same call(s) that already succeeded,
		// making no progress (a reasoning model can spin here). Count both as
		// unproductive: after several such turns give ONE forceful "rethink" nudge
		// with an escape hatch to answer in words; if it's STILL stuck after that,
		// stop and hand the user a one-tap steering suggestion.
		sig := callSig(assistant.ToolCalls)
		repeating := sig != "" && sig == lastSig
		lastSig = sig
		if nErr == len(assistant.ToolCalls) || repeating {
			if stuck++; stuck >= maxStuckTurns {
				if !nudged {
					msgs = append(msgs, llm.User(stuckNudge))
					a.emit("nudge", map[string]any{})
					stuck, nudged = 0, true
				} else {
					const msg = "Stopped — it kept repeating the same tool calls without progress (or they kept failing) even after a nudge to rethink. Steer it (a different approach), or use a stronger model."
					tr.Final, tr.Stopped = msg, true
					tr.Messages = msgs
					a.emit("final", map[string]any{"text": msg, "suggest": stuckSuggest, "exhausted": true})
					a.remember(goal, msg)
					return tr, nil
				}
			}
		} else {
			// A productive turn (a tool call succeeded and it's not a verbatim repeat)
			// clears BOTH the counter and the already-nudged latch: real progress
			// earns a fresh nudge budget, so a couple of early errors followed by good
			// work don't insta-stop the next time the model briefly stumbles.
			stuck, nudged = 0, false
		}
	}

	tr.Messages = msgs
	tr.Stopped = true // ran out of steps before a clean answer
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

// callSig is a stable signature of a turn's tool calls (name + raw args), so two
// turns making the exact same calls compare equal — the tell for a no-progress
// loop. Empty for a turn with no calls.
func callSig(calls []llm.ToolCall) string {
	if len(calls) == 0 {
		return ""
	}
	var b strings.Builder
	for _, c := range calls {
		b.WriteString(c.Name)
		b.WriteByte('|')
		b.WriteString(c.Arguments)
		b.WriteByte('\n')
	}
	return b.String()
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
	// Strip leading markdown/bullet decoration so "**NEXT:**", "- NEXT:", "› NEXT:"
	// are recognized too — otherwise the decorated line leaks into the answer.
	bare := strings.TrimLeft(last, "*_~#>-•·› \t")
	if !strings.HasPrefix(strings.ToUpper(bare), "NEXT:") {
		return text, ""
	}
	suggestion = strings.TrimSpace(strings.Trim(bare[len("NEXT:"):], " \t\"'`*_~"))
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
const stuckNudge = `You're repeating the same tool call(s) without making progress. Stop and re-read the last result: if it errored, it says exactly what's wrong (the call format, or a missing param) — fix that; if it succeeded, you already have what you need, so act on it or finish. Take a different approach. If you genuinely cannot proceed, reply in ONE sentence explaining what's blocking you, and do NOT call a tool.`

// stuckSuggest is offered to the user (one tap) when even the nudge didn't help.
const stuckSuggest = "take a different approach — outline the steps first"

// refusalNudge is the one forceful push-back when a chat model dodges an action
// task — pasting file contents or claiming it can't touch the filesystem —
// instead of using its tools.
const refusalNudge = `You changed nothing — you only described changes or pasted file contents. You are NOT a plain chat model here: you are an agent with working tools in THIS session — file (write/edit/append), run, git — that modify real files on disk, and the user has authorized them. Do not paste file contents, and never say you lack file access (it is false). Make every change now by calling the file tool (write or edit) for each file, then run anything relevant and report the real result.`

// goalReturn re-states the goal when the judge finds it unmet, keeping the
// objective in focus (recency) instead of letting it sink under the transcript,
// and naming the gap so the model finishes the remaining work with tools.
func goalReturn(goal, missing string) string {
	s := "The GOAL is NOT complete yet — keep going. Do the remaining work now with tools (don't stop early, don't just describe it), and only finish once it's actually done."
	if strings.TrimSpace(missing) != "" {
		s += "\n\nStill missing: " + strings.TrimSpace(missing)
	}
	return s + "\n\nGOAL: " + goal
}

// judgeGoal asks the model, in a fresh side call (no tools), whether the goal is
// actually met given the work just finished. It returns done plus a one-line gap
// to feed back when it isn't. On any error or unparseable reply it defaults to
// done=true — a judge that can't decide must not trap the agent in the loop.
func (a *Agent) judgeGoal(ctx context.Context, goal, result string) (bool, string) {
	reply, err := a.llm.Chat(ctx, []llm.Message{
		llm.System(judgeSystem),
		llm.User("GOAL:\n" + goal + "\n\nWHAT THE AGENT DID / ITS FINAL ANSWER:\n" + clip(result, 2000)),
	}, nil)
	if err != nil {
		slog.Debug("goal judge failed", "err", err)
		return true, ""
	}
	line := strings.TrimSpace(reply.Content)
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = strings.TrimSpace(line[:i]) // first line only — small models ramble
	}
	upper := strings.ToUpper(line)
	if rest, ok := cutVerdict(upper, line, "MORE"); ok {
		return false, rest
	}
	if _, ok := cutVerdict(upper, line, "DONE"); ok {
		return true, ""
	}
	slog.Debug("goal judge unparseable", "reply", clip(line, 120))
	return true, "" // can't tell → don't trap the loop
}

// cutVerdict matches a "DONE"/"MORE[: gap]" verdict at the start of the judge's
// line (case-insensitive) and returns the trailing gap text.
func cutVerdict(upper, orig, word string) (string, bool) {
	if !strings.HasPrefix(upper, word) {
		return "", false
	}
	rest := strings.TrimSpace(orig[len(word):])
	rest = strings.TrimLeft(rest, ":-—. ")
	return rest, true
}

// judgeSystem instructs the side-call judge. Tight on purpose: a small local model
// must answer in one parseable line.
const judgeSystem = `You are a strict acceptance checker. Given a GOAL and what an agent did, decide if the goal is FULLY met. Reply with ONE line, nothing else:
- "DONE" if the goal is fully and verifiably accomplished.
- "MORE: <what is still missing, in a few words>" if anything is incomplete, untested, or only described instead of done.
Be skeptical: describing a change instead of making it, or leaving it untested, is NOT done.`

// looksLikeRefusal reports whether a no-tool-call reply is a chat model dodging
// the work — pasting file/code in a fence, or claiming it can't reach the
// filesystem — rather than a real answer. Used only to push back once.
func looksLikeRefusal(s string) bool {
	if strings.Contains(s, "```") { // pasted file/code instead of writing it
		return true
	}
	low := strings.ToLower(s)
	for _, p := range refusalMarkers {
		if strings.Contains(low, p) {
			return true
		}
	}
	return false
}

var refusalMarkers = []string{
	"have access to your", "have direct access", "access to the file system",
	"access to your file", "as a language model", "as an ai", "i'm unable to",
	"i am unable to", "can't directly", "cannot directly", "can't modify files",
	"cannot modify files", "provide you with the", "updated versions of",
	"не имею доступа", "нет доступа к файл", "языковая модель", "не могу напрямую",
	"скопируйте", "полные обновлённые версии", "не имею возможности",
}

// runToolCalls executes every call from one assistant turn, keeping results in
// the emitted order, and reports how many errored (for stuck-loop detection).
// Calls run concurrently ONLY when nothing in the batch has side effects; if any
// call writes files / runs a command / changes git / spawns a sub-agent, the
// whole batch runs sequentially so the calls can't race each other (or the shared
// filesystem / usage ledger).
func (a *Agent) runToolCalls(ctx context.Context, calls []llm.ToolCall) ([]llm.Message, int) {
	out := make([]llm.Message, len(calls))
	errs := make([]bool, len(calls))
	// Concurrency: a single call, or a side-effecting batch, runs sequentially so
	// calls can't race the filesystem/ledger — EXCEPT a pure fan-out of `agent`
	// spawns, which are independent sub-agents (own dirs/clients/usage-locking), so
	// we run those in parallel. A mixed batch (spawns + writes) stays sequential.
	concurrent := len(calls) > 1 && (!a.anyMutating(calls) || a.allAgentCalls(calls))
	if !concurrent {
		for i, c := range calls {
			out[i], errs[i] = a.execOne(ctx, c)
		}
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

// allAgentCalls reports whether every call in the batch is an `agent` spawn — a
// pure fan-out that is safe to run in parallel (each sub-agent is independent).
func (a *Agent) allAgentCalls(calls []llm.ToolCall) bool {
	for _, c := range calls {
		if c.Name != "agent" {
			return false
		}
	}
	return len(calls) > 0
}

// anyMutating reports whether the batch has any side-effecting call — a mutating
// tool action, or an `agent` spawn (sub-agents touch files / the ledger).
func (a *Agent) anyMutating(calls []llm.ToolCall) bool {
	for _, c := range calls {
		if c.Name == "agent" {
			return true
		}
		action, _ := parseArgs(c.Arguments)
		if a.reg.Mutates(c.Name, action) {
			return true
		}
	}
	return false
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
	if a.tr == nil {
		return
	}
	if a.label != "" { // a sub-agent — tag every event so the UI can group it
		if fields == nil {
			fields = map[string]any{}
		}
		fields["agent"] = a.label
	}
	a.tr.Emit(kind, fields)
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
		// Fold in any sibling fields the model split out to the top level
		// ({"action":"edit","path":"x","params":{"find":..}}); params wins.
		for k, v := range m {
			if k == "action" || k == "params" {
				continue
			}
			if _, exists := p[k]; !exists {
				p[k] = v
			}
		}
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

// unwrapEnvelope salvages answers from models that emit their whole chat-message
// envelope as the content — e.g. {"role":"assistant","content":{"text":"…"}} or
// {"role":"assistant","content":"…"} — returning just the inner text. It only
// fires when the content IS exactly such an object (role=="assistant" + a
// content/text field), so a normal answer (even one containing JSON) is untouched.
func unwrapEnvelope(s string) string {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "{") || !strings.HasSuffix(t, "}") {
		return s
	}
	var m map[string]json.RawMessage
	if json.Unmarshal([]byte(t), &m) != nil {
		return s
	}
	var role string
	if json.Unmarshal(m["role"], &role) != nil || role != "assistant" {
		return s
	}
	if inner := envelopeText(m["content"]); inner != "" {
		return inner
	}
	if inner := envelopeText(m["text"]); inner != "" {
		return inner
	}
	return s
}

// envelopeText pulls a string out of a raw value that is either a JSON string or
// an object with a "text"/"content" string field.
func envelopeText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return ""
	}
	for _, k := range []string{"text", "content"} {
		if json.Unmarshal(obj[k], &s) == nil && s != "" {
			return s
		}
	}
	return ""
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
