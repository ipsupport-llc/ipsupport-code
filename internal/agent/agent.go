// Package agent runs the reason → act → observe loop against an llm.Chatter,
// dispatching fat tools, injecting learned pitfalls into tool errors, and
// tracing every step.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"

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
	llm llm.Chatter
	reg *tool.Registry
	kb  *knowledge.KB
	tr  trace.Tracer
	// detached marks an agent whose task was force-abandoned (a wedged request).
	// Its remaining work must not touch shared UI/session state, so emit and
	// remember become no-ops. A fresh agent takes over; this one is orphaned.
	detached atomic.Bool
	system   string
	maxSteps int
	// contextWindow is the active connection's real context size (0 = unknown),
	// used ONLY to keep a single long task's own growing tool-call trail inside
	// it (see trimIfNearWindow) — auto-compact (Compact) is the analogous
	// backstop BETWEEN tasks, not within one.
	contextWindow int

	history    []llm.Message
	maxHistory int
	planMode   bool
	label      string // non-empty for a sub-agent; tags its events so the UI can group them

	// beforeTurn, if set, is called at the top of every loop iteration and its
	// messages are folded into the working set before the next model call. It's
	// the seam for side-channel steering (/btw) that must reach the model between
	// turns WITHOUT interrupting the run. Must be safe to call concurrently.
	beforeTurn func() []llm.Message

	// Goal pursuit: when the model finalizes but a judge (judgeGoal) decides the
	// goal isn't met, it's re-fed the goal and continues, up to maxReturns (a TTL).
	maxReturns int
	// nudgeIdle: after a re-feed, if the model finishes without doing any work,
	// push it once (instead of silently giving up) before accepting the finish.
	nudgeIdle bool
	// asides, if set, is drained between steps of a running task; each returned
	// question gets a one-turn, no-tools answer (the /btw side-channel) without
	// derailing the task.
	asides func() []string
	// archiver, if set, durably records every entry remember() commits to
	// history — see Archiver.
	archiver Archiver
	// historyGen counts discontinuous changes to history: Reset, SetHistory,
	// Compact's replacement, and remember's rolling-window trim all bump it —
	// an ordinary append does not. A checkpoint that captures this alongside its
	// history length can tell, structurally, whether that length still indexes
	// the same history it was taken against: any caller that discontinuously
	// changes history (however it does so, now or in the future) automatically
	// invalidates every outstanding checkpoint, without needing to separately
	// remember to say so.
	historyGen atomic.Int64
}

// Archiver durably records every (goal, final answer + actions digest) pair
// remember() commits to session history — the same text that goes into the
// live session memory, but never trimmed or folded by Compact, so the
// `history` tool can always recover something a compaction summary has since
// shortened. Nil (the default) just means there's nothing to recall from.
type Archiver interface {
	Archive(goal, entry string)
}

// SetArchiver wires a durable record of session memory (see Archiver).
func (a *Agent) SetArchiver(ar Archiver) { a.archiver = ar }

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
func (a *Agent) Reset() { a.history = nil; a.historyGen.Add(1) }

// SetSystem swaps the base system prompt (e.g. after learning new project facts),
// so the next run uses it without a full re-wire.
func (a *Agent) SetSystem(s string) { a.system = s }

// System returns the current base system prompt.
func (a *Agent) System() string { return a.system }

// SetMaxHistory overrides how many recent session messages remember() keeps
// verbatim before silently cutting the oldest ones — no LLM recap, no chance
// to preserve anything (default 16, set in New). Cutting the front of the
// prompt breaks a local server's KV-cache prefix reuse exactly like a summary
// compact does, so a low cap makes that happen on every single turn once a
// session runs long. Memory "raw" callers raise this a lot, turning the trim
// into a rare safety backstop instead of everyday routine.
func (a *Agent) SetMaxHistory(n int) { a.maxHistory = n }

// SetContextWindow tells the agent the active connection's real context size
// (tokens), so a single long-running task can watch its OWN growing tool-call
// trail against it (trimIfNearWindow) instead of only being bounded by
// Compact/remember's cross-task cap, which never runs mid-task. 0 (the
// default) disables this — no window known, nothing to check against.
func (a *Agent) SetContextWindow(n int) { a.contextWindow = n }

// SetGoalLoop configures goal pursuit: when the model finalizes, a judge decides
// whether the goal is met; if not, re-feed the goal and keep going, up to
// maxReturns times (a TTL). 0 disables it — one run, the model's finish stands.
// nudgeIdle gives a model that re-finalizes without working after a re-feed one
// push before giving up.
func (a *Agent) SetGoalLoop(maxReturns int, nudgeIdle bool) {
	a.maxReturns = maxReturns
	a.nudgeIdle = nudgeIdle
}

// SetLabel tags this agent as a sub-agent: the label is attached to every event
// it emits (as the "agent" field) so the UI can group a sub-agent's progress on
// its own status line during a parallel fan-out.
func (a *Agent) SetLabel(s string) { a.label = s }

// SetBeforeTurn registers a hook called at the top of every loop iteration; the
// messages it returns are appended to the working set before the next model
// call. Used for /btw side-channel steering. The hook must be concurrency-safe —
// it's invoked from the run goroutine while the UI goroutine may be feeding it.
func (a *Agent) SetBeforeTurn(fn func() []llm.Message) { a.beforeTurn = fn }

// SetAsides registers the /btw side-channel: fn is drained between steps of a
// running task, and each question gets a one-turn, no-tools answer (emitted as an
// "aside" event) using the live conversation, without steering the task.
func (a *Agent) SetAsides(fn func() []string) { a.asides = fn }

// asidePrompt frames a /btw side question so the model answers it in one turn and
// doesn't try to act on it.
const asidePrompt = "The user asked a quick SIDE question while the main task keeps running. Answer it in one turn from the conversation so far — do NOT call tools, do NOT change anything, just answer concisely.\n\nQuestion: "

// askAside makes a one-shot, no-tools answer to a side question against base (the
// live message set), returning the answer text.
func (a *Agent) askAside(ctx context.Context, base []llm.Message, question string) string {
	msgs := append(append([]llm.Message(nil), base...), llm.User(asidePrompt+question))
	reply, err := a.llm.Chat(ctx, msgs, nil) // nil tools — a single answering turn
	if err != nil {
		return "(couldn't answer the aside: " + err.Error() + ")"
	}
	return strings.TrimSpace(reply.Content)
}

// AnswerAside answers a side question when no task is running, against base —
// a snapshot of the session (system prompt + history) the caller must capture
// synchronously (e.g. via System()+History()) BEFORE calling this, rather than
// having it read a.system/a.history itself: those fields have no lock, and a
// caller that answers the aside on its own goroutine (so the UI stays
// responsive) could otherwise race a concurrent Reset/SetHistory/SetSystem
// call mutating them from the main goroutine while no task is running.
func (a *Agent) AnswerAside(ctx context.Context, base []llm.Message, question string) string {
	return a.askAside(ctx, base, question)
}

// SetPlanMode toggles plan mode. In plan mode the agent investigates with
// read-only tools and proposes a plan; mutating tool calls are refused, so it
// can't change anything until switched back to auto.
func (a *Agent) SetPlanMode(on bool) { a.planMode = on }

// PlanMode reports whether plan mode is on.
func (a *Agent) PlanMode() bool { return a.planMode }

// SessionLen reports how many remembered messages are in the current session.
func (a *Agent) SessionLen() int { return len(a.history) }

// MaxHistory reports the current trim cap (see SetMaxHistory).
func (a *Agent) MaxHistory() int { return a.maxHistory }

// ContextWindow reports the context window this agent was told about (see
// SetContextWindow); 0 means none is known.
func (a *Agent) ContextWindow() int { return a.contextWindow }

// History returns a copy of the session conversation (for persistence).
func (a *Agent) History() []llm.Message { return append([]llm.Message(nil), a.history...) }

// SetHistory restores a session conversation (e.g. loaded from disk, or
// carried forward into a rebuilt Agent) — a discontinuous replacement, so it
// bumps historyGen (see HistoryGen).
func (a *Agent) SetHistory(h []llm.Message) {
	a.history = append([]llm.Message(nil), h...)
	a.historyGen.Add(1)
}

// TruncateHistory cuts history back to its first n entries (a no-op if n is
// already >= the current length) — /rewind's own way of undoing turns
// recorded after a checkpoint. Unlike SetHistory, this only ever removes
// entries from the END, leaving every earlier entry byte-for-byte as it was,
// so it does NOT bump historyGen: any OTHER checkpoint whose captured length
// is <= n still indexes that identical, untouched prefix and stays valid —
// e.g. an earlier checkpoint the user might still want to rewind to next.
func (a *Agent) TruncateHistory(n int) {
	if n < 0 {
		n = 0
	}
	if n < len(a.history) {
		a.history = append([]llm.Message(nil), a.history[:n]...)
	}
}

// HistoryGen returns the current history generation: a counter bumped by every
// discontinuous change to history (Reset, SetHistory, Compact, and remember's
// rolling-window trim) but not by an ordinary append. A checkpoint that
// captured this at take-time, alongside a history length, can compare it here
// to tell — structurally — whether that length still indexes the same
// history, instead of relying on every caller capable of discontinuously
// changing history to separately remember to invalidate outstanding
// checkpoints.
func (a *Agent) HistoryGen() int64 { return a.historyGen.Load() }

// SeedHistoryGen carries the history generation counter forward from a
// previous Agent, so rebuilding the stack (a /permissions or /skills toggle,
// /login, /model) doesn't reset it to 0 and make an already-invalidated
// checkpoint from before the rebuild spuriously valid again.
func (a *Agent) SeedHistoryGen(g int64) { a.historyGen.Store(g) }

// Compact summarizes the session so far into a short recap and replaces the
// history with it, freeing context while keeping continuity. Returns how many
// messages were compacted (0 if there was nothing worth compacting).
//
// The deterministic action digests actionsDigest already baked into each
// remembered entry — file paths touched, commands run, and WHY a command
// failed — are preserved verbatim alongside the LLM's own prose summary, not
// left to its retelling alone. Caught live: a weak model asked to summarize
// many near-identical retries into a few sentences blurred or dropped a
// specific fact like "go.mod already exists" — the compacted history read
// like a fresh start, and the very next task blindly repeated the exact
// command that had already failed every time.
func (a *Agent) Compact(ctx context.Context) (int, error) {
	if len(a.history) < 2 {
		return 0, nil
	}
	var b strings.Builder
	var digests []string
	seenDigest := map[string]bool{}
	for _, m := range a.history {
		switch m.Role {
		case "user":
			b.WriteString("User: " + m.Content + "\n")
		case "assistant":
			if strings.TrimSpace(m.Content) != "" {
				b.WriteString("Assistant: " + m.Content + "\n")
			}
			if d := extractActionsDigest(m.Content); d != "" && !seenDigest[d] {
				seenDigest[d] = true
				digests = append(digests, d)
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
	summary := "[Summary of earlier conversation]\n" + reply.Content
	if len(digests) > 0 {
		summary += "\n\n(exact record of actions across those turns, kept verbatim regardless of the summary above — do not repeat a command marked FAILED, it will fail the same way again:\n" +
			strings.Join(digests, "\n") + ")"
	}
	a.history = []llm.Message{
		{Role: "user", Content: summary},
		{Role: "assistant", Content: "Got it — I have that context."},
	}
	a.historyGen.Add(1)
	return n, nil
}

// extractActionsDigest pulls the "(actions this turn — ...)" suffix
// actionsDigest appends to a remembered entry's content, or "" if it has none.
func extractActionsDigest(content string) string {
	const marker = "\n\n(actions this turn"
	i := strings.Index(content, marker)
	if i < 0 {
		return ""
	}
	return strings.TrimSpace(content[i:])
}

// inTaskTrimRatio is how full the estimated prompt must be (relative to
// contextWindow) before trimIfNearWindow starts trimming — high enough that
// it's a rare backstop for a genuinely long, complex task, not everyday
// routine (Compact/remember's cross-task cap is the everyday mechanism;
// contextWindow only exists for the mid-task case those two never see).
const inTaskTrimRatio = 0.85

// trimKeepRecent is how many of the most recent messages trimIfNearWindow
// never touches, regardless of size — the model is actively working with
// this part of the conversation right now.
const trimKeepRecent = 6

// trimMinResultSize is the smallest tool-result content trimIfNearWindow will
// bother trimming (bytes) — an already-small result isn't worth the churn.
const trimMinResultSize = 500

// estimateMsgTokens is a rough, deterministic token-count proxy: the same
// ~4-bytes-per-token convention used elsewhere in this codebase (see
// internal/llm's promptEstimate, TestCatalogTokenBudget) — good enough to
// decide "getting close", not meant to be exact.
func estimateMsgTokens(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n / 4
}

// trimIfNearWindow keeps a SINGLE long task's own growing tool-call trail from
// silently outgrowing the model's real context window mid-run — the auto-
// compact/remember mechanisms in cmd/agent only ever check BETWEEN tasks, so a
// genuinely complex task (many tool-call rounds) could otherwise organically
// fill the window with nothing watching until the NEXT task got a chance to
// notice.
//
// It trims deterministically, not with an LLM summary: once past
// inTaskTrimRatio of contextWindow, it walks the OLDEST eligible messages
// (never the system prompt, the original goal, assistant text, or the most
// recent trimKeepRecent messages) and shrinks large tool RESULT contents down
// to a short placeholder, stopping as soon as the estimate is back under the
// threshold. This is safe specifically because tools are idempotent: if the
// model still needs that detail, it can just re-run the call — unlike an
// LLM-written recap, there's no risk of silently dropping or misremembering a
// fact the model is still relying on. Mutates msgs in place; returns bytes
// freed (0 if nothing was trimmed).
func trimIfNearWindow(msgs []llm.Message, contextWindow int) int {
	limit := int(float64(contextWindow) * inTaskTrimRatio)
	if estimateMsgTokens(msgs) < limit {
		return 0
	}
	protectFrom := len(msgs) - trimKeepRecent
	freed := 0
	for i := 0; i < protectFrom; i++ {
		m := &msgs[i]
		if m.Role != "tool" || len(m.Content) < trimMinResultSize {
			continue
		}
		freed += len(m.Content)
		m.Content = fmt.Sprintf("[%d bytes of this tool result trimmed to stay within the context window — re-run the call if you still need the detail]", len(m.Content))
		freed -= len(m.Content)
		if estimateMsgTokens(msgs) < limit {
			break
		}
	}
	return freed
}

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

// Detach orphans the agent: its emit and remember become no-ops, so a wedged
// run that later unblocks can't scribble on the UI or the (now fresh) session.
func (a *Agent) Detach() { a.detached.Store(true) }

// Reattach undoes Detach (used only if the agent swap it was part of failed).
func (a *Agent) Reattach() { a.detached.Store(false) }

// remember appends the goal and its final answer to the session, trimming to
// the most recent maxHistory messages. msgs is the full turn just finished
// (all tool calls/results) — actionsDigest pulls a short, deterministic list of
// files touched and commands run out of it and appends that to the STORED
// assistant text (not to final, which is what the user already saw), so the
// next task's context — and a later summary of it — knows what really
// happened on disk instead of relying on the model's own final-answer prose.
func (a *Agent) remember(goal, final string, msgs []llm.Message) {
	if a.detached.Load() {
		return
	}
	entry := final + actionsDigest(msgs)
	if a.archiver != nil {
		a.archiver.Archive(goal, entry)
	}
	a.history = append(a.history, llm.User(goal), llm.Message{Role: "assistant", Content: entry})
	if a.maxHistory > 0 && len(a.history) > a.maxHistory {
		a.history = append([]llm.Message(nil), a.history[len(a.history)-a.maxHistory:]...)
		a.historyGen.Add(1)
	}
}

// actionsDigest scans a finished turn's tool calls for a short, deterministic
// record of what actually changed: file paths touched (write/edit/append/mkdir)
// and top-level shell commands run — each command tagged with why it failed,
// if it did. Without that, a stuck-loop task that stops after repeating a
// failing command leaves the NEXT task's cross-task memory saying only that
// the command was "run", never that it failed or why — so a weak model has no
// signal to avoid blindly repeating it and has to rediscover the same failure
// from scratch. Empty when nothing mutating happened.
//
// A call's result is found by POSITION, not by ToolCall.ID: runToolCalls
// always appends a turn's results immediately after its assistant message, in
// the same order as its ToolCalls, so the message right after an assistant
// message's Nth tool call is always that call's own result — regardless of
// what (if anything) the model/gateway put in the ID field. A weak local
// model's OpenAI-compat endpoint often leaves tool_call ids empty or reuses
// the same one across calls, which silently broke ID-based lookups on some
// turns but not others.
func actionsDigest(msgs []llm.Message) string {
	var files, cmds []string
	seenFile, seenCmd := map[string]bool{}, map[string]bool{}
	for i, m := range msgs {
		if len(m.ToolCalls) == 0 {
			continue
		}
		for j, tc := range m.ToolCalls {
			action, params := parseArgs(tc.Arguments)
			switch tc.Name {
			case "file":
				if action == "read" || action == "list" || action == "find" || action == "search" {
					continue // read-only — not something the next task needs to be told "already happened"
				}
				if p, _ := params["path"].(string); p != "" && !seenFile[p] {
					seenFile[p] = true
					files = append(files, p)
				}
			case "run":
				if c, _ := params["command"].(string); c != "" && !seenCmd[c] {
					seenCmd[c] = true
					entry := clip(c, 100) // long enough that a compound "mkdir && cd && go mod init X" isn't cut before the part that actually identifies it
					if k := i + 1 + j; k < len(msgs) && msgs[k].Role == "tool" {
						if reason := runFailureReason(msgs[k].Content); reason != "" {
							entry += " — FAILED: " + reason
						}
					}
					cmds = append(cmds, entry)
				}
			}
		}
	}
	if len(files) == 0 && len(cmds) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n(actions this turn —")
	if len(files) > 0 {
		fmt.Fprintf(&b, " files touched: %s;", strings.Join(files, ", "))
	}
	if len(cmds) > 0 {
		fmt.Fprintf(&b, " commands run: %s;", strings.Join(cmds, "; "))
	}
	b.WriteString(")")
	return b.String()
}

// runFailureReason extracts a short, concrete reason a run.shell call failed —
// the run tool's own result content is "exit N\n<output>" (see internal/tool/
// run.go), so the first line of output after a non-zero exit is normally the
// actual error message. Returns "" for a successful call ("exit 0"), an
// untracked call (no matching result, or a shape this can't parse), so the
// digest entry stays exactly as it was before this function existed.
func runFailureReason(content string) string {
	if content == "" || strings.HasPrefix(content, "exit 0") {
		return ""
	}
	_, body, ok := strings.Cut(content, "\n")
	if !ok {
		return ""
	}
	line, _, _ := strings.Cut(body, "\n")
	if line = strings.TrimSpace(line); line != "" {
		return clip(line, 80)
	}
	return ""
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
	idleNudged := false       // already pushed a no-progress model once since the last re-feed?
	for step := 0; step < a.maxSteps; step++ {
		tr.Steps = step + 1

		// Side-channel steering (/btw): fold any notes the user dropped mid-run into
		// the working set before this turn's model call, so they land between
		// iterations instead of interrupting the stream.
		if a.beforeTurn != nil {
			if extra := a.beforeTurn(); len(extra) > 0 {
				msgs = append(msgs, extra...)
			}
		}
		// /btw side questions: answer each in one no-tools turn from the live
		// context, then continue the task (the answer isn't fed back into it).
		if a.asides != nil {
			for _, q := range a.asides() {
				a.emit("aside", map[string]any{"q": q, "a": a.askAside(ctx, msgs, q)})
			}
		}

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
				tr.Returns = returns
				tr.Final = stopNote(msgs, cancelled, err)
				a.emit("final", map[string]any{"text": tr.Final})
				if acted {
					a.remember(goal, tr.Final, msgs)
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
			"args", toolCallArgs(assistant.ToolCalls),
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
			if !a.planMode && a.maxReturns > 0 && returns < a.maxReturns {
				switch {
				case actedSinceReturn && acted:
					verdict, missing := a.judgeGoal(ctx, goal, clean)
					switch verdict {
					case judgeMore:
						returns++
						actedSinceReturn, idleNudged = false, false
						msgs = append(msgs, llm.User(goalReturn(goal, missing)))
						a.emit("continue", map[string]any{"return": returns, "of": a.maxReturns, "missing": missing})
						continue
					case judgeDone:
						goalMet = true // only an explicit DONE marks the goal verifiably met
						a.emit("judge", map[string]any{"done": true})
					}
					// judgeUnclear → accept this final (don't trap the loop), but leave
					// goalMet false: an unverifiable judge is NOT a confirmed success.
				case a.nudgeIdle && !idleNudged && returns > 0:
					// The goal was just re-fed but the model finished WITHOUT doing any
					// work this turn. Rather than silently give up, push it once.
					idleNudged = true
					msgs = append(msgs, llm.User(idleNudge))
					a.emit("nudge", map[string]any{"idle": true})
					continue
				}
			}
			// The goal loop ran and re-fed at least once, but the judge never
			// confirmed the goal met — say so plainly instead of implying success.
			goalStalled := a.maxReturns > 0 && returns > 0 && !goalMet
			if strings.TrimSpace(clean) == "" {
				// Blank final turn. If the model already did work via tools, say it
				// finished (the changes/output are above); otherwise it produced
				// nothing — flag that instead of ending on silence.
				switch {
				case goalStalled:
					clean = fmt.Sprintf("(goal not confirmed complete — stopped after %d/%d continues; the model finished without further progress. `/goal go` to keep pushing, or switch to a stronger model with /model.)", returns, a.maxReturns)
				case acted:
					clean = "(done — finished without a written summary; see the changes/output above.)"
				default:
					clean = "(the model returned an empty reply — no answer and no tool call. Try rephrasing, or pick a stronger model with /model.)"
				}
				suggest = ""
			} else if goalStalled {
				clean += "\n\n(note: the goal was not confirmed complete — `/goal go` to keep pushing.)"
			}
			tr.Final = clean
			tr.Messages = msgs
			tr.Returns, tr.GoalMet = returns, goalMet
			a.emit("final", map[string]any{"text": clean, "suggest": suggest})
			a.remember(goal, clean, msgs)
			return tr, nil
		}

		// Intermediate turn: show the model's reasoning text (if any) alongside
		// the tool calls it's about to make.
		a.emit("assistant", map[string]any{"content": assistant.Content, "tool_calls": len(assistant.ToolCalls)})
		acted = true
		actedSinceReturn = true
		results, nErr := a.runToolCalls(ctx, assistant.ToolCalls)
		msgs = append(msgs, results...)
		if a.contextWindow > 0 {
			if freed := trimIfNearWindow(msgs, a.contextWindow); freed > 0 {
				a.emit("context_trim", map[string]any{"bytes_freed": freed})
			}
		}

		// A model burns steps either by repeating calls that all fail (e.g. empty
		// action) OR by repeating the exact same call(s) that already succeeded,
		// making no progress (a reasoning model can spin here). Count both as
		// unproductive: after several such turns give ONE forceful "rethink" nudge
		// with an escape hatch to answer in words; if the very NEXT turn is still
		// unproductive, stop. We do NOT reset the counter on the nudge — a degenerate
		// model (esp. one that thinks for minutes per turn) shouldn't get a fresh full
		// budget to keep flailing; only real progress (the else branch) earns that.
		sig := callSig(assistant.ToolCalls)
		repeating := sig != "" && sig == lastSig
		lastSig = sig
		if nErr == len(assistant.ToolCalls) || repeating {
			if stuck++; stuck >= maxStuckTurns {
				if !nudged {
					msgs = append(msgs, llm.User(stuckNudgeFor(repeating)))
					a.emit("nudge", map[string]any{})
					nudged = true // keep stuck high: one more dud turn now stops it
				} else {
					const msg = "Stopped — it kept repeating the same tool calls without progress (or they kept failing) even after a nudge to rethink. Steer it (a different approach), or use a stronger model."
					tr.Final, tr.Stopped = msg, true
					tr.Returns = returns
					tr.Messages = msgs
					a.emit("final", map[string]any{"text": msg, "suggest": stuckSuggest, "exhausted": true})
					a.remember(goal, msg, msgs)
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
	tr.Returns = returns
	clean, suggest := splitSuggestion(lastAssistantContent(msgs))
	tr.Final = clean
	a.emit("final", map[string]any{"text": clean, "suggest": suggest, "exhausted": true})
	a.remember(goal, clean, msgs)
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

// toolCallArgs renders each call's raw argument JSON (clipped) for the debug
// log — the first thing you need when a model's calls misbehave.
func toolCallArgs(calls []llm.ToolCall) []string {
	if len(calls) == 0 {
		return nil
	}
	args := make([]string, len(calls))
	for i, c := range calls {
		args[i] = clip(c.Arguments, 200)
	}
	return args
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
	// The prompt says to skip the line entirely when nothing fits, but a model
	// sometimes fills it with an honest "nothing to suggest" statement instead
	// (observed: "NEXT: — (no specific next step)") — that's not a real
	// suggestion to offer Tab-acceptable in the input, it's the model saying
	// there isn't one. Unwrap a fully-parenthesized form the same way as the
	// bracketed placeholder above, then treat an actual "no suggestion"
	// statement as no suggestion.
	core := strings.TrimSpace(strings.TrimLeft(suggestion, "—–- \t"))
	if strings.HasPrefix(core, "(") && strings.HasSuffix(core, ")") {
		core = strings.TrimSpace(core[1 : len(core)-1])
	}
	if core == "" || strings.HasPrefix(strings.ToLower(core), "no ") ||
		strings.EqualFold(core, "none") || strings.EqualFold(core, "n/a") {
		suggestion = ""
	}
	if nl < 0 {
		return "", suggestion
	}
	return strings.TrimRight(trimmed[:nl], " \n"), suggestion
}

// maxStuckTurns is how many consecutive all-error tool turns trigger the rethink
// nudge (and, if it doesn't help, the stop).
const maxStuckTurns = 3

// stuckNudgeRepeat is injected when the model sent the exact same tool
// call(s) as last turn — a literal repeat, whether it keeps failing or
// keeps (uselessly) succeeding again.
const stuckNudgeRepeat = `You're repeating the same tool call(s) without making progress. Stop and re-read the last result: if it errored, it says exactly what's wrong (the call format, or a missing param) — fix that; if it succeeded, you already have what you need, so act on it or finish. Take a different approach. If you genuinely cannot proceed, reply in ONE sentence explaining what's blocking you, and do NOT call a tool.`

// stuckNudgeFailing is injected when several turns in a row each failed with
// a DIFFERENT tool call — not a literal repeat (telling the model it's
// "repeating" here would be false and undermine the rest of the nudge), but
// still zero progress. Guessing a new shape each time instead of reading the
// error is the actual failure mode this is pushing back on.
const stuckNudgeFailing = `You've made several different tool calls in a row and every one of them failed. Stop guessing new shapes — re-read the LAST error message closely: it says exactly what's wrong (unknown tool/action, or a missing param) and usually shows the exact shape expected. Fix that specific problem. If you genuinely cannot proceed, reply in ONE sentence explaining what's blocking you, and do NOT call a tool.`

// stuckNudgeFor picks the accurate framing for why the stuck counter tripped —
// see stuckNudgeRepeat / stuckNudgeFailing.
func stuckNudgeFor(repeating bool) string {
	if repeating {
		return stuckNudgeRepeat
	}
	return stuckNudgeFailing
}

// stuckSuggest is offered to the user (one tap) when even the nudge didn't help.
const stuckSuggest = "take a different approach — outline the steps first"

// refusalNudge is the one forceful push-back when a chat model dodges an action
// task — pasting file contents or claiming it can't touch the filesystem —
// instead of using its tools.
const refusalNudge = `You changed nothing — you only described changes or pasted file contents. You are NOT a plain chat model here: you are an agent with working tools in THIS session — file (write/edit/append), run, git — that modify real files on disk, and the user has authorized them. Do not paste file contents, and never say you lack file access (it is false). Make every change now by calling the file tool (write or edit) for each file, then run anything relevant and report the real result.`

// idleNudge is the single push for a model that re-reads the goal after a re-feed
// but then finishes without touching a tool — do the next step, don't just stop.
const idleNudge = `You re-read the GOAL but did nothing this turn — no tool call, no change on disk. Don't stop and don't just describe what to do: take the next concrete step now with your tools (file, run, git) toward the goal, then keep going. Only finish if the goal is genuinely already complete — and if so, state explicitly what's done and why.`

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

// judgeVerdict is the acceptance-checker's answer: the goal is met, needs more, or
// the reply couldn't be read as either.
type judgeVerdict int

const (
	judgeUnclear judgeVerdict = iota // couldn't parse a verdict — accept but don't claim success
	judgeDone                        // explicitly met
	judgeMore                        // explicitly incomplete
)

// judgeGoal asks the model, in a fresh side call (no tools), whether the goal is
// actually met given the work just finished. Returns a tri-state so the caller can
// treat "can't tell" differently from "confirmed done": on a transport error or an
// unparseable reply it returns judgeUnclear — accept the final to avoid trapping
// the loop, but do NOT record the goal as verifiably met.
func (a *Agent) judgeGoal(ctx context.Context, goal, result string) (judgeVerdict, string) {
	reply, err := a.llm.Chat(ctx, []llm.Message{
		llm.System(judgeSystem),
		llm.User("GOAL:\n" + goal + "\n\nWHAT THE AGENT DID / ITS FINAL ANSWER:\n" + clip(result, 2000)),
	}, nil)
	if err != nil {
		slog.Debug("goal judge failed", "err", err)
		return judgeUnclear, ""
	}
	return parseVerdict(reply.Content)
}

var (
	moreToken = regexp.MustCompile(`(?i)\bMORE\b`) // \b avoids matching "MOREOVER"
	doneToken = regexp.MustCompile(`(?i)\bDONE\b`)
)

// parseVerdict reads the judge's whole reply (not just the first line) and biases
// skeptical: any boundary-delimited MORE means not-done (with the gap that follows
// it); only a MORE-free reply mentioning DONE counts as met. This keeps a rambling
// weak judge from either false-accepting or matching "MOREOVER".
func parseVerdict(s string) (judgeVerdict, string) {
	if loc := moreToken.FindStringIndex(s); loc != nil {
		gap := strings.TrimLeft(strings.TrimSpace(s[loc[1]:]), ":-—. \t")
		if i := strings.IndexByte(gap, '\n'); i >= 0 {
			gap = strings.TrimSpace(gap[:i])
		}
		clipped, _ := textutil.Clip(gap, 160)
		return judgeMore, clipped
	}
	if doneToken.MatchString(s) {
		return judgeDone, ""
	}
	slog.Debug("goal judge unparseable", "reply", clip(strings.TrimSpace(s), 120))
	return judgeUnclear, ""
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
			// A cancellation mid-batch (esc during a multi-call turn) must stop
			// the REST of the batch too — execOne only sees ctx on its own
			// call, so without this check a cancelled run still dispatched
			// every remaining mutation in the batch before returning.
			if ctx.Err() != nil {
				out[i], errs[i] = llm.ToolResult(c.ID, c.Name, "cancelled — not run"), true
				continue
			}
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
		if hints := a.hints(c.Name, action, res.Content); hints != "" {
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

// hints pulls matching learned pitfalls for a failed tool call. A pitfall is
// only surfaced when its error pattern actually occurs in THIS error —
// otherwise a loosely keyword-matched lesson (e.g. a "missing path" fix shown
// on a "no action" error) just misleads a weak model. Many domains reuse the
// exact same generic validation text across different actions (file's
// "missing required param(s): path" fires for write, edit, AND append alike),
// so a lesson recorded under one action's Context (free text written by the
// reflecting LLM, e.g. "file: edit") must not surface for a DIFFERENT action's
// identical error — see contextMatchesAction.
func (a *Agent) hints(domain, action, errText string) string {
	if a.kb == nil {
		return ""
	}
	low := strings.ToLower(errText)
	knownActions := a.reg.Actions(domain)
	var b strings.Builder
	for _, p := range a.kb.Query(domain, errText, 3) {
		if p.ErrorPattern == "" || !strings.Contains(low, strings.ToLower(p.ErrorPattern)) {
			continue
		}
		if !contextMatchesAction(p.Context, action, knownActions) {
			continue
		}
		if b.Len() == 0 {
			b.WriteString("Hints from past runs:")
		}
		fmt.Fprintf(&b, "\n- when you saw %q while %s, this worked: %s", p.ErrorPattern, p.Context, p.ProvenFix)
	}
	return b.String()
}

// contextMatchesAction rejects a pitfall whose freeform Context clearly names
// a DIFFERENT action of the same domain than the one that just failed (e.g.
// Context "file: edit" when the current action is "write"). Context isn't a
// structured field — it's free text a reflecting LLM wrote — so this can only
// reject a clear mismatch; a Context naming no action at all (or the current
// one) is always kept, since there's no reliable signal it's wrong.
func contextMatchesAction(context, action string, knownActions []string) bool {
	if action == "" || len(knownActions) < 2 {
		return true // nothing to disambiguate against
	}
	low := strings.ToLower(context)
	for _, other := range knownActions {
		if strings.EqualFold(other, action) {
			continue
		}
		if strings.Contains(low, strings.ToLower(other)) {
			return false
		}
	}
	return true
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
	if a.tr == nil || a.detached.Load() {
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
			// Fold in top-level siblings too (e.g. a top-level "path" beside a
			// stringified params), matching the map case — else they're lost.
			for k, v := range m {
				if k == "action" || k == "params" {
					continue
				}
				if _, exists := inner[k]; !exists {
					inner[k] = v
				}
			}
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
	s = strings.TrimSpace(s)
	if m := decodeObjStrict(s); m != nil {
		return m
	}
	// Some models wrap the JSON in their own tool-call convention instead of
	// emitting it bare (reported live: the whole arguments string was literally
	// "<parameter=params>\n{\"url\": \"...\"}\n</parameter>") — a tagging scheme
	// this project never asked for leaking through from different training.
	// Recovering the embedded object is far more useful than giving up outright,
	// and can't make a genuinely malformed call any worse: if there's no valid
	// object hiding in there either, this still returns nil like before.
	if i, j := strings.IndexByte(s, '{'), strings.LastIndexByte(s, '}'); i >= 0 && j > i {
		if m := decodeObjStrict(s[i : j+1]); m != nil {
			return m
		}
	}
	return nil
}

func decodeObjStrict(s string) map[string]any {
	var m map[string]any
	if json.Unmarshal([]byte(s), &m) != nil {
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
