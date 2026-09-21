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
	// Missing carries the judge's own "what's left" assessment (judgeMore's
	// gap text) out to the caller when a goal-loop run ends WITHOUT a clean
	// finalize to judge in the usual way — currently only set by the
	// stuck-stop path (see Run), so a persisted "incomplete" goal carries a
	// real hint instead of nothing at all. Empty when no such judge call ran,
	// or when it said done/unclear.
	Missing string
	// PromptTokens is the real conversation's prompt size, snapshotted right
	// after the last successful MAIN-turn llm.Chat call (before judgeGoal, which
	// shares the same client absent a dedicated judge/reflect model, gets a
	// chance to overwrite the client's own last-request reading). Callers that
	// need to know how full the context actually is (e.g. auto-compact) should
	// read this instead of asking the client directly.
	PromptTokens int
	// Productive is this run's own final hadProductiveTurn value — whether at
	// least one tool call succeeded and wasn't a verbatim repeat. The caller
	// (finishGoal) ORs this into the standing goal's persisted Progressed flag,
	// so a LATER resume of the same goal (/goal go) that stumbles again
	// immediately still lets the give-up judge run — see SetPriorGoalProgress.
	Productive bool
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
	// maxStuckTurns is how many consecutive unproductive turns earn a rethink
	// nudge before the run gives up — see SetMaxStuckTurns and
	// DefaultMaxStuckTurns.
	maxStuckTurns int
	// contextWindow is the active connection's real context size (0 = unknown),
	// used ONLY to keep a single long task's own growing tool-call trail inside
	// it (see trimIfNearWindow) — auto-compact (Compact) is the analogous
	// backstop BETWEEN tasks, not within one.
	contextWindow int

	// historyMu guards history: SessionLen/History let a caller (e.g. the TUI
	// rendering /status) inspect the session from a different goroutine than
	// the one running Run(), while remember() (called from Run) appends to it
	// concurrently. Brief critical sections only — never held across a
	// model/tool call.
	historyMu  sync.Mutex
	history    []llm.Message
	maxHistory int
	planMode   bool
	// riskObserver is the optional shadow-mode hook (see SetRiskObserver). nil
	// in every path that does not wire one, which is every test in this package.
	riskObserver func(tool, action string, params map[string]any)
	label        string // non-empty for a sub-agent; tags its events so the UI can group them

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
	// priorGoalProgress carries forward whether an EARLIER attempt at the same
	// standing goal ever had a productive turn — see SetPriorGoalProgress.
	priorGoalProgress bool
	// asides, if set, is drained between steps of a running task; each returned
	// question gets a one-turn, no-tools answer (the /btw side-channel) without
	// derailing the task.
	asides func() []string
	// archiver, if set, durably records every entry remember() commits to
	// history — see Archiver.
	archiver Archiver
	// historyGen counts discontinuous changes to history: Reset, SetHistory,
	// and Compact's replacement all bump it — an ordinary append does not, and
	// (see frontTrim) neither does remember's routine rolling-window trim, which
	// only ever shifts a prefix rather than replacing history wholesale. A
	// checkpoint that captures this alongside its history length can tell,
	// structurally, whether that length still indexes the same history it was
	// taken against: any caller that discontinuously changes history (however it
	// does so, now or in the future) automatically invalidates every outstanding
	// checkpoint, without needing to separately remember to say so.
	historyGen atomic.Int64
	// frontTrim cumulatively counts every message remember()'s routine FIFO trim
	// has ever dropped from the front of history. Unlike historyGen, this alone
	// doesn't invalidate a checkpoint: a checkpoint's captured history length can
	// be remapped forward by the delta in this counter since capture (see
	// cmd/agent's checkpointValid/effectiveHistLen) to still point at the right
	// place after a routine trim, instead of being discarded outright the way a
	// genuine reshape (Reset/SetHistory/Compact) must be.
	frontTrim atomic.Int64

	// lastPrompt is the prompt size of the most recent MAIN model turn, in
	// tokens. Distinct from the client's own Context(), which reports whatever
	// request went out last — and the judge, the reflection pass, Compact and
	// /btw asides all share that client, so every one of them repaints it with
	// its own (much smaller) prompt. Reported live: right after "/compact" the
	// UI's context meter dropped from 134% to 1%, which was the size of the
	// compaction request itself, not of the session. The auto-compact DECISION
	// was already protected from this (see the host's lastRealContext); the
	// meter was not, because it also wants to move DURING a task, which that
	// end-of-task snapshot cannot do. This moves on every main turn and on
	// nothing else, so it satisfies both.
	lastPrompt atomic.Int64

	// judgeLLM is the goal judge's own connection, nil when it shares a.llm —
	// see SetJudgeLLM.
	judgeLLM llm.Chatter

	// judgeCriteria is the user's own acceptance criteria, appended to the
	// judge's instruction — see judgeSystemWith.
	judgeCriteria string

	// goalGap is what the judge said was missing when this goal was last
	// attempted — see SetGoalGap.
	goalGap string

	// goalText is the STANDING goal's own text — what the judge accepts against
	// and what a re-feed puts back in front of the model. Empty means "whatever
	// this run was asked to do", which is right when the run IS the goal.
	goalText string
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
	return &Agent{llm: l, reg: reg, kb: kb, tr: tr, system: system, maxSteps: maxSteps, maxHistory: 16, maxStuckTurns: DefaultMaxStuckTurns}
}

// SetMaxStuckTurns overrides how many consecutive unproductive turns are
// tolerated before a rethink nudge (see DefaultMaxStuckTurns). n <= 0 resets
// to the default — same "0 = auto/default" convention as SetContextWindow.
func (a *Agent) SetMaxStuckTurns(n int) {
	if n <= 0 {
		n = DefaultMaxStuckTurns
	}
	a.maxStuckTurns = n
}

// SetPriorGoalProgress records whether an earlier attempt at the goal this Run
// call is about to pursue ever had a productive turn (hadProductiveTurn resets
// to false on every Run call, even though the session it resumes into is the
// same one) — the caller (cmd/agent's goalLoopBudget/finishGoal) should
// pass the standing goal's persisted Progressed flag right before Run, so a
// give-up that stumbles again immediately on a resume still lets the judge
// assess the real progress made in an earlier attempt instead of skipping it
// every single time. Reported live: "неправильный вызов тула - разорвал гоал" —
// /goal go after a stuck-stop could stumble again and never once reach a judge.
func (a *Agent) SetPriorGoalProgress(b bool) { a.priorGoalProgress = b }

// Reset clears the session conversation memory.
func (a *Agent) Reset() {
	a.historyMu.Lock()
	a.history = nil
	a.historyMu.Unlock()
	a.historyGen.Add(1)
}

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

// SetGoalGap carries the judge's own account of what was still missing at the
// END of the previous attempt at this goal, so a resumed run does not start
// blind.
//
// The gap was computed by the judge, persisted to goal.json, shown in
// /goal status — and then never read by any run path. A goal resumed after a
// give-up re-derived its own shortfall from scratch, at the cost of a fresh
// judge round, while the answer sat on disk. Empty when there is none.
func (a *Agent) SetGoalGap(text string) { a.goalGap = strings.TrimSpace(text) }

// SetGoalText sets the standing goal's own text as the acceptance target,
// independent of what any one run was asked to do.
//
// The goal loop is in force for EVERY prompt while a goal stands, and without
// this the judge was handed the run's own prompt as the goal. So with "implement
// authentication and tests" outstanding, a follow-up "read README.md" was judged
// as if READING THE README were the goal — the judge correctly answered DONE,
// and the caller then closed the authentication goal on the strength of it. The
// acceptance target has to be the goal, not the errand.
func (a *Agent) SetGoalText(text string) { a.goalText = strings.TrimSpace(text) }

// acceptanceTarget is what the judge decides about, and what a re-feed restates:
// the standing goal when there is one, else this run's own request.
func (a *Agent) acceptanceTarget(runGoal string) string {
	if a.goalText != "" {
		return a.goalText
	}
	return runGoal
}

// SetJudgeLLM points the goal judge at its own connection instead of the main
// one. nil (the default) means the judge shares a.llm.
//
// Reported live: eleven judge calls in a single run, every one of them
// "reply=\"\" reasoning=<long>". The judge inherited the main model's reasoning
// settings along with its connection, so on a reasoning model it thought as hard
// about a yes/no acceptance check as the main model did about the task, spent
// its entire output budget in reasoning_content, and never reached Content. The
// reflection pass has had its own scope for exactly this reason; the judge had
// none, and no way to be told to stop thinking.
func (a *Agent) SetJudgeLLM(c llm.Chatter) { a.judgeLLM = c }

// judgeChatter is the connection judge calls go out on.
func (a *Agent) judgeChatter() llm.Chatter {
	if a.judgeLLM != nil {
		return a.judgeLLM
	}
	return a.llm
}

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
// having it read a.system/a.history itself: a.system has no lock (a caller
// answering the aside on its own goroutine, so the UI stays responsive, could
// otherwise race a concurrent SetSystem call from the main goroutine), and even
// though History() is itself guarded (see historyMu), reading it separately
// from System() here could still pair a NEW system prompt with an OLD history
// (or vice versa) if a Reset/SetHistory/SetSystem lands in between — the
// caller's single synchronous snapshot keeps the pair consistent.
func (a *Agent) AnswerAside(ctx context.Context, base []llm.Message, question string) string {
	return a.askAside(ctx, base, question)
}

// SetRiskObserver installs a hook called just before every tool call is
// dispatched, with the call already parsed. It exists for shadow-mode risk
// scoring: the observer scores the call and logs what it thought, and whatever
// it concludes changes nothing here.
//
// A plain func of primitives rather than an interface over the risk package: an
// agent that imported the classifier (or the permission policy the classifier
// is compared against) would drag both into every test that builds one, to
// support a feature that is by construction allowed to have no effect.
func (a *Agent) SetRiskObserver(f func(tool, action string, params map[string]any)) {
	a.riskObserver = f
}

// SetPlanMode toggles plan mode. In plan mode the agent investigates with
// read-only tools and proposes a plan; mutating tool calls are refused, so it
// can't change anything until switched back to auto.
func (a *Agent) SetPlanMode(on bool) { a.planMode = on }

// PlanMode reports whether plan mode is on.
func (a *Agent) PlanMode() bool { return a.planMode }

// SessionLen reports how many remembered messages are in the current session.
func (a *Agent) SessionLen() int {
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	return len(a.history)
}

// PromptTokens reports the prompt size of the most recent MAIN model turn (see
// lastPrompt). 0 before the first turn.
func (a *Agent) PromptTokens() int { return int(a.lastPrompt.Load()) }

// MaxHistory reports the current trim cap (see SetMaxHistory).
func (a *Agent) MaxHistory() int { return a.maxHistory }

// ContextWindow reports the context window this agent was told about (see
// SetContextWindow); 0 means none is known.
func (a *Agent) ContextWindow() int { return a.contextWindow }

// History returns a copy of the session conversation (for persistence).
func (a *Agent) History() []llm.Message {
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	return append([]llm.Message(nil), a.history...)
}

// SetHistory restores a session conversation (e.g. loaded from disk, or
// carried forward into a rebuilt Agent) — a discontinuous replacement, so it
// bumps historyGen (see HistoryGen).
func (a *Agent) SetHistory(h []llm.Message) {
	a.historyMu.Lock()
	a.history = append([]llm.Message(nil), h...)
	a.historyMu.Unlock()
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
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	if n < len(a.history) {
		a.history = append([]llm.Message(nil), a.history[:n]...)
	}
}

// HistoryGen returns the current history generation: a counter bumped by every
// discontinuous change to history (Reset, SetHistory, Compact) but not by an
// ordinary append, nor by remember's routine rolling-window trim (see
// FrontTrimCount for that). A checkpoint that captured this at take-time,
// alongside a history length, can compare it here to tell — structurally —
// whether that length still indexes the same history, instead of relying on
// every caller capable of discontinuously changing history to separately
// remember to invalidate outstanding checkpoints.
func (a *Agent) HistoryGen() int64 { return a.historyGen.Load() }

// FrontTrimCount returns the total number of messages remember()'s routine
// FIFO trim has ever dropped from the front of history, cumulative across the
// agent's whole lifetime. Distinct from HistoryGen: a routine trim only shifts
// where index 0 points, so a checkpoint's captured history length can be
// remapped forward by the delta in this counter since it was captured, rather
// than being invalidated outright the way HistoryGen's discontinuous changes
// must be.
func (a *Agent) FrontTrimCount() int64 { return a.frontTrim.Load() }

// SeedHistoryGen carries the history generation counter forward from a
// previous Agent, so rebuilding the stack (a /permissions or /skills toggle,
// /login, /model) doesn't reset it to 0 and make an already-invalidated
// checkpoint from before the rebuild spuriously valid again.
func (a *Agent) SeedHistoryGen(g int64) { a.historyGen.Store(g) }

// Compact summarizes the session so far into a short recap and replaces the
// history with it, freeing context while keeping continuity. Returns how many
// messages were compacted (0 if there was nothing worth compacting).
//
// focus is the user's own steer for THIS compaction ("/compact keep the SIP
// details"), empty for the automatic one. It only reweights what the prose
// summary dwells on — the action digests below are appended deterministically
// after the model has answered, so no focus, however aggressive, can drop the
// record of which files were touched or which commands failed.
//
// The deterministic action digests actionsDigest already baked into each
// remembered entry — file paths touched, commands run, and WHY a command
// failed — are preserved verbatim alongside the LLM's own prose summary, not
// left to its retelling alone. Caught live: a weak model asked to summarize
// many near-identical retries into a few sentences blurred or dropped a
// specific fact like "go.mod already exists" — the compacted history read
// like a fresh start, and the very next task blindly repeated the exact
// command that had already failed every time.
func (a *Agent) Compact(ctx context.Context, focus string) (int, error) {
	a.historyMu.Lock()
	if len(a.history) < 2 {
		a.historyMu.Unlock()
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
		}
		// Checked regardless of role: Compact's OWN digest block (below) is
		// embedded in a "user"-role summary message, not "assistant" — a later
		// Compact call must still re-harvest it, or the exact-action records it
		// carries evaporate the moment the LLM's own prose summary drops them.
		if d := extractActionsDigest(m.Content); d != "" && !seenDigest[d] {
			seenDigest[d] = true
			digests = append(digests, d)
		}
	}
	a.historyMu.Unlock()
	reply, err := a.llm.Chat(ctx, []llm.Message{
		llm.System(compactSystem(focus)),
		llm.User(b.String()),
	}, nil)
	if err != nil {
		return 0, err
	}
	summary := "[Summary of earlier conversation]\n" + reply.Content
	if len(digests) > 0 {
		// Starts with actionsDigestMarker (not bespoke wording) so a LATER Compact
		// call's scan above recognizes and re-harvests this whole block too.
		// NOT "do not repeat a command marked FAILED, it will fail the same way
		// again", which is what this said. A command's past failure is an
		// observation about the state it ran against, not a property of the
		// command: `go test ./...` fails before the fix and passes after it, and
		// the old wording told the model never to run it again — directly
		// against a judge that requires tests to have been RUN. What the record
		// is actually good for is not blindly REPEATING a failure without
		// changing anything first.
		summary += actionsDigestMarker + " — exact record of actions across those turns, kept verbatim regardless of the summary above. A FAILED entry says what went wrong at the time, not that the command is forbidden: don't re-run one unchanged expecting a different result, but DO re-run it once you've addressed the cause (a verification step is worth repeating after a fix):\n" +
			strings.Join(digests, "\n") + ")"
	}
	a.historyMu.Lock()
	n := len(a.history)
	a.history = []llm.Message{
		{Role: "user", Content: summary},
		{Role: "assistant", Content: "Got it — I have that context."},
	}
	a.historyMu.Unlock()
	a.historyGen.Add(1)
	return n, nil
}

// compactSystem builds the recap instruction, optionally steered by the user's
// own focus for this one compaction. The steer says PRIORITIZE, not "discard
// everything else": a recap that obeys "only keep X" literally can strand the
// next task without the context it needs to continue, and the user asking for a
// focus is asking for emphasis, not amnesia.
func compactSystem(focus string) string {
	const base = "Summarize the conversation so far into a compact recap that preserves the key facts, decisions, files touched, and context needed to keep going. A few sentences, no preamble."
	f := strings.TrimSpace(focus)
	if f == "" {
		return base
	}
	return base + "\n\nThe user asked for this recap to focus on: " + f +
		"\nGive that priority and detail; compress everything else hard, but still keep whatever is needed to continue the work."
}

// actionsDigestMarker is the literal marker actionsDigest's writer side always
// prepends its digest block with, and extractActionsDigest's reader side scans
// for — a single shared constant so writer and reader can never drift apart.
// Compact's own digest block (see Compact) is deliberately written with this
// SAME marker, so a LATER Compact call's scan (which uses extractActionsDigest)
// re-harvests it too, not just the per-turn digests actionsDigest produces —
// otherwise Compact's own exact-action records would only ever survive a
// single compaction.
const actionsDigestMarker = "\n\n(actions this turn"

// extractActionsDigest pulls the "(actions this turn — ...)" suffix
// actionsDigest appends to a remembered entry's content, or "" if it has none.
func extractActionsDigest(content string) string {
	i := strings.Index(content, actionsDigestMarker)
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
// decide "getting close", not meant to be exact. Counts ToolCalls[].Arguments
// too, not just Content — client.go's toWire() resends a tool call's Arguments
// verbatim on every subsequent request, so a large write/edit/run argument
// payload is just as much a real, resent cost as message Content is.
func estimateMsgTokens(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
		for _, tc := range m.ToolCalls {
			n += len(tc.Arguments)
		}
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
// fact the model is still relying on.
//
// If that first pass still isn't enough — the protected trimKeepRecent
// messages alone are bigger than the whole budget (e.g. one huge tool result
// among them) — a second, overflow-fallback pass trims further into the
// protected zone too, sparing only the single most recent message. Nothing
// short of that is left untouchable when the alternative is silently blowing
// through the entire context window.
//
// sentEst is the byte estimate of the messages exactly as they were sent, and
// realTokens the server's own prompt_tokens for that same request (0 when
// unknown); together they calibrate the estimate (see tokenScale).
//
// Mutates msgs in place; returns bytes freed (0 if nothing was trimmed).
func trimIfNearWindow(msgs []llm.Message, contextWindow, sentEst, realTokens int) int {
	limit := int(float64(contextWindow) * inTaskTrimRatio)
	scale := tokenScale(sentEst, realTokens)
	if sizedInTokens(msgs, scale) < limit {
		return 0
	}
	freed := trimOldToolResults(msgs, limit, scale, len(msgs)-trimKeepRecent)
	if sizedInTokens(msgs, scale) >= limit {
		freed += trimOldToolResults(msgs, limit, scale, len(msgs)-1)
	}
	return freed
}

// tokenScale calibrates estimateMsgTokens against what the server actually
// charged for the last main turn.
//
// Reported live, with the debug log to prove it: a run on a 32.8k-token window
// went 5k → 20k → 44k tokens in two steps, blew past the window, and the model
// degenerated into echoing its own prompt. trimIfNearWindow never fired. Its
// arithmetic was self-consistent and wrong: two captured HTML pages, each
// bounded to 50 000 bytes, estimate at 4 bytes/token to ~25k — just under the
// 27 880 trigger — while the server reported 43 816 for the same messages.
// HTML and minified JS tokenize at roughly half that assumed density, and that
// is exactly the content that blows a window up. The true figure was already in
// hand (Run snapshots prompt_tokens every turn); the decision was simply being
// made against a guess instead. One ratio folds the truth back in, and covers
// the tool schemas and other per-request overhead that never appear in msgs at
// all. Clamped, so one odd reading can't send it trimming the whole run away.
// sentEst must be the estimate of the messages as they were SENT, not as they
// are now: the reading describes that request, and by the time trimming runs
// this turn's results have already been appended. Dividing the old reading by
// the new, larger estimate produced a ratio below 1 and shrank the very number
// it was meant to correct — so a set that had just outgrown the window measured
// smaller than before and no trim happened.
func tokenScale(sentEst, realTokens int) float64 {
	est := sentEst
	if realTokens <= 0 || est <= 0 {
		return 1
	}
	scale := float64(realTokens) / float64(est)
	if scale < 0.5 {
		scale = 0.5
	}
	if scale > 4 {
		scale = 4
	}
	return scale
}

// sizedInTokens is estimateMsgTokens corrected by the calibration above.
func sizedInTokens(msgs []llm.Message, scale float64) int {
	return int(float64(estimateMsgTokens(msgs)) * scale)
}

// trimOldToolResults walks msgs[0:protectFrom] oldest-first, shrinking large
// tool RESULT contents to a short placeholder (via trimPlaceholder) until the
// estimate drops under limit or the range is exhausted. Returns bytes freed.
func trimOldToolResults(msgs []llm.Message, limit int, scale float64, protectFrom int) int {
	freed := 0
	for i := 0; i < protectFrom; i++ {
		m := &msgs[i]
		if m.Role != "tool" || len(m.Content) < trimMinResultSize {
			continue
		}
		before := len(m.Content)
		m.Content = trimPlaceholder(m.Content)
		freed += before - len(m.Content)
		if sizedInTokens(msgs, scale) < limit {
			break
		}
	}
	return freed
}

// trimPlaceholder replaces a tool result's content with a short placeholder,
// while preserving just enough of the content ahead of it that runFailureReason
// (and therefore actionsDigest's "— FAILED: ..." tag) still extracts the SAME
// failure signal it would have from the untrimmed content. trimIfNearWindow
// runs mid-task, before remember() ever sees the result, so whatever isn't
// carried forward here is gone for good by the time the digest is built: the
// run tool's content is "exit N\n<output>" (see internal/tool/run.go), so the
// first line carries the exit-status check runFailureReason keys off of
// (HasPrefix "exit 0") and the next line carries the reason text it extracts.
func trimPlaceholder(content string) string {
	placeholder := fmt.Sprintf("[%d bytes of this tool result trimmed to stay within the context window — re-run the call if you still need the detail]", len(content))
	firstLine, rest, ok := strings.Cut(content, "\n")
	if !ok {
		return placeholder
	}
	firstLine = clip(firstLine, 80)
	reasonLine, _, _ := strings.Cut(rest, "\n")
	if reasonLine = strings.TrimSpace(reasonLine); reasonLine == "" {
		return firstLine + "\n" + placeholder
	}
	return firstLine + "\n" + clip(reasonLine, 80) + "\n" + placeholder
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
	a.historyMu.Lock()
	a.history = append(a.history, llm.User(goal), llm.Message{Role: "assistant", Content: entry})
	if a.maxHistory > 0 && len(a.history) > a.maxHistory {
		dropped := len(a.history) - a.maxHistory
		a.history = append([]llm.Message(nil), a.history[len(a.history)-a.maxHistory:]...)
		a.frontTrim.Add(int64(dropped))
	}
	a.historyMu.Unlock()
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
	var files []string
	seenFile := map[string]bool{}
	// First-seen order (cmdOrder), latest-seen outcome (cmdEntry, overwritten on
	// each repeat): a command retried later in the SAME task — e.g. it failed,
	// got fixed, then succeeded — must be recorded with its LATEST result, not
	// frozen on its first occurrence. Otherwise a fixed-and-passing command stays
	// permanently tagged FAILED, and Compact strengthens that stale record into
	// an instruction to never repeat it — actively wrong.
	var cmdOrder []string
	cmdEntry := map[string]string{}
	for i, m := range msgs {
		if len(m.ToolCalls) == 0 {
			continue
		}
		for j, tc := range m.ToolCalls {
			action, params, _ := parseArgs(tc.Arguments)
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
				if c, _ := params["command"].(string); c != "" {
					entry := clip(c, 100) // long enough that a compound "mkdir && cd && go mod init X" isn't cut before the part that actually identifies it
					if k := i + 1 + j; k < len(msgs) && msgs[k].Role == "tool" {
						if reason := runFailureReason(msgs[k].Content); reason != "" {
							entry += " — FAILED: " + reason
						}
					}
					if _, ok := cmdEntry[c]; !ok {
						cmdOrder = append(cmdOrder, c)
					}
					cmdEntry[c] = entry
				}
			}
		}
	}
	if len(files) == 0 && len(cmdOrder) == 0 {
		return ""
	}
	cmds := make([]string, len(cmdOrder))
	for i, c := range cmdOrder {
		cmds[i] = cmdEntry[c]
	}
	var b strings.Builder
	b.WriteString(actionsDigestMarker + " —")
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
func (a *Agent) Run(ctx context.Context, goal string) (tr Transcript, err error) {
	a.emit("goal", map[string]any{"text": goal})
	hist := a.History() // guarded snapshot — see historyMu
	msgs := make([]llm.Message, 0, len(hist)+3)
	msgs = append(msgs, llm.System(a.system))
	if a.planMode {
		msgs = append(msgs, llm.System(planDirective))
	}
	msgs = append(msgs, hist...) // session memory
	msgs = append(msgs, llm.User(goal))
	tools := a.reg.OpenAITools()
	slog.Debug("run start", a.debugArgs("goal", clip(goal, 120), "tools", toolNames(tools), "plan_mode", a.planMode)...)
	// Reported live: no single log line said how/why a run ended — reading it
	// off "model turn"/"stuck check" lines meant inferring it indirectly (e.g.
	// a tool_calls=[] turn with no following "stuck check" line MIGHT be a
	// clean finish, or might be about to hit an invisible goal-loop judge
	// call — see the "goal judge" log added below). A named return + defer
	// covers every return path (cancellation, a mid-run Chat error, a clean
	// finalize, the stuck-stop, and step-exhaustion) without having to
	// remember to add a log call at each one individually.
	defer func() {
		slog.Debug("run end", a.debugArgs("steps", tr.Steps, "stopped", tr.Stopped, "cancelled", tr.Cancelled,
			"goal_met", tr.GoalMet, "returns", tr.Returns, "productive", tr.Productive, "err", err, "final", clip(tr.Final, 200))...)
	}()

	stuck, nudged := 0, false
	lastSig := ""             // signature of the previous turn's tool calls (loop detection)
	acted := false            // did the model call any tool this run?
	actedSinceReturn := false // acted since the last goal re-feed (don't burn a return on no progress)
	// hadProductiveTurn distinguishes "attempted some tool call" (acted) from
	// "at least one turn actually succeeded, wasn't a repeat" — a stuck-stop
	// where EVERY turn failed/repeated has acted=true too (a failed call still
	// counts as "acted"), which isn't a real signal that a judge call on a
	// stuck-stop would be worth its cost; a genuinely productive turn is.
	hadProductiveTurn := false
	// This run's own hadProductiveTurn resets to false regardless of what an
	// earlier attempt at the same goal did — recorded into tr.Productive here
	// (LIFO: runs before the "run end" log defer above, which reads it) so
	// finishGoal can OR it into the standing goal's persisted Progressed flag.
	defer func() { tr.Productive = hadProductiveTurn }()
	returns := 0              // goal re-feeds so far (TTL = a.maxReturns)
	lastMissing := ""         // the final judge verdict's "what's still missing", once the TTL can buy no more attempts
	goalMet := false          // the judge confirmed the goal was met
	refusalNudged := false    // already pushed back on a "can't edit / here are the files" dodge?
	emptyNudged := false      // already pushed back on a totally blank first reply?
	degenerateNudged := false // already pushed back on ONE collapsed generation?
	idleNudged := false       // already pushed a no-progress model once since the last re-feed?
	promptTokens := 0         // last known real prompt size from a MAIN-turn Chat call — see Transcript.PromptTokens
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

		sentEst := estimateMsgTokens(msgs) // what this request actually carried, for tokenScale below
		assistant, err := a.llm.Chat(ctx, msgs, tools)
		// One generation collapsing into repetition is not the run failing. The
		// transport is fine, the context is usually nowhere near full, and every
		// other flaky-reply case here already gets exactly one retry — an empty
		// reply, a refusal, the judge's own unreadable verdict. Reported live: a
		// 5-step run against a 128-step budget ended because the fifth generation
		// repeated a phrase, at 11% of the context window, with a standing goal
		// that then went unassessed and unmentioned. Push back once, the same way
		// the repeated-tool-call nudge does, and only give up if the very next
		// turn collapses too.
		if err != nil && ctx.Err() == nil && !degenerateNudged && llm.IsDegenerateOutput(err) {
			slog.Debug("degenerate turn", a.debugArgs("step", step+1, "err", err)...)
			msgs = append(msgs, llm.User(degenerateNudge))
			a.emit("nudge", map[string]any{})
			degenerateNudged = true
			continue
		}
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
				// Three paths end a run early and two of them account for the goal;
				// this one said nothing at all. Reported live: a run with a standing
				// goal died here on one collapsed generation, and the ending never
				// mentioned the goal — no verdict, no "/goal go", nothing. From the
				// user's seat the goal had simply vanished, though it was still on
				// disk. Cancellation is excluded: esc means give control back, not
				// spend another model call.
				if !cancelled {
					met, missing := a.judgeOnGiveUp(ctx, a.acceptanceTarget(goal), returns, hadProductiveTurn,
						"turn_failed", stopNote(msgs, false, err)+actionsDigest(msgs)+a.priorGapNote(returns), msgs)
					goalMet = met
					if !met {
						tr.Missing = firstNonEmpty(missing, lastMissing)
						lastMissing = tr.Missing
					}
					tr.GoalMet = met
					// The judge said it isn't done and the budget is untouched, so
					// the goal has not ended by any of the three things that end it.
					// A model-level failure — a generation that collapsed — says
					// nothing about whether the goal is reachable; re-feed and keep
					// going. A TRANSPORT failure is the machine being unavailable,
					// not the model failing the goal, and no amount of re-feeding
					// changes that: the verdict is recorded and the run ends, with
					// the goal left standing and resumable.
					if !met && returns < a.maxReturns && llm.IsDegenerateOutput(err) {
						returns++
						actedSinceReturn, idleNudged, degenerateNudged = false, false, false
						msgs = append(msgs, llm.User(goalReturn(a.acceptanceTarget(goal), missing)))
						a.emit("continue", map[string]any{"return": returns, "of": a.maxReturns, "missing": missing})
						continue
					}
					tr.Final += goalNotConfirmedNote(a.maxReturns, met, tr.Missing)
				}
				tr.PromptTokens = promptTokens
				a.emit("final", map[string]any{"text": tr.Final})
				if acted {
					a.remember(goal, tr.Final, msgs)
				}
				return tr, nil
			}
			tr.PromptTokens = promptTokens
			return tr, fmt.Errorf("llm chat (step %d): %w", step+1, err)
		}
		// Snapshot the real conversation's fullness right after this MAIN-turn call
		// succeeds — BEFORE judgeGoal (below, same iteration) gets a chance to run.
		// judgeGoal shares this same a.llm when no dedicated judge/reflect model is
		// configured, and its own (typically much shorter) Chat call would
		// otherwise clobber the client's last-request Context() reading before
		// anyone reads it for auto-compact sizing.
		if cr, ok := a.llm.(interface{ Context() int }); ok {
			promptTokens = cr.Context()
			a.lastPrompt.Store(int64(promptTokens))
		}
		// At IPS_LOG=debug this shows exactly what the model returned each turn —
		// the actual tool calls, or text with NO tool calls (e.g. a chat model
		// refusing to edit instead of calling file.edit). reasoning/finish_reason
		// are the difference between "content empty, no idea why" and an actual
		// diagnosis: a model that reasoned its way to nothing, one the server cut
		// off (finish_reason=length) vs. one that just quietly matched "stop".
		// context_tokens is this same turn's prompt size (see promptTokens above),
		// so a degenerate-empty-reply pattern can be correlated with the context
		// actually being near full — the exact failure mode auto-compact exists
		// to prevent.
		slog.Debug("model turn", "step", step+1,
			"tool_calls", toolCallNames(assistant.ToolCalls),
			"args", toolCallArgs(assistant.ToolCalls),
			"content", clip(strings.TrimSpace(assistant.Content), 240),
			"reasoning", clip(strings.TrimSpace(assistant.Reasoning), 500),
			"finish_reason", assistant.FinishReason,
			"context_tokens", promptTokens)
		assistant.Content = unwrapEnvelope(assistant.Content) // salvage envelope-as-content leaks
		msgs = append(msgs, assistant)

		// A reply with no tool calls IS the final answer — emit only "final"
		// (emitting "assistant" too would render the same text twice). A turn
		// whose ONLY tool call is the bare "done" signal (see tool.NewDone)
		// counts the same way: some models are so habituated to always calling
		// something that they never produce a real no-tool-call reply, and
		// invent a workaround instead (reported live: repeated shell echoes of
		// "TASK COMPLETE"/"EXIT 0"). done gives them a correct channel for
		// exactly that. Deliberately narrow: done arriving ALONGSIDE other real
		// tool calls does NOT intercept — those still need normal dispatch.
		// Independent of the goal judge's own done/more tools (judgeTools) —
		// this only changes how the MAIN model signals "I think I'm finished";
		// the judge still separately decides whether the goal is actually met.
		if len(assistant.ToolCalls) == 0 || isDoneOnly(assistant.ToolCalls) {
			// The assistant message is already in msgs, done's call included, and
			// this branch never dispatches it — so every path below that CONTINUES
			// (the judge's re-feed, the idle nudge, the empty-reply and refusal
			// nudges) left an assistant tool_call with no matching tool result in
			// the conversation. toWire sends it verbatim and nothing repairs it, so
			// a backend that enforces the pairing rejects the next request and
			// pursuit ends for a protocol reason, short of the TTL. Answer the call
			// here, before anything can continue past it — and it matters more now
			// that the judge is shown these same messages.
			if isDoneOnly(assistant.ToolCalls) {
				msgs = append(msgs, llm.ToolResult(assistant.ToolCalls[0].ID, "done",
					"(noted — this only signals you think you're finished; whether the goal is met is decided separately)"))
			}
			clean, suggest := splitSuggestion(assistant.Content)
			// Reported live: a single genuinely empty reply (no content, no tool
			// calls) on the very first turn — before anything productive happened —
			// used to finalize the WHOLE run immediately, goal or not, with no retry
			// at all. Every other flaky-reply case in this codebase already gets one
			// retry (the judge's own empty-reply retry, the refusal nudge right
			// below) — this is the same class of problem and deserves the same
			// treatment before being accepted as a real (if silent) answer.
			if !a.planMode && !emptyNudged && !acted && strings.TrimSpace(clean) == "" {
				msgs = append(msgs, llm.User(emptyReplyNudge))
				a.emit("nudge", map[string]any{})
				emptyNudged = true
				continue
			}
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
			//
			// While goal mode is on, the JUDGE decides — every time, on every
			// ending. The only things that finish a goal are an explicit DONE, the
			// TTL running out, and the user.
			//
			// This used to be gated on `acted`, on the argument that a run calling
			// no tool is a chat reply and shouldn't be dragged into the loop. The
			// argument was about the model ANSWERING IN PROSE; the gate asked
			// whether a tool ran, which is not the same question. A reply with no
			// text AND no tool call is not a conversational turn — it is nothing at
			// all — and it was ending goal runs outright. Reported live: a goal
			// with a 255-refeed budget spent zero of them and handed control back
			// after two empty replies, and the user had to paste the whole task in
			// again.
			//
			// So the gate is now what it always meant: skip only a reply that
			// actually SAID something without doing anything — an answer to a
			// question asked alongside the goal. /btw exists for those, and /goal
			// off turns this off entirely.
			//
			// NOT gated on returns < maxReturns. The TTL bounds how many times the
			// goal may be RE-FED, not whether the last attempt it paid for gets
			// looked at: with ttl 1, the model's one re-feed could finish the work
			// and the run still ended "not confirmed", because returns had already
			// reached the cap and the judge was skipped entirely. Funding an
			// attempt and then refusing to grade it is the one combination that
			// makes no sense.
			chatReply := !acted && strings.TrimSpace(clean) != ""
			if !a.planMode && a.maxReturns > 0 && !chatReply {
				// The goal was re-fed and the model finished WITHOUT doing any work
				// since. Push it once first: a nudge is cheaper than a judge call and
				// spends no return.
				if a.nudgeIdle && !idleNudged && returns > 0 && !actedSinceReturn {
					idleNudged = true
					msgs = append(msgs, llm.User(idleNudge))
					a.emit("nudge", map[string]any{"idle": true})
					continue
				}
				// Everything else goes to the judge — including a model that
				// finalizes idle a SECOND time, with the nudge already spent.
				//
				// Reported live: "(goal not confirmed complete — stopped after 3/255
				// continues; the model finished without further progress.)" — the run
				// ended with 252 returns unspent and no judge call at all. The old
				// switch only judged when actedSinceReturn was true, so a model that
				// idled once, got nudged, and idled again matched NEITHER case and
				// fell straight out to the final. Same disease as an unclear verdict
				// ending the run: the goal died neither on an explicit DONE, nor on
				// the TTL, nor on the user turning it off.
				{
					// judgeGoal is an isolated call with no access to msgs — when the
					// final turn itself has empty/uninformative content (a model that
					// did real tool-call work but wrote no summary), clean alone told
					// it nothing to judge. actionsDigest gives it the real record
					// (files touched / commands run) instead, same fix already applied
					// to the give-up paths (see judgeOnGiveUp).
					verdict, missing := a.judgeGoal(ctx, a.acceptanceTarget(goal), clean+actionsDigest(msgs)+a.priorGapNote(returns), msgs)
					// Reported live: the judge's own decision (a separate LLM call)
					// was entirely invisible in the debug log for its two NORMAL
					// verdicts — only its failure ("goal judge failed") and
					// unparseable-reply cases were logged, so a "more"/"done" judge
					// call left no trace of ever having happened, making a
					// tool_calls=[] turn's true fate unreadable from the log alone.
					slog.Debug("goal judge", a.debugArgs("verdict", verdict, "missing", missing, "return", returns, "of", a.maxReturns, "reason", "normal")...)
					// judgeDone is the ONLY verdict that ends a goal run early. Both
					// judgeMore and judgeUnclear mean the same thing — the goal is not
					// confirmed met — so both re-feed it and keep going, up to the TTL.
					//
					// Reported live, with the debug log to prove it: a judge whose own
					// reply came back empty on BOTH attempts (finish_reason=length —
					// its reasoning ate the entire output budget, nothing reached
					// Content) returned judgeUnclear, and judgeUnclear used to accept
					// the final and end the run: "verdict=unclear return=1 of=255",
					// i.e. a goal abandoned with 254 returns still unspent. The goal
					// thus died neither because it was met, nor because the budget ran
					// out, nor because the user turned it off — but because the judge
					// couldn't speak. A judge that can't answer has confirmed nothing;
					// silence is not acceptance. The only things that end goal pursuit
					// are an explicit DONE, the TTL, and the user (/goal off).
					if verdict == judgeDone {
						goalMet = true // only an explicit DONE marks the goal verifiably met
						a.emit("judge", map[string]any{"done": true})
					} else if lastMissing = missing; returns < a.maxReturns {
						returns++
						actedSinceReturn, idleNudged = false, false
						msgs = append(msgs, llm.User(goalReturn(a.acceptanceTarget(goal), missing)))
						a.emit("continue", map[string]any{"return": returns, "of": a.maxReturns, "missing": missing})
						continue
					}
					// lastMissing is set above for BOTH branches: the gap is the
					// judge's account of this run's shortfall whether or not the
					// budget could still buy another attempt, and it has to
					// survive into the transcript either way.
				}
			}
			// The goal loop was in force, real work happened (acted), but the judge
			// never confirmed the goal met — say so plainly instead of implying
			// success. Deliberately NOT gated on returns > 0 (a prior re-feed):
			// reported live, a run whose very FIRST judge call came back unclear
			// (returns never incremented) still finalized on a bare "(done —
			// finished without a written summary...)" with zero indication the
			// goal was left unconfirmed — the exact same problem the give-up
			// paths' goalNotConfirmedNote already solves, just missing here on the
			// normal finalize path. Still gated on acted, though: a run where
			// NOTHING ever happened (the empty-reply case just above) is better
			// described by its own more specific message than by this one.
			goalStalled := a.maxReturns > 0 && acted && !goalMet
			if strings.TrimSpace(clean) == "" {
				// Blank final turn. If the model already did work via tools, say it
				// finished (the changes/output are above); otherwise it produced
				// nothing — flag that instead of ending on silence.
				switch {
				case goalStalled:
					clean = fmt.Sprintf("(goal not confirmed complete — stopped after %d/%d continues; the model finished without further progress. `/goal go` to keep pushing, or switch to a stronger model with /model.)", returns, a.maxReturns)
					if lastMissing != "" {
						clean += "\n\n(still missing: " + lastMissing + ")"
					}
				case assistant.FinishReason == "length":
					// Reported live: a reasoning model spent its ENTIRE output budget
					// thinking and never reached Content — the server cut it off
					// (finish_reason=length) at a prompt size nowhere near the context
					// window, so this has nothing to do with auto-compact. Ranked
					// ABOVE the acted case: a cut-off reply is not a silent success,
					// and the generic advice below is wrong on both counts —
					// rephrasing doesn't move an output cap, and a stronger model
					// burns more of it. Name the actual cap and the actual knob.
					clean = "(the model hit its output limit mid-reply (finish_reason=length) and produced nothing — a reasoning model can spend the whole budget thinking. Raise `max output` in /config.)"
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
			// The normal finalize path never recorded the gap, so a goal that ran
			// out of TTL persisted an EMPTY "what's missing" — the judge's own
			// one-line account of the shortfall, computed and then dropped on the
			// floor. Only the give-up paths were setting it.
			if !goalMet {
				tr.Missing = lastMissing
			}
			tr.PromptTokens = promptTokens
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
			if freed := trimIfNearWindow(msgs, a.contextWindow, sentEst, promptTokens); freed > 0 {
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
		// Forensic trail for the "Stopped — it kept repeating..." message below:
		// reported live, a debug log showed the visible transcript (a failing run,
		// a file write, then a SUCCESSFUL run) followed immediately by that stop
		// message, with no visible nudge line in between — looking like a false
		// trigger. Without this line there was no way to tell whether stuck/nudged
		// carried over from turns earlier than whatever was pasted, or whether the
		// reset-on-progress logic below actually ran. Logged every turn (not just
		// on a stuck/nudge/stop decision) so the counter's whole history is
		// reconstructable, not just its final value.
		allFailed := nErr == len(assistant.ToolCalls)
		slog.Debug("stuck check", a.debugArgs("step", step+1, "all_failed", allFailed, "repeating", repeating, "stuck_before", stuck, "nudged", nudged)...)
		if allFailed || repeating {
			if stuck++; stuck >= a.maxStuckTurns {
				if !nudged {
					msgs = append(msgs, llm.User(stuckNudgeFor(repeating)))
					a.emit("nudge", map[string]any{})
					nudged = true // keep stuck high: one more dud turn now stops it
				} else {
					msg := "Stopped — it kept repeating the same tool calls without progress (or they kept failing) even after a nudge to rethink. Steer it (a different approach), or use a stronger model."
					// Reported live: a goal-pursuing task that got stuck here was
					// unconditionally marked "incomplete" with no assessment at all —
					// the judge only ever ran from the CLEAN finalize path below, so
					// real partial progress before the stall left no trace beyond
					// the bare "incomplete" status. See judgeOnGiveUp.
					// judgeGoal is an ISOLATED side call with no access to msgs — it never
					// saw "above" despite this text once claiming it should assess it.
					// actionsDigest gives it something real to go on instead: the actual
					// files touched / commands run (and why they failed), the same record
					// remember() below stores as cross-task memory.
					met, missing := a.judgeOnGiveUp(ctx, a.acceptanceTarget(goal), returns, hadProductiveTurn, "stuck_stop",
						"(the agent got stuck repeating or failing tool calls before producing a coherent final answer.)"+actionsDigest(msgs)+a.priorGapNote(returns),
						msgs)
					// Reported live: a single bad tool call, repeated until this stop,
					// read as having killed the whole standing goal ("неправильный вызов
					// тула - разорвал гоал") — nothing on screen said the goal itself was
					// still alive and resumable. This fires even with no productive turn
					// at all (hadProductiveTurn=false skips the judge above, but the goal
					// wasn't abandoned — see goalNotConfirmedNote).
					msg += goalNotConfirmedNote(a.maxReturns, met, missing)
					tr.Final, tr.Stopped = msg, true
					tr.Returns = returns
					tr.GoalMet = met
					tr.Missing = missing
					tr.Messages = msgs
					tr.PromptTokens = promptTokens
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
			hadProductiveTurn = true
		}
	}

	tr.Messages = msgs
	tr.Stopped = true // ran out of steps before a clean answer
	tr.Returns = returns
	clean, suggest := splitSuggestion(lastAssistantContent(msgs))
	judgeResult := clean
	if strings.TrimSpace(clean) == "" {
		// Reported live: a task that burns its whole step budget on turns that
		// never produce non-empty content (all reasoning/tool-calls — e.g. a
		// long unproductive exploration) ended in TOTAL SILENCE in the TUI: no
		// message on screen, nothing in the log distinguishing this ending from
		// any other. runOne (the plain/one-shot path) already had its own
		// fallback text for this exact case; the TUI path only ever sees this
		// same emitted "final" event's text and renders nothing when it's
		// empty — so a real, common ending mode was completely invisible
		// there. Setting it here, at the source, means every caller gets it.
		clean = fmt.Sprintf("(no final answer — step budget exhausted after %d steps)", a.maxSteps)
		judgeResult = "(the agent ran out of its step budget without ever writing a final answer.)"
	}
	// Same reasoning as the stuck-stop path (see judgeOnGiveUp): running out
	// of steps is just as much a "give up" as getting stuck repeating calls,
	// and was marked "incomplete" with zero assessment just the same. judgeGoal
	// is an isolated call with no access to msgs, so actionsDigest gives it
	// something real to go on (files touched / commands run) instead of just
	// the model's own possibly-empty final text.
	met, missing := a.judgeOnGiveUp(ctx, a.acceptanceTarget(goal), returns, hadProductiveTurn, "step_exhaustion",
		judgeResult+actionsDigest(msgs)+a.priorGapNote(returns), msgs)
	tr.GoalMet = met
	tr.Missing = missing
	clean += goalNotConfirmedNote(a.maxReturns, met, missing)
	tr.Final = clean
	tr.PromptTokens = promptTokens
	a.emit("final", map[string]any{"text": clean, "suggest": suggest, "exhausted": true})
	a.remember(goal, clean, msgs)
	return tr, nil
}

// clip shortens s for a debug log line (rune-safe).
func clip(s string, n int) string { out, _ := textutil.Clip(s, n); return out }

// clipMarked is clip with the cut made VISIBLE. Evidence handed to the judge
// must never be silently partial: an unmarked excerpt makes a complete report
// look like it is missing the sections that sat past the cut, and hides a
// disqualifying line that came after an otherwise clean prefix. The judge can
// reason about "there is more I am not being shown"; it cannot reason about an
// excerpt it believes is the whole thing.
func clipMarked(s string, n int) string {
	out, truncated := textutil.Clip(s, n)
	if !truncated {
		return out
	}
	return out + fmt.Sprintf("\n…[cut here — %d bytes in total]", len(s))
}

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

// isDoneOnly reports whether a turn's ONLY tool call was the bare "done"
// signal (see tool.NewDone) — the agent loop treats such a turn exactly like a
// plain no-tool-call finalize. Deliberately narrow: done arriving ALONGSIDE
// other real tool calls does not count — those still need normal dispatch and
// a paired tool-result message, which ending the run here can't provide.
func isDoneOnly(calls []llm.ToolCall) bool {
	return len(calls) == 1 && calls[0].Name == "done"
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

// DefaultMaxStuckTurns is how many consecutive unproductive turns (every call
// failed, or the exact same call(s) repeated) trigger the rethink nudge (and,
// if it doesn't help, the stop) when SetMaxStuckTurns hasn't set an explicit
// value. Raised from an earlier, more aggressive 3: reported live, a model
// working through the workspace jail from a few different angles (an
// absolute path, then a couple of cwd variants) got cut off before it had
// room to actually work through the problem — a real backstop against a
// truly looping model still exists (the outer step budget), so a more
// patient default here costs little.
const DefaultMaxStuckTurns = 8

// IsHarnessMessage reports whether a user-role message is something THIS PROGRAM
// injected — a goal re-feed, a nudge, a steer or job note — rather than anything
// the user typed.
//
// Everything the harness pushes at the model arrives as role "user", and
// internal/reflect labelled every one of them "GOAL:" when summarising a run for
// the learning pass. A goal run with four judge rounds therefore showed the
// reflecting model five different "GOAL:" lines, three of which were our own
// scaffolding — and a distilled "fact" could come out reading "You're repeating
// the same tool call(s) without making progress", which is our text, about our
// nudge, stored as a durable truth about the user's project.
func IsHarnessMessage(content string) bool {
	c := strings.TrimSpace(content)
	for _, p := range harnessPrefixes {
		if strings.HasPrefix(c, p) {
			return true
		}
	}
	return false
}

// harnessPrefixes are the opening words of every message the agent injects. Kept
// as prefixes of the real constants (not copies) so a reworded nudge cannot
// silently start reading as user intent again.
var harnessPrefixes = []string{
	goalReturnOpening,
	stuckNudgeRepeat[:40],
	stuckNudgeFailing[:40],
	emptyReplyNudge[:40],
	refusalNudge[:40],
	idleNudge[:40],
	// Delivered background-job results and /btw notes. The doc comment above
	// always claimed these were covered and the list never included them, so a
	// finished sub-agent's entire answer — and every /btw the user dropped
	// mid-run — reached the learning pass labelled as the user's own goal.
	// Matched on the bracketed opening these are built with (cmd/agent/jobs.go).
	"[background job #",
	"[by the way]",
}

// goalReturnOpening is the fixed opening of every goal re-feed (see goalReturn).
const goalReturnOpening = "The GOAL is NOT complete yet"

// degenerateNudge is injected when ONE generation collapsed into repetition and
// the transport aborted it. Deliberately concrete about what happened: a model
// told only "try again" tends to resume the same sentence it was stuck on.
const degenerateNudge = `Your last reply collapsed into repeating the same text over and over, and was cut off. Do not continue or restate it. Look at the LAST tool result you received, decide the single next concrete step from there, and take it with one tool call — or, if the work is finished, say so in one short sentence.`

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

// emptyReplyNudge is the one push-back on a totally blank first reply (no
// content, no tool calls) — a flaky/cut-off response, not a real "nothing to
// do" answer, so it gets one retry before being accepted at face value.
const emptyReplyNudge = `Your last reply was empty — no answer and no tool call. Try again: either call a tool to make progress, or write an actual answer.`

// refusalNudge is the one forceful push-back when a chat model dodges an action
// task — pasting file contents or claiming it can't touch the filesystem —
// instead of using its tools.
const refusalNudge = `You changed nothing — you only described changes or pasted file contents. You are NOT a plain chat model here: you are an agent with working tools in THIS session — file (write/edit/append), run, git — that modify real files on disk, and the user has authorized them. Do not paste file contents, and never say you lack file access (it is false). Make every change now by calling the file tool (write or edit) for each file, then run anything relevant and report the real result.`

// idleNudge is the single push for a model that re-reads the goal after a re-feed
// but then finishes without touching a tool — do the next step, don't just stop.
const idleNudge = `You re-read the GOAL but did nothing this turn — no tool call, no change on disk. Don't stop and don't just describe what to do: take the next concrete step now with your tools (file, run, git) toward the goal, then keep going. Only finish if the goal is genuinely already complete — and if so, state explicitly what's done and why.`

// priorGapNote tells the judge what the LAST attempt at this goal was found to
// be missing, on this run's first judge call only. It is context, not a verdict:
// the judge still decides from the evidence in front of it, but it no longer has
// to rediscover a shortfall that was already established and written down.
// Silent once this run has produced a verdict of its own (returns > 0), which is
// fresher than anything carried over.
func (a *Agent) priorGapNote(returns int) string {
	if returns > 0 || a.goalGap == "" {
		return ""
	}
	return "\n\n(an earlier attempt at this same goal was judged incomplete, with this still missing: " + a.goalGap + " — check whether it has since been done.)"
}

// GoalReturnForTest exposes the goal re-feed's exact wording to tests in other
// packages, so internal/reflect can assert that this program's own injected
// messages are not mistaken for what the user asked for.
func GoalReturnForTest(goal, missing string) string { return goalReturn(goal, missing) }

// goalReturn re-states the goal when the judge finds it unmet, keeping the
// objective in focus (recency) instead of letting it sink under the transcript,
// and naming the gap so the model finishes the remaining work with tools.
func goalReturn(goal, missing string) string {
	s := goalReturnOpening + " — keep going. Do the remaining work now with tools (don't stop early, don't just describe it), and only finish once it's actually done."
	if strings.TrimSpace(missing) != "" {
		s += "\n\nStill missing: " + strings.TrimSpace(missing)
	}
	return s + "\n\nGOAL: " + goal
}

// judgePrompt is what the acceptance checker is asked. It is the run's OWN
// conversation, verbatim, under the judge's system prompt — not a reconstruction
// of it.
//
// It used to be a summary built by hand: the newest tool result per target,
// deduped by file path, each clipped, inside a byte budget. Every fix to the
// judge for a week was another thing that summary had dropped — file contents,
// what a write actually wrote, an edit's replacement text, the previous
// attempt's gap, an emptied file reading as its old contents, a failed call
// reading as a success — and two more were queued: background sub-agent results
// (they arrive as user-role messages, which the scan never looked at) and
// results the context trimmer had already overwritten. That was a worse copy of
// the conversation, rebuilt badly, one bug at a time. The conversation is right
// there.
//
// It is already bounded: trimIfNearWindow keeps the working set inside the
// model's context window, and this is a separate request, so it costs one extra
// turn's worth of prompt per finalize and nothing in the main loop.
//
// The main model's own system prompt is dropped — it carries the agent's
// instructions, its tools and the project's learned notes, none of which the
// judge should be reasoning from — and the judge's own takes its place.
func judgePrompt(criteria, goal, result string, convo []llm.Message) []llm.Message {
	out := []llm.Message{llm.System(judgeSystemWith(criteria))}
	for _, m := range convo {
		if m.Role == "system" {
			continue
		}
		out = append(out, m)
	}
	return append(out, llm.User("Everything above is the agent's working record for this run.\n\nGOAL:\n"+goal+
		"\n\nWHAT THE AGENT SAYS IT DID:\n"+clipMarked(result, 2000)+
		"\n\nNow give your verdict."))
}

// judgeUsage reads a chatter's cumulative token counters when it exposes them,
// for the before/after snapshot around a judge call. A chatter that doesn't
// (a test fake, a future transport) reports nothing and the log simply omits
// the figures.
func judgeUsage(c llm.Chatter) (prompt, completion int) {
	if u, ok := c.(interface{ Usage() (int, int) }); ok {
		return u.Usage()
	}
	return 0, 0
}

// judgeSpend is what one judge call cost, given the counters read before it.
// Cumulative counters mean this is a delta; when the judge shares the main
// client (no dedicated judge connection) nothing else runs between the two
// reads, so the delta is still exactly this call.
func judgeSpend(c llm.Chatter, p0, c0 int) (prompt, completion int) {
	p1, c1 := judgeUsage(c)
	return p1 - p0, c1 - c0
}

// clipTail keeps the LAST n bytes of s, marking the cut — the mirror of clip,
// for text whose conclusion is at the end (see the judge's reasoning log).
func clipTail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}

// judgeVerdict is the acceptance-checker's answer: the goal is met, needs more, or
// the reply couldn't be read as either.
type judgeVerdict int

const (
	judgeUnclear judgeVerdict = iota // couldn't parse a verdict — treated as "not met" (see Run)
	judgeDone                        // explicitly met
	judgeMore                        // explicitly incomplete
)

func (v judgeVerdict) String() string {
	switch v {
	case judgeDone:
		return "done"
	case judgeMore:
		return "more"
	default:
		return "unclear"
	}
}

// judgeOnGiveUp runs one extra judge call when a goal-pursuing run gives up
// WITHOUT a clean finalize to judge in the usual way (the stuck-stop and
// step-exhaustion paths both call this). Reported live: both endings used to
// mark the goal "incomplete" unconditionally, with zero assessment, even
// when real progress happened first — the judge only ever ran from the
// normal "model produced a chat-only reply" finalize path.
//
// Gated exactly like that normal path (goal loop on, returns left) PLUS
// (hadProductiveTurn OR priorGoalProgress): a give-up where NOTHING ever
// actually succeeded — in THIS run or an earlier attempt at the same standing
// goal — isn't worth an extra judge call for; the answer is obviously "not
// done". priorGoalProgress matters because hadProductiveTurn resets to false
// on every Run call: without it, resuming via /goal go after a stuck-stop and
// immediately stumbling again (the same bad tool call, say) would never once
// reach a judge, even though real work happened in an earlier attempt at this
// same goal. reason tags the "goal judge" debug log so it's clear which
// give-up path triggered it.
func (a *Agent) judgeOnGiveUp(ctx context.Context, goal string, returns int, hadProductiveTurn bool, reason, result string, convo []llm.Message) (met bool, missing string) {
	// NOT gated on returns >= maxReturns. That gate was removed from the normal
	// path with the argument that funding an attempt and then refusing to grade
	// it is the one combination that makes no sense — and then left standing
	// here, on the two endings where the run produced the least on its own. With
	// the default TTL of 6, a run that takes its sixth re-feed and then burns its
	// step budget got no assessment at all.
	if a.planMode || a.maxReturns == 0 || !(hadProductiveTurn || a.priorGoalProgress) {
		return false, ""
	}
	verdict, missing := a.judgeGoal(ctx, goal, result, convo)
	slog.Debug("goal judge", a.debugArgs("verdict", verdict, "missing", missing, "return", returns, "of", a.maxReturns, "reason", reason)...)
	met = verdict == judgeDone
	// judgeUnclear carries no real signal (met=false, missing="") — stay silent on
	// it, same as the normal finalize path above, which only ever emits "judge" on
	// a definite DONE. Without this, a give-up run's judge call was invisible on
	// screen: the caller only ever learned the verdict later, via goalState.Missing
	// surfaced through an explicit /goal or /status.
	if met || missing != "" {
		a.emit("judge", map[string]any{"done": met, "missing": missing})
	}
	return met, missing
}

// firstNonEmpty returns the first non-blank of its arguments.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// goalNotConfirmedNote points a give-up ending (stuck-stop or step-exhaustion)
// back at `/goal go` whenever it happened while pursuing an active goal loop
// (maxReturns > 0) and the judge didn't confirm it met — including when the
// judge never even ran (hadProductiveTurn=false skips judgeOnGiveUp entirely,
// but the goal itself wasn't abandoned). Mirrors the wording the normal
// finalize path already uses for its own "goalStalled" case, so a give-up
// doesn't read as the standing goal having silently died — only as unconfirmed
// and resumable.
func goalNotConfirmedNote(maxReturns int, met bool, missing string) string {
	if maxReturns == 0 || met {
		return ""
	}
	if missing != "" {
		return fmt.Sprintf("\n\n(note: the goal was not confirmed complete — missing: %s. `/goal go` to keep pushing.)", missing)
	}
	return "\n\n(note: the goal was not confirmed complete — `/goal go` to keep pushing.)"
}

// judgeGoal asks the model, in a fresh side call (no tools), whether the goal is
// actually met given the work just finished. Returns a tri-state so the caller can
// treat "can't tell" differently from "confirmed done": on a transport error or an
// unparseable reply it returns judgeUnclear, which never marks the goal met (only
// an explicit DONE does) — see Run for what the callers do with it.
//
// One retry on judgeUnclear: reported live, a weak local judge model
// occasionally returns a totally empty reply ("goal judge unparseable
// reply=\"\""), which otherwise permanently stalls the goal as "incomplete" on
// pure noise — a single cheap, tools-free retry costs little and can turn a
// wrongly-unclear verdict into a real one.
func (a *Agent) judgeGoal(ctx context.Context, goal, result string, convo []llm.Message) (judgeVerdict, string) {
	// Reported live: from the TUI, the isolated judge call was indistinguishable
	// from the main model still "thinking" — no visible sign a separate check was
	// even happening, let alone which one. One event per judgeGoal call (not per
	// judgeGoalOnce attempt) — the internal retry on judgeUnclear is plumbing,
	// not something the user needs a second "now judging" line for.
	a.emit("judging", map[string]any{})
	verdict, missing := a.judgeGoalOnce(ctx, goal, result, convo)
	if verdict == judgeUnclear {
		verdict, missing = a.judgeGoalOnce(ctx, goal, result, convo)
	}
	return verdict, missing
}

func (a *Agent) judgeGoalOnce(ctx context.Context, goal, result string, convo []llm.Message) (judgeVerdict, string) {
	// Snapshotted around the call so the log can say how much the judge actually
	// spent. Reported live: nine judge calls in one run, every one
	// "finish_reason=length", and no way to tell whether the cut came from the
	// request's own max_tokens (raise it) or from the context window filling up
	// (raising it changes nothing) — the two are indistinguishable without the
	// completion size.
	p0, c0 := judgeUsage(a.judgeChatter())
	reply, err := a.judgeChatter().Chat(ctx, judgePrompt(a.judgeCriteria, goal, result, convo), judgeTools())
	if err != nil {
		slog.Debug("goal judge failed", a.debugArgs("err", err)...)
		return judgeUnclear, ""
	}
	judgeP, judgeC := judgeSpend(a.judgeChatter(), p0, c0)
	verdict, missing, source := parseJudgeReply(reply)
	// Where the verdict came from. Caught live: a run came back "verdict=done"
	// with the model having produced nothing, and the log could not say whether
	// the judge had committed to that in Content, called done(), or merely left
	// the word somewhere in a draft — which is the difference between the
	// judge being wrong and this code being wrong.
	slog.Debug("goal judge reply", a.debugArgs(
		"verdict", verdict, "source", source, "finish_reason", reply.FinishReason,
		"prompt_tokens", judgeP, "completion_tokens", judgeC)...)
	if verdict == judgeUnclear {
		// Reasoning logged alongside Content: a reasoning-capable local model
		// can stream its whole verdict into Reasoning and leave Content empty
		// — without this, a future occurrence still couldn't tell "genuinely
		// empty" from "reasoned it out but never committed it to Content".
		// The TAIL of the reasoning, not its head: a judge that thinks its way to
		// a verdict puts it at the END, and clipping from the front showed only
		// the restatement of the goal every time — enough to see it was thinking,
		// never enough to see whether it concluded. finish_reason separates "it
		// stopped without answering" from "the server cut it off mid-thought",
		// which need different fixes.
		slog.Debug("goal judge unparseable", a.debugArgs(
			"reply", clip(strings.TrimSpace(reply.Content), 120),
			"finish_reason", reply.FinishReason,
			"reasoning_tail", clipTail(strings.TrimSpace(reply.Reasoning), 300))...)
	}
	return verdict, missing
}

var (
	moreToken = regexp.MustCompile(`(?i)\bMORE\b`) // \b avoids matching "MOREOVER"
	doneToken = regexp.MustCompile(`(?i)\bDONE\b`)
)

// judgeTools offers the judge model a structured OR alternative to writing
// "DONE"/"MORE: ..." as plain text — not a replacement for it. A judge model
// that's itself habituated to always calling a tool (the same instinct behind
// the "TASK COMPLETE"/echo problem this session traced on the MAIN model) can
// express its verdict as a real function call instead of fighting that
// instinct to produce free text, which is exactly the failure mode behind
// "goal judge unparseable reply=\"\"". Either shape works; parseJudgeReply
// checks both.
func judgeTools() []map[string]any {
	return []map[string]any{
		{"type": "function", "function": map[string]any{
			"name":        "done",
			"description": "The goal is fully and verifiably met.",
			"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
		}},
		{"type": "function", "function": map[string]any{
			"name":        "more",
			"description": "The goal is NOT fully met yet.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"missing": map[string]any{"type": "string", "description": "what is still missing, in a few words"},
				},
				"required": []string{"missing"},
			},
		}},
	}
}

// parseJudgeReply reads the judge's answer — a done/more tool call OR the
// plain-text "DONE"/"MORE: ..." convention, whichever the judge model actually
// produced. Tool calls are checked first (when present, they're the more
// reliable shape); free text is the fallback for a model that just answers.
func parseJudgeReply(reply llm.Message) (verdict judgeVerdict, missing, source string) {
	// Every tool call is weighed, not just the first. A reply carrying both
	// done() and more() is not a confirmation that happens to have a caveat
	// attached — it is a judge that has not settled, and taking whichever came
	// first meant the verdict flipped with the order of the array. MORE wins,
	// the same way it does in text.
	sawDone, sawMore, moreMissing := false, false, ""
	for _, tc := range reply.ToolCalls {
		switch tc.Name {
		case "done":
			sawDone = true
		case "more":
			var args struct {
				Missing string `json:"missing"`
			}
			_ = json.Unmarshal([]byte(tc.Arguments), &args)
			if clipped, _ := textutil.Clip(strings.TrimSpace(args.Missing), 160); moreMissing == "" {
				moreMissing = clipped
			}
			sawMore = true
		default:
			// The judge has exactly two tools; anything else is it trying to go
			// and look at something. Reported live: three judge calls in a row
			// came back finish_reason=tool_calls with no verdict, the judge
			// saying in its own words "I need to check the current state of
			// main.go after the edit" — it was asking for evidence it had not
			// been given, and the call was dropped without even naming it.
			slog.Debug("goal judge asked for a tool it doesn't have", "tool", tc.Name, "args", clip(tc.Arguments, 120))
		}
	}
	switch {
	case sawMore:
		return judgeMore, moreMissing, "tool"
	case sawDone:
		// A done() call alongside text that says otherwise is the same
		// unsettled answer in a different shape.
		if v, missing := parseVerdict(reply.Content); v == judgeMore {
			return judgeMore, missing, "tool+content"
		}
		return judgeDone, "", "tool"
	}
	if verdict, missing := parseVerdict(reply.Content); verdict != judgeUnclear {
		return verdict, missing, "content"
	}
	// Nothing usable in Content — look in the thinking. Reported live, eleven
	// judge calls in one run, every single one "reply=\"\" reasoning=<long>":
	// the judge shares the main model's connection AND its reasoning settings,
	// so on a reasoning model it thinks as hard about a yes/no acceptance check
	// as the main model does about the task, spends its whole output budget in
	// reasoning_content, and never reaches Content. Its conclusion was in that
	// text the whole time, logged and thrown away.
	if verdict, missing := parseReasonedVerdict(reply.Reasoning); verdict != judgeUnclear {
		return verdict, missing, "reasoning"
	}
	return judgeUnclear, "", ""
}

// verdictLine matches a line the judge wrote as its actual verdict — "DONE",
// "MORE: no report yet", "**DONE**" — anchored to the start of a line on
// purpose. The loose token match parseVerdict uses is safe on Content (one
// deliberate line) but not on free-running thought, where "the task is not done"
// would read as DONE. A line that BEGINS with the token is a verdict; the same
// word inside a sentence is not.
var verdictLine = regexp.MustCompile("(?i)^[\\s>*_`\"'\\-]*(DONE|MORE)\\b[\\s:.\u2014-]*(.*)$")

// parseReasonedVerdict recovers a MORE — and only a MORE — from the judge's
// reasoning when it never wrote a verdict to Content.
//
// Recovering DONE from thinking was tried and is wrong, caught live within the
// hour: a run where the model read the code, ran it, found the API it depends on
// dead, and was then cut off mid-sentence (finish_reason=length, no content, no
// tool calls) came back "verdict=done … goal_met=true". Nothing had been fixed.
// A line reading "Done." or "Done: read the code" is an utterly ordinary thing
// to find inside free-running thought — a checklist, a note to self — and no
// amount of anchoring distinguishes that from a verdict, because in thinking it
// ISN'T one.
//
// The asymmetry is the point, and it is the same rule the whole goal loop runs
// on: only an EXPLICIT confirmation ends a goal. Content and a done() tool call
// are explicit; a fragment of a draft is not. "Not done yet" read out of a draft
// costs nothing if wrong — the goal keeps going, bounded by the TTL and endable
// by the user. "Done" read out of a draft ends the goal on work never finished.
func parseReasonedVerdict(reasoning string) (judgeVerdict, string) {
	return scanVerdictLines(reasoning, false)
}

// parseVerdict reads the judge's whole reply (not just the first line) and biases
// skeptical: any boundary-delimited MORE means not-done (with the gap that follows
// it); only a MORE-free reply mentioning DONE counts as met. This keeps a rambling
// weak judge from either false-accepting or matching "MOREOVER".
func parseVerdict(s string) (judgeVerdict, string) {
	return scanVerdictLines(s, true)
}

// scanVerdictLines reads a judge reply line by line and returns the verdict it
// COMMITTED to. A line must BEGIN with the token to count: "DONE",
// "MORE: no report", "**DONE**". Any MORE anywhere wins over any DONE.
//
// acceptDone=false is the reasoning variant: a draft may be scanned for MORE,
// never for DONE (see parseReasonedVerdict).
//
// The anchoring is the whole point, and it was missing here far longer than it
// was missing from the draft path. The old rule — "any bare done token, as long
// as no more token appears" — accepted every one of these as a confirmed goal:
//
//	"The task is not done yet."
//	"The agent said \"DONE\", but the report is absent."
//	an evidence block echoed back that happens to contain the word
//
// A judge that says the work is NOT done must never be read as saying it is.
// Anything that isn't a committed verdict is unclear, which keeps the goal
// going — the only safe direction to be wrong in.
func scanVerdictLines(s string, acceptDone bool) (judgeVerdict, string) {
	done := false
	for _, line := range strings.Split(s, "\n") {
		m := verdictLine.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		if strings.EqualFold(m[1], "more") {
			gap := strings.TrimLeft(strings.TrimSpace(m[2]), ":-—. \t")
			clipped, _ := textutil.Clip(gap, 160)
			return judgeMore, clipped
		}
		done = true
	}
	if done && acceptDone {
		return judgeDone, ""
	}
	// Logging moved to the caller (judgeGoalOnce), which also has the reply's
	// Reasoning field and the agent's sub-agent label to tag it with.
	return judgeUnclear, ""
}

// judgeSystem instructs the side-call judge. Tight on purpose: a small local model
// must answer in one parseable line.
const judgeSystem = `You are a strict acceptance checker. Given a GOAL, what an agent did, and EVIDENCE (the actual results of its tool calls), decide if the goal is FULLY met. Answer either by calling a tool — done() if fully and verifiably accomplished, or more(missing: "...") if anything is incomplete, untested, or only described instead of done — or, if you prefer, reply with ONE line of plain text instead, nothing else:
- "DONE" if the goal is fully and verifiably accomplished.
- "MORE: <what is still missing, in a few words>" if anything is incomplete, untested, or only described instead of done.

You are shown the agent's whole working record for this run: what it was asked, every tool call it made, and every result those calls returned.

Judge from the RESULTS. A tool result is what actually happened — file contents written or read back, command output, exit codes. If the results show the work done, that IS the verification: say DONE.

The agent's own words are not evidence. Its narration, its plans, its reasoning and its closing summary are all claims about the work, made by the party being judged. "I fixed it", "the report is complete", "all tests pass" prove nothing on their own — look for the result that backs the claim, and if there isn't one, say so.

You have no tools and cannot inspect anything yourself, so never ask for a check you are unable to perform. "Verify that it works" is not a missing piece. If something is genuinely absent from the record, name that specific thing instead.

Be skeptical of claims with nothing behind them: a change only described, or an artifact never shown, is NOT done.`

// judgeSystemWith appends the user's own acceptance criteria to the judge's
// instruction. Deliberately ADDITIVE and placed last: the GOAL stays the thing
// being accepted, the criteria only say what to look for while deciding, and the
// EVIDENCE stays what the decision is made from. Extra criteria can make the
// judge stricter about a project's own standards — "a change isn't done until
// the tests were actually RUN, not just written" — without any of them being
// able to redefine, replace or dilute the goal itself.
func judgeSystemWith(criteria string) string {
	c := strings.TrimSpace(criteria)
	if c == "" {
		return judgeSystem
	}
	return judgeSystem + "\n\nThe user has additional acceptance criteria for this project. Apply them ON TOP of the goal — they add to what counts as done, they never replace the goal or excuse any part of it:\n" + c
}

// SetJudgeCriteria sets the user's own acceptance criteria, folded into the
// judge's instruction on every judge call (see judgeSystemWith). Empty clears.
func (a *Agent) SetJudgeCriteria(text string) { a.judgeCriteria = text }

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
	for i, e := range errs {
		out[i].IsError = e // carried so evidence can tell an attempt from an accomplishment
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
		action, _, _ := parseArgs(c.Arguments)
		if a.reg.Mutates(c.Name, action) {
			return true
		}
	}
	return false
}

func (a *Agent) execOne(ctx context.Context, c llm.ToolCall) (llm.Message, bool) {
	action, params, argWarn := parseArgs(c.Arguments)
	a.emit("tool_call", map[string]any{"tool": c.Name, "action": action, "params": params})

	// Shadow-mode risk scoring, if anything is listening. Called with the parsed
	// call and nothing else: the observer decides what to score and what to
	// compare it against, so this package stays free of both the classifier and
	// the permission policy. It cannot influence what happens next — the return
	// value is discarded and the call proceeds exactly as it would have.
	if f := a.riskObserver; f != nil {
		f(c.Name, action, params)
	}

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
	// parseArgs could see that params WERE sent but couldn't recover anything from
	// them (see its warn return). Lead with that: the dispatch error can only
	// report the empty params it actually received ("you gave {}"), which reads to
	// the model as a flat contradiction of what it just sent, and it will keep
	// re-sending the same broken shape trying to satisfy it.
	if res.IsError && argWarn != "" {
		content = argWarn + "\n" + content
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
	shown := 0
	// Query with NO cap, and cut to three only after the eligibility tests below.
	// The cap used to be applied by Query, which ranks on loose word overlap,
	// while the real gates here are an exact substring and a matching action —
	// so three lessons that merely shared vocabulary could take all the slots
	// and then each fail the substring test, leaving the model with no hint at
	// all while a genuinely matching lesson sat unexamined.
	for _, p := range a.kb.Query(domain, errText, 0) {
		if shown == 3 {
			break
		}
		if p.ErrorPattern == "" || !strings.Contains(low, strings.ToLower(p.ErrorPattern)) {
			continue
		}
		if !contextMatchesAction(p.Context, action, knownActions) {
			continue
		}
		if b.Len() == 0 {
			b.WriteString("Hints from past runs:")
		}
		shown++
		a.kb.MarkUsed(p) // this lesson was actually surfaced — see KB.MarkUsed
		// A dead-end lesson (knowledge.KindAvoid) must NOT be introduced as
		// something that worked — the whole point of recording one is that the
		// approach kept failing, and "this worked: <the thing that never worked>"
		// would push the model straight back into the loop the lesson exists to
		// break. Lessons stored before Kind existed have Kind == "" and keep the
		// original wording.
		if p.Kind == knowledge.KindAvoid {
			fmt.Fprintf(&b, "\n- when you saw %q while %s, retrying the same call did NOT work — do this instead: %s", p.ErrorPattern, p.Context, p.ProvenFix)
		} else {
			fmt.Fprintf(&b, "\n- when you saw %q while %s, this worked: %s", p.ErrorPattern, p.Context, p.ProvenFix)
		}
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

// debugArgs prepends an "agent" tag to a slog.Debug call's key/value pairs
// when this is a sub-agent (a.label != ""), mirroring the tag emit already
// puts on every UI event. Without this, "run start"/"run end"/"stuck
// check"/"goal judge" were textually indistinguishable in IPS_LOG=debug output
// between the top-level run and any spawned sub-agent — reported live: a real
// debug log's "run end" line couldn't be attributed with certainty to either
// one, because its own "run start" (which would have carried the goal text)
// had scrolled out of view.
func (a *Agent) debugArgs(args ...any) []any {
	if a.label == "" {
		return args
	}
	return append([]any{"agent", a.label}, args...)
}

// parseArgs decodes a tool-call argument string into (action, params), tolerating
// the three shapes weak models emit: params as a nested object, params as a
// JSON-encoded STRING (double-encoded — a very common mistake), or the per-action
// fields flattened at the top level. Action may live at the top level or inside
// a stringified params blob.
//
// The third return value is a diagnostic for the ONE case where params were
// visibly sent but could not be recovered at all (a stringified params blob that
// isn't valid JSON): empty otherwise, and only surfaced to the model when the
// resulting call actually errors — see execOne.
func parseArgs(raw string) (action string, params map[string]any, warn string) {
	m := decodeObj(raw)
	if m == nil {
		return "", map[string]any{}, ""
	}
	action, _ = m["action"].(string)

	// A model can leak its own "<parameter=NAME>value</parameter>" tool-call
	// convention into just the "action" field's own string value, rather than
	// wrapping the whole arguments blob (which decodeObj's fallback above
	// already recovers) — reported live, twice, real transcripts: action=
	// "<parameter=action>\nlist\n</parameter>" (a bare action name) and action=
	// "<parameter=params>\n{\"command\": \"ls -la\", \"cwd\": \"...\"}" (a whole
	// params object, mislabeled but still landing in "action"). Before this
	// fix the entire garbled string became the action value; for run (a
	// single-action domain) garbledActionAsParam then ran the literal tag
	// text as a shell command, producing a real "sh: syntax error" instead of
	// a self-correction, and for file it surfaced as an unknown action.
	if newAction, extra := recoverActionTag(action); extra != nil {
		action = ""
		if a, ok := extra["action"].(string); ok {
			action = a
			delete(extra, "action")
		}
		for k, v := range extra {
			if _, exists := m[k]; !exists {
				m[k] = v
			}
		}
	} else {
		action = newAction
	}

	// A singleton array wrapping one object ("params":[{...}]) has exactly one
	// unambiguous reading — unwrap it before the switch below so it's handled
	// the same as a plain object. Anything else (0 or 2+ elements, or a
	// non-object element) is genuinely ambiguous: leave it alone, which falls
	// through to the flattened branch below and yields empty params — a
	// normal "missing param" dispatch error rather than a guess.
	if arr, ok := m["params"].([]any); ok && len(arr) == 1 {
		if obj, ok := arr[0].(map[string]any); ok {
			m["params"] = obj
		}
	}

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
		return action, p, ""
	case string: // params double-encoded as a JSON string — decode it
		inner := decodeObj(p)
		if inner == nil {
			// The CLOSING half of a leaked "<parameter=NAME>value</parameter>" tag,
			// landing in params with its opening half stripped:
			//
			//	"params": "command>\ngit clone https://github.com/..."
			//	"params": "path>\n/Users/roman220/test-remote/coding-agent-test"
			//
			// There is no JSON in there for decodeObj or recoverActionTag to find —
			// the value is a bare string — so both gave up and the model was told
			// its JSON was malformed, which it was not. Measured in one real run:
			// 11 of 84 turns went to this, the model re-deriving the right shape
			// over three turns and relapsing a turn later, every time.
			//
			// The fourth salvage of this class in this file (recoverActionTag,
			// unwrapEnvelope, recoverTextToolCall) — a model whose chat template
			// and tool-calling format disagree is a normal thing to meet locally.
			if name, val, ok := splitParamTag(p); ok {
				slog.Debug("recovered params from a chat-template fragment", "param", name, "raw", clip(p, 120))
				inner = map[string]any{name: val}
			}
		}
		if inner == nil {
			// Reported live, ten turns in a row: a model sent params as a JSON
			// *string* whose inner JSON didn't parse (a long file body cut off
			// mid-document), so decodeObj failed and the call fell through to the
			// flattened branch below — yielding EMPTY params and a dispatch error
			// reading "file.write needs {path}; you gave {}". The model could see
			// in its own transcript that it HAD sent path and content, read the
			// error as nonsense ("the error says I'm giving {} but I am passing
			// path"), and retried the identical broken shape until the run died.
			// The error wasn't wrong about what arrived — it just described the
			// wreckage instead of the crash. Say what actually happened.
			warn = fmt.Sprintf("note: \"params\" arrived as a JSON string rather than an object, and that string is not valid JSON (truncated or mis-escaped) — nothing could be read out of it, which is why the call below saw no params at all. Send params as a real JSON object, e.g. {\"action\": \"write\", \"params\": {\"path\": \"x.md\", \"content\": \"...\"}}. What arrived was: %s", clip(p, 200))
			break
		}
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
		return action, inner, ""
	}

	// Flattened: everything except action/params is a param.
	params = map[string]any{}
	for k, v := range m {
		if k != "action" && k != "params" {
			params[k] = v
		}
	}
	return action, params, warn
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

// strayParamTagRe matches a parameter NAME that arrived glued to its value
// instead of as a JSON key, in either shape a leaked tool-call template leaves
// behind. paramTagRe (below) handles the intact "<parameter=NAME>" tag; these
// are the mangled siblings, with no "<parameter=" prefix left to key on:
//
//	"command>\ngit clone …"   the tag's closing bracket survived
//	"command: git clone …"    only the separator survived
//	"path: /Users/…/main.go"
//
// Both were measured in real runs, a day apart, from the same task — so
// matching only one of them fixes the log you happen to be holding.
//
// The name must start with a LETTER, so a value leading with a shell redirect
// ("2>&1 …") is not read as a parameter called "2". After a COLON the separator
// must be followed by whitespace, which is what keeps "https://example.com" from
// being read as a parameter called "https" — after ">" it need not be, since a
// ">" cannot occur in a parameter name at all.
var strayParamTagRe = regexp.MustCompile(`(?s)^\s*([A-Za-z][A-Za-z0-9_.-]*)(?:>\s*|:\s+)(.*)$`)

// splitParamTag pulls (name, value) out of that fragment. Only ever consulted
// for a params string that already failed to parse as JSON, so it cannot
// reinterpret a call that was going to work.
func splitParamTag(s string) (name, value string, ok bool) {
	m := strayParamTagRe.FindStringSubmatch(s)
	if m == nil {
		return "", "", false
	}
	v := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(m[2]), "</parameter>"))
	return m[1], v, v != ""
}

// textToolCallRe matches a whole tool call a model wrote as TEXT instead of
// emitting it through the tool-calling protocol: the function name, and the body
// that carries its parameters.
//
// Reported live, and it cost a whole run: the model decided to call file.list,
// wrote the call into its reasoning as
//
//	<tool_call><function=file><parameter=action>list</parameter>
//	<parameter=params>{}</parameter></function></tool_call>
//
// and returned finish_reason=stop with empty Content and no ToolCalls. From our
// side that is an empty reply, so it got the empty-reply nudge, repeated itself
// verbatim, and the run ended having done nothing — while the user watched a
// perfectly well-formed intent being thrown away twice.
//
// This is the third salvage of the same class already in this file
// (recoverActionTag for a tag inside the arguments, unwrapEnvelope for a whole
// chat envelope emitted as content). A model whose chat template and
// tool-calling format disagree is a normal thing to meet locally.
var (
	textToolCallRe = regexp.MustCompile(`(?s)<function=([A-Za-z0-9_.-]+)\s*>(.*?)</function>`)
	textParamRe    = regexp.MustCompile(`(?s)<parameter=([A-Za-z0-9_.-]+)\s*>(.*?)</parameter>`)
)

// recoverTextToolCall rebuilds a tool call a model wrote as text.
//
// Only ever consulted for a reply that is otherwise EMPTY — no content, no tool
// calls — which is what makes it safe: a model discussing a tool call in prose
// still has prose, and nothing here touches it. When the alternative is "the
// model said nothing at all", acting on the intent it plainly expressed is
// strictly better than discarding it.
func recoverTextToolCall(reply llm.Message) []llm.ToolCall {
	for _, src := range []string{reply.Content, reply.Reasoning} {
		m := textToolCallRe.FindStringSubmatch(src)
		if m == nil {
			continue
		}
		args := map[string]any{}
		for _, p := range textParamRe.FindAllStringSubmatch(m[2], -1) {
			name, raw := p[1], strings.TrimSpace(p[2])
			// A parameter whose value is itself JSON (params={...}) has to go in
			// as JSON, not as a string — parseArgs would otherwise see the
			// double-encoded shape it already has to work around.
			if obj := decodeObjStrict(raw); obj != nil {
				args[name] = obj
				continue
			}
			args[name] = raw
		}
		if len(args) == 0 {
			// A named function with no readable parameters is still a better
			// guess than nothing: dispatch will name what's missing.
			args = map[string]any{}
		}
		encoded, err := json.Marshal(args)
		if err != nil {
			return nil
		}
		return []llm.ToolCall{{ID: "recovered-1", Name: m[1], Arguments: string(encoded)}}
	}
	return nil
}

// paramTagRe matches a model's own "<parameter=NAME>value</parameter>" tool-
// call convention (see recoverActionTag). The closing tag is optional — a
// model sometimes truncates it — so the capture group runs to end-of-string
// either way.
var paramTagRe = regexp.MustCompile(`(?s)^<parameter=[^>]*>\s*(.*?)\s*(?:</parameter>\s*)?$`)

// recoverActionTag unwraps a "<parameter=NAME>value</parameter>" tag found in
// the "action" field's own value (as opposed to wrapping the whole arguments
// blob, which decodeObj already handles). If the unwrapped value is itself a
// JSON object, it was meant as params, not an action name — those keys are
// returned in extra for the caller to fold in (its own "action" key, if any,
// wins). Otherwise the trimmed inner text is the real action name, returned
// as newAction with extra == nil. A string not shaped like this tag at all is
// returned unchanged with extra == nil.
func recoverActionTag(action string) (newAction string, extra map[string]any) {
	trimmed := strings.TrimSpace(action)
	if !strings.HasPrefix(trimmed, "<parameter=") {
		return action, nil
	}
	m := paramTagRe.FindStringSubmatch(trimmed)
	if m == nil {
		return action, nil
	}
	inner := strings.TrimSpace(m[1])
	if obj := decodeObj(inner); obj != nil {
		return "", obj
	}
	return inner, nil
}

func lastAssistantContent(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" && strings.TrimSpace(msgs[i].Content) != "" {
			return msgs[i].Content
		}
	}
	return ""
}
