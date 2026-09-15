// Package reflect runs the post-task learning pass: it asks the model to distill
// durable lessons from a finished transcript and returns them as pitfalls for
// the knowledge base. Parse failures yield no lessons (not an error); only a
// transport failure surfaces as ReflectionError.
package reflect

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// ReflectionError wraps a host-level failure of the reflection pass (the model
// call itself failing).
type ReflectionError struct{ Err error }

func (e *ReflectionError) Error() string { return "reflection: " + e.Err.Error() }
func (e *ReflectionError) Unwrap() error { return e.Err }

// Reflector distills lessons from a transcript using a Chatter. Lite uses a
// simpler, facts-only prompt — for a small local model that loops on the full
// two-part ask.
type Reflector struct {
	LLM  llm.Chatter
	Lite bool
}

// New constructs a Reflector.
func New(l llm.Chatter) *Reflector { return &Reflector{LLM: l} }

// Lessons is what a reflection pass distills: general tool pitfalls (saved to
// the workspace's lesson store) and durable facts about THIS project (saved
// alongside them, folded into the prompt next time). Both are per workspace and
// both are wiped by /clear.
type Lessons struct {
	Pitfalls []knowledge.Pitfall
	Facts    []string
	// Parsed reports whether a JSON object was actually read out of the model's
	// reply. It separates "the model looked and found nothing to learn" from "we
	// could not read what it said" — two different failures with two different
	// fixes, which were indistinguishable because both surfaced as an empty
	// Lessons and neither was logged at all.
	Parsed bool
	// Reply is what came back when nothing could be parsed, for the log. Empty
	// once Parsed is true.
	Reply string
}

const reflectPrompt = `You review a finished run by a tool-using agent and extract two things for next time, as ONE JSON object:
{"pitfalls": [...], "facts": [...]}

"pitfalls" — general lessons about USING THE TOOLS, each: {"domain" (file|run|git|web|calc|agent|mcp|skill), "kind" ("fix" or "avoid"), "error_pattern", "context", "proven_fix"}.

  kind "fix" — an error was hit and a later action actually fixed it. "proven_fix" = what worked.
  kind "avoid" — an error was hit and the SAME approach was tried again and kept failing, right up to the end of the run. "proven_fix" = what to do DIFFERENTLY next time. Never write the failing approach itself as the advice.

  "error_pattern" — copy a substring out of the error text itself, long enough that only this KIND of failure contains it. Never a generic wrapper like an exit code alone ("exit 1"), which every failed command has regardless of cause.
  "context" — the tool action it happened during, e.g. "file: write".

  HARD RULE for "error_pattern" and "proven_fix": no value taken from this run — no file path, filename, directory, project name, URL, or command argument. A lesson keyed on those never matches again, and one quoting them actively misleads a future run in a different project. Write the SHAPE of the problem, not this instance of it.

  Emit a pitfall only for a genuine tool-usage failure. Use [] if none qualifies.

"facts" — short, durable, reusable facts about THIS project worth remembering next time: build/test/run commands, where things live, conventions, gotchas. Solid reusable facts only, not one-off details.

Use [] for an empty list. Return ONLY the JSON object.`

// reflectFactsLite and reflectPitfallsLite split the full two-part ask into two
// single-purpose calls for a small model. Lite used to mean facts ONLY, on the
// reasoning that a weak model loops on "give me pitfalls AND facts in one JSON"
// — true about the combined ask, but the wrong conclusion: it silently removed
// the lesson mechanism from exactly the models it exists for. A weak local model
// repeating the same broken tool call every run is the whole reason the store is
// there; a strong hosted one rarely needs it. Two short asks, each with one
// output shape, keep the weak model on rails without giving up half the output.
const reflectFactsLite = `From the finished agent run below, list a few short, durable facts about THIS project worth remembering next time — build/test/run commands, where files live, conventions. Reply with ONLY this JSON, nothing else: {"facts": ["...", "..."]}. Use {"facts": []} if there's nothing solid. Do not explain.`

const reflectPitfallsLite = `The agent run below may have hit tool errors. Report what to do differently next time. Reply with ONLY this JSON, nothing else:
{"pitfalls": [{"domain": "...", "kind": "...", "error_pattern": "...", "context": "...", "proven_fix": "..."}]}

"domain" is the tool that failed: file, run, git, web, calc, agent, mcp or skill.
"kind" is "fix" if an error was hit and a later action fixed it — "proven_fix" is what worked.
"kind" is "avoid" if the same thing was tried again and kept failing — "proven_fix" is what to do INSTEAD. Never repeat the failing approach as the advice.
"error_pattern" is a few words copied from the error text itself.
"context" is the action it happened during, like "file: write".
Never put a file path, filename, directory or project name in "error_pattern" or "proven_fix" — describe the shape of the problem, not this one case.

Use {"pitfalls": []} if the run hit no tool errors. Do not explain.`

// reflectStuckPrompt is used when a run was STOPPED by the harness — the
// stuck-stop or the step budget — rather than finishing.
//
// Those runs used to be excluded from reflection entirely (`if !tr.Stopped`),
// which made the "avoid" lesson kind unreachable by construction: it exists for
// "the same approach was tried again and kept failing right up to the end of the
// run", and that is precisely the transcript that was thrown away. Any run that
// ended cleanly enough to be reflected on is one where the model got PAST the
// failure — a fix.
//
// Deliberately narrower than the normal pass: one avoid lesson at most, and no
// facts at all. A run that ended in repeated failure is the least trustworthy
// source of "durable truths about this project" in the system, and the lesson it
// does produce will be injected exactly when the model is next struggling — so
// it must be about the call that demonstrably kept failing, and nothing else.
const reflectStuckPrompt = `The agent run below was STOPPED by the harness: it kept repeating failing tool calls, or ran out of steps. It did NOT recover.

Report at most ONE lesson, about a tool call that visibly failed more than once here. Reply with ONLY this JSON, nothing else:
{"pitfalls": [{"domain": "...", "kind": "avoid", "error_pattern": "...", "context": "...", "proven_fix": "..."}]}

"domain" is the tool that failed: file, run, git, web, calc, agent, mcp or skill.
"error_pattern" is a few words copied from the error text itself.
"context" is the action it happened during, like "file: write".
"proven_fix" is what to do DIFFERENTLY. Never repeat the failing approach as the advice.
Never put a file path, filename, directory or project name in "error_pattern" or "proven_fix".

Only report what the transcript SHOWS failing repeatedly. If nothing failed more than once, or you cannot tell why it failed, reply {"pitfalls": []} — a confident guess about a failure you did not diagnose is worse than no lesson. Do not explain.`

// Reflect distills lessons from t. A turn with no tool use (a plain chat) has
// nothing to learn, so it skips the model call — no point making a small model
// reason over an empty run.
//
// Lite runs two narrow calls instead of one combined one (see the prompts
// above); a failure of either is not fatal to the other, since half the lessons
// beats none.
func (r *Reflector) Reflect(ctx context.Context, t agent.Transcript) (Lessons, error) {
	if !usedTools(t) {
		return Lessons{}, nil
	}
	summary := summarize(t)
	if strings.TrimSpace(summary) == "" {
		return Lessons{}, nil
	}
	// A harness-stopped run gets the narrow pass (see reflectStuckPrompt) on any
	// provider: what it has to teach is one dead end, not project facts.
	if t.Stopped {
		reply, err := r.LLM.Chat(ctx, []llm.Message{
			llm.System(reflectStuckPrompt),
			llm.User(summary),
		}, nil)
		if err != nil {
			return Lessons{}, &ReflectionError{Err: err}
		}
		out := parseLessons(reply.Content)
		out.Facts = nil // never from a run that ended in failure
		if len(out.Pitfalls) > 1 {
			out.Pitfalls = out.Pitfalls[:1]
		}
		for i := range out.Pitfalls {
			// The prompt asks for "avoid" and the shape only makes sense that
			// way here: nothing in this transcript was proven to work.
			out.Pitfalls[i].Kind = knowledge.KindAvoid
		}
		return out, nil
	}
	if r.Lite {
		return r.reflectLite(ctx, summary)
	}
	reply, err := r.LLM.Chat(ctx, []llm.Message{
		llm.System(reflectPrompt),
		llm.User(summary),
	}, nil)
	if err != nil {
		return Lessons{}, &ReflectionError{Err: err}
	}
	return parseLessons(reply.Content), nil
}

// reflectLite asks for facts and pitfalls separately. Only a failure of BOTH
// calls surfaces as an error: one weak-model reply that comes back unusable
// shouldn't throw away the other half that parsed fine.
func (r *Reflector) reflectLite(ctx context.Context, summary string) (Lessons, error) {
	var out Lessons
	factsReply, factsErr := r.LLM.Chat(ctx, []llm.Message{
		llm.System(reflectFactsLite),
		llm.User(summary),
	}, nil)
	if factsErr == nil {
		l := parseLessons(factsReply.Content)
		out.Facts, out.Parsed, out.Reply = l.Facts, l.Parsed, l.Reply
	}
	pitReply, pitErr := r.LLM.Chat(ctx, []llm.Message{
		llm.System(reflectPitfallsLite),
		llm.User(summary),
	}, nil)
	if pitErr == nil {
		l := parseLessons(pitReply.Content)
		out.Pitfalls = l.Pitfalls
		// Either half being readable counts as understood; the unread one's tail
		// is what the log needs to show.
		out.Parsed = out.Parsed || l.Parsed
		if !l.Parsed && out.Reply == "" {
			out.Reply = l.Reply
		}
	}
	if factsErr != nil && pitErr != nil {
		return Lessons{}, &ReflectionError{Err: factsErr}
	}
	if factsErr != nil || pitErr != nil {
		slog.Debug("reflect lite half failed", "facts_err", factsErr, "pitfalls_err", pitErr)
	}
	return out, nil
}

// summarize compacts a transcript into the error→recovery→outcome shape the
// reflection prompt expects.
// summaryBudget caps the whole transcript summary. Reported by review: this had
// no total bound at all while judgeEvidence capped itself — a 40-step run
// produced an ~40k-character prompt, which on the small local model the lite
// path targets overruns the window, returns unparseable output, and makes the
// entire pass (two calls) buy nothing. The END of a run is kept: that is where
// the recovery, the outcome and the last state are.
const summaryBudget = 12000

func summarize(t agent.Transcript) string {
	var b strings.Builder
	for _, m := range t.Messages {
		switch m.Role {
		case "user":
			// Only what the USER actually asked is the goal. Everything this
			// program injects arrives as role "user" too — goal re-feeds and
			// nudges — and labelling those "GOAL:" told the learning pass that
			// our own scaffolding was the user's intent, which it could then
			// distill into a "fact" about the project.
			if agent.IsHarnessMessage(m.Content) {
				fmt.Fprintf(&b, "HARNESS (not the user): %s\n", oneLine(m.Content))
				continue
			}
			fmt.Fprintf(&b, "GOAL: %s\n", oneLine(m.Content))
		case "assistant":
			if len(m.ToolCalls) > 0 {
				for _, tc := range m.ToolCalls {
					fmt.Fprintf(&b, "CALL %s: %s\n", tc.Name, oneLine(tc.Arguments))
				}
			} else if strings.TrimSpace(m.Content) != "" {
				fmt.Fprintf(&b, "ASSISTANT: %s\n", oneLine(m.Content))
			}
		case "tool":
			fmt.Fprintf(&b, "RESULT(%s): %s\n", m.Name, oneLine(m.Content))
		}
	}
	if t.Final != "" {
		fmt.Fprintf(&b, "FINAL: %s\n", oneLine(t.Final))
	}
	// The judge's own verdict is the sharpest signal in the run and the learning
	// pass never saw it: the transcript it was handed carried GoalMet, Returns
	// and Missing on the very struct, all ignored. A run that took four judge
	// rounds to be accepted, or was never accepted at all, teaches something
	// quite different from a one-shot success.
	if t.Returns > 0 || t.GoalMet || strings.TrimSpace(t.Missing) != "" {
		fmt.Fprintf(&b, "JUDGE: goal met=%v after %d re-feed(s)", t.GoalMet, t.Returns)
		if m := strings.TrimSpace(t.Missing); m != "" {
			fmt.Fprintf(&b, "; last unmet: %s", oneLine(m))
		}
		b.WriteString("\n")
	}
	return clipTail(b.String(), summaryBudget)
}

// clipTail keeps the LAST n bytes, marking the cut — the end of a run is where
// the recovery and the outcome are, so an over-long transcript loses its
// exploratory beginning rather than its conclusion.
func clipTail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "…[earlier steps omitted]\n" + s[len(s)-n:]
}

// usedTools reports whether the run actually called any tool.
func usedTools(t agent.Transcript) bool {
	for _, m := range t.Messages {
		if m.Role == "tool" {
			return true
		}
	}
	return false
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ⏎ ")
	if clipped, truncated := textutil.Clip(s, 500); truncated {
		s = clipped + "…"
	}
	return s
}

// validDomain is the set of tool names a lesson can be keyed on — the KB only
// surfaces a pitfall when its domain equals the failing tool's name, so a lesson
// with any other domain (e.g. "shell", "general") is stored but never retrieved.
var validDomain = map[string]bool{
	"file": true, "run": true, "git": true, "web": true,
	"calc": true, "agent": true, "mcp": true, "skill": true,
}

func parseLessons(content string) Lessons {
	parsedEmpty := false
	for _, candidate := range jsonObjectCandidates(content) {
		var raw struct {
			Pitfalls []struct {
				Domain       string `json:"domain"`
				Kind         string `json:"kind"`
				ErrorPattern string `json:"error_pattern"`
				Context      string `json:"context"`
				ProvenFix    string `json:"proven_fix"`
			} `json:"pitfalls"`
			Facts []string `json:"facts"`
		}
		if err := json.Unmarshal([]byte(candidate), &raw); err != nil {
			continue
		}
		var out Lessons
		for _, p := range raw.Pitfalls {
			domain := strings.ToLower(strings.TrimSpace(p.Domain))
			pattern := strings.TrimSpace(p.ErrorPattern)
			if domain == "" || strings.TrimSpace(p.ProvenFix) == "" {
				continue
			}
			if !validDomain[domain] {
				continue // a domain the KB can never match on is dead weight that still ages toward pruning
			}
			if knowledge.IsGenericErrorPattern(pattern) {
				continue // "exit N" alone can't discriminate this failure from any other
			}
			// The prompt already forbids carrying a value from this run into a
			// lesson, and a model ignored it — the resulting entry quoted another
			// project's file path and misled a later, unrelated run (see
			// knowledge.IsProjectSpecific). Enforce it here rather than trusting
			// the instruction, and say what was dropped: a silently rejected
			// lesson is indistinguishable from reflection never producing one.
			if knowledge.IsProjectSpecific(pattern) || knowledge.IsProjectSpecific(p.ProvenFix) {
				slog.Debug("lesson rejected", "reason", "project-specific", "domain", domain,
					"error_pattern", pattern, "proven_fix", p.ProvenFix)
				continue
			}
			out.Pitfalls = append(out.Pitfalls, knowledge.Pitfall{
				Domain: domain, Kind: normalizeKind(p.Kind), ErrorPattern: pattern,
				Context: p.Context, ProvenFix: p.ProvenFix,
			})
		}
		for _, f := range raw.Facts {
			if s := strings.TrimSpace(f); s != "" {
				out.Facts = append(out.Facts, s)
			}
		}
		// Keep scanning past a decoy/empty object (e.g. a format-example `{}` the
		// model emits before the real one) — only a candidate with actual content wins.
		if len(out.Pitfalls) > 0 || len(out.Facts) > 0 {
			out.Parsed = true
			return out
		}
		// A well-formed but empty object IS an answer: the model looked and found
		// nothing. Remember that we understood it, in case no later candidate
		// carries content.
		parsedEmpty = true
	}
	return Lessons{Parsed: parsedEmpty, Reply: unparsedReply(parsedEmpty, content)}
}

// unparsedReply is the tail of a reply nothing could be read out of, for the
// log — empty when the reply WAS understood and simply said there was nothing.
func unparsedReply(parsed bool, content string) string {
	if parsed {
		return ""
	}
	return clipTail(strings.TrimSpace(content), 200)
}

// normalizeKind maps the prompt's "kind" onto Pitfall.Kind. Only an explicit
// "avoid" becomes KindAvoid; "fix", an omitted field (a model following the old
// prompt, or a lesson replayed from an older store) and any unrecognized value
// all mean the default "a fix that worked". Erring that way is deliberate: a
// dead end mislabeled as a fix reads as bad advice, but a fix mislabeled as a
// dead end tells the model to stop doing the thing that actually works.
func normalizeKind(kind string) string {
	if strings.EqualFold(strings.TrimSpace(kind), knowledge.KindAvoid) {
		return knowledge.KindAvoid
	}
	return ""
}

// jsonObjectCandidates returns every substring of s that starts at a '{' and
// decodes as a complete JSON object, in order — tolerating prose around it.
func jsonObjectCandidates(s string) []string {
	var out []string
	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		var raw json.RawMessage
		if err := json.NewDecoder(strings.NewReader(s[i:])).Decode(&raw); err == nil && len(raw) > 0 && raw[0] == '{' {
			out = append(out, string(raw))
		}
	}
	return out
}
