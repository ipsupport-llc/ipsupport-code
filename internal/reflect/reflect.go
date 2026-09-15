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

// reflectPromptLite is the small-model variant: facts only (the more useful
// half), terse, to avoid the looping a weak model does on the full two-part ask.
const reflectPromptLite = `From the finished agent run below, list a few short, durable facts about THIS project worth remembering next time — build/test/run commands, where files live, conventions. Reply with ONLY this JSON, nothing else: {"facts": ["...", "..."]}. Use {"facts": []} if there's nothing solid. Do not explain.`

// Reflect distills lessons from t. A turn with no tool use (a plain chat) has
// nothing to learn, so it skips the model call — no point making a small model
// reason over an empty run.
func (r *Reflector) Reflect(ctx context.Context, t agent.Transcript) (Lessons, error) {
	if !usedTools(t) {
		return Lessons{}, nil
	}
	summary := summarize(t)
	if strings.TrimSpace(summary) == "" {
		return Lessons{}, nil
	}
	prompt := reflectPrompt
	if r.Lite {
		prompt = reflectPromptLite
	}
	reply, err := r.LLM.Chat(ctx, []llm.Message{
		llm.System(prompt),
		llm.User(summary),
	}, nil)
	if err != nil {
		return Lessons{}, &ReflectionError{Err: err}
	}
	return parseLessons(reply.Content), nil
}

// summarize compacts a transcript into the error→recovery→outcome shape the
// reflection prompt expects.
func summarize(t agent.Transcript) string {
	var b strings.Builder
	for _, m := range t.Messages {
		switch m.Role {
		case "user":
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
	return b.String()
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
			return out
		}
	}
	return Lessons{}
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
