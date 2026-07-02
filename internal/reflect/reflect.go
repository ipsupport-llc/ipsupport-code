// Package reflect runs the post-task learning pass: it asks the model to distill
// durable lessons from a finished transcript and returns them as pitfalls for
// the knowledge base. Parse failures yield no lessons (not an error); only a
// transport failure surfaces as ReflectionError.
package reflect

import (
	"context"
	"encoding/json"
	"fmt"
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

// Lessons is what a reflection pass distills: environment-general tool pitfalls
// (saved to the global knowledge base) and durable facts about THIS project
// (saved per workspace, folded into the prompt next time).
type Lessons struct {
	Pitfalls []knowledge.Pitfall
	Facts    []string
}

const reflectPrompt = `You review a finished run by a tool-using agent and extract two things for next time, as ONE JSON object:
{"pitfalls": [...], "facts": [...]}

"pitfalls" — environment-general tool lessons. Each: {"domain" (file|run|web|calc), "error_pattern" (short substring of the error), "context", "proven_fix" (the concrete fix that worked)}. Include ONE only where an error was hit AND a later action fixed it. EXCLUDE anything specific to this project/path.

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
			if domain == "" || strings.TrimSpace(p.ProvenFix) == "" {
				continue
			}
			if !validDomain[domain] {
				continue // a domain the KB can never match on is dead weight that still ages toward pruning
			}
			out.Pitfalls = append(out.Pitfalls, knowledge.Pitfall{
				Domain: domain, ErrorPattern: p.ErrorPattern, Context: p.Context, ProvenFix: p.ProvenFix,
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
