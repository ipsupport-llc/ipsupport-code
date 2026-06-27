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

// Reflector distills lessons from a transcript using a Chatter.
type Reflector struct{ LLM llm.Chatter }

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
	reply, err := r.LLM.Chat(ctx, []llm.Message{
		llm.System(reflectPrompt),
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
			if strings.TrimSpace(p.Domain) == "" || strings.TrimSpace(p.ProvenFix) == "" {
				continue
			}
			out.Pitfalls = append(out.Pitfalls, knowledge.Pitfall{
				Domain: p.Domain, ErrorPattern: p.ErrorPattern, Context: p.Context, ProvenFix: p.ProvenFix,
			})
		}
		for _, f := range raw.Facts {
			if s := strings.TrimSpace(f); s != "" {
				out.Facts = append(out.Facts, s)
			}
		}
		return out // first candidate that parses as the object wins
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
