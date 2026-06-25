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

const reflectPrompt = `You review a finished run by a tool-using agent and extract durable lessons for next time.
Return ONLY a JSON array of objects with keys: "domain", "error_pattern", "context", "proven_fix".
- "domain" must be one of: file, run, web, calc.
- Include a lesson ONLY where the agent hit an error AND a later action fixed it.
- "error_pattern" is a short substring of the error; "proven_fix" is the concrete fix that worked.
- Keep lessons environment-general (tool/OS behaviour). EXCLUDE anything specific to this project, path, or task.
- If there is nothing durable to learn, return exactly [].`

// Reflect returns the durable pitfalls learned from t. A turn with no tool use
// (a plain chat exchange) has nothing to learn, so it skips the model call
// entirely — no point making a small model reason over an empty run.
func (r *Reflector) Reflect(ctx context.Context, t agent.Transcript) ([]knowledge.Pitfall, error) {
	if !usedTools(t) {
		return nil, nil
	}
	summary := summarize(t)
	if strings.TrimSpace(summary) == "" {
		return nil, nil
	}
	reply, err := r.LLM.Chat(ctx, []llm.Message{
		llm.System(reflectPrompt),
		llm.User(summary),
	}, nil)
	if err != nil {
		return nil, &ReflectionError{Err: err}
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
	if len(s) > 500 {
		s = s[:500] + "…"
	}
	return s
}

func parseLessons(content string) []knowledge.Pitfall {
	for _, candidate := range jsonArrayCandidates(content) {
		var raw []struct {
			Domain       string `json:"domain"`
			ErrorPattern string `json:"error_pattern"`
			Context      string `json:"context"`
			ProvenFix    string `json:"proven_fix"`
		}
		if err := json.Unmarshal([]byte(candidate), &raw); err != nil {
			continue
		}
		var out []knowledge.Pitfall
		for _, r := range raw {
			if strings.TrimSpace(r.Domain) == "" || strings.TrimSpace(r.ProvenFix) == "" {
				continue
			}
			out = append(out, knowledge.Pitfall{
				Domain:       r.Domain,
				ErrorPattern: r.ErrorPattern,
				Context:      r.Context,
				ProvenFix:    r.ProvenFix,
			})
		}
		return out // first candidate that parses as an array wins
	}
	return nil
}

// jsonArrayCandidates returns every substring of s that starts at a '[' and
// decodes as a complete JSON array, in order. This tolerates prose around the
// array — and prose that itself contains brackets — unlike a naive
// first-'['-to-last-']' slice.
func jsonArrayCandidates(s string) []string {
	var out []string
	for i := 0; i < len(s); i++ {
		if s[i] != '[' {
			continue
		}
		var raw json.RawMessage
		if err := json.NewDecoder(strings.NewReader(s[i:])).Decode(&raw); err == nil && len(raw) > 0 && raw[0] == '[' {
			out = append(out, string(raw))
		}
	}
	return out
}
