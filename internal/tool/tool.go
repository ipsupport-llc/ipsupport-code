// Package tool defines the fat-tool abstraction the agent exposes to the model:
// one tool per domain, each taking an (action, params) pair. It also holds the
// shared helpers — argument validation, parameter coercion, and the typed
// ToolError used for host-level failures.
package tool

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// Result is a tool's outcome as the model sees it. Content is always model-
// facing text. IsError marks a failure the model should react to. Err is set
// only for host/infra failures (a *ToolError) so the host can log the typed
// cause while the model still just reads Content.
type Result struct {
	Content string
	IsError bool
	Err     error
	Diff    string // optional unified diff to display (set by file edits)
}

// Ok is a successful result.
func Ok(s string) Result { return Result{Content: s} }

// Err is a model-recoverable failure (bad input, denied, not found) — text only.
func Err(s string) Result { return Result{Content: s, IsError: true} }

// Fail is a host/infra failure: the model sees msg, the host gets a typed
// *ToolError via Result.Err.
func Fail(tool, action, msg string, cause error) Result {
	te := &ToolError{Tool: tool, Action: action, Err: cause}
	if msg == "" {
		msg = te.Error()
	}
	return Result{Content: msg, IsError: true, Err: te}
}

// ToolError wraps a host-level tool failure.
type ToolError struct {
	Tool   string
	Action string
	Err    error
}

func (e *ToolError) Error() string {
	return fmt.Sprintf("tool %s/%s: %v", e.Tool, e.Action, e.Err)
}
func (e *ToolError) Unwrap() error { return e.Err }

// Approver answers an interactive permission prompt for a policy "ask" decision.
type Approver interface {
	Approve(kind, detail string) bool
}

// Tool is one fat domain tool.
type Tool interface {
	Name() string
	Description() string
	Actions() []string
	Call(ctx context.Context, action string, params map[string]any) Result
	// Mutates reports whether an action changes state (writes a file, runs a
	// command, alters git). Plan mode blocks these; read-only actions stay open.
	Mutates(action string) bool
}

func isEmpty(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(x) == ""
	case []any:
		return len(x) == 0
	}
	return false
}

// Str reads a string param (JSON values arrive as any), coercing non-strings via
// fmt. Missing → "".
func Str(p map[string]any, key string) string {
	v, ok := p[key]
	if !ok || v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

// Int reads an integer param, tolerating JSON float64 and numeric strings.
func Int(p map[string]any, key string, def int) int {
	switch v := p[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return def
}

// Bool reads a boolean param, tolerating the string and numeric forms weak
// models emit ("true"/"yes"/"1", or a non-zero number) instead of a real JSON
// bool. Missing or unrecognized → false.
func Bool(p map[string]any, key string) bool {
	switch v := p[key].(type) {
	case bool:
		return v
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "true", "yes", "y", "1":
			return true
		}
	case float64:
		return v != 0
	}
	return false
}
