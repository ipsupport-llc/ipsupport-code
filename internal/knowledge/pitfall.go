// Package knowledge is the agent's persistent memory: a JSON-backed store of
// Pitfalls (lessons) it has learned. It is the mechanism that lets the agent
// actually remember experience across restarts — the store is read on startup
// and written after each task's reflection pass.
package knowledge

import (
	"regexp"
	"strings"
)

// Pitfall is one durable lesson: when ErrorPattern is seen in Domain while doing
// Context, ProvenFix is what worked. Hits counts reuse, for ranking and pruning.
// Added/LastSeen (YYYY-MM-DD) drive age-based pruning — LastSeen bumps each time
// the lesson recurs, so a still-relevant lesson stays fresh.
type Pitfall struct {
	Domain       string `json:"domain"`
	ErrorPattern string `json:"error_pattern"`
	Context      string `json:"context"`
	ProvenFix    string `json:"proven_fix"`
	// Kind is "" (a fix that actually worked — the original and default shape) or
	// KindAvoid (a dead end: the same approach was retried and kept failing, and
	// ProvenFix says what to do differently). Reported live: a run died after ten
	// identical failing tool calls, and the ONE lesson worth keeping from it —
	// "re-sending this shape never works" — could not be expressed at all, because
	// the store only ever recorded fixes that succeeded. Omitted from JSON when
	// empty, so lessons written before this field existed load unchanged.
	Kind     string `json:"kind,omitempty"`
	Hits     int    `json:"hits"`
	Added    string `json:"added,omitempty"`
	LastSeen string `json:"last_seen,omitempty"`
}

// KindAvoid marks a lesson about an approach that never worked, as opposed to a
// fix that did. The two must render differently wherever a lesson is shown to a
// model: telling it "this worked: <the thing that never worked>" is worse than
// saying nothing.
const KindAvoid = "avoid"

// genericErrorPattern matches an ErrorPattern that's pure tool-wrapper noise
// rather than anything about the actual failure — "exit 1" is the run tool's
// own generic prefix on EVERY failed command, so a lesson keyed on it alone
// would "match" (and mislead on) any unrelated failure.
var genericErrorPattern = regexp.MustCompile(`(?i)^exit \d+$`)

// IsGenericErrorPattern reports whether pattern is too generic to usefully key
// a lesson on (see genericErrorPattern). Checked both when a new lesson is
// about to be stored (internal/reflect) and whenever existing lessons are
// queried (Query below) — a pitfall stored before this check existed, or by
// any other future write path, must not surface either.
func IsGenericErrorPattern(pattern string) bool {
	return genericErrorPattern.MatchString(strings.TrimSpace(pattern))
}

// projectPath matches a token that names a file inside a directory —
// "cmd/agent/main.go", "nemotron-extreme-quant/PLAN.md". Requires BOTH a
// separator and an extension, so genuinely general advice survives:
// "go test ./..." (no extension), "--prefix=/usr/local" (no extension) and a
// bare "main.go" (no directory) are all left alone.
var projectPath = regexp.MustCompile(`[\w.-]+/[\w.-]*\.\w+`)

// IsProjectSpecific reports whether s carries a value from the run it was
// learned in rather than a general lesson. Reported live, from a real debug
// log: a stored lesson read "Provide proper path parameter: {"path":
// "nemotron-extreme-quant/PLAN.md", "content": "..."}" and surfaced — in a
// DIFFERENT project — on a failure whose real cause was the encoding of the
// params blob, not the path. The model quoted that hint back in its own
// reasoning and kept retrying the wrong thing. The reflection prompt already
// says to exclude anything project-specific; a model that ignores it must not
// be able to poison the store anyway, so this is enforced on the way in.
func IsProjectSpecific(s string) bool {
	return projectPath.MatchString(s)
}
