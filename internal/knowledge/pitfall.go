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
	Hits         int    `json:"hits"`
	Added        string `json:"added,omitempty"`
	LastSeen     string `json:"last_seen,omitempty"`
}

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
