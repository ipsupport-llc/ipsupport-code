// Package knowledge is the agent's persistent memory: a JSON-backed store of
// Pitfalls (lessons) it has learned. It is the mechanism that lets the agent
// actually remember experience across restarts — the store is read on startup
// and written after each task's reflection pass.
package knowledge

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
