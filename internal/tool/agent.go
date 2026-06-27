package tool

import "context"

// SpawnFunc runs a scoped task on a sub-agent (a fresh agent loop on another
// model/provider) and returns its final answer. The host provides it because it
// owns config, provider resolution, approval, and the agent build. It returns an
// error for an unknown profile/provider, a missing key, or a denied spawn.
type SpawnFunc func(ctx context.Context, task, profile, provider, model string) (string, error)

// NewAgent is the `agent` fat tool: delegate a self-contained task to a different
// model/provider and get its answer back — a second opinion, or another model's
// strength (e.g. reviewing the same code across models). Sub-agents can't spawn
// their own sub-agents (the host gives them a catalog without this tool).
func NewAgent(spawn SpawnFunc) Tool {
	return NewDomain(DomainSpec{
		Name:    "agent",
		Summary: "Delegate a self-contained task to a sub-agent on another model/provider; returns its answer (a second opinion, or another model's strength).",
		NotHere: "Only for handing a whole task to a different model — do your own work with file/run/git.",
		Actions: []Action{{
			Name:   "run",
			Params: []Param{Req("task", "str"), Opt("profile", "str", ""), Opt("provider", "str", ""), Opt("model", "str", "")},
			Note:   "(profile=a configured profile, or give provider+model; task must be self-contained — the sub-agent can't see this chat)",
			Run: func(ctx context.Context, a Args) Result {
				out, err := spawn(ctx, a.Str("task"), a.Str("profile"), a.Str("provider"), a.Str("model"))
				if err != nil {
					return Err("sub-agent: " + err.Error())
				}
				return Ok(out)
			},
		}},
	})
}
