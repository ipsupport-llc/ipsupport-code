package tool

import "context"

// SpawnFunc runs a delegated task on a sub-agent — a fresh LLM session for the
// named profile (which carries the provider+model), in a given directory — and
// returns its final answer. The host provides it because it owns the profiles,
// provider resolution, approval, and the agent build. It errors on an unknown
// profile, a missing key, a bad directory, or a denied spawn.
type SpawnFunc func(ctx context.Context, profile, task, dir string) (string, error)

// NewAgent is the `agent` fat tool: delegate a self-contained task to one of the
// configured profiles (each a model the user curated) — a second opinion, or
// another model's strength (e.g. reviewing the same code across models),
// optionally in another directory. Sub-agents can't spawn their own sub-agents
// (the host gives them a catalog without this tool). Issue several calls in one
// turn to fan a task out across profiles; they run in parallel.
//
// spawnBg is the fire-and-forget variant: it starts the task as a detached
// background job and returns an immediate acknowledgement; the host delivers
// the job's result to the model at the start of a later turn.
func NewAgent(spawn, spawnBg SpawnFunc) Tool {
	return NewDomain(DomainSpec{
		Name:    "agent",
		Summary: "Delegate a self-contained task to a sub-agent — a separate LLM session for a configured profile, optionally in another directory; returns its answer.",
		NotHere: "Only for handing a whole task to another model — do your own work with file/run/git.",
		Actions: []Action{{
			Name:   "run",
			Params: []Param{Req("profile", "str"), Req("task", "str"), Opt("dir", "str", ""), Opt("background", "bool", "false")},
			Note:   "(profile=one of the configured profiles; dir=working directory e.g. ~/proj, defaults to here; task must be self-contained — the sub-agent can't see this chat; background=true starts it as a detached job — the call returns at once and the result is delivered to you on a later turn)",
			Run: func(ctx context.Context, a Args) Result {
				run := spawn
				if a.Bool("background") && spawnBg != nil {
					run = spawnBg
				}
				out, err := run(ctx, a.Str("profile"), a.Str("task"), a.Str("dir"))
				if err != nil {
					return Err("sub-agent: " + err.Error())
				}
				return Ok(out)
			},
		}},
	})
}
