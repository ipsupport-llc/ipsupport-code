---
name: subagents
description: Delegate work to other models with the agent tool — fan a task out across profiles and merge the results.
when: the user wants a second opinion, a cross-model review, or part of the work handled by another model or in another project directory
---
The agent tool hands a whole, self-contained task to a sub-agent: a separate LLM
session for one of the user's configured profiles. Use it deliberately:

1. Pick a profile. Profiles are the only delegate targets, and they're listed in
   your system prompt under "Sub-agents you can delegate to" (and via the user's
   /agents). If there are none, tell the user to add one: /agents add <name>
   <provider> [model]. Don't invent a profile name.
2. Call it, e.g.:
   {"action": "run", "params": {"profile": "grok", "task": "<task>", "dir": "~/proj"}}
   - task MUST be self-contained: the sub-agent cannot see this conversation, so
     put every file path, the scope, and what to produce into it.
   - dir is optional — it points the sub-agent at another project (defaults to the
     current workspace). The sub-agent is jailed to that directory.
3. To use several models (a cross-model review, or a tie-breaker), issue several
   agent calls IN ONE turn with different profiles — they run in parallel, each on
   its own status line. Keep the scope concrete (a few files), since each
   sub-agent has a limited step/context budget.
4. When they return, synthesize ONE answer: merge identical findings, note which
   models agree, order by severity, and call out disagreements — e.g. "grok
   flagged X at file:line; claude flagged Y; both agree on Z". Don't just paste
   each sub-agent's reply back.

Notes: a sub-agent reads and writes files and uses git, but only runs shell
commands if the user enabled it (/agents exec on). Each spawn asks for approval
unless the user relaxed it (/permissions agents on). For a review with no edits,
switch to plan mode first so every sub-agent stays read-only.
