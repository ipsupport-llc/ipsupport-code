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
   - ALWAYS pass `dir` = the working directory of the SPECIFIC project you're
     working on (e.g. the repo root). The sub-agent is jailed to that directory.
     If you omit it the sub-agent inherits the session's workspace, which may be a
     home directory or an unrelated folder — never turn a sub-agent loose on a
     whole home dir. If you don't know the project's path, find it first (file
     list / pwd) and pass it.
   - task MUST be self-contained: the sub-agent cannot see this conversation, so
     put every file path (relative to `dir`), the scope, and what to produce into it.
3. To use several models (a cross-model review, or a tie-breaker), issue several
   agent calls IN ONE turn with different profiles — they run in parallel, each on
   its own status line. Keep the scope concrete (a few files), since each
   sub-agent has a limited step/context budget.
4. When they return, synthesize ONE answer: merge identical findings, note which
   models agree, order by severity, and call out disagreements — e.g. "grok
   flagged X at file:line; claude flagged Y; both agree on Z". Don't just paste
   each sub-agent's reply back.

Keep your OWN context lean: instead of reading dozens of files into this
conversation, delegate a big exploration to a sub-agent and have it return a short
summary ("map the auth flow in <dir>, list the files and what each does"). You pay
its context, not yours — then act on the summary.

Notes: a sub-agent reads and writes files and uses git, but only runs shell
commands if the user enabled it (/agents exec on). Each spawn asks for approval
unless the user relaxed it (/permissions agents on). For a review with no edits,
switch to plan mode first so every sub-agent stays read-only.
