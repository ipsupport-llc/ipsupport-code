---
name: review
description: Review code across several models via sub-agents, then synthesize one merged list.
when: the user asks for a code review, a second opinion, or a review "across models"
---
To review code with more than one model, delegate to sub-agents and combine the
results — don't just review it yourself once:

1. Find the reviewers: the user's configured agent profiles (the ones `/agents`
   lists). If none are configured, tell the user to add some under "agents" in
   config.json (each a `provider` + `model`), then continue with whatever exists.
2. For EACH reviewer profile, call the agent tool, e.g.:
   {"action": "run", "params": {"profile": "<name>", "dir": "<project dir>",
   "task": "Review <scope> for real bugs, correctness, concurrency, and design
   risks. Report each finding as file:line — what is wrong — why. No style nits."}}
   ALWAYS pass `dir` = the project's working directory (the repo root). Without it
   the sub-agent inherits the session workspace, which may be a home directory —
   so it would "review" the wrong tree. Keep the scope concrete (a few packages or
   files), not "the whole repo at once" — a sub-agent has a limited budget.
3. Synthesize ONE deduplicated list from all reviewers: group identical findings,
   note how many reviewers agree on each, order by severity, and call out
   disagreements. Don't just concatenate the transcripts.

If you only intend to review (no edits), switch to plan mode first so every
sub-agent stays read-only. Report the merged findings, not each raw reply.
