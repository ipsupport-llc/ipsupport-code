# 0009. LLM sub-agents: depth one, their own policy, their own observer

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

Delegating to another model — a stronger one for review, several for a fan-out —
is useful. Unbounded recursion, shared mutable state between parallel agents, and
one agent's approvals being attributed to another are not.

## Decision

The `agent` tool spawns a sub-agent on any configured profile (`/agents add`):

- **Depth 1** — a sub-agent cannot spawn its own.
- It inherits plan/auto mode, reads and writes files and uses git, but runs shell
  commands **only if** `/agents exec on`.
- Every spawn asks for approval until `/permissions agents on` — except one
  whose working directory is outside the workspace jail, which always asks.
- Each sub-agent is built with its **own** policy and its **own** risk observer
  (`newSubAgent`), so an approval answered inside it is learned against its own
  call, not the parent's spawn. Per-call context (such as the risk assessment)
  travels on `context.Context`, the only thing correct under concurrency.
- Its tokens are recorded in `/usage` like any other. A spawn can run as a
  **background job** (`/jobs`) whose result is folded into the parent's next step.

## Consequences

- Parallel sub-agents share nothing mutable through the app struct; anything
  per-call is carried on the context.
- A sub-agent is jailed to its own working directory. One pointed outside the
  parent's workspace is a decision the user makes every time; relaxed spawn
  approvals do not cover it. *(Corrected 2026-10-04: this said a sub-agent
  can't escape the parent's jail, which a directory argument always could.)*
