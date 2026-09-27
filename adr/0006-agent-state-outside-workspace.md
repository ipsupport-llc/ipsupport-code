# 0006. The agent's own state lives outside the workspace

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The agent used to keep its runtime state — standing goal, learned facts, lessons,
prompt history, sessions — in the project's `.agent/` directory. That put its
private state on the filesystem it reads. Observed live: a model listing the
project found `.agent/goal.json`, read it, and began echoing the goal text back
until the repetition detector killed the run. It also cluttered users'
repositories.

## Decision

Files the **agent writes for itself** live in
`~/.config/ipsupport-code/state/<workspace-name>-<hash>/` (`config.StateDir`),
keyed by the workspace's absolute path so two checkouts of the same project keep
separate state. Per-session files are scoped by the session name.

Files the **user writes** — `.agent/config.json`, `system.md`, `judge.md`,
`compact.md`, `instructions.md` — stay in the workspace: they are project
content and belong under its version control.

An existing install is migrated once on start (`migrateLegacyState`): the
agent's own files are **moved**, not copied — a stale copy is the hazard this
exists to remove — and nothing already at the destination is overwritten.

State shared between concurrently running sessions (the usage ledger, the
knowledge base, the risk feedback ledger) is written under a cross-process lock
(`internal/filelock`) and via atomic replace (`internal/atomicfile`).

## Consequences

- Moving a checkout starts it with clean state rather than silently inheriting
  another tree's goal.
- Nothing new may be written into the workspace for the agent's own bookkeeping.
  `.agent/` in this repository stays gitignored except `config.example.json`.
