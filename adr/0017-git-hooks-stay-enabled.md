# 0017. The git tool runs the checkout's hooks and filters

- **Status:** Accepted
- **Date:** 2026-10-04

## Context

The full-project review (Muse) showed that an approved `git commit`, `add`,
`checkout`, `pull` or `push` runs the repository's hooks and filter drivers —
`.git/hooks/*`, `filter.*.clean` in `.git/config` — as shell commands, outside
the `run` sandbox and without an approval of their own. A hostile checkout gets
code execution from one approved git action.

Turning them off (`-c core.hooksPath=/dev/null`, no filters) closes that, and
breaks what hooks are for: formatters, linters, commit-message checks, LFS.
A commit the agent makes would then skip the checks the user's own commits get.

## Decision

Hooks and filters stay enabled. The file tool cannot create or edit them —
`.git/` is under the deny floor (ADR-0004) — and a shell command that could is
an approved `run` of its own. So a hook in the checkout is one the user, or the
repository they chose to work in, put there. `core.fsmonitor` and `textconv`
stay disabled, since they run on read-only actions (status, diff) the user never
approves.

## Consequences

- Working in an untrusted repository means trusting its hooks, as it does for
  the user typing `git commit` themselves. This is stated here, not hidden.
- A repository cloned by the agent arrives without hooks (git does not clone
  them), so the risk is limited to checkouts that already had them.
- Revisit if a per-workspace "untrusted" mode is added: that mode would disable
  hooks and filters.
