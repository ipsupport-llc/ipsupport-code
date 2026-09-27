# 0004. The permission policy is the gate, with a floor nobody can lower

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The agent executes model-chosen actions on a real machine. The project it works
in may not be fully trusted — it is a checkout, and its own config file is part
of it.

## Decision

`internal/policy` decides, per tool and action, **allow / ask / deny**:

- A **deny** glob blocks, an **allow** glob runs without asking, otherwise the
  **default** applies. Run-command deny globs match anywhere in the command;
  file globs are path-aware and confined to the workspace **jail**.
- A **deny floor** (`rm -rf`, `sudo`, secrets, `.git`, `.env`, …) is always
  unioned in. Config adds to it; it cannot remove it.
- Mutating actions **ask** by default. `a` at a prompt allows that whole
  *category* for the session, in memory only; `/permissions … on` is the
  persistent relaxation and is written to the workspace config.
- The workspace config (`<workspace>/.agent/config.json`) wins over the user's
  global config for everything **except the `llm` connection and `providers`**:
  a checkout may tighten or loosen what the agent does, but it may not redirect
  the model endpoint while the user's real API key is sent to it.

## Consequences

- Everything else (sandbox, risk scoring) is layered **under** or **beside** the
  policy, never instead of it (ADR-0005, ADR-0011).
- Approval categories are a security boundary, not UI grouping: a category that
  folded two trust classes together would let one "allow session" cover both
  (see ADR-0010).
