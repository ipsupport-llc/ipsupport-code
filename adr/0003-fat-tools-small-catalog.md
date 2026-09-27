# 0003. Few fat tools with a small catalog

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

Small local models route poorly across dozens of tools, and every tool's schema is
paid for in prefill on every step. A long tool list is slower and less accurate on
exactly the models this agent targets.

## Decision

One tool per domain — `file`, `run`, `git`, `web`, `calc`, plus `agent`, `mcp`,
`skill`, `help`, `history` — each called as `{"action": ..., "params": {...}}`
(`internal/tool`). A declarative `Domain` generates each tool's schema, help text
and validation from one definition. The whole catalog is kept to roughly 1k
tokens; growth is treated as a cost, not a free addition.

When a tool call fails, a matching lesson from past runs is injected into the
error the model sees (ADR-0007), so the model doesn't need a separate
"how do I use this" round trip.

## Consequences

- New capability goes in as an **action** on an existing tool where it fits
  (e.g. `git` clone/pull/push are actions of `git`), not as a new tool.
- Catalog size has little headroom; every new action or parameter description has
  to earn its bytes.
- Per-action permission rules are natural: the policy sees `tool` + `action`
  (ADR-0004).
