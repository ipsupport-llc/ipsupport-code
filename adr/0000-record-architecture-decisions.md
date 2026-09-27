# 0000. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The project has grown past the point where its load-bearing choices can live in
commit messages, code comments and the heads of the people who made them. Several
of them — where state lives, what a workspace may override, why external agents
are gated separately — are easy to undo by accident with a change that looks
harmless.

## Decision

Keep Architecture Decision Records in `adr/`, one Markdown file per decision,
numbered in order: `NNNN-short-title.md`. Each has **Status**, **Context**,
**Decision** and **Consequences**. `adr/README.md` is the index.

The first set (0001–0013) was written at once, on the date above, to capture the
project as it stands. They describe existing behaviour; each one names the code
that implements it.

A decision is never edited into saying something else. When one changes, write a
new ADR that supersedes it and set the old one's status to
`Superseded by NNNN`.

## Consequences

- A change that contradicts an accepted ADR has to say so — in a new ADR — rather
  than slip through as a refactor.
- The ADRs are only as good as their upkeep: a PR that changes one of these
  behaviours updates or supersedes the record in the same PR.
