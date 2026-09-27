# 0008. In goal mode, a separate judge decides when the work is done

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

A model asked "are you done?" by its own loop tends to say yes. For a
multi-turn objective that makes "finished" mean "the model stopped".

## Decision

`/goal <text>` sets a persisted goal (ADR-0006). When the agent thinks it is
finished, a **judge** — a separate model call (`internal/agent`) — decides
whether the goal is actually met. If not, the goal is re-fed with the gap the
judge named, up to a TTL (default 6), inside the usual esc / stuck-loop /
runaway guards and a hard step cap. **When goal mode is on, the judge decides**;
the agent's own claim of completion is not enough.

Guards against the loop itself: an unparseable judge reply counts as done; a turn
that called no tools is never judged or re-fed; a re-fed run that does no work
gets one nudge, then the loop stops. A loop that stops without the judge
confirming success says so plainly. Plain tasks run once with no judge.

## Consequences

- Goals cost extra model calls; plain tasks don't pay them.
- The judge prompt is user-replaceable (`judge.md` in the workspace).
