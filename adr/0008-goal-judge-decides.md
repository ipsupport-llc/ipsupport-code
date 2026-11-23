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

Guards against the loop itself: an unparseable judge reply is asked once more,
then counts as **not met** and the goal is re-fed — a judge that cannot answer
has confirmed nothing; a re-fed run that finishes without doing any work gets one
nudge, and if it idles again it goes to the judge like any other finish. A
continuation judge (ADR-0015) that answers with a tool call has not answered, nor
has a judge that calls done() while asking for a tool it doesn't have. A
loop that stops without the judge confirming success says so plainly. Plain
tasks run once with no judge. A prose reply with no tool call is judged when the
run is the goal itself, and not when it answers a side question asked while the
goal stands.

The judge has a third answer, ASK: the agent stopped on a question only the user
can answer. The run then ends without a re-feed, the goal stands, and the user's
reply carries it on. The judge decides this, not the agent — an agent given a way
to ask would ask instead of working — and a question the agent could settle
itself is MORE. Any MORE outranks an ASK, and an ASK outranks DONE.

*Corrected 2026-10-04:* this section first said an unparseable reply counts as
done and a turn with no tool calls is never judged. Neither has been true since
the judge became the only way to end a goal run early.

## Consequences

- Goals cost extra model calls; plain tasks don't pay them.
- The judge prompt is user-replaceable (`judge.md` in the workspace).
