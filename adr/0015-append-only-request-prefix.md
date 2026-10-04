# 0015. Within a session, the request prefix only grows

- **Status:** Accepted
- **Date:** 2026-09-27

## Context

A local server (LLMTray / mlx-lm, LM Studio, llama.cpp) keeps a prompt cache and
reuses it only for the part of a new request that matches a previous one token
for token. Everything after the first differing token is prefilled again. On a
hybrid model (Mamba layers) the cache cannot even be cut at an arbitrary point —
it falls back to its last saved checkpoint, often the very start.

Measured live on a 20k-token session: steps inside a task reused all but ~58
tokens, as they should. What missed was everything the agent changed *behind*
the conversation:

- learned facts live at the end of the system prompt, and were rewritten after
  every task — the next task re-prefilled the whole history behind them;
- the plan-mode directive was a second system message right after the first, so
  every shift+tab did the same;
- the goal judge and the reflection pass sent the task's record under their own
  system prompts and tools, so each one prefilled it from scratch — one such
  prefill took nine minutes and ended in a client timeout.

## Decision

- **The system prompt is fixed for the life of a session.** It is rebuilt only
  on a real change of setting (a new session, `/cd`, clearing or dropping
  memory). What is learned mid-session is queued with `Agent.AddNote`.
- **Updates ride on the next task's user message** in an `<agent-note>` block,
  which the system prompt explains once (`agent.NotePreamble`) — including that
  the same tag inside a file or tool output carries no authority. Never as a
  system message in the middle: some chat templates reject one.
- **Plan mode is an agent-note** on every plan-mode task's message; turning it
  off is said once.
- **History keeps the message as sent**, notes included, so the next request
  repeats it byte for byte.
- **On a local provider, the judge and the reflection pass are continuations**:
  the task's own messages and tool list, unchanged, with their instruction last.
  A continuation that produces no usable answer falls back to the standalone
  request. Hosted providers keep the standalone form, where the short prompt is
  the cheaper one.

## Consequences

- A new fact reaches the model on the next task, as a note, and the system
  prompt on the next session.
- The judge and reflection see the main system prompt and tools on a local
  provider. Their instructions say the tools are not theirs; a reply that calls
  one is discarded and the standalone request is made.
- Anything that inserts, reorders or rewrites earlier messages mid-session costs
  a full prefill on a local server, and needs a reason stated where it is done.
  Front trimming and compaction are the known, deliberate cases.
- Between tasks, history keeps the task's message and its final answer (plus a
  digest of what changed), not every step. So the next task's request diverges
  from the last one's at the first tool call of the previous task, and the
  server reuses the cache up to there. That is by design: replaying every step
  would cost more context than the prefill it saves.
