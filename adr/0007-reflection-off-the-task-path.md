# 0007. Learn from each run through a reflection pass, off the task path

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

A small local model repeats the same mistakes run after run. Fine-tuning it per
user is not practical; telling it what went wrong last time is.

## Decision

After a task, `internal/reflect` runs a second model pass that distils:

- **lessons** — environment-general tool pitfalls, stored in
  `~/.config/ipsupport-code/knowledge.json` (`internal/knowledge`), and
- **facts** — durable things about this project (build/test commands, where
  things live, conventions), stored in the workspace's state directory
  (ADR-0006).

Both are folded into the next run's prompt, and a lesson that matches a failing
tool call is injected straight into that error. Lessons track when they were last
seen; `/knowledge` prunes stale ones and `retain <days>` auto-purges. Facts are
presented to the model as what was worked out before — useful, not axioms.

Reflection runs on its **own goroutine** after the answer is shown
(`startReflect`); it never blocks the next task. A pass that lands while another
task is running is held and applied when that task ends, so nothing rewrites the
stores under a running task.

## Consequences

- The quality of what is learned is bounded by the reflecting model; a separate
  reflection profile (`ReflectProfile`) can point it at a stronger one.
- The stores need pruning, or they accrete junk; that is part of the design, not
  an afterthought.
- Every step is also appended to `traces.jsonl` (`internal/trace`) — the raw
  material for any later offline learning.
