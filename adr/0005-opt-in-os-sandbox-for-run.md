# 0005. The OS sandbox is opt-in and confines `run` only

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The policy decides *whether* a command runs. Once it runs, it can touch anything
the user can. OS sandboxes can confine that, but they differ per platform, need
recent kernels on Linux, and can break build tools that write outside the
project.

## Decision

`internal/sandbox` wraps **shell (`run`) commands** in the platform's mechanism
when the user turns it on (`sandbox auto`, or `seatbelt` / `landlock`
explicitly): Seatbelt on macOS, Landlock on Linux (5.13+; network rules need
6.7+). Writes are confined to the workspace plus temp dirs and the user cache;
the network follows `/offline`; reads are unchanged.

It is **off by default**. Where no supported mechanism exists, commands run
unconfined, as before. The file tool is not wrapped — it is already confined by
the policy jail. External CLI agents are **not** sandboxed (ADR-0010).

## Consequences

- Turning it on can't be a regression for someone who didn't ask for it.
- CI runs the real mechanisms: a macOS job for Seatbelt, Linux for Landlock.
- Stronger confinement (reads, other tools) is a future decision, not implied by
  this one.
