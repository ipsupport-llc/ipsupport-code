# 0010. External CLI agents run outside the sandbox and are approved separately

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

Locally installed coding agents — Codex, Claude Code, Muse, Gemini CLI, Qwen
Code, aider, Goose, OpenCode — are useful delegates. They bring their own tools
and their own permission systems, and we can't see inside them: our policy jail
doesn't apply, the OS sandbox doesn't wrap them, and `/rewind` can't see their
edits.

## Decision

They are **external profiles** (`/agents add-tool`, `cmd/agent/external_agent.go`):
the CLI is executed in the target directory in its **non-interactive** mode, with
the task substituted into `{task}`. A small built-in catalog knows each one's
headless launch shape (`codex exec`, `claude -p`, `muse exec`, …), so adding one is
one word; the full form takes any launch line.

- Every launch asks its **own** approval under the `external agent` category —
  deliberately not a `spawn …` kind, so an earlier allow-session for ordinary
  sub-agents never covers an unsandboxed agent. The prompt shows the full task.
- Output is bounded: only the **tail** of stdout (where a CLI agent's final answer
  is) and a `git diff --stat` go back to the parent — never the full patch.
- A timeout (15 min default, per-profile override) and process-group kill stop a
  CLI that hangs waiting for input or leaves children behind.

## Consequences

- An external agent is trusted as much as the user trusts that CLI; our guarantees
  stop at its process boundary. The docs say so where they describe it.
- A new CLI agent is supported by adding one catalog line — once its headless
  mode has been run for real and found to print its answer on stdout.
