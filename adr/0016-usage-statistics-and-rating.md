# 0016. Anonymous usage statistics, on for new installs, and an in-app rating

- **Status:** Accepted
- **Date:** 2026-10-04

## Context

We want to know how many people run ipsupport-code, on what, and which of its
features they use, and to let them rate it — without undermining a tool whose
point is your own model on your own machine. The server side already exists for
LLMTray: `POST /api/telemetry` and `POST /api/reviews` on ipsupport.us
(ipsupport-api ADRs 4, 6, 9, 10), extended for a command-line product in its
ADR 12. LLMTray's ADR 15 settled the client behaviour, and this follows it.

## Decision

- **On by default for a new install; an update never turns it on.** First-run
  setup — a brand-new machine with no settings — writes `telemetry: true` and
  prints what is sent and how to turn it off. An install that existed before
  this build and never chose is written `false` at its first launch
  (`settleTelemetryDefault`). A piped first run that never sees setup decides
  nothing: CI and containers don't count themselves.
- **Global only.** `telemetry` is restored after the workspace merge, like the
  model connection (ADR-0004): a checkout can't turn it on for the user.
- **Paused, not forgotten,** under `DO_NOT_TRACK`, `/offline`, or a development
  build (anything but a release tag or a nightly; `IPS_API_BASE` points a dev
  build at a test server). **Off forgets**: the install ID and unsent counters
  are deleted. On again means a new install ID.
- **What is counted** (`internal/telemetry`): per local day, `tasks`,
  `tool_calls`, `goals`, `subagents`, `external_agents`, `mcp`, `skills`,
  `plan_mode`, `local_model`, `cloud_model`, and the coarse model family (only
  the last path component of a model name is looked at). The machine: OS,
  architecture, OS version, Apple chip, memory snapped to a shipped size,
  language. Never code, prompts, commands, tool arguments or output, paths,
  model names, keys or provider URLs.
- **When.** Counters live in `telemetry.json` in the config directory, shared by
  every session under a file lock. A day is collected while it lasts and sent
  at the first launch after it ends, oldest first; nothing runs on a timer. Days older than seven
  (counted from the UTC day, as the server does) or after today are dropped.
  204 sent; 429 waits for `Retry-After`; 408, 5xx and no answer wait an
  hour; either wait holds back launches before it ends. Any other 4xx drops
  the day. The lock is not held across the network.
- **`/telemetry`** shows how the last send went (accepted, refused and why,
  or failed), the days waiting, and the exact JSON of each waiting report.
- **`/rate <1-5> <words> [--name <you>]`** submits a review with an
  `Idempotency-Key`, validated locally against the server's limits first. A
  dim one-line suggestion may follow a finished task — after the first day,
  never once rated or after `/rate never`, and showing it snoozes it for two
  weeks. Not in piped mode, where it would land in a script's output.
- The website shows the published reviews, read from the browser (ipsupport-api
  ADR 13).

## Consequences

- Numbers undercount: existing installs report only if their users turn it on.
- A model name never leaves the machine, so the families are coarse by design;
  a new family needs both the client's mapping and the server's allowlist.
- Reviews of a tool that drives other tools will often name them, which the
  moderation prompt escalates to a human — more ratings arrive by email first.
