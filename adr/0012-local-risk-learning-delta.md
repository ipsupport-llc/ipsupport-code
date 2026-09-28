# 0012. Adapt the risk model per workspace from approvals, without touching the base

- **Status:** Accepted; amended by [0014](0014-approvals-are-verdicts-not-labels.md)
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

A user's approvals are real labels for their own environment. Learning from them
is valuable; letting that corrupt the shipped model, or survive a model upgrade
it no longer fits, is not.

## Decision

- The embedded base model is **immutable**. Local learning writes a sparse
  per-workspace **delta** (`internal/risk/tuned.go`) layered over it, with
  proximal decay and a cap on how far any weight can move.
- Only an approval a **person actually answered** is learned from. Esc, a killed
  job or closed stdin return "no" just like a refusal, and are not treated as one
  (`ApproveAnswered`); where it can't be known, nothing is learned.
- Two files, two roles: `risk-feedback.jsonl` is the **durable ledger** of raw,
  model-independent feedback; `risk-delta.bin` is a **disposable cache**, refused
  unless it was learned against this exact base — same feature space, same
  labels, and the same weights (a fingerprint of the model file). A refused
  delta is dropped, not rebuilt; the workspace relearns from new answers.
- Saving happens off the approval path and is flushed at shutdown. `/risk` shows
  the state; `/risk reset` drops the delta, in memory and on disk under the same
  lock a save writes under, so a save already in flight cannot bring it back.

## Consequences

- A new base model never inherits a delta built for another, even one that
  keeps the labels and features (added 2026-09-27: before the fingerprint, a
  retrained model applied the old corrections on top of weights that had
  already learned them). The ledger is what carries the user's feedback
  forward, into offline retraining.
- Approving a call whose side effect was *intended* taught the delta that the
  side effect did not happen. Resolved in ADR-0014: the delta adjusts only
  whether a call is flagged here, never the label scores.
