# 0014. An approval is a verdict, not a label

- **Status:** Accepted
- **Date:** 2026-09-27
- **Amends:** [0012](0012-local-risk-learning-delta.md)

## Context

The risk model's labels describe what a call **does** — `destructive`,
`credential_access`, `external_side_effect`, … (ADR-0011). An answer at an
approval prompt says something else: whether the call was **acceptable**, here,
now. ADR-0012 treated the two as one. An approved flagged call pushed the labels
that fired toward zero, and the feedback log recorded it as `["safe"]`, ready
for the trainer.

So approving an intended `git push` taught the workspace's delta that git push
has no external side effect, and an approved `rm -rf build/` would, once
retrained on, teach the base model that it deletes nothing. The answer is true;
the lesson drawn from it was not.

## Decision

- **Label scores stay the base model's word.** `Tuned.Assess` reports every
  label exactly as the base scores it (`Assessment.Scores`, and the headline
  before corrections as `BaseRisk`).
- **The local delta adjusts only the headline** — `Assessment.Risk`, the number
  that decides whether a call is flagged and shown. The learning rule, rate,
  decay and cap are unchanged; what they now move is attention in this
  workspace, not a claim about the call. Informational labels, which never reach
  the headline, are not learned at all.
- **The feedback log records verdicts.** Each row carries `verdict`
  (`approved` / `refused`) and the labels the model `scored`, and no `labels`.
  `scripts/train_risk.py` skips every `source: "approval"` row, including the
  older ones that wrote `["safe"]`. A row becomes a training example only when
  a person labels it and sets `source: "manual"`.
- The shadow log prints both numbers, `risk` and `base`, so the effect of the
  corrections is visible next to what the model thinks the call does.

## Consequences

- Approving `git push` a few times stops the warning in that workspace while
  `/risk` and the log still say it has an external side effect.
- A label the base model genuinely gets wrong (`cat README.md` scored as
  `credential_access`) can be quieted by answers, but is fixed only by
  retraining the base on a hand-labelled row. That is the intended split: local
  answers tune attention, labels are facts, and facts need a person to state
  them.
- Existing deltas keep working — the same numbers, now applied to the headline
  only. Existing feedback logs are safe to pass to the trainer: their approval
  rows are skipped.
