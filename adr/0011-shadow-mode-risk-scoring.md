# 0011. Score tool-call risk with a small local model, in shadow mode first

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The policy is glob-based: it knows `rm -rf` is dangerous but not that
`cat ~/.ssh/id_rsa` is. A learned signal could close some of that gap — or add
friction without adding safety. Which one is unknown until measured.

## Decision

`internal/risk` scores every tool call with a hashed-feature multi-label linear
model, `sigmoid(Wx+b)` over word and character n-grams, six labels
(`destructive`, `sandbox_escape`, `credential_access`, `network`,
`external_side_effect`, `safe`). The feature vector is L2-normalized.

- **Shadow mode:** it blocks nothing. It logs its verdict next to what the policy
  did and counts the disagreements (`allowed-but-flagged`,
  `gated-but-unremarkable`) — that comparison is what decides whether it ever
  gets to gate anything. A call it has an opinion on shows it on the tool-call
  line and at the approval prompt; routine calls show nothing.
- **Inference is pure Go**, weights (768 KB of float32) embedded with `go:embed`
  (ADR-0001), ~28 µs per call. `IPS_RISK=off` disables it;
  `IPS_RISK_MODEL=<path>` swaps the model without a rebuild.
- **The model file is self-describing:** it carries its format version, feature
  config, label names and which labels are *informational* (reported but not part
  of the headline score — `network` is one). A new file with a different feature
  space or label set loads with no code change.
- **Training is offline Python, stdlib only** (`scripts/train_risk.py`), on a
  generated compositional dataset (`scripts/gen_risk_dataset.py`) that crosses
  every verb with every argument class and holds out whole **paths**, not rows.
  Its vocabulary is vendored from upstream sources by `scripts/fetch_risk_vocab.py`;
  a monthly workflow proposes refreshes as a PR and refuses one that regresses
  the metrics. CI checks the committed dataset matches its generator. Go and
  Python feature hashing are held together by shared test vectors.

## Consequences

- The decision to let it **gate** anything is deferred to a later ADR, to be made
  on the shadow numbers.
- Model changes are data changes: regenerate, retrain, commit `model.bin`.
