# Architecture Decision Records

How and why `ipsupport-code` is built the way it is. Format and rules:
[0000](0000-record-architecture-decisions.md). The first set was recorded on
2026-09-27 to capture the project as it stood; each record names the code behind it.

| # | Decision | Status |
|---|---|---|
| [0000](0000-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0001](0001-single-static-go-binary.md) | Ship one static Go binary with no runtime dependencies | Accepted |
| [0002](0002-openai-compatible-wire-protocol.md) | Speak one wire protocol: OpenAI-compatible chat completions, local first | Accepted |
| [0003](0003-fat-tools-small-catalog.md) | Few fat tools with a small catalog | Accepted |
| [0004](0004-permission-policy-and-deny-floor.md) | The permission policy is the gate, with a floor nobody can lower | Accepted |
| [0005](0005-opt-in-os-sandbox-for-run.md) | The OS sandbox is opt-in and confines `run` only | Accepted |
| [0006](0006-agent-state-outside-workspace.md) | The agent's own state lives outside the workspace | Accepted |
| [0007](0007-reflection-off-the-task-path.md) | Learn from each run through a reflection pass, off the task path | Accepted |
| [0008](0008-goal-judge-decides.md) | In goal mode, a separate judge decides when the work is done | Accepted |
| [0009](0009-llm-subagents-depth-one.md) | LLM sub-agents: depth one, their own policy, their own observer | Accepted |
| [0010](0010-external-cli-agents.md) | External CLI agents run outside the sandbox and are approved separately | Accepted |
| [0011](0011-shadow-mode-risk-scoring.md) | Score tool-call risk with a small local model, in shadow mode first | Accepted |
| [0012](0012-local-risk-learning-delta.md) | Adapt the risk model per workspace from approvals, without touching the base | Accepted; amended by 0014 |
| [0013](0013-releases-and-self-update.md) | Distribute through GitHub Releases with a checksum-verified self-update | Accepted |
| [0014](0014-approvals-are-verdicts-not-labels.md) | An approval is a verdict, not a label | Accepted |
| [0015](0015-append-only-request-prefix.md) | Within a session, the request prefix only grows | Accepted |
| [0016](0016-usage-statistics-and-rating.md) | Anonymous usage statistics, on for new installs, and an in-app rating | Accepted |
| [0017](0017-git-hooks-stay-enabled.md) | The git tool runs the checkout's hooks and filters | Accepted |
| [0018](0018-risk-dataset-and-honest-eval.md) | The risk model is measured on real commands, and its training data is generated from a grammar | Accepted |
