# 0001. Ship one static Go binary with no runtime dependencies

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The agent runs on developers' own machines, next to their own model. Anything it
needs at runtime — a Python, a Node, shared libraries, a container — is something
that can be missing, the wrong version, or broken by the project it is working on.

## Decision

`ipsupport-code` is a single Go binary built with `CGO_ENABLED=0 -trimpath`
for linux/amd64, linux/arm64, darwin/amd64, darwin/arm64, windows/amd64 and windows/arm64
(`Makefile`: `release`, `archives`). Everything it needs at runtime is compiled or
`go:embed`-ed in — including the risk model's weights (see ADR-0011). Tools that
would usually shell out are built in: the `file` tool's `search` is its own
regex grep, `web` converts HTML to Markdown in-process.

Python exists in the repository only for **offline** work (`scripts/`: the risk
vocabulary fetch, dataset generator and trainer). Nothing the binary does at
runtime calls it.

## Consequences

- Install is "download one file and verify its SHA-256" (`scripts/install.sh`,
  `docs/install.ps1`), and self-update is "replace one file" (ADR-0013).
- Features that need native code (CGO) are out unless they can be done in pure
  Go or through the OS itself — the sandbox uses pure-Go Landlock bindings on
  Linux and the system's `sandbox-exec` on macOS (ADR-0005).
- Heavy ML runtimes are out; the risk model is a linear model because inference
  must be a few lines of Go (ADR-0011).
