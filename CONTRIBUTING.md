# Contributing

Thanks for taking a look. This is a small, focused tool — contributions that keep
it small and sharp are very welcome.

## Development

Pure Go, no codegen, no CGO. You need a Go toolchain matching `go.mod`.

```sh
make build      # host binary at dist/ipsupport-code
make race       # all tests with the race detector
make fmtcheck   # gofmt check (CI fails on unformatted code)
make vet        # go vet
make release    # cross-compile every target into dist/
```

CI runs `fmtcheck`, `vet`, `race`, and a cross-compile of all targets on every
push and PR. Please make sure `make fmtcheck vet race` is green before opening a
PR.

## Guidelines

- **Tests with the change.** New behaviour or a bug fix comes with a test; for a
  bug, a test that reproduces it first.
- **Keep it minimal.** Solve the task at hand; avoid speculative abstractions and
  flags nobody asked for. Match the surrounding style.
- **Mind the prompt budget.** The tool catalog and system prompt ship in every
  request to a small local model — `internal/tool` has a token-budget test guarding
  the catalog. Don't bloat them.
- **English** in code, comments, and commits.

## Releases

Maintainers cut a release by tagging: `git tag v0.1.0 && git push origin v0.1.0`.
The `release` workflow builds every target and publishes a GitHub Release. A
rolling `nightly` pre-release tracks `main`.
