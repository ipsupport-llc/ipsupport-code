# 0013. Distribute through GitHub Releases with a checksum-verified self-update

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

A single binary (ADR-0001) still has to get onto machines and stay current,
without a package manager in between and without trusting an unverified
download.

## Decision

Tagged builds publish per-platform archives and `checksums.txt` to GitHub
Releases. The install scripts and `ipsupport-code update` / `/update`
(`internal/selfupdate`) verify the SHA-256 before replacing the binary in place.

Two channels: **stable** (default) and **nightly**, switched with
`update stable|nightly` and saved in the config. A startup check prints a
one-line notice when a newer build is out on the channel; it is skipped with
`update_check false` or `/offline on`, and dev builds are never nagged.

Release binaries are never committed to the repository; the only binaries in
git are small assets the product itself needs (site images, the risk model's
weights).

## Consequences

- Shipping is: merge, tag, let the release workflow build and upload.
- The version is stamped at build time (`-X main.version`), so a build always
  knows which release it is.
