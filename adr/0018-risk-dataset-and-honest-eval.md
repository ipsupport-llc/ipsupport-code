# 0018. The risk model is measured on real commands, and its training data is generated from a grammar

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The risk model (ADR-0011) is trained on `scripts/risk_dataset.jsonl`, which
`scripts/gen_risk_dataset.py` builds from hand-written lists. In one day three
fixes came from the same cause: the model learned the words in those lists,
not what a command does. `example.com` meant a side effect, `/dev/` meant a
wiped disk, an unfamiliar URL leaned destructive, and Windows did not exist
for it at all. Every fix was one more list, carrying its own bias.

The holdout comes from the same generator, so it measures how well the model
learned the generator. A precision of 1.00 there says little when the training
data and the test share their blind spots.

## Decision

**1. An honest evaluation set, never trained on.** Real commands, labelled by a
person:

- `scripts/risk_eval.jsonl`, in the repository: commands real agents ran,
  sampled from public trajectories (Terminal-bench, Apache-2.0), and Windows and
  macOS commands from tldr-pages (CC-BY-4.0), which have no public agent
  trajectories. Each row names its source.
- A private set from the user's own sessions (`traces.jsonl`), labelled the
  same way and kept on that machine. It never enters the repository: it is
  real operator data.

The trainer reports the honest set separately from the generated holdout, and
a new model ships only if it is no worse on the honest set. The two never
share a command: the generator drops one that is in the honest set, and the
trainer leaves out of the report any row that is training data. Programs may
be shared — `choco`, `reg`, `netsh` are both generator vocabulary and in the
set — so the set measures new commands, not new programs; measuring unseen
programs is part of step 3, when the vocabulary comes from real sources.

*Amended 2026-10-07:* this first said a tldr program in the set would never be
generator vocabulary, which was not true of the set as built.

**2. Labels are derived, not assigned.** Each generated example records what
the command does (read, write, delete, execute, install, escalate, persist,
network read, network write) and what it acts on (project, build output,
system, secret, remote). One table maps that pair to the six labels, so two
examples cannot contradict each other.

**3. A grammar, not lists.** A command is a program, flags (those that change
the action, such as `-X POST`, `-r`, `/s`, and neutral ones such as `-s`,
`-L`), typed arguments (a path of a class, a URL, a package, a registry key, a
service) and a tail (`2>&1`, `> /dev/null`, `| head`, `| jq`, quoting). Tails
are neutral by construction.

**4. Vocabulary from real sources.** Programs and flags per platform (Linux,
macOS, Windows, common), hosts from a list of real sites and API path shapes,
paths from each platform's filesystem layout, secrets from gitleaks.

**5. Pairs that differ in one thing.** Each dangerous example has a safe twin
one change away: `POST`/`GET`, `| sh`/`| head`, `/dev/sda`/`/dev/null`,
`rm -rf /`/`rm -rf build/`, `defaults write`/`defaults read`.

**6. The model knows the OS.** The call text carries the platform (`os=…`),
in Go and in the trainer together, since it is part of the wire format.

**7. Data, not code.** Programs, flags and vocabularies live in per-platform
data files; the generator only combines them. CI retrains and evaluates.

The order is the numbering: without the honest set (1) nothing else can be
measured, so it comes first and records the current model's baseline.

## Consequences

- Labelling the honest set is manual and takes judgement; contentious rows are
  decided by the maintainer, not by the model.
- Public sources need attribution, kept in the eval file's rows and in
  `scripts/README` notes.
- Each retrain changes the model fingerprint, so local corrections are dropped
  (ADR-0012). That is the existing behaviour, not a new cost.
- The private set can only be checked where it lives; a release from CI is
  measured on the public set alone.
