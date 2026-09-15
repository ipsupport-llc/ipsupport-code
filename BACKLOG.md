# Backlog

Known work that is designed but not built. Each entry says what is wrong, how it
was found, what the fix is, and **what the fix would break** — this subsystem has
been made worse twice by well-meant changes, so an item without a stated downside
is not ready to be worked on.

Closed items are deleted, not archived: git history is the archive.

---

## 1. Facts are asserted, never verified — "claim + check"

**What is wrong.** A learned fact is a bare string. Nothing establishes that it
was ever true, and nothing notices when it stops being true. A real run produced
five, of which one was a fact; the rest were a diary of that run, in the past
tense, and all five went into the system prompt of every later task.

Two reviews (codex + a second agent, run separately) agreed on the diagnosis:

> The main defect is promoting a run summary into authoritative project memory
> without checking usefulness or continued truth. Better wording can reduce diary
> entries; it cannot resolve stale or contradictory facts.

`#313` fixed the *degradation* mechanism (a re-derived fact now moves back into
the injected window instead of being discarded) and `#314` made facts readable
and removable. Neither makes a fact *checkable*.

This is now the largest item here: the judge-side entries that used to sit
alongside it were closed by showing the judge the conversation itself (`#316`),
which deleted the hand-built evidence summary and with it the two queued fixes
for what that summary dropped.

**The design.** Store a fact as a claim plus the means to re-establish it:

```json
{
  "claim": "Makefile has 'run' and 'all' targets, both calling go run main.go",
  "check": {"file": "Makefile", "contains": "go run main.go"}
}
```

- **The diary is excluded structurally, not lexically.** No check can be written
  for "Switched to type.fit" — it is an event, not a property — so it cannot be
  stored. This matters because both reviews explicitly rejected a word filter:
  *"Generated files must not be edited"* is a legitimate rule a tense/keyword
  blacklist would reject, while *"The project uses type.fit"* is relabelled
  narration that would pass one.
- **Stale facts remove themselves.** File-backed checks are re-run before
  injection — a file read and a substring test, no model call. A fact whose check
  fails is deleted, not flagged.
- **The heading can go back to asserting.** `#314` had to weaken it to "notes
  … not re-checked since" precisely because nothing re-checks them.

**What it breaks.**
- Facts established by *execution* rather than by file content ("the integration
  suite needs a reachable Docker daemon") have no cheap check. Mitigation: allow
  a command check, run it only at write time, and mark the fact
  `verified <date>, not re-checked`.
- A model can write a check that always passes (`contains: "a"`). Mitigation: a
  minimum length, and it is visible in `/knowledge`.
- **Migration is the real risk.** `loadFacts` swallows a JSON unmarshal error
  *silently* (`if json.Unmarshal(data, &f) == nil`), so changing the on-disk
  shape would make every existing user's facts vanish with no message. The reader
  must try the old `[]string` shape first and keep unverifiable legacy entries
  until they are re-derived.
- Far fewer facts will be stored. Given the current one-in-five ratio, that is
  the point, but it will look like a regression at first.

---

## 2. `knowledge budget` — one knob, measured in prompt, not in entries

**What is wrong.** `maxFacts` (30) and `maxInjectedFacts` (15) are constants.
Neither is settable, and the count is the wrong unit: fifteen entries is 300
tokens or 3000 depending on their length, which matters on a 32k window and not
at all on a 200k one.

This was deliberately deferred until `#313`, because before that fix "the most
recent N" selected the *diary*, and a knob would only have let the user choose
how much of it to keep. With re-derivation now refreshing a fact's position,
"most recent" means "most recently confirmed" and the knob becomes meaningful.

**The design.** A single `/config` row under **Learning**, following the
`max_steps` / `max_history` pattern already in the codebase: `0 = auto` (a small
fraction of the context window), then fixed presets.

**What it breaks.** Little, but it is the smallest item here and should not be
done before the ones above — a budget over unverified facts is a knob on the
wrong quantity.

---

## 3. Lessons are keyed on error text, which is unstable

**What is wrong.** A pitfall is retrieved only when the stored `error_pattern` is
a literal substring of a later error *and* `contextMatchesAction` passes. Both
reviews found the same consequences:

- `contextMatchesAction` rejects on English, not structure: a context reading
  `"file: search — couldn't find the pattern"` is rejected for action `search`
  because it contains the word `find`. Same for `"git: commit after add"` on
  `commit`, `"web: fetch a search result page"` on `fetch`.
- `parseLessons` accepts an empty `error_pattern`, which `hints` then always
  rejects — a lesson that can never fire, stored and aging toward purge.
- `parseLessons` does not verify the proposed pattern actually occurred in the
  error it was distilled from, so a paraphrase is stored and never matches.
- `execOne` deliberately bypasses hints for `"no action given"` errors, though
  reflection can and does store lessons about them.

**The fix.** Narrow `contextMatchesAction` to a delimited-token test, and reject
at write time a pattern that does not appear in the transcript it came from.

**What it breaks.** A delimited-token test lets through a genuinely mismatched
context when the reflecting model wrote `"file: edit"` for a `write` failure.
Requiring the pattern to appear in the transcript needs the transcript at store
time, which `reflectAndStore` has but `KB.Add` does not — a signature change
across two packages.

---

## 4. A lesson cannot describe a mistake that did not fail

**What is wrong.** `hints` fires inside `if res.IsError`, so the lesson store can
only ever describe errors. The mistakes that cost the most are not errors:

| mistake | why the shape misses it |
|---|---|
| a call that succeeded and was wrong (`file.write` clobbering a file that wanted `edit`) | no error, no retrieval path |
| an omission ("you never re-read the file, so the judge could not verify it") | there is no failing call to attach it to |
| wrong ordering across tools | `domain` is a single tool name |
| "the tests pass but the wrong requirement was implemented" | nothing failed |

**Not designed yet.** Any trigger other than "a tool call failed" changes when
hints are injected, which changes the prompt on successful turns — the one place
the current design is careful to stay out of. Both reviews flagged the shape as
limiting; neither proposed a mechanism that pays for itself. It is here so it is
not rediscovered as new.
