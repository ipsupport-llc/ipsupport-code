---
name: git-flow
description: Branch off main, commit at the end, never push red.
when: making changes in a git repository
---
Keep version control clean:

- Don't commit straight to the default branch — create a topic branch first.
- Make the change complete and working before committing; group related edits.
- Write a short imperative commit subject (e.g. "Fix off-by-one in parser").
- Run the tests and only push when they're green. Never push a broken build.
- Don't commit secrets, build artifacts, or unrelated formatting churn.
