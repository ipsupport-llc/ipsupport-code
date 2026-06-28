---
name: plan
description: Track a multi-step task in .agent/plan.md so nothing is forgotten and you can resume without being told what's left.
when: a task has several steps, or the user says "continue" / "you didn't finish it"
---
For anything with more than a couple of steps, keep a checklist on disk and work it
to completion — so both you and the user can see what's done and what isn't.

1. At the start, write the plan to `.agent/plan.md` with the file tool — a short
   markdown checklist, one line per real step:
   ```
   - [ ] add the endpoint
   - [ ] wire the route
   - [ ] write a test
   ```
   Keep it to the actual steps (≈8 max), not narration.
2. As you FINISH each step, mark it done with file edit — change `- [ ] that step`
   to `- [x] that step`. Update it as you go, not all at the end.
3. Before you stop, read `.agent/plan.md`. If any `- [ ]` items remain, keep going —
   don't end the turn with unfinished items unless you're genuinely blocked (then
   say what's blocking and what's left).
4. Resuming — when the user says "continue" or "you didn't finish": read
   `.agent/plan.md` FIRST, then pick up the first unchecked item. Don't restart
   from scratch or re-do checked items.

The plan file is the shared source of truth for progress; keep it honest and in
sync with what you've actually done.
