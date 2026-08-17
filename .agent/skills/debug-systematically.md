---
name: debug-systematically
description: Reproduce, isolate, find the root cause before changing code.
when: something is broken, failing, or behaving unexpectedly
---
Do not patch on a guess. Work the problem:

1. Reproduce it reliably — find the smallest input or command that triggers it.
2. Read the actual error and the code around it. Form ONE hypothesis about the
   cause and predict what you'd see if it were true.
3. Test that hypothesis (add a print, run a command, read a file) before editing.
4. Fix the root cause, not the symptom. Then reproduce again to confirm it's gone.

If two attempts at the same fix fail, the hypothesis is wrong — step back.
