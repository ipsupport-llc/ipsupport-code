---
name: test-first
description: Write a failing test before the code, then make it pass.
when: adding a feature or fixing a bug in a project that has tests
---
Work test-first whenever the project has a test suite:

1. Write (or extend) a test that captures the desired behaviour — for a bug, one
   that reproduces it. Run it and watch it FAIL for the expected reason.
2. Write the minimum code to make that test pass. Nothing speculative.
3. Run the whole suite. Keep it green before moving on.

Report the test you added and the run result, not just "done".
