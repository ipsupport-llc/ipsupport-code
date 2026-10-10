---
name: minimal-code
description: Write the least code that solves the task; nothing speculative.
when: writing or changing code
---
Favour the simplest thing that works:

- Solve exactly what was asked — no features, flags, or abstractions nobody
  requested.
- Don't add error handling for impossible cases or "flexibility" for a single
  use. One use does not need an abstraction.
- Match the surrounding code's style instead of imposing your own.
- Touch only what the task requires; don't refactor or reformat adjacent code.
- If a change could be half the size, rewrite it smaller.
