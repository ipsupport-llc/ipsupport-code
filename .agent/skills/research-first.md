---
name: research-first
description: Look up real docs before guessing an unfamiliar API or error.
when: unsure about an API, library, flag, or an unfamiliar error message
---
Don't invent API signatures or flags from memory. When unsure:

1. Use the web tool — search the official docs, or stackexchange for the exact
   error message in quotes.
2. Prefer primary sources (official docs, the project's README) over blog noise.
3. Confirm the signature/flag exists before you call it; if a call fails, read
   the error — it usually names the right usage — and adjust.

A minute of lookup beats three failed tool calls on a guessed API.
