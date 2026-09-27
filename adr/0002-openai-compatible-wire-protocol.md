# 0002. Speak one wire protocol: OpenAI-compatible chat completions, local first

- **Status:** Accepted
- **Date:** 2026-09-27 (recorded retrospectively — describes what the code does today)

## Context

The agent is meant to drive the user's own model, which today means an
OpenAI-compatible server — LLMTray, LM Studio, Ollama, vLLM, a LiteLLM proxy —
and, when the user wants one, a cloud provider. Every cloud provider we care about
also exposes an OpenAI-compatible endpoint.

## Decision

`internal/llm` has one client, speaking `/v1/chat/completions` with native tool
calling and streaming. Everything that reasons or reflects depends only on the
`Chatter` interface. Providers — `local` plus the presets `openai`, `anthropic`,
`grok`, `groq`, `openrouter`, `zai` (`internal/config`), or any custom entry under
`providers` — differ only in base URL, model and key. Anthropic is reached
through its OpenAI-compatible endpoint, not a separate adapter.

Local is the default. A keyless local server is a first-class provider. A
provider name that doesn't resolve is an error at startup, never a silent fall
back to local while the status line shows the name the user typed.

## Consequences

- Adding a provider is a config entry, not code.
- Provider-specific features that exist only in a native API (not in the
  OpenAI-compatible surface) are not available.
- Tolerance for gateway quirks lives in the one client — e.g. tool-call arguments
  arriving as a JSON-encoded string.
