# ipsupport-code — design spec

**Date:** 2026-06-24
**Repo:** `github.com/ipsupport-llc/ipsupport-code` (local `~/gh/ipsupport-code`)
**Status:** approved, implementing

## 1. Purpose

A small self-learning agent for **LM Studio** running locally on a Mac (Apple
Silicon or Intel). It is meant for micro-tasks driven by a small local model —
not a replacement for a frontier model, but a useful local hands-on agent that
gets better over time by remembering what worked.

Single static Go binary. `ipsupport-code "goal"` runs one-shot; with no argument
it opens a REPL. It talks to LM Studio's OpenAI-compatible server
(`http://localhost:1234/v1`) using **native function-calling**.

Lineage: this is a Go rethink of a Python prototype (fat domain tools +
knowledge base of pitfalls + reflection loop). The fat-tool ergonomics are
ported from the operator's `mcp-weather-simple` (`fat_tools_lean.py`):
`(action, params)` signatures, contract-in-docstring, self-correcting error
messages, `NOT here →` cross-references, and tight catalog-token budget.

## 2. Core ideas

1. **Real agent loop** against LM Studio, native tool-calling only.
2. **Fat tools by domain.** One tool per domain, `action` discriminator inside.
   Keeps the per-turn tool catalog tiny (~1k tokens) so small CPU/GPU models on
   a Mac prefill fast and route well.
3. **Help is proactive.** A small model will rarely call `help` on its own. So
   learned lessons (`Pitfall`) are **auto-injected into tool-error observations**
   at the moment a tool fails. `help(domain)` also stays available as an explicit
   escape hatch.
4. **Two lines of help.** First line: the tools self-correct (wrong action →
   "that action belongs to domain X"; missing param → list all missing). Second
   line: learned pitfalls layered on top.
5. **Reflection learns.** After a task ends, a second LM Studio pass reviews the
   transcript and distills 0+ durable, environment-general lessons into a
   persisted JSON knowledge base. Available on the next run.
6. **Catalog-token discipline** is load-bearing — the central constraint, per the
   mcp-weather lesson.
7. **Safety is workspace-configurable** — a small model running shell on the
   user's Mac must be gated by an allow/deny policy that lives in the workspace.

## 3. Agent loop

```
system prompt + tiny tool catalog + user goal
└─ loop (until max_steps):
   POST /v1/chat/completions {messages, tools}
   ├─ response has tool_calls? → for each call:
   │   1) validate args (self-correct: check_action / require)
   │   2) policy gate (allow / ask y-n / deny)
   │   3) execute → observation
   │   4) on error → augment observation with matching Pitfalls from KB
   │   append tool result message; continue loop
   └─ response is content only (no tool_calls)? → final answer; stop
after loop → reflection pass → persist new lessons
```

Stop conditions: final answer (assistant message with no tool calls), `max_steps`
reached, or user abort (Ctrl-C / "stop" in REPL).

## 4. Fat tools

Style mirrors `fat_tools_lean.py`: each domain is one tool taking `action` plus a
`params` object; the contract for each action lives in the tool description; each
description ends with `NOT here →` pointers to sibling domains. Errors are
self-correcting.

- **file** *(policy + jail gated)*
  - `read` `{path}`
  - `write` `{path, content}`
  - `append` `{path, content}`
  - `list` `{path?}` — directory listing
  - `mkdir` `{path}`
  - (no `delete` in v1 — dangerous; add later behind a hard deny gate if needed.)
- **run**
  - `shell` `{command, cwd?}` → `{stdout, stderr, exit_code}`. Policy gated.
- **web**
  - `search` `{query, limit?}` — DuckDuckGo HTML endpoint, no key.
  - `fetch` `{url}` — fetch a page, return readable Markdown (readability +
    html-to-markdown).
  - `stackexchange` `{query, site?, tag?, limit?}` — `api.stackexchange.com`
    search/excerpts, open API (no key at low volume). `site` defaults to
    `stackoverflow`.
- **help**
  - `lessons` `{domain}` — return learned pitfalls for a domain (escape hatch).
- **calc**
  - `eval` `{expression}` — AST-safe arithmetic (port of the mcp-weather
    whitelist). Cheap; small models are unreliable at multi-digit/chained math.

## 5. Self-learning

```go
type Pitfall struct {
    Domain       string `json:"domain"`
    ErrorPattern string `json:"error_pattern"` // substring/keywords to match observed errors
    Context      string `json:"context"`
    ProvenFix    string `json:"proven_fix"`
    Hits         int    `json:"hits"` // times reused; for ranking/pruning
}
```

- **KnowledgeBase** — JSON file, **global** by default
  (`~/.config/ipsupport-code/knowledge.json`), because most lessons (shell, fs,
  web, OS quirks) are cross-project. Path overridable in config.
  - `Query(domain, errText) []Pitfall` — filter by domain, rank by keyword
    overlap between `errText` and each pitfall's `ErrorPattern`/`Context`, return
    top N.
  - `Add(p Pitfall)` — dedupe by `(domain, normalized error_pattern)`; bump
    `Hits` instead of duplicating.
  - No embeddings in v1 (YAGNI). Keyword overlap is the matcher.
- **Help injection** — when a tool returns an error, the agent calls
  `Query(domain, errText)` and appends matches to the observation:
  ```
  ERROR: <err>
  Hints from past runs:
  - When you saw "<error_pattern>" while <context>, this worked: <proven_fix>
  ```
- **Reflection** — after the task, send a compacted transcript (the error →
  recovery pairs and the final outcome) to LM Studio asking for durable,
  environment-general lessons as a JSON array of
  `{domain, error_pattern, context, proven_fix}`. Parse leniently, dedupe, persist.
  Prompt explicitly forbids project-specific noise.

## 6. Safety — workspace-configurable policy

Config lives in the workspace at `.agent/config.json`:

```json
{
  "llm": {
    "base_url": "http://localhost:1234/v1",
    "model": "qwen2.5-7b-instruct",
    "temperature": 0.2,
    "max_steps": 12
  },
  "run": {
    "default": "ask",
    "allow": ["ls*", "cat*", "git status", "go build*", "go test*"],
    "deny":  ["rm -rf*", "sudo*", "*| sh", "*curl*sh*"]
  },
  "file": {
    "default": "ask",
    "jail": ".",
    "allow_write": ["**/*.go", "notes/**"],
    "deny_write":  [".git/**", "**/*secret*", "**/.env*"]
  }
}
```

Per-action resolution:

1. Matches a **deny** glob → blocked; return an error to the model, never execute.
2. Matches an **allow** glob → execute without prompting.
3. Otherwise → **default**: `ask` (interactive y/n) | `allow` | `deny`.

Plus optional **jail**: for `file` operations (and `run` cwd), the resolved
absolute path must stay under the jail root; escapes (`..`, symlink, absolute
outside) are blocked. `jail: ""` disables the jail.

This is the operator's "hybrid 1+3 via wildcards, 2 as an option, configurable in
the workspace itself": allow/deny wildcards give auto-allow + confirm + hard-block
in one model, and the jail gives the sandbox option.

Globs: `**` path matching via `bmatcuk/doublestar`. Shell-command matching uses
the same glob engine against the raw command string (prefix/wildcard).

## 7. Layout

```
ipsupport-code/
  go.mod                       module github.com/ipsupport-llc/ipsupport-code
  Makefile                     build / release matrix / test
  cmd/agent/main.go            CLI / REPL entrypoint
  internal/
    llm/client.go              LM Studio OpenAI-compatible client (native tool_calls)
    agent/loop.go              orchestration loop
    tool/tool.go               Tool interface, catalog/schema build, require/check_action
    tool/registry.go           registry + dispatch
    tool/file.go run.go web.go help.go calc.go
    knowledge/kb.go pitfall.go persisted lessons
    reflect/reflect.go         post-task learning pass
    policy/policy.go           workspace permission engine
    config/config.go           global + workspace config load
  eval/cases.json eval.go      tool-routing eval harness (stub in v1)
  README.md
  .agent/config.example.json
```

## 8. Build (cross-compile from Linux)

All dependencies are pure Go, so `CGO_ENABLED=0` cross-compiles cleanly to every
target from the Linux dev box.

```makefile
PKG := ./cmd/agent
BIN := ipsupport-code
PLATFORMS := linux/amd64 darwin/amd64 darwin/arm64

release:
	@for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -o dist/$(BIN)-$$os-$$arch $(PKG); \
	done

build:
	go build -o dist/$(BIN) $(PKG)
```

Artifacts: `linux-amd64` (dev/test), `darwin-amd64` (Intel Mac), `darwin-arm64`
(Apple Silicon). On the Mac: copy the binary, `xattr -d com.apple.quarantine
./ipsupport-code-darwin-arm64` if Gatekeeper-quarantined, and have LM Studio's
server running on `localhost:1234`. CI runs `release` and attaches all three.

## 9. Dependencies

- `github.com/bmatcuk/doublestar/v4` — `**` globs for policy.
- `github.com/go-shiori/go-readability` — strip page boilerplate before convert.
- `github.com/JohannesKaufmann/html-to-markdown` — fetch → clean Markdown.
- Everything else stdlib (`net/http`, `encoding/json`, `os/exec`, `go/ast` etc.).
- DuckDuckGo and StackExchange are plain `net/http` + `encoding/json`.

## 10. Testing

TDD throughout.

- **policy** — deny precedence, allow auto-pass, default fallthrough, jail escape
  (`..`, absolute-outside, symlink).
- **tool** — dispatch, `check_action` (wrong-domain hint), `require` (missing-param
  list), per-tool action coverage.
- **knowledge** — dedupe on add, `Query` keyword ranking, JSON round-trip.
- **agent loop** — driven by an `httptest` fake LM Studio returning canned
  `tool_calls`, then a final content message; assert tools fire, observations feed
  back, error→pitfall injection happens, loop terminates.
- **calc** — arithmetic correctness + rejection of unsafe expressions.
- **eval** — tool-routing harness (`cases.json`), stub in v1.

## 11. Out of scope (v1)

- Embeddings / semantic pitfall matching (keyword overlap is enough to start).
- `file.delete`.
- Text-protocol tool-calling fallback (native only, per decision).
- Per-workspace (vs global) knowledge base split.
- Multi-model routing, streaming UI niceties.

## 12. Decisions (settled)

- Repo name: `ipsupport-code`.
- Tool-calling: native function-calling only.
- Web/Q&A: search + fetch + StackExchange API.
- Safety: workspace `.agent/config.json` with allow/deny globs + optional jail.
- KB: global JSON file. Config: JSON (zero-dep, inspectable), not YAML.
- `calc` included; `file.delete` excluded.
- Build targets: `linux/amd64`, `darwin/amd64`, `darwin/arm64`.

## 13. Refinements (2026-06-24)

Three follow-up requests, adapted from Python idioms to right-sized Go.

### 13.1 Async / non-blocking — goroutines + context, not asyncio

Go has no asyncio; the equivalent is `context.Context` + goroutines.

- `context.Context` is threaded through `llm.Chatter.Chat` and `tool.Tool.Call`,
  giving per-step timeouts and Ctrl-C cancellation. A hung `web.fetch` or
  `run.shell` is cancellable; the REPL never freezes.
- When one assistant turn returns multiple `tool_calls`, they execute
  concurrently (a goroutine per call; results reassembled in the model's emitted
  order). The common case — one call per turn from a small model — runs inline.
- We deliberately do NOT add a task runtime / event loop; context + goroutines
  cover the requirement without the ceremony.

### 13.2 Error model — typed errors at boundaries, Results for the model

Two tiers, kept distinct on purpose:

- **Model-recoverable** failures (bad path, denied, missing param, wrong action)
  stay `tool.Result{IsError:true}` *text* so the loop continues and the model
  self-corrects. These are NOT Go errors.
- **Host/infra** failures are typed Go errors: `tool.ToolError{Tool,Action,Err}`,
  `knowledge.KnowledgeError{Op,Path,Err}`, `reflect.ReflectionError{Err}` — each
  implements `error` + `Unwrap` for `errors.As`. `tool.Result` gains an optional
  `Err error` carrying a `*ToolError` on host failure, so the host logs the typed
  cause while the model still sees `Content`. Reflection *parse* failures return
  `[]` (no error); only transport failures yield `ReflectionError`.

### 13.3 Provider swappability — one interface, no LangChain

The model sits behind `llm.Chatter` (`Chat(ctx, msgs, tools) (Message, error)`).
Both the agent's reasoning and the `reflect.Reflector` depend only on `Chatter`.

- LM Studio, OpenAI, and an OpenAI-compatible **LiteLLM proxy** are the SAME
  `llm.OpenAIClient` — swap via `base_url` + `api_key` in config, zero code.
- Anthropic (or any non-OpenAI API) is a small adapter struct implementing
  `Chatter`, added when needed.
- We do NOT pull LangChain/LiteLLM into Go — a one-method interface is the whole
  abstraction; a framework dependency would be over-engineering for a micro-agent.

### 13.4 Logging & trace — the training dataset

Two separate streams:

- **Operational logging** via stdlib `log/slog` to stderr, level from config/env
  (`IPS_LOG=debug|info|warn`). Replaces every `print`/`fmt.Println` debug line.
- **Decision trace** via `internal/trace`: every step of a task — `goal`,
  `assistant` (model text + tool calls), `tool_call`, `observation`, `final`,
  `lesson` — is appended as a JSONL record tagged with a run id and RFC3339 time
  to `~/.config/ipsupport-code/traces.jsonl` (path configurable). One run = one
  trajectory; the file is the future training dataset (state → action →
  observation → outcome shape). `agent.Agent` and `reflect.Reflector` take a
  nil-safe tracer; the CLI wires the real one. Interactive approver prompts stay
  on stdout (UI, not logging).
