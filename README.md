# ipsupport-code

A small **self-learning local agent** for [LM Studio](https://lmstudio.ai). It
drives a local model through a reason → act → observe loop over a handful of fat
tools (`file`, `run`, `web`, `help`, `calc`), recovers from tool errors using
lessons it learned on past runs, and — after each task — reflects and writes new
lessons to disk so it actually gets better over time.

It will not match a frontier model. It is built for **micro-tasks** on your own
machine, with your own model, under a permission policy you control.

## How it works

- **Native tool calling.** Talks to LM Studio's OpenAI-compatible server
  (`http://localhost:1234/v1`) and lets the model call tools natively.
- **Fat tools.** One tool per domain, each taking `{"action": ..., "params": {...}}`.
  The catalog stays tiny (~1k tokens) so small models prefill fast and route well.
- **Proactive help.** When a tool fails, matching lessons from the knowledge base
  are injected straight into the error the model sees — it doesn't have to ask.
- **Reflection.** After a task, a second model pass distills durable lessons into
  `~/.config/ipsupport-code/knowledge.json`. They're available next run.
- **Trace = dataset.** Every step (goal, tool call, observation, final, lesson)
  is appended as JSONL to `~/.config/ipsupport-code/traces.jsonl` — your future
  training dataset.

## Build

Everything is pure Go, so any target cross-compiles from any host with
`CGO_ENABLED=0`.

```sh
make release   # → dist/ipsupport-code-{linux-amd64,darwin-amd64,darwin-arm64}
make build     # host binary at dist/ipsupport-code
make test      # all tests
```

## Run on your Mac

1. In LM Studio, load a tool-calling model (e.g. `qwen2.5-7b-instruct`) and start
   the local server on port `1234`.
2. Copy the matching binary over (`darwin-arm64` for Apple Silicon, `darwin-amd64`
   for Intel). If macOS quarantines it:
   ```sh
   xattr -d com.apple.quarantine ./ipsupport-code-darwin-arm64
   ```
3. Run a one-shot task, or a REPL:
   ```sh
   ./ipsupport-code-darwin-arm64 "use calc to compute (1234*9)+sqrt(2)"
   ./ipsupport-code-darwin-arm64            # REPL; 'exit' to quit
   ./ipsupport-code-darwin-arm64 -C ~/proj "summarize what main.go does"
   ```

## Configuration

Drop a `.agent/config.json` in your workspace (see
[`.agent/config.example.json`](.agent/config.example.json)). It's merged over safe
defaults, so you only set what you want to change.

Permissions for `run` and `file` resolve per action: a **deny** glob blocks, an
**allow** glob runs without asking, otherwise the **default** (`ask` / `allow` /
`deny`) applies. File ops are confined to `jail` (set `""` to disable). Run-command
globs are flat (`*` spans anything); file globs are path-aware (`**`, `*.go`).

Point it at OpenAI or a LiteLLM proxy instead of LM Studio by changing
`llm.base_url` / `llm.api_key` — same client.

Logging level: `IPS_LOG=debug|info|warn|error` (default `warn`) to stderr.

## Layout

```
cmd/agent        CLI / REPL
internal/llm      LM Studio client + Chatter interface
internal/agent    the reason→act→observe loop
internal/tool     fat tools: file, run, web, help, calc
internal/policy   workspace permission engine (+ jail)
internal/knowledge persistent pitfall store
internal/reflect  post-task lesson distillation
internal/trace    JSONL decision trace (the dataset)
internal/config   config load/merge
```

See [`docs/superpowers/specs`](docs/superpowers/specs) for the design and
[`docs/superpowers/plans`](docs/superpowers/plans) for the implementation plan.
