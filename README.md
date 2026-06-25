# ipsupport-code

A small **self-learning local agent** for [LM Studio](https://lmstudio.ai). It
drives a local model through a reason → act → observe loop over a handful of fat
tools (`file`, `run`, `git`, `web`, `help`, `calc`), recovers from tool errors
using lessons it learned on past runs, and — after each task — reflects and writes
new lessons to disk so it actually gets better over time.

The interactive REPL is a Bubble Tea **TUI**: live-streamed tool calls and
observations, a spinner while the model thinks, in-place `y/n` approval prompts,
and a status bar (model · workspace · tokens). Piped/non-interactive runs fall
back to a plain line REPL.

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
- **Session memory.** Within a REPL session it remembers the conversation (your
  goals and its answers carry across turns), so "what did we just do?" works.
  `/new` clears it.
- **Project instructions.** On startup it reads a `CLAUDE.md` (or `AGENTS.md`, or
  `.agent/instructions.md`) from the workspace and folds it into the system prompt,
  so the model follows your project's conventions.
- **Trace = dataset.** Every step (goal, tool call, observation, final, lesson)
  is appended as JSONL to `~/.config/ipsupport-code/traces.jsonl` — your future
  training dataset and session log.

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
   ./ipsupport-code-darwin-arm64            # REPL
   ./ipsupport-code-darwin-arm64 -C ~/proj "summarize what main.go does"
   ```

### REPL commands

In the REPL, anything not starting with `/` is run as a task. Slash commands:

| command | what |
|---|---|
| `/status` | config, knowledge base, and trace paths |
| `/usage` | session counters + token usage |
| `/login` | (re)configure server URL / model / key, then reload |
| `/new` | clear the session conversation memory |
| `/goal <task>` | run a task explicitly |
| `/loop [n] <task>` | run a task `n` times (default 3) so lessons compound |
| `/help` | command list |
| `/exit`, `/quit` | leave |

## Configuration

**First run.** On its first interactive start (no user config yet) it asks for the
server URL, model, and a couple of settings, then writes them to
`~/.config/ipsupport-code/config.json`. Re-run setup any time with `-init`. A
non-interactive first run (piped/CI) skips the prompt and uses defaults.

Settings come from two JSON files merged over safe defaults:

- **`~/.config/ipsupport-code/config.json`** — machine-level: the `llm` connection
  (server URL, model, key). Written by first-run setup.
- **`<workspace>/.agent/config.json`** — per-project: the permission policy (see
  [`.agent/config.example.json`](.agent/config.example.json)). The workspace file
  wins over the user file.

Each file only needs the keys you want to change; everything else keeps its
default.

Permissions for `run` and `file` resolve per action: a **deny** glob blocks, an
**allow** glob runs without asking, otherwise the **default** (`ask` / `allow` /
`deny`) applies. File ops are confined to `jail` (set `""` to disable).
Run-command **deny** globs match *anywhere* in the command (so `rm -rf*` catches
`cd x && rm -rf /`) and ignore extra whitespace; **allow** globs match the whole
command. File globs are path-aware (`**`, `*.go`). A built-in protective deny
floor (`rm -rf*`, `sudo*`, `mkfs*`, `.git/**`, `**/*secret*`, …) is always
enforced — your config adds to it, it can't remove it. The `jail` confines the
file tool; a shell command can still `cd` elsewhere, so lean on `run.deny`/`ask`
for shell.

Point it at OpenAI or a LiteLLM proxy instead of LM Studio by changing
`llm.base_url` / `llm.api_key` — same client.

Logging level: `IPS_LOG=debug|info|warn|error` (default `warn`) to stderr.

## Layout

```
cmd/agent        CLI / REPL
internal/llm      LM Studio client + Chatter interface
internal/agent    the reason→act→observe loop
internal/tool     fat tools: file, run, git, web, help, calc
internal/policy   workspace permission engine (+ jail)
internal/knowledge persistent pitfall store
internal/reflect  post-task lesson distillation
internal/trace    JSONL decision trace (the dataset)
internal/config   config load/merge
cmd/agent         CLI, plain REPL, and the Bubble Tea TUI
```
