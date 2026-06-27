# ipsupport-code

[![ci](https://github.com/ipsupport-llc/ipsupport-code/actions/workflows/ci.yml/badge.svg)](https://github.com/ipsupport-llc/ipsupport-code/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/ipsupport-llc/ipsupport-code?sort=semver)](https://github.com/ipsupport-llc/ipsupport-code/releases)
[![license](https://img.shields.io/github/license/ipsupport-llc/ipsupport-code)](LICENSE)
[![go](https://img.shields.io/github/go-mod/go-version/ipsupport-llc/ipsupport-code)](go.mod)

**Website: [ipsupport-llc.github.io/ipsupport-code](https://ipsupport-llc.github.io/ipsupport-code/)**

A small **self-learning local coding agent** for [LM Studio](https://lmstudio.ai),
in a single static binary. It drives a local model through a reason → act →
observe loop over a handful of fat tools (`file`, `run`, `git`, `web`, `calc`),
recovers from tool errors using lessons it learned on past runs, and — after each
task — reflects and writes new lessons to disk so it actually gets better over
time.

The interactive UI is a Claude-Code-style **TUI** (Bubble Tea): live-streamed
tool calls and observations, markdown answers, syntax-highlighted diffs, a
**plan / auto** mode toggle, non-stealing approval prompts, and a status bar
showing the model, context usage, and tokens. Piped/non-interactive runs fall
back to a plain line REPL.

It will not match a frontier model. It's built for **micro-tasks** on your own
machine, with your own model, under a permission policy you control.

> Not affiliated with Anthropic's Claude Code; the name reflects a similar
> terminal-agent UX.

## Install

**One-liner** (macOS/Linux) — auto-detects your platform, verifies the SHA-256,
installs to `~/.local/bin`, and prints how to run it (and how to add it to PATH
if needed):

```sh
curl -fsSL https://raw.githubusercontent.com/ipsupport-llc/ipsupport-code/main/scripts/install.sh | sh
```

That installs the latest **nightly**; append `-s -- latest` for the newest stable
release, `-s -- v0.3.1` for a specific tag, or a second arg for a custom path.

**Or download the archive** for your platform from the
[latest release](https://github.com/ipsupport-llc/ipsupport-code/releases/latest)
(or the rolling [`nightly`](https://github.com/ipsupport-llc/ipsupport-code/releases/tag/nightly)):
`darwin-arm64` (Apple Silicon), `darwin-amd64` (Intel Mac), `linux-amd64`,
`linux-arm64`, `windows-amd64`. Each release also ships `checksums.txt` (SHA-256).

```sh
tar -xzf ipsupport-code_*_darwin-arm64.tar.gz   # .zip on Windows
chmod +x ipsupport-code
# macOS may quarantine a downloaded binary:
xattr -d com.apple.quarantine ./ipsupport-code
./ipsupport-code -version
```

**Or build from source** (pure Go, `CGO_ENABLED=0`, cross-compiles from any host):

```sh
make build     # host binary at dist/ipsupport-code
make release   # every target into dist/
go install github.com/ipsupport-llc/ipsupport-code/cmd/agent@latest  # installs as `agent`
```

## Providers — local or external

The agent and tools are model-agnostic, so you can point them at any
OpenAI-compatible API and switch in one command — run a strong cloud model for a
hard task, then drop back to your local one:

```
/ai                       list providers (local + openai, grok, groq, openrouter)
/ai key openai sk-…       add an API key for a provider
/ai openai                switch to it   ·   /ai local   back to LM Studio
/model                    list models   ·   /model gpt-4o   pick   ·   /model sonnet   filter (great for OpenRouter)
```

Switching keeps your session, tokens, and mode. Keys live in
`~/.config/ipsupport-code/config.json` (written `chmod 600`) or fall back to the
env var (`OPENAI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`).
`/config` opens an interactive settings panel: **↑↓** to move, **Enter** to
cycle a value in place (provider, mode, permissions, run timeout, color, channel)
or jump to the right flow (model, key, rename), **esc** to close — changes apply
and save as you make them, no hand-editing JSON.

## Updating

The binary updates itself in place from GitHub Releases:

```sh
ipsupport-code update            # update on the current channel
ipsupport-code update nightly    # switch to the nightly channel (saved) and update
ipsupport-code update stable     # switch back to stable
```

`/update` does the same from inside the REPL. On startup it does a quick,
best-effort check and prints a one-line notice if a newer build is out on your
channel (`stable` by default — set `"channel": "nightly"` in
`~/.config/ipsupport-code/config.json`, or switch with `update nightly`). Local
dev builds are never nagged.

## Quick start

1. In LM Studio, load a **tool-calling** model (e.g. `qwen2.5-7b-instruct`) and
   start the local server on port `1234`.
2. First interactive run walks you through the server URL, model, and context
   window; settings are saved to `~/.config/ipsupport-code/config.json` (re-run
   with `-init`).
3. Run a one-shot task, or open the REPL:

   ```sh
   ./ipsupport-code "use calc to compute (1234*9)+sqrt(2)"
   ./ipsupport-code                      # interactive TUI
   ./ipsupport-code -C ~/proj "summarize what main.go does"
   ```

## Modes: auto vs plan

Toggle with **shift+tab** (or `/plan` / `/auto`); the current mode shows at the
bottom of the screen.

- **auto** (default) — the agent executes the task with tools.
- **plan** — read-only: it investigates and proposes a numbered plan, and every
  state-changing tool call is **blocked at the engine**, so it can't touch
  anything until you switch back to auto.

## Permissions

Mutating actions ask for approval by default. The prompt **doesn't steal your
input** — keep typing, then press **↑** to answer (←→ / `y`/`n`, Enter confirms).
A non-overridable deny floor (`rm -rf`, `sudo`, secrets, `.git`, `.env`, …) is
always enforced.

Tired of approving every write? `/permissions files on` auto-allows
non-destructive file ops in the workspace (the deny floor still applies); same
for `/permissions run on`. The choice is saved to the workspace config.

## Skills

On-demand instruction packs — the user-extensible version of guides-on-demand.
Only an **enabled** skill adds a single line to the system prompt; the model
loads a skill's full instructions on demand, so the base prompt stays lean no
matter how many you install. Five curated skills ship in the binary
(`test-first`, `debug-systematically`, `git-flow`, `research-first`,
`minimal-code`), seeded **disabled** so you opt in.

```
/skills                       list installed skills (on/off)
/skills on git-flow           enable one
/skills install <url|git>     add a .md by URL, or every skill in a git repo
/skills remove <name>
```

## Context & auto-compact

The status bar shows `ctx 4.1k/8k` — the size of the last prompt vs. the model's
context window. The window is **auto-detected** from LM Studio's
`/api/v0/models`; set `llm.context_window` to override (0 disables auto-compact).
When the prompt passes ~75% of the window the session is **auto-compacted** into
a short summary to free room (run it any time with `/compact`).

## REPL commands

Anything not starting with `/` is run as a task. Tab completes commands.

| command | what |
|---|---|
| `/plan`, `/auto` | plan mode (propose only) vs auto mode (execute) — also shift+tab |
| `/skills` | list / toggle / install on-demand instruction packs |
| `/permissions` | relax approval for non-destructive file/shell actions |
| `/status` | config, knowledge base, and trace paths |
| `/usage` | session counters + token history (by day, by provider/model) |
| `/login` | (re)configure server URL / model / key, then reload |
| `/new` | clear the session conversation memory |
| `/clear` | fresh start — clear the screen and the session |
| `/compact` | summarize the session so far to free up context |
| `/color [name]` | change the TUI frame color (cycles if no name) |
| `/rename <name>` | rename the agent (saved in settings) |
| `/loop <interval> [xN] <task>` | re-run a task on an interval (e.g. `/loop 5m <task>`, `/loop 30s x10 <task>`); **esc** stops it |
| `/help` | command list |
| `/exit`, `/quit` | leave |

The input is multi-line: paste a whole block (e.g. a YAML snippet) and it keeps
its line breaks, the box grows and word-wraps instead of scrolling on one line,
and **alt+enter** (or **ctrl+j**) inserts a newline by hand. **Enter** submits.

While a task runs the input stays live: Enter **queues** a follow-up (pinned
above the input until it runs), **↑** pulls the last queued message back to edit
or drop, and **esc** cancels.

**Shell.** `/shell` (or `!`, or Enter on an empty prompt) drops you into an
interactive shell in the workspace — do things by hand, exit to return. `!cmd`
runs a single command and shows its output. These are your commands, not the
agent's, so they aren't gated by the permission policy.

**Custom system prompt.** The built-in engine prompt is deliberately tiny; you
can replace it with `.agent/system.md` (per project) or
`~/.config/ipsupport-code/system.md` (global). `ipsupport-code -dump-prompt`
prints the default to start from (`> .agent/system.md`). Your `CLAUDE.md`,
environment, and skills are still appended after it. `/status` shows which
prompt is in effect. (A bloated prompt makes a small model call tools worse —
edit at your own risk.)

## How it works

- **Native tool calling.** Talks to LM Studio's OpenAI-compatible server and lets
  the model call tools natively. Point `llm.base_url` / `llm.api_key` at OpenAI or
  a LiteLLM proxy instead — same client.
- **Fat tools.** One tool per domain, each `{"action": ..., "params": {...}}`. The
  catalog stays tiny (~1k tokens) so small models prefill fast and route well; a
  declarative `Domain` generates each tool's schema, help, and validation.
- **Proactive help.** When a tool fails, a matching lesson from past runs is
  injected straight into the error the model sees — it doesn't have to ask.
- **Reflection.** After a task, a second model pass distills durable lessons into
  `~/.config/ipsupport-code/knowledge.json` (env-general tool pitfalls) and durable
  **facts** about the current project (build/test/run commands, where things live,
  conventions) into `<workspace>/.agent/facts.json` — folded into the prompt next run.
- **Code search.** The `file` tool's `search` action greps the workspace by regex
  (`file:line: match`), skipping VCS/dep/build dirs and binaries — no external `grep`.
- **Session memory.** Remembers your goals and its answers across turns and across
  restarts (`.agent/session.json`, per workspace). `/new` wipes it.
- **Resilience.** Exponential-backoff retry on transient 5xx/network errors, an
  idle watchdog that aborts a silently-stalled stream, and a stuck-loop guard.
- **Project instructions.** Reads a `CLAUDE.md` / `AGENTS.md` / `.agent/instructions.md`
  from the workspace into the system prompt.
- **Trace = dataset.** Every step (goal, tool call, observation, final, lesson) is
  appended as JSONL to `~/.config/ipsupport-code/traces.jsonl`.
- **Usage ledger.** Token spend is recorded per day and per provider/model to
  `~/.config/ipsupport-code/usage.json`; `/usage` shows the history.

## Configuration

Settings merge over safe defaults from two JSON files:

- **`~/.config/ipsupport-code/config.json`** — machine-level: the `llm` connection
  (server URL, model, key, `context_window`). Written by first-run setup.
- **`<workspace>/.agent/config.json`** — per-project: the permission policy (see
  [`.agent/config.example.json`](.agent/config.example.json)). Wins over the user
  file.

`run.timeout_seconds` caps how long a shell command may run (default 60s); raise
it for slow builds/test suites, or let the model pass a larger per-call `timeout`.

Permissions for `run` and `file` resolve per action: a **deny** glob blocks, an
**allow** glob runs without asking, otherwise the **default** (`ask`/`allow`/`deny`)
applies. Run-command deny globs match *anywhere* in the command (so `rm -rf*`
catches `cd x && rm -rf /`); file globs are path-aware (`**`, `*.go`) and confined
to `jail`. The protective deny floor is always unioned in — your config adds to
it, it can't remove it.

Logging: `IPS_LOG=debug|info|warn|error` (default `warn`) to stderr.

## Layout

```
cmd/agent          CLI, plain REPL, and the Bubble Tea TUI
internal/llm        LM Studio client (streaming, retry, context detection)
internal/agent      the reason → act → observe loop (+ plan mode)
internal/tool       fat tools: file, run, git, web, calc, skill, help
internal/skill      downloadable, toggleable instruction packs
internal/policy     workspace permission engine (+ jail, deny floor)
internal/knowledge  persistent pitfall store
internal/reflect    post-task lesson + project-fact distillation
internal/trace      JSONL decision trace (the dataset)
internal/config     config load/merge
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs gofmt, `go vet`, the race suite,
and a cross-compile of every target on each push and PR.

## License

[MIT](LICENSE) © ipsupport-llc
