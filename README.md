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

Local by default, but the same loop runs **any OpenAI-compatible provider**
(OpenAI, Anthropic, Grok, Groq, OpenRouter) — switch in one command — and it can
**delegate a task to a sub-agent on a different model**, so a local agent can hand
a review to a frontier one and compare. See [Providers](#providers--local-or-external)
and [Sub-agents](#sub-agents).

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

**Windows** (PowerShell) — installs to `%LOCALAPPDATA%\Programs\ipsupport-code`,
verifies the SHA-256, and adds it to your user PATH:

```powershell
iex (irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1)
```

That's the nightly; for a channel/tag use the scriptblock form (so it can take an
arg): `& ([scriptblock]::Create((irm https://ipsupport-llc.github.io/ipsupport-code/install.ps1))) latest`.

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

**Add a provider by hand.** Any OpenAI-compatible endpoint works as a first-class
provider straight from the config — no command needed, and **no API key required**
for a keyless local server (Ollama, vLLM, a second LM Studio). Add it under
`providers` and point `provider` at it:

```json
{ "provider": "ollama",
  "providers": { "ollama": { "base_url": "http://localhost:11434/v1", "model": "qwen2.5-coder:7b" } } }
```

It's then active at launch and switchable via `/ai ollama` and `/config`.

Switching keeps your session, tokens, and mode. Keys live in
`~/.config/ipsupport-code/config.json` (written `chmod 600`) or fall back to the
env var (`OPENAI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`).
`/config` opens an interactive settings panel: **↑↓** to move, **Enter** to
cycle a value in place (provider, mode, permissions, run timeout, color, channel)
or jump to the right flow (model, key, rename), **esc** to close — changes apply
and save as you make them, no hand-editing JSON.

## Sub-agents

Define a **profile** — a named model — and the agent gains an `agent` tool: it can
**delegate a self-contained task to a sub-agent on that model** and use the answer.
A second opinion, another model's strength, or the same code reviewed across 2–3
models at once. Run your main loop on a local model, then have it fan a review out
to a couple of frontier models and merge the findings.

A profile is the only way to delegate (and so the curated list of what the
assistant may spawn). Build one interactively — provider → model list → name —
from `/config`, or on the command line:

```text
/agents add grok   openrouter x-ai/grok-4.3     # omit the model to list them
/agents add architect openai gpt-4o
/agents                                          # list profiles
/agents rm grok                                  # remove one
```

Then just ask — e.g. *"review internal/tool across grok and architect, then merge."*
The assistant calls `agent(profile, task, dir)`: `dir` (optional, `~` expanded)
points a sub-agent at **another project** — it gets its own jail there, so it can
work on `~/other-repo` without leaving the rest of your disk exposed. A pure
fan-out of several profiles **runs in parallel**, each on its own live status line;
the main model then merges the results into one answer.

Safety: every spawn **asks for approval** (even local ones cost compute) until you
relax it with `/permissions agents on`. Sub-agents read/write files and use git but
**only run shell commands if you enable it** with `/agents exec on`. They inherit
the current plan/auto mode, can't spawn their own sub-agents (depth 1), and their
tokens are recorded in `/usage` like any other.

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

**Working directory.** Launched from a parent dir (or `~`)? `/cd <subdir>` points
the session at your project — relative file/run/git paths resolve there, and
sub-agents inherit it as their default `dir`, so you set the path once instead of
repeating it. It stays inside the workspace jail.

**Offline?** `/offline on` cuts all internet egress — the web tool refuses with a
clear "no internet" message and the startup update check is skipped. Your local
model (LM Studio on localhost) keeps working. `/offline off` re-enables it.

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

**Which session?** Memory is saved per **workspace** (the `-C` dir, default the
current one) and per **agent name** (default `ipsupport-code`). A one-shot or
piped run silently **continues** the saved thread; the interactive TUI opens a
**navigable chooser** (↑↓ to pick a saved session, **enter** to open, **d** to
delete, **esc** for the newest) when any exist. To start clean or pick a thread
from the CLI:

```sh
./ipsupport-code -new "fresh task"              # ignore the saved session
./ipsupport-code -session review "look at X"    # a separate named thread (review.json)
```

In the TUI, `/sessions` lists/switches/deletes threads, `/new <name>` starts a
fresh named thread (the old one stays in `/sessions`), and `/new` clears the
active one.

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
matter how many you install. Eight curated skills ship in the binary
(`test-first`, `debug-systematically`, `git-flow`, `research-first`,
`minimal-code`, `review` — multi-model review via sub-agents — `subagents`, how to
delegate and fan work out, and `plan`, a `.agent/plan.md` checklist so multi-step
work resumes itself), seeded **disabled** so you opt in. Built-in skills refresh
on upgrade unless you've edited them.

```
/skills                       list installed skills (on/off)
/skills on git-flow           enable one
/skills install <url|git>     add a .md by URL, or every skill in a git repo
/skills remove <name>
```

## MCP servers

Connect [Model Context Protocol](https://modelcontextprotocol.io) servers and the
agent gains their tools — but through **one** proxy tool, not by dumping every
server's schema into the prompt (which would swamp a small model's context). Add
them to `~/.config/ipsupport-code/config.json`:

```jsonc
"mcp_servers": {
  "fs":     { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"] },
  "gh":     { "command": "github-mcp-server", "env": { "GITHUB_TOKEN": "…" } },
  "remote": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer …" } }
}
```

Two transports: **stdio** (a local subprocess via `command`) and **HTTP** (a
remote `url`, with `headers` for auth — e.g. `Authorization: Bearer`). `/mcp`
lists the servers and their tools. The model uses the `mcp` tool — `list` to
discover, `schema` to see a tool's inputs, `call` to run one. Servers connect
lazily on first use; **every `mcp call` asks for approval** (it's external code).
Sub-agents don't get MCP.

## When a local model misbehaves

Small/“thinking” local models can loop in their own reasoning or over-think. The
levers, in order:

- **`/reasoning low`** (or `minimal`/`off`) — trims the model's reasoning. There's
  no universal API for this, so it's **per-model** and stored in its provider's own
  shape (`reasoning_effort` for OpenAI, `reasoning:{effort}` for OpenRouter,
  `chat_template_kwargs:{enable_thinking}` for Qwen/LM Studio). `/reasoning low` on
  the current model writes the right shape for known providers; for others set it
  raw in `config.json` under `reasoning` (keyed by `<provider>` or
  `<provider>/<model>`).
- **A runaway turn is auto-stopped.** A single turn generating far past the context
  window (looping) aborts with a clear message instead of streaming for minutes.
  Press **esc** to cancel anything sooner.
- **`/reflect off`** — skip the post-task learning pass if it's where a weak model
  loops (the status shows *“task done — distilling lessons”* so you can tell the
  task itself already finished). Or **`/reflect <profile>`** to run learning on a
  stronger model, and **`/reasoning reflect low`** to give the learning pass its
  own (leaner) reasoning setting.
- **`/skills on plan`** — for long multi-step work, the model keeps a checklist so
  it resumes instead of drifting.

## Rewind

`/rewind` opens a list of the session's steps — pick one (↑↓, enter) to roll back
to **before** it ran: every file that step (or its sub-agents) changed is restored
to its prior content, files it created are removed, and the conversation is trimmed
back. Checkpoints are taken at the start of each turn, so you can rewind no matter
how a turn ended (finished, esc, a loop, an error). Shell commands, git, and
network calls **can't** be undone — only files and the chat. (REPL: `/rewind`
lists, `/rewind <n>` applies.) Snapshots live for the session.

## Goals

A **goal** is a multi-turn objective — not a single turn. `/goal <text>` sets one
and starts pursuing it: the agent works, and when it thinks it's finished a
**judge** (a separate model call) decides whether the goal is *actually* met. If
it isn't, the goal is **re-fed** to the agent — kept in focus, with the gap the
judge named — and it keeps going. This repeats up to a **TTL** (`/goal ttl <n>`,
default 6 re-feeds) before it gives up, on top of the usual esc / stuck / runaway
guards and a hard step cap.

The goal is a first-class, persisted object: it lives in `.agent/goal.json`, so it
survives a restart. `/goal` shows the standing goal and its status; `/goal go`
resumes it; `/goal clear` drops it; `/goal off` disables the loop (a goal then runs
as a single pass). Plain tasks (anything you type that isn't a goal) run as one
pass with no judge overhead — only an explicit goal gets the loop.

The judge defaults to done on any unparseable reply, so a confused model can't trap
the agent in the loop. The whole thing is gated on real progress: a turn that calls
no tools is never judged or re-fed.

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
| `/goal <text>` | set & pursue a multi-turn goal (judge re-feeds until met); `go` · `clear` · `ttl <n>` · `off` |
| `/skills` | list / toggle / install on-demand instruction packs |
| `/permissions` | relax approval for non-destructive file/shell actions |
| `/status` | config, knowledge base, and trace paths |
| `/usage` | token spend + **estimated $** (today / 7d / 30d / all, by day, by model); `clear` · `purge <days>` · `retain <days>` |
| `/login` | (re)configure server URL / model / key, then reload |
| `/new` | clear the session conversation memory |
| `/clear` | fresh start — clear the screen and the session |
| `/compact` | summarize the session so far to free up context |
| `/color [name]` | change the TUI frame color (cycles if no name) |
| `/rename <name>` | rename the agent (saved in settings) |
| `/sessions` | list / switch / delete saved sessions (per agent name) |
| `/agents` | manage sub-agent profiles: `add` / `rm` / `exec` (models the agent tool delegates to) |
| `/loop <interval> [xN] <task>` | re-run a task on an interval (e.g. `/loop 5m <task>`, `/loop 30s x10 <task>`); **esc** stops it |
| `/help` | command list |
| `/exit`, `/quit` | leave |

The input is multi-line: paste a whole block (e.g. a YAML snippet) and it keeps
its line breaks, the box grows and word-wraps instead of scrolling on one line,
and **alt+enter** (or **ctrl+j**) inserts a newline by hand. **Enter** submits.

While a task runs the input stays live: Enter **queues** a follow-up (pinned
above the input until it runs), **↑** pulls the last queued message back to edit
or drop, and **esc** cancels.

**Shell.** `/shell` (or `!`) drops you into an
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
  Each lesson tracks when it was last seen (bumped on recurrence); `/knowledge`
  reports the store and `clear` / `purge <days>` / `retain <days>` prune stale ones
  (`retain` auto-purges on startup) so the memory doesn't accrete junk forever.
- **Code search.** The `file` tool's `search` action greps the workspace by regex
  (`file:line: match`), skipping VCS/dep/build dirs and binaries — no external `grep`.
- **Session memory.** Remembers your goals and its answers across turns and across
  restarts, kept per workspace **and per agent name** (`.agent/sessions/<name>.json`).
  On startup the TUI shows a **navigable chooser** of saved sessions (↑↓ / enter /
  d) — and on restore it replays the recent exchanges so you pick up where you
  left off. `/new <name>` starts a fresh named thread; `/new` wipes the active one.
- **Resilience.** Exponential-backoff retry on transient 5xx/network errors, an
  idle watchdog that aborts a silently-stalled stream, and a stuck-loop guard.
- **Project instructions.** Reads a `CLAUDE.md` / `AGENTS.md` / `.agent/instructions.md`
  from the workspace into the system prompt.
- **Trace = dataset.** Every step (goal, tool call, observation, final, lesson) is
  appended as JSONL to `~/.config/ipsupport-code/traces.jsonl`.
- **Usage ledger.** Token spend is recorded per day and per provider/model to
  `~/.config/ipsupport-code/usage.json` and accumulates across runs; `/usage`
  shows today / 7-day / 30-day / all-time rollups, with `clear`, `purge <days>`,
  and a `retain <days>` retention window.

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
