# Fine-tuning a local model on ipsupport-code's tool-calling schema

A small/quantized local model can garble the tool-call shape ipsupport-code
expects — wrapping `params` in a needless array, leaking a `<parameter=NAME>`
tagging convention from different training into the `action` field, or dumping
a raw string into `action` instead of using `params`. The registry recovers
several of these shapes automatically (see `internal/agent/agent.go`'s
`parseArgs`/`recoverActionTag` and `internal/tool/registry.go`'s
`garbledActionAsParam`), but the more reliable fix for a *specific* model you
run regularly is to fine-tune it on the exact schema, so it stops needing the
recovery path at all.

This is a from-scratch SFT (supervised fine-tuning) pass, not RLHF/DPO — the
goal is narrow: teach a model that already knows how to code to *emit this
one JSON shape reliably*, not to teach it new domain knowledge.

## Fine-tune the base model, not the GGUF

Fine-tune the base/instruct checkpoint (fp16/bf16 safetensors from Hugging
Face), not the quantized `.gguf` you run in LM Studio — you can't train a
quantized model directly. After training:

1. Merge the LoRA adapter into the base weights.
2. Convert to GGUF with llama.cpp's `convert_hf_to_gguf.py`.
3. Quantize with `llama-quantize` back to whatever level you run today.
4. Point LM Studio at the new GGUF and spot-check it against real tasks.

## Method: LoRA/QLoRA SFT

A full fine-tune is unnecessary for this and likely won't fit local hardware;
a LoRA (rank 16–64 is typically plenty for a formatting habit, not new
knowledge) is enough. Established local-friendly options:

- **[Unsloth](https://github.com/unslothai/unsloth)** — usually the easiest
  path for a single local GPU, well-optimized memory-wise.
- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** /
  **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — more
  configurable via YAML, both handle Nemotron-family models.

## Dataset shape

JSONL, one conversation per line, OpenAI-style `messages`. Ground every
system prompt and tool-call shape in what ipsupport-code actually sends —
don't invent variations.

**System message** — the exact built-in prompt
(`agent.DefaultSystemPrompt()`), with the environment line
`systemPrompt()` appends at request time:

```
You are the engine inside ipsupport-code, a local terminal coding agent. You run in a loop and act ONLY through tools; the user sees your tool calls and results.

- You CAN edit files. The file tool's write/edit/append actions modify REAL files on disk in this workspace, and the user has already authorized you to use them. NEVER claim you "only have read/run", "can't modify files", or that the user must "enable file editing" / open a different mode — that is false. To change code, just call file (action edit, or write); for a multi-file change, do one file at a time.
- DO the task with tools: write/edit files with file, run commands with run, use git/web/calc. NEVER just tell the user how to do it ("create a file", "chmod +x", "here's how…") — describing steps instead of doing them is a failure. (e.g. "make a hello script and run it" → file.write then run.shell, then report the output.)
- If what you built is runnable, RUN it yourself with run and report the real output. Don't hand back a "how to test it" recipe — that's the user doing your job.
- Each call is {"action": <name>, "params": {...}}. On an error, read it — it names the fix or the right tool — and retry.
- Small local model in a terminal: be brief. Finish with a one-line summary of what you did — not a tutorial, and not a menu of optional features to add — and no tool call.
- After that summary, add ONE last line exactly: "NEXT: <one short next step the user might want>" (≤6 words; skip the line if nothing fits).

Environment: you are running on <os>; your working directory is <dir>. Relative paths resolve there — and by default this is a HARD JAIL: no tool (file, run's cwd, git) can reach a path outside it, an absolute path elsewhere is rejected, not silently redirected. If a task genuinely needs a different directory, say so — don't keep retrying different absolute paths or cwd values, they'll all fail the same way. Use commands that exist on this OS — on darwin prefer vm_stat/top/sw_vers over Linux-only tools like free.
```

**Assistant tool-call turns** use the `tool_calls` field (name + an
`Arguments` string), never plain text describing a call:

```json
{
  "role": "assistant",
  "tool_calls": [
    {"id": "c1", "type": "function", "function": {
      "name": "file",
      "arguments": "{\"action\":\"write\",\"params\":{\"path\":\"main.go\",\"content\":\"package main\\n\"}}"
    }}
  ]
}
```

**Tool results** go back as `{"role": "tool", "tool_call_id": "c1", "content": "wrote main.go (13 bytes)"}`.

## The real tool catalog

Every training example's `action`/`params` must come from this table (pulled
straight from each domain's `DomainSpec` in `internal/tool/*.go` — read the
source directly if it's changed since):

| Tool | Action | Required params | Optional params (default) | Note |
|---|---|---|---|---|
| `file` | `read` | `path:str` | `offset:int(0)`, `limit:int(0)` | line window, big files |
| | `write` | `path:str` | `content:str("")` | overwrites; omit content = empty file |
| | `append` | `path:str`, `content:str` | | |
| | `edit` | `path:str` | `find:str("")`, `replace:str("")`, `replace_all:bool("")`, `edits:str("")` | 1st match; replace_all=all; or edits=JSON `[{find,replace},…]` |
| | `list` | | `path:str(".")` | |
| | `find` | `pattern:str` | `path:str(".")` | glob names, e.g. `**/*.go` |
| | `search` | `query:str` | `path:str(".")` | regex; `file:line:` match |
| | `mkdir` | `path:str` | | |
| `run` | `shell` | `command:str` | `cwd:str("")`, `timeout:int("")` | timeout=seconds before kill |
| `git` | `init` | | | start a repo |
| | `status` | | | |
| | `diff` | | `path:str("")`, `staged:bool("")` | |
| | `log` | | `n:int("15")` | |
| | `show` | | `ref:str("HEAD")` | |
| | `add` | `paths:str` | | space-separated |
| | `commit` | `message:str` | | |
| | `branch` | | `name:str("")` | none=list, name=create |
| | `checkout` | `ref:str` | | |
| `web` | `search` | `query:str` | `limit:int(8)` | |
| | `fetch` | `url:str` | | |
| | `stackexchange` | `query:str` | `site:str("stackoverflow")`, `tag:str("")`, `limit:int(5)` | |
| `calc` | `calculate` | `expression:str` | | `+ - * / %` + parens; sqrt/cbrt/pow/abs/floor/ceil/round/log/log2/log10/exp/sin/cos/tan/hypot/min/max; pi/e/tau |
| `help` | `lessons` | `domain:str` | | one of: file, run, git, web, calc |
| `skill` | `list` | | | enabled skills |
| | `load` | `name:str` | | returns full instructions |
| `agent` | `run` | `profile:str`, `task:str` | `dir:str("")`, `background:bool(false)` | delegates to a sub-agent |
| `history` | `recent` | | `n:int(5)` | last N tasks this session |
| | `search` | `query:str` | | archived tasks matching query |
| `mcp` | `list` | | | all MCP servers+tools |
| | `schema` | `server:str`, `tool:str` | | a tool's input schema |
| | `call` | `server:str`, `tool:str` | `args:str("")` | args=JSON object |

## Data you likely already have

ipsupport-code already logs real usage in a few places — mine these before
writing anything synthetic:

- **`~/.config/ipsupport-code/traces.jsonl`** — `DefaultTracePath()` is
  literally doc-commented as *"the training dataset location"*. One JSON
  record per `Emit()` call: `tool_call {tool, action, params}`,
  `observation {tool, action, is_error, content}`, `assistant {content,
  tool_calls}`, plus `final`/`nudge`/`continue`/`judge`/`goal`. This is
  already a real, already-correctly-parsed (tool, action, params) → (result)
  log.
- **Session files** (`.agent`/wherever `saveSession()` writes for your
  workspace) — full `llm.Message` history per saved session, including
  `ToolCalls` and tool-result messages, i.e. complete real multi-turn
  conversations, garbled-then-corrected turns included.
- **The knowledge/pitfalls store** (`internal/knowledge`) — `Pitfall{Domain,
  ErrorPattern, Context, ProvenFix}` — a curated (error → fix) list the agent
  already learned on its own; a ready-made source of negative→positive pairs.

## Known failure modes worth specifically covering

These are real, live-diagnosed shapes this project's own registry now
recovers automatically — training the model past them directly (so recovery
never has to fire) is the actual win:

1. **Array-wrapped params**: `{"action":"write","params":[{"path":"README.md","content":"x"}]}` → should be `{"action":"write","params":{"path":"README.md","content":"x"}}`.
2. **Whole-arguments tag leak**: raw arguments `"<parameter=params>\n{\"url\": \"https://example.com\"}\n</parameter>"` → should be bare `{"action":"fetch","params":{"url":"https://example.com"}}`.
3. **Bare action-name tag leak**: `{"action":"<parameter=action>\nlist\n</parameter>","params":{"path":"."}}` → should be `{"action":"list","params":{"path":"."}}`.
4. **Object tag leak into `action`** (seen on single-action tools like `run`): `{"action":"<parameter=params>\n{\"command\": \"ls -la\", \"cwd\": \"/some/dir\"}\n</parameter>"}` → should be `{"action":"shell","params":{"command":"ls -la","cwd":"/some/dir"}}`.
5. **Raw command/expression dumped into `action`**: action string literally `"echo hi"` or `"2+2"` with empty `params` → should be `{"action":"shell","params":{"command":"echo hi"}}` / `{"action":"calculate","params":{"expression":"2+2"}}`.

For each, a training pair of (correct call for that exact scenario) is more
useful than the garbled form itself — you're teaching the right habit, not
teaching the model to recognize its own mistake.

## Practical size and mix

A narrow formatting fix like this typically needs low hundreds to a couple
thousand well-distributed examples, not tens of thousands — this is fixing a
habit, not teaching new knowledge. Aim for:

- Every action in the table above represented at least a handful of times.
- A mix of single-call and multi-call turns (the model emitting 2–3 tool
  calls in one turn is common and normal — see `internal/agent/agent.go`'s
  `runToolCalls`).
- Full multi-turn examples: request → reasoning → tool call → tool result →
  either another tool call or a final one-line answer with the `NEXT:` line.
- A deliberate concentration of examples around whichever of the 5 failure
  modes above your own traces show most often for your specific model.

## Suggested workflow

1. Turn on real logging (`IPS_LOG=debug`, or just use ipsupport-code
   normally for a while) so `traces.jsonl` and session files accumulate real
   examples, including your model's own real mistakes.
2. Write a short converter (a few dozen lines) turning `traces.jsonl` +
   session files into the `messages`-array JSONL shape above. Keep it
   simple — filter to `tool_call`/`observation`/`assistant` records already
   in the right shape, discard anything already caught by the recovery
   heuristics (an already-garbled call that got recovered isn't a good
   training example — a *clean* equivalent call is).
3. Optionally supplement with synthetic examples from a stronger model,
   prompted with the real tool table above (not an invented one) for
   coverage on actions your own logs haven't exercised yet.
4. Fine-tune with Unsloth/Axolotl/LLaMA-Factory — a LoRA rank of 16–64,
   a handful of epochs, is a reasonable starting point for a schema-only fix.
5. Merge, quantize back to GGUF, reload in LM Studio, and re-run the same
   real tasks that used to trip it up.
