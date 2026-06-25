# ipsupport-code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-learning local agent binary for LM Studio with fat domain tools, proactive help, reflection, and a workspace-configurable permission policy.

**Architecture:** Single Go binary, CLI/REPL, talks to LM Studio's OpenAI-compatible server via native function-calling. An agent loop dispatches fat tools (`file`/`run`/`web`/`help`/`calc`); tool errors are augmented with matching learned pitfalls from a persisted JSON knowledge base; after each task a reflection pass distills new lessons. A workspace `.agent/config.json` gates shell/file ops with allow/deny globs and an optional jail.

**Tech Stack:** Go 1.26, stdlib (`net/http`, `encoding/json`, `os/exec`, `go/ast`), `bmatcuk/doublestar/v4`, `go-shiori/go-readability`, `JohannesKaufmann/html-to-markdown`.

## Global Constraints

- Module: `github.com/ipsupport-llc/ipsupport-code`; Go 1.26.
- Native function-calling only (no text-protocol fallback).
- `CGO_ENABLED=0`; all deps pure Go; build targets `linux/amd64`, `darwin/amd64`, `darwin/arm64`.
- Config is JSON (no YAML). Knowledge base is a global JSON file by default (`~/.config/ipsupport-code/knowledge.json`).
- Fat-tool style: one tool per domain, `(action, params)` shape, contract in the description, self-correcting errors, tiny catalog.
- Action names are globally unique across tools (enables "action X belongs to tool Y" hints).
- English only in repo content. TDD, frequent commits, DRY, YAGNI.

## File Structure

| File | Responsibility |
|---|---|
| `internal/config/config.go` | Config types; `Default()`; `Load(workspace)` |
| `internal/policy/policy.go` | Permission engine: `Run`/`Write`/`Read` decisions + jail |
| `internal/knowledge/pitfall.go` | `Pitfall` type |
| `internal/knowledge/kb.go` | `KB` load/save/add(dedupe)/query(rank) |
| `internal/tool/tool.go` | `Tool` interface, `Result`, `Approver`, `Require`, schema build |
| `internal/tool/registry.go` | `Registry`, `OpenAITools()`, `Dispatch()` with cross-tool correction |
| `internal/tool/calc.go` | calc tool (AST-safe arithmetic) |
| `internal/tool/file.go` | file tool (policy + jail gated) |
| `internal/tool/run.go` | run tool (policy gated) |
| `internal/tool/web.go` | web tool (search/fetch/stackexchange) |
| `internal/tool/help.go` | help tool (reads KB) |
| `internal/llm/client.go` | LM Studio OpenAI-compatible client, native tool_calls |
| `internal/agent/agent.go` | orchestration loop, error→pitfall injection |
| `internal/reflect/reflect.go` | post-task lesson distillation |
| `cmd/agent/main.go` | CLI/REPL wiring, stdin approver |
| `Makefile` | build / release matrix / test |
| `.agent/config.example.json` | example workspace config |
| `README.md` | usage + "Run on your Mac" |

---

### Task 1: Config

**Files:** Create `internal/config/config.go`, `internal/config/config_test.go`.

**Interfaces — Produces:**
```go
type LLM struct {
    BaseURL     string  `json:"base_url"`
    Model       string  `json:"model"`
    Temperature float64 `json:"temperature"`
    MaxSteps    int     `json:"max_steps"`
    APIKey      string  `json:"api_key,omitempty"`
}
type RunPolicy struct {
    Default string   `json:"default"`           // ask|allow|deny
    Allow   []string `json:"allow"`
    Deny    []string `json:"deny"`
}
type FilePolicy struct {
    Default    string   `json:"default"`
    Jail       string   `json:"jail"`            // "" disables
    AllowWrite []string `json:"allow_write"`
    DenyWrite  []string `json:"deny_write"`
}
type Config struct {
    LLM       LLM        `json:"llm"`
    Run       RunPolicy  `json:"run"`
    File      FilePolicy `json:"file"`
    KBPath    string     `json:"kb_path,omitempty"`
    Workspace string     `json:"-"`              // resolved abs path
}
func Default() Config
func Load(workspace string) (Config, error)      // merge <workspace>/.agent/config.json over Default()
func DefaultKBPath() string                       // ~/.config/ipsupport-code/knowledge.json
```

**Behavior:** `Default()` sets base_url `http://localhost:1234/v1`, temperature 0.2, max_steps 12, run/file default `ask`. `Load` reads the file if present (absent = defaults), unmarshals over a `Default()` value, resolves `Workspace` to abs, fills `KBPath` from `DefaultKBPath()` if empty.

- [ ] Test: `Load` on a dir with no `.agent/config.json` returns `Default()` values + resolved workspace + default KBPath.
- [ ] Test: `Load` merges a partial config (only `run.allow` set) without zeroing other defaults (max_steps stays 12).
- [ ] Implement, run tests, commit.

---

### Task 2: Policy engine

**Files:** Create `internal/policy/policy.go`, `internal/policy/policy_test.go`.

**Interfaces — Consumes:** `config.Config`, `config.RunPolicy`, `config.FilePolicy`.
**Produces:**
```go
type Decision int
const ( Allow Decision = iota; Ask; Deny )
type Engine struct { /* run, file, jailRoot */ }
func New(c config.Config) (*Engine, error)        // resolves jail to abs under workspace
func (e *Engine) Run(command string) Decision
func (e *Engine) Write(path string) (Decision, error)  // jail-check then glob/default
func (e *Engine) Read(path string) error                // jail-check only; nil if allowed
func (e *Engine) Resolve(path string) (string, error)   // abs path, error if escapes jail
```

**Behavior:**
- `Run`: deny glob → `Deny`; allow glob → `Allow`; else parse default. Glob via `doublestar.Match(pattern, command)`.
- `Write`: first `Resolve` (jail) — escape → error; then deny_write → `Deny`; allow_write → `Allow`; else default.
- `Read`: `Resolve` only; nil within jail, error if escape.
- `Resolve`: if jailRoot empty, just abs-clean; else abs-clean and require `strings.HasPrefix(abs, jailRoot+sep)` or `abs==jailRoot`. Use `filepath.EvalSymlinks` best-effort on the parent to catch symlink escape; if parent unresolvable (new file) fall back to lexical clean.
- Default-string → Decision: `allow`→Allow, `deny`→Deny, anything else→Ask.

- [ ] Test: deny beats allow (command matches both → Deny).
- [ ] Test: allow glob → Allow; unmatched → default (Ask).
- [ ] Test: `Write` to `allow_write` glob inside jail → Allow; to `deny_write` → Deny; jail escape (`../x`) → error.
- [ ] Test: jail disabled (`jail:""`) → `Resolve` never errors.
- [ ] Implement, run tests, commit.

---

### Task 3: Knowledge base

**Files:** Create `internal/knowledge/pitfall.go`, `internal/knowledge/kb.go`, `internal/knowledge/kb_test.go`.

**Produces:**
```go
type Pitfall struct {
    Domain       string `json:"domain"`
    ErrorPattern string `json:"error_pattern"`
    Context      string `json:"context"`
    ProvenFix    string `json:"proven_fix"`
    Hits         int    `json:"hits"`
}
type KB struct { /* path, pitfalls */ }
func Open(path string) (*KB, error)              // missing file = empty KB
func (k *KB) Add(p Pitfall) bool                  // dedupe by (domain, norm(error_pattern)); bump Hits; true if new
func (k *KB) Query(domain, errText string, limit int) []Pitfall  // domain filter + keyword-overlap rank
func (k *KB) Save() error                          // mkdir -p dir, write JSON
func (k *KB) All() []Pitfall
```

**Behavior:** `norm` = lowercased, trimmed, collapsed whitespace. `Query` scores each same-domain pitfall by count of shared lowercased word-tokens (len≥3) between `errText` and `ErrorPattern+" "+Context`; drop zero-score; sort desc by score then Hits; cap at limit.

- [ ] Test: `Add` same domain+pattern twice → second returns false, `Hits` incremented, len stays 1.
- [ ] Test: `Query` returns higher-overlap pitfall first and excludes other-domain entries.
- [ ] Test: `Save` then `Open` round-trips pitfalls.
- [ ] Implement, run tests, commit.

---

### Task 4: Tool framework

**Files:** Create `internal/tool/tool.go`, `internal/tool/registry.go`, `internal/tool/registry_test.go`.

**Produces:**
```go
type Result struct { Content string; IsError bool }
func Ok(s string) Result      { return Result{Content: s} }
func Err(s string) Result     { return Result{Content: s, IsError: true} }

type Approver interface { Approve(kind, detail string) bool }  // for policy "ask"

type Tool interface {
    Name() string
    Description() string
    Actions() []string
    Call(action string, params map[string]any) Result
}
func Require(params map[string]any, keys ...string) error      // lists all missing

type Registry struct { /* tools, actionToTool */ }
func NewRegistry(ts ...Tool) *Registry
func (r *Registry) OpenAITools() []map[string]any              // function schema per tool
func (r *Registry) Dispatch(name, action string, params map[string]any) Result
```

**Behavior:**
- `OpenAITools`: per tool a `{type:"function", function:{name, description, parameters:{type:"object", properties:{action:{type:"string", enum:Actions()}, params:{type:"object"}}, required:["action"]}}}`.
- `Dispatch`: unknown tool → `Err`. Action not owned by tool: if another tool owns it → `Err("action %q belongs to tool %q, not %q; call %q instead")`; else `Err("%s: unknown action %q; valid: %v")`. Owned → `tool.Call`.
- `Require`: missing = value nil/""/empty list → `error` listing all.

- [ ] Test: `Dispatch` routes a valid (tool,action) to the tool.
- [ ] Test: wrong-tool action returns the "belongs to" hint naming the correct tool.
- [ ] Test: `OpenAITools` emits one function per tool with the action enum populated.
- [ ] Implement, run tests, commit.

---

### Task 5: calc tool

**Files:** Create `internal/tool/calc.go`, `internal/tool/calc_test.go`.

**Produces:** `func NewCalc() Tool` — Name `calc`, Actions `["calculate"]`, param `{expression}`.

**Behavior:** evaluate arithmetic with `go/parser`+`go/ast` over a whitelist (binary `+ - * / %`, unary `- +`, parens, float/int literals, `math` funcs via a small name map: sqrt/pow/abs/floor/ceil/round/log/exp/sin/cos/tan, consts pi/e). Reject identifiers/calls not in the whitelist with a clear error. `calculate` requires `expression`.

- [ ] Test: `calculate {"expression":"3847*29"}` → `111563`.
- [ ] Test: `sqrt(2)+1` ≈ 2.41421 (string-formatted).
- [ ] Test: unsafe `os.Exit(1)` style / unknown ident → IsError with a clear message.
- [ ] Test: missing `expression` → IsError listing the missing param.
- [ ] Implement, run tests, commit.

---

### Task 6: file tool

**Files:** Create `internal/tool/file.go`, `internal/tool/file_test.go`.

**Consumes:** `*policy.Engine`, `Approver`.
**Produces:** `func NewFile(p *policy.Engine, a Approver) Tool` — Name `file`, Actions `["read","write","append","list","mkdir"]`.

**Behavior:**
- `read {path}`: `policy.Read`; on jail error → Err; else read file, return content (cap large files, note truncation).
- `write {path,content}` / `append`: `policy.Write` → Deny/error → Err; Ask → `a.Approve("write", path)`; false → Err("denied by user"); then write/append.
- `mkdir {path}`: treated as Write decision on the dir path.
- `list {path?}`: default `.`; `policy.Read`; return names with `/` suffix for dirs.
- All `Require` their params.

- [ ] Test (stub approver always-true, jail=tmpdir): `write` then `read` round-trips content.
- [ ] Test: `write` outside jail → IsError (jail escape).
- [ ] Test: approver returning false on an `ask` write → IsError "denied".
- [ ] Test: `list` returns created entries.
- [ ] Implement, run tests, commit.

---

### Task 7: run tool

**Files:** Create `internal/tool/run.go`, `internal/tool/run_test.go`.

**Consumes:** `*policy.Engine`, `Approver`.
**Produces:** `func NewRun(p *policy.Engine, a Approver) Tool` — Name `run`, Actions `["shell"]`.

**Behavior:** `shell {command, cwd?}`: `policy.Run(command)` → Deny → Err; Ask → `a.Approve("run", command)`; false → Err; Allow/approved → run via `exec.Command("sh","-c",command)` with `cwd` clamped into jail (default workspace), 60s timeout, capture stdout+stderr+exit, return a compact block. `Require` command.

- [ ] Test (allow-all policy, true approver): `shell {"command":"echo hi"}` → output contains `hi`, exit 0.
- [ ] Test: deny-glob command → IsError, not executed (sentinel file not created).
- [ ] Test: `ask` + approver false → IsError "denied".
- [ ] Implement, run tests, commit.

---

### Task 8: web tool

**Files:** Create `internal/tool/web.go`, `internal/tool/web_test.go`.

**Produces:** `func NewWeb(hc *http.Client) Tool` — Name `web`, Actions `["search","fetch","stackexchange"]`. Injectable `*http.Client` so tests point at `httptest`.

**Behavior:**
- `search {query, limit?}`: GET `https://html.duckduckgo.com/html/?q=…`, parse result anchors/snippets, return top N as `title — url\n snippet`.
- `fetch {url}`: GET url, run `go-readability` then `html-to-markdown`, cap length, return markdown.
- `stackexchange {query, site?, tag?, limit?}`: GET `https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q=…&site=…` (default `stackoverflow`), JSON-decode, return `title — link (score, answered?)` lines.
- Base URLs are package vars so tests can override to the httptest server.

- [ ] Test: `search` against an httptest server serving canned DuckDuckGo HTML → parses expected titles/urls.
- [ ] Test: `fetch` against httptest HTML → returns markdown containing the heading text, no tags.
- [ ] Test: `stackexchange` against httptest JSON → returns the question titles+links.
- [ ] Implement, run tests, commit.

---

### Task 9: help tool

**Files:** Create `internal/tool/help.go`, `internal/tool/help_test.go`.

**Consumes:** `*knowledge.KB`.
**Produces:** `func NewHelp(kb *knowledge.KB) Tool` — Name `help`, Actions `["lessons"]`.

**Behavior:** `lessons {domain}`: `kb.Query(domain, "", 20)` (empty errText → return all for domain, ranked by Hits); format as bullet lines; empty → "no lessons recorded for <domain> yet". (Adjust `Query` so empty `errText` yields all same-domain pitfalls ranked by Hits.)

- [ ] Test: with two pitfalls seeded for domain `run`, `lessons {"domain":"run"}` lists both.
- [ ] Test: unknown domain → "no lessons" message.
- [ ] Implement (incl. empty-errText branch in `Query`), run tests, commit.

---

### Task 10: LM Studio client

**Files:** Create `internal/llm/client.go`, `internal/llm/client_test.go`.

**Consumes:** `config.LLM`.
**Produces:**
```go
type ToolCall struct { ID, Name, Arguments string }   // Arguments = raw JSON string
type Message struct {
    Role       string     `json:"role"`
    Content    string     `json:"content"`
    ToolCalls  []ToolCall `json:"-"`
    ToolCallID string     `json:"-"`
    Name       string     `json:"-"`
}
type Client struct { /* baseURL, model, apiKey, temp, http */ }
func New(c config.LLM) *Client
func (c *Client) Chat(ctx context.Context, msgs []Message, tools []map[string]any) (Message, error)
```

**Behavior:** POST `<base_url>/chat/completions` with `{model, temperature, messages, tools, tool_choice:"auto"}`. Marshal messages to OpenAI shape (assistant tool_calls; role `tool` carries `tool_call_id`+`name`+`content`). Parse `choices[0].message` back into `Message` incl. `tool_calls[].function.{name,arguments}`. Bearer header only if APIKey set. The base URL/client are overridable for tests.

- [ ] Test: against httptest returning a `tool_calls` completion → `Chat` yields a Message with the parsed ToolCall (name+arguments).
- [ ] Test: against httptest returning a plain content completion → Message.Content set, no tool calls.
- [ ] Test: request body includes the tools array and model.
- [ ] Implement, run tests, commit.

---

### Task 11: Agent loop

**Files:** Create `internal/agent/agent.go`, `internal/agent/agent_test.go`.

**Consumes:** `tool.Registry`, `knowledge.KB`, an LLM interface, system prompt, max_steps.
**Produces:**
```go
type LLMClient interface { Chat(context.Context, []llm.Message, []map[string]any) (llm.Message, error) }
type Transcript struct { Messages []llm.Message; Final string; Steps int }
type Agent struct { /* llm, reg, kb, sys, maxSteps */ }
func New(l LLMClient, reg *tool.Registry, kb *knowledge.KB, system string, maxSteps int) *Agent
func (a *Agent) Run(ctx context.Context, goal string) (Transcript, error)
func DefaultSystemPrompt() string
```

**Behavior:** seed messages `[system, user(goal)]`. Loop ≤ maxSteps: `Chat(msgs, reg.OpenAITools())`; append assistant msg. If no tool_calls → set `Final = content`, stop. Else for each tool_call: parse `{action, params}` from Arguments (lenient: tolerate params at top level by folding non-`action` keys into params); `reg.Dispatch(name, action, params)`; if `IsError` → `kb.Query(name, content, 3)` and append `\nHints from past runs:\n- …` to the content; append a `role:"tool"` message with tool_call_id. Continue. On maxSteps exhaustion, `Final` = last assistant content (may be empty). Policy gating happens inside file/run tools, not here.

- [ ] Test (fake LLM scripted: step1 returns a `calc.calculate` tool_call, step2 returns final content): `Run` fires the tool, feeds the observation, ends with the final text; `Steps==2`.
- [ ] Test (fake LLM returns a tool_call that errors; KB seeded with a matching pitfall): the tool message content contains the injected hint.
- [ ] Test: a tool_call naming an action from another tool surfaces the "belongs to" correction in the tool message.
- [ ] Implement, run tests, commit.

---

### Task 12: Reflection

**Files:** Create `internal/reflect/reflect.go`, `internal/reflect/reflect_test.go`.

**Consumes:** agent `LLMClient`, `agent.Transcript`, `knowledge.Pitfall`.
**Produces:** `func Reflect(ctx context.Context, l agent.LLMClient, t agent.Transcript) ([]knowledge.Pitfall, error)`.

**Behavior:** build a compact transcript summary (tool errors + the following recovery + final outcome); ask the model (no tools) for a JSON array of `{domain, error_pattern, context, proven_fix}` of durable, environment-general lessons (explicitly exclude project-specific noise); extract the first JSON array from the reply and unmarshal leniently; return `[]` on parse failure rather than erroring the run.

- [ ] Test (fake LLM returns a JSON array of one lesson): `Reflect` returns one `Pitfall` with the parsed fields.
- [ ] Test (fake LLM returns prose with no JSON): returns empty slice, no error.
- [ ] Implement, run tests, commit.

---

### Task 13: CLI wiring, Makefile, README, example config

**Files:** Create `cmd/agent/main.go`, `Makefile`, `.agent/config.example.json`, `README.md`. Add deps to `go.mod`.

**Behavior:** `main`: resolve workspace (`-C` flag or cwd); `config.Load`; `knowledge.Open(cfg.KBPath)`; `policy.New`; build tools (`NewFile`/`NewRun` with a stdin y/n `Approver`, `NewWeb(http.DefaultClient)`, `NewHelp(kb)`, `NewCalc()`); `tool.NewRegistry`; `llm.New`; `agent.New`. If args → one-shot goal; else REPL loop reading lines. After each goal: print `Final`, run `reflect.Reflect`, `kb.Add` each (report new lessons), `kb.Save()`. Stdin approver prints `kind: detail` and reads `y/N`. Makefile = the `build`/`release`/`test` targets from the spec. README = quickstart + "Run on your Mac" (xattr, LM Studio on :1234). `.agent/config.example.json` = the spec's example.

- [ ] `go build ./...` and `go vet ./...` clean.
- [ ] `make release` produces all three `dist/` binaries.
- [ ] Smoke: `./dist/ipsupport-code-linux-amd64 "use calc to compute 2+2"` against a running LM Studio (or documented as manual) returns 4. (CI-skippable; verify locally.)
- [ ] Commit.

---

## Self-Review

- **Spec coverage:** loop (T11), fat tools file/run/web/help/calc (T5-T9), KB + injection + reflection (T3,T11,T12), policy + jail + workspace config (T1,T2), native LM Studio client (T10), build matrix + Mac notes (T13). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; each task has concrete tests and behavior.
- **Type consistency:** `tool.Result`/`Ok`/`Err`, `tool.Approver`, `Registry.Dispatch`, `knowledge.KB.Query(domain,errText,limit)`, `llm.Message`/`ToolCall`, `agent.LLMClient`/`Transcript` are used consistently across T4–T13. `Query` empty-errText behavior is introduced in T3 and refined in T9 (noted there).
