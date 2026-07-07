// Command ipsupport-code is a self-learning local agent for LM Studio. With a
// goal argument it runs one task; with none on a terminal it opens a Bubble Tea
// TUI (plain line REPL when piped). After each task it reflects and persists
// what it learned.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/term"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/mcp"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/reflect"
	"github.com/ipsupport-llc/ipsupport-code/internal/selfupdate"
	"github.com/ipsupport-llc/ipsupport-code/internal/skill"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
	"github.com/ipsupport-llc/ipsupport-code/internal/trace"
	"github.com/ipsupport-llc/ipsupport-code/internal/usage"
)

// version is stamped at build time via -ldflags "-X main.version=…" (GoReleaser
// and `make release`); "dev" for a plain `go build`.
var version = "dev"

func main() {
	var (
		workspace   string
		doInit      bool
		showVersion bool
		dumpPrompt  bool
		newSession  bool
		sessionName string
	)
	flag.StringVar(&workspace, "C", ".", "workspace directory")
	flag.BoolVar(&doInit, "init", false, "re-run first-time setup (server URL, API key, model)")
	flag.BoolVar(&showVersion, "version", false, "print version and exit")
	flag.BoolVar(&dumpPrompt, "dump-prompt", false, "print the built-in system prompt and exit (e.g. > .agent/system.md to start editing)")
	flag.BoolVar(&newSession, "new", false, "start a fresh session (don't restore the saved one)")
	flag.StringVar(&sessionName, "session", "", "use a named session (a separate saved thread)")
	flag.Usage = printUsage
	flag.Parse()
	if showVersion {
		fmt.Println("ipsupport-code", version)
		return
	}
	if dumpPrompt {
		fmt.Println(agent.DefaultSystemPrompt())
		return
	}
	// Subcommand dispatch — one clear surface for the non-task verbs, rather than
	// ad-hoc arg checks. `init` needs the reader, so it's handled just below.
	if args := flag.Args(); len(args) >= 1 {
		switch args[0] {
		case "version":
			fmt.Println("ipsupport-code", version)
			return
		case "update":
			runUpdate(args[1:])
			return
		case "config":
			runConfig(workspace, args[1:])
			return
		case "help":
			printUsage()
			return
		}
	}
	setupLogging()

	reader := bufio.NewReader(os.Stdin)
	if args := flag.Args(); len(args) >= 1 && args[0] == "init" {
		maybeInit(reader, true) // `init` subcommand: (re-)run setup and exit
		return
	}
	maybeInit(reader, doInit)

	app, cleanup, err := build(workspace, reader)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer cleanup()

	// -session selects a named thread for this run (its own saved file); the
	// agent's identity in the prompt follows the name.
	if sessionName != "" {
		app.cfg.Name = sessionName
		app.ag.SetSystem(app.systemPrompt())
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	switch {
	case strings.TrimSpace(strings.Join(flag.Args(), " ")) != "":
		if !newSession {
			app.loadSession() // one-shot: silently continue the saved session
		}
		app.runOne(ctx, strings.TrimSpace(strings.Join(flag.Args(), " ")))
	case isTTY():
		app.startNew = newSession             // the TUI shows an in-screen session chooser (unless -new)
		if sessionName != "" && !newSession { // -session: go straight to that named thread
			app.loadSession()
			app.sessionRestored = app.ag.SessionLen() > 0
		}
		if err := app.runTUI(ctx); err != nil {
			fmt.Fprintln(os.Stderr, "tui:", err)
		}
	default:
		if !newSession {
			app.loadSession() // piped: silently continue
		}
		app.repl(ctx)
	}
}

// runUpdate downloads and installs a newer binary from GitHub Releases for the
// configured channel (an optional "stable"/"nightly" arg switches and saves it).
func runUpdate(args []string) {
	cfg, _ := config.Load(".")
	channel := cfg.Channel
	if channel == "" {
		channel = selfupdate.Stable
	}
	if len(args) >= 1 {
		switch args[0] {
		case selfupdate.Stable, selfupdate.Nightly:
			channel = args[0]
			if err := config.SaveChannel(channel); err != nil {
				fmt.Fprintln(os.Stderr, "warning: channel not saved:", err)
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown channel %q — use 'stable' or 'nightly'\n", args[0])
			os.Exit(1)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	rel, err := selfupdate.Latest(ctx, selfupdate.Repo, channel, http.DefaultClient)
	if err != nil {
		fmt.Fprintln(os.Stderr, "update:", err)
		os.Exit(1)
	}
	if rel.Version == version {
		fmt.Printf("already up to date — %s (%s channel)\n", version, channel)
		return
	}
	fmt.Printf("updating %s → %s (%s channel)…\n", version, rel.Version, channel)
	path, err := selfupdate.Apply(ctx, rel, http.DefaultClient)
	if err != nil {
		fmt.Fprintln(os.Stderr, "update:", err)
		os.Exit(1)
	}
	fmt.Printf("done — %s is now %s\n", path, rel.Version)
}

// startupNotice runs freshnessNotice under a short timeout (best-effort).
func (a *app) startupNotice(ctx context.Context) string {
	cctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	return a.freshnessNotice(cctx)
}

// freshnessNotice returns a one-line "newer build available" message, or "" when
// up to date, on a local (dev) build, or if the check fails. Best-effort.
func (a *app) freshnessNotice(ctx context.Context) string {
	if !a.cfg.UpdateCheck { // update check disabled (pinned / package-managed install)
		return ""
	}
	if a.cfg.Offline { // offline mode: never reach GitHub to check for updates
		return ""
	}
	// Only released binaries have a clean version ("v0.1.0" / "nightly-…"); a local
	// build is "dev" or a git-describe ("v0.1.0-20-g<sha>", "-dirty") — skip those
	// so a developer build doesn't nag about being "outdated".
	if version == "dev" || strings.Contains(version, "dirty") || strings.Contains(version, "-g") {
		return ""
	}
	channel := a.cfg.Channel
	if channel == "" {
		channel = selfupdate.Stable
	}
	rel, err := selfupdate.Latest(ctx, selfupdate.Repo, channel, http.DefaultClient)
	if err != nil || rel.Version == "" || rel.Version == version {
		return ""
	}
	return fmt.Sprintf("a newer %s build is available: %s (you're on %s) — run `update`", channel, rel.Version, version)
}

// app bundles everything one process needs across tasks, plus session counters.
type app struct {
	cfg       config.Config
	workspace string
	kb        *knowledge.KB
	usage     *usage.Store
	skills    *skill.Store
	reader    *bufio.Reader

	fileTracer trace.Tracer  // JSONL dataset
	uiTracer   trace.Tracer  // live TUI (nil in plain mode)
	tracer     trace.Tracer  // composite, set in wire()
	approver   tool.Approver // stdin (plain) or the TUI bridge (the actual prompt)

	// sessionAllow is the in-memory "don't ask again this session" set, keyed by
	// approval category (file/run/git/spawn/mcp). Granted from an approval prompt,
	// never persisted; cleared on /new and /clear. gatedApprover consults it.
	sessionMu    sync.Mutex
	sessionAllow map[string]bool

	promptHist []string // submitted inputs (tasks + /commands), oldest→newest, persisted per workspace for ↑ recall across restarts

	jobMu  sync.Mutex // guards jobs/jobSeq — background sub-agent runs (see jobs.go)
	jobs   []*job
	jobSeq int

	btwMu      sync.Mutex // guards pendingBtw — /btw side-channel steering notes
	pendingBtw []string   // user asides awaiting the running task's next turn (see jobs.go drainBtw)

	costMu         sync.Mutex // guards sessionCostUSD (parallel sub-agent spawns accrue too)
	sessionCostUSD float64    // estimated spend this process run, for the SessionBudgetUSD guard

	client          *llm.OpenAIClient
	ag              *agent.Agent
	pol             *policy.Engine // host policy/jail; sub-agents in a dir get their own
	workdir         string         // absolute session working dir (set by /cd); "" = workspace
	subReg          *tool.Registry // tools for sub-agents (no `agent` tool → no recursion)
	spawnMu         sync.Mutex     // serializes spawn-approval prompts during parallel fan-out
	spawnSeq        atomic.Int64   // unique id per sub-agent spawn (for grouping its UI events)
	mcpMu           sync.Mutex     // guards the lazy MCP client cache
	mcpClients      map[string]*mcp.Client
	ckptMu          sync.Mutex    // guards checkpoints / the in-progress one
	checkpoints     []*checkpoint // per-turn file+history snapshots for /rewind (session lifetime)
	curCkpt         *checkpoint   // the checkpoint being filled during the running turn
	instrSrc        string        // project instructions file in effect, "" if none
	promptSrc       string        // "built-in" or the system.md override path
	facts           []string      // durable project facts learned over time (per workspace)
	planMode        bool          // plan (propose) vs auto (execute); survives re-wire
	goal            goalState     // standing goal pursued by the judge loop (per workspace)
	windowDetected  bool          // got the real loaded context window (vs a default/guess)
	sessionRestored bool          // a saved session was restored at startup (TUI renders a recap)
	tui             bool          // running the TUI (detect the context window off-thread, not inline)
	startNew        bool          // -new: skip the startup chooser, begin a fresh session

	tasks, steps, toolCalls int
	lastPrompt, lastCompl   int // client usage snapshot for per-task ledger deltas
}

func build(workspace string, reader *bufio.Reader) (*app, func(), error) {
	cfg, err := config.Load(workspace)
	if err != nil {
		return nil, nil, fmt.Errorf("load config: %w", err)
	}
	kb, err := knowledge.Open(cfg.KBPath)
	if err != nil {
		slog.Warn("knowledge base unreadable; starting empty", "err", err)
		kb, _ = knowledge.Open("")
	}
	skills, err := skill.Open(cfg.SkillsPath, http.DefaultClient)
	if err != nil {
		slog.Warn("skills unavailable", "err", err)
	}
	usageStore, err := usage.Open(cfg.UsagePath)
	if err != nil {
		slog.Warn("usage ledger unreadable; starting empty", "err", err)
		usageStore, _ = usage.Open("")
	}

	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, usage: usageStore, skills: skills, reader: reader}
	a.approver = &stdinApprover{r: reader, app: a} // set after: it references the app for session-allow
	cleanup := func() { a.closeMCP() }             // shut down any launched MCP servers on exit
	a.applyUsageRetention()                        // honor usage_retention_days on startup
	a.applyKnowledgeRetention()                    // honor knowledge_retention_days on startup
	if ft, err := trace.NewFileTracer(cfg.TracePath, newRunID()); err != nil {
		slog.Warn("trace disabled", "err", err)
	} else {
		a.fileTracer = ft
		cleanup = func() { a.closeMCP(); _ = ft.Close() }
	}
	a.loadFacts()      // learned project facts → folded into the prompt by wire()
	a.loadGoal()       // standing goal (if any) → resumable across restarts
	a.loadPromptHist() // ↑ recall spans past runs (persisted per workspace)
	if err := a.wire(); err != nil {
		return nil, nil, err
	}
	// The prior session is restored by the caller: interactively via chooseSession
	// (restore/new/delete), or auto-loaded in non-interactive modes.
	a.detectContextWindow() // ask LM Studio for the real window (auto-compact sizing)
	return a, cleanup, nil
}

// activeLLM resolves the connection for the active provider. "" / "local" is the
// LM Studio connection (cfg.LLM); any other name is an external provider preset
// (template + key/env merged).
func (a *app) activeLLM() config.LLM {
	if a.cfg.Provider == "" || a.cfg.Provider == "local" {
		return a.cfg.LLM
	}
	if l, ok := config.ResolveProvider(a.cfg, a.cfg.Provider); ok {
		return l
	}
	return a.cfg.LLM
}

func (a *app) isLocal() bool { return a.cfg.Provider == "" || a.cfg.Provider == "local" }

// hasSubagentTargets reports whether the `agent` tool should exist: only when at
// least one profile is configured (a profile is the sole way to delegate, so it
// is also the user's curated list of what the assistant may spawn).
func (a *app) hasSubagentTargets() bool {
	return len(a.cfg.Agents) > 0
}

// spawnAgent runs a delegated task on a sub-agent: it resolves the profile (which
// carries the provider+model), optionally re-roots to a directory (its own jail),
// asks approval unless the spawn policy is relaxed, builds a fresh agent loop (the
// host tools minus `agent` so it can't recurse — and minus run unless spawn.exec
// is on; it inherits the current plan/auto mode), runs it, records its tokens,
// and returns its final answer. Safe to call concurrently (fan-out).
func (a *app) spawnAgent(ctx context.Context, profile, task, dir string) (string, error) {
	if strings.TrimSpace(task) == "" {
		return "", fmt.Errorf("task is required")
	}
	profile = strings.TrimSpace(profile)
	if profile == "" {
		return "", fmt.Errorf("profile is required — configured: %s", a.profilesOrHint())
	}
	resolved, ok := a.resolveProfileName(profile)
	if !ok {
		return "", fmt.Errorf("unknown profile %q — configured: %s", profile, a.profilesOrHint())
	}
	profile = resolved
	p := a.cfg.Agents[profile]
	if p.Kind == "external" { // a local CLI agent, not one of our LLM sub-agents
		return a.spawnExternalAgent(ctx, profile, p, task, dir)
	}
	provider := p.Provider
	if provider == "" {
		provider = "local"
	}

	// Resolve the connection for the profile's provider.
	var llmCfg config.LLM
	if provider == "local" {
		llmCfg = a.cfg.LLM
	} else {
		rp, rok := config.ResolveProvider(a.cfg, provider)
		if !rok {
			return "", fmt.Errorf("profile %q: unknown provider %q", profile, provider)
		}
		if rp.APIKey == "" {
			return "", fmt.Errorf("profile %q: %s has no API key — add one with /ai key %s <token>", profile, provider, provider)
		}
		llmCfg = rp
	}
	if p.Model != "" {
		llmCfg.Model = p.Model
	}

	// Resolve the working directory (default: the session workspace). The path may
	// point anywhere — ~ is expanded, relatives resolve against the session — but
	// the sub-agent gets its OWN jail rooted there, so it still can't escape it.
	subReg, subWorkspace := a.subReg, a.effectiveDir()
	if d := strings.TrimSpace(dir); d != "" {
		root, err := a.resolveSpawnDir(d)
		if err != nil {
			return "", err
		}
		if fi, statErr := os.Stat(root); statErr != nil || !fi.IsDir() {
			return "", fmt.Errorf("dir %q is not a directory", dir)
		}
		subCfg := a.cfg
		subCfg.Workspace = root
		subCfg.File.Jail = "." // keep the jail — confine the sub-agent to its own dir
		subPol, pErr := policy.New(subCfg)
		if pErr != nil {
			return "", pErr
		}
		subReg, subWorkspace = a.buildSubReg(subPol), root
	}

	// Ask before spawning unless the policy is relaxed. "ask" (default) guards
	// every spawn — even local ones still cost compute, and a runaway main model
	// could fan out endlessly. Serialize the prompt so parallel fan-out spawns ask
	// one at a time instead of racing on the approver.
	if a.cfg.Spawn.Default != "allow" {
		a.spawnMu.Lock()
		approved := a.approveGated("spawn agent", fmt.Sprintf("%s · %s · %s\n  task: %s", profile, llmCfg.Model, subWorkspace, task))
		a.spawnMu.Unlock()
		if !approved {
			return "", fmt.Errorf("spawn denied by user")
		}
	}

	id := fmt.Sprintf("sub%d", a.spawnSeq.Add(1)) // groups this sub-agent's UI events
	client := llm.NewOpenAIClient(a.withReasoning(llmCfg, provider, ""))
	sub := agent.New(client, subReg, a.kb, a.tracer, a.subAgentPrompt(subWorkspace, p.Prompt), llmCfg.MaxSteps)
	sub.SetPlanMode(a.planMode) // inherit the current mode
	sub.SetLabel(id)
	a.emit("subagent", map[string]any{"agent": id, "profile": profile, "provider": provider, "model": llmCfg.Model, "dir": subWorkspace, "task": oneLine(task, 80)})

	tr, err := sub.Run(ctx, task)
	if a.usage != nil { // the sub-agent's spend counts too
		pt, ct := client.Usage()
		a.usage.Add(today(), provider, llmCfg.Model, pt, ct)
		a.addSessionCost(llmCfg.Model, pt, ct) // sub-agent spend counts toward /budget too
		_ = a.usage.Save()
	}
	done := map[string]any{"agent": id, "profile": profile, "ok": err == nil}
	if err != nil {
		done["error"] = oneLine(err.Error(), 60)
	}
	a.emit("subagent_done", done)
	if err != nil {
		return "", err
	}
	return tr.Final, nil
}

// resolveProfileName matches the model's (often fumbled) profile argument to a
// configured profile, tolerantly — small models mistype long names. Order: exact
// (case-insensitive) → the only profile if there's just one → a unique
// case-insensitive substring → the nearest by edit distance (if unambiguous).
// Returns false only when it's genuinely ambiguous or there are no profiles.
func (a *app) resolveProfileName(name string) (string, bool) {
	names := agentProfileNames(a.cfg)
	if len(names) == 0 {
		return "", false
	}
	for _, n := range names { // exact, case-insensitive
		if strings.EqualFold(n, name) {
			return n, true
		}
	}
	if len(names) == 1 { // only one target — whatever it typed, it means this one
		return names[0], true
	}
	lc := strings.ToLower(strings.TrimSpace(name))
	var subs []string
	for _, n := range names {
		ln := strings.ToLower(n)
		if strings.Contains(ln, lc) || strings.Contains(lc, ln) {
			subs = append(subs, n)
		}
	}
	if len(subs) == 1 {
		return subs[0], true
	}
	best, bestDist, ties := "", 1<<30, 0
	for _, n := range names {
		if d := levenshtein(strings.ToLower(n), lc); d < bestDist {
			best, bestDist, ties = n, d, 1
		} else if d == bestDist {
			ties++
		}
	}
	if ties == 1 && bestDist <= 3 { // a single clear near-match
		return best, true
	}
	return "", false
}

// levenshtein is the edit distance between two strings (small inputs: profile
// names), for tolerant matching.
func levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(rb)]
}

// profilesOrHint lists configured profile names, or a hint to make one.
func (a *app) profilesOrHint() string {
	if names := agentProfileNames(a.cfg); len(names) > 0 {
		return strings.Join(names, ", ")
	}
	return "none yet — add one in /config"
}

// resolveSpawnDir turns a sub-agent's dir argument into an absolute path: ~ and
// ~/… expand to the home dir, a relative path resolves against the session
// workspace. The path may point anywhere — the sub-agent is jailed to it later.
func (a *app) resolveSpawnDir(dir string) (string, error) {
	if dir == "~" || strings.HasPrefix(dir, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, strings.TrimPrefix(dir, "~"))
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(a.effectiveDir(), dir)
	}
	return filepath.Clean(dir), nil
}

// buildSubReg builds a sub-agent's tool registry against a given policy: the host
// tools except `agent` (so sub-agents can't recurse) and except help (a usage
// helper they don't need). The run (shell) tool is included only when spawn.exec
// is on — the sharpest capability to hand an autonomous sub-agent.
func (a *app) buildSubReg(pol *policy.Engine) *tool.Registry {
	tools := []tool.Tool{tool.NewFile(pol, gatedApprover{a}, a.snapFile)}
	if a.cfg.Spawn.Exec {
		tools = append(tools, tool.NewRun(pol, gatedApprover{a}, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second))
	}
	tools = append(tools, tool.NewGit(pol, gatedApprover{a}), tool.NewWeb(http.DefaultClient, a.cfg.Offline), tool.NewCalc())
	if a.skills != nil && a.skills.HasEnabled() {
		tools = append(tools, tool.NewSkill(a.skills))
	}
	return tool.NewRegistry(tools...)
}

// subAgentPrompt builds a sub-agent's system prompt: the dedicated sub-agent base
// (deliberately different from the interactive one), its working directory, any
// project instructions found there, the enabled-skills index, and the profile's
// role. It does NOT inject the host's learned facts — those belong to the host
// workspace, not the directory the sub-agent was pointed at.
func (a *app) subAgentPrompt(workspace, role string) string {
	out := agent.SubAgentSystemPrompt()
	out += fmt.Sprintf(
		"\n\nEnvironment: you are running on %s; your working directory is %s. Use commands that exist on this OS. All file/run/git paths resolve in that directory.",
		runtime.GOOS, workspace)
	if text, src := loadInstructions(workspace); text != "" {
		out += "\n\n## Project instructions (from " + src + ") — follow these:\n" + text
	}
	if a.skills != nil {
		if idx := a.skills.Index(); idx != "" {
			out += "\n\n## Skills (load full instructions with the skill tool when the topic fits):\n" + idx
		}
	}
	if strings.TrimSpace(role) != "" {
		out += "\n\n## Your role\n" + role
	}
	return out
}

// subagentTargetsPrompt is the dynamic roster injected into the main assistant's
// system prompt so it knows WHO it can delegate to — the configured profiles.
// Only emitted when a profile exists (so it costs nothing otherwise), and kept to
// a couple of lines.
func (a *app) subagentTargetsPrompt() string {
	names := agentProfileNames(a.cfg)
	if len(names) == 0 {
		return ""
	}
	parts := make([]string, 0, len(names))
	external := false
	for _, n := range names {
		p := a.cfg.Agents[n]
		if p.Kind == "external" {
			external = true
			parts = append(parts, fmt.Sprintf("%s→external·%s", n, p.Command))
			continue
		}
		m := p.Model
		if m == "" {
			m = "default"
		}
		parts = append(parts, fmt.Sprintf("%s→%s·%s", n, p.Provider, m))
	}
	prompt := "\n\n## Sub-agents you can delegate to (the agent tool)\n" +
		"Profiles (call agent with profile=<name>): " + strings.Join(parts, ", ") + "\n" +
		"ALWAYS pass dir=<the specific project's directory> (the repo root) — without it a sub-agent inherits this session's workspace, which may be a home dir. Fan a task out across several profiles in one turn — they run in parallel — then merge their findings. To keep your own context lean, delegate a big exploration to a sub-agent and take back just its summary. For a LONG task you don't need immediately (a big external review, a broad exploration), add background=true: the call returns at once, you keep working, and the job's result is delivered to you at the start of a later turn — never wait or poll for it."
	if external {
		prompt += "\nexternal·<command> profiles are autonomous local CLI coding agents (not our models): give them one complete, self-contained task; they edit files themselves and you get back their output tail plus a change summary."
	}
	return prompt
}

// agentsCommand manages sub-agent profiles: list, add (listing the provider's
// models when the model is omitted), remove, and toggle shell exec. Profiles are
// the only way to delegate, so they're also the curated list of allowed targets.
func (a *app) agentsCommand(ctx context.Context, rest string) []string {
	sub, arg := splitCommand(rest)
	switch sub {
	case "":
		return a.agentsLines()
	case "add", "set":
		return a.agentsAdd(ctx, arg)
	case "add-tool", "add-external":
		return a.agentsAddExternal(arg)
	case "rm", "remove", "del", "delete":
		return a.agentsRemove(arg)
	case "exec":
		return a.agentsExec(arg)
	default:
		return []string{"usage: /agents [add <name> <provider> [model]] [add-tool <name> <command> [args…]] [rm <name>] [exec on|off]"}
	}
}

// agentsAddExternal registers a locally installed CLI coding agent (codex, claude,
// aider…) as an external sub-agent profile. It runs OUTSIDE our sandbox — its own
// tools and permissions, edits invisible to /rewind — so every launch asks its own
// approval (the "external agent" category, separate from ordinary spawns).
func (a *app) agentsAddExternal(arg string) []string {
	fields := strings.Fields(arg)
	if len(fields) == 0 { // bare add-tool: scan PATH for the known CLI agents
		out := []string{"known CLI agents (✓ = installed):"}
		for _, c := range externalCatalog {
			mark, hint := "—", "not in PATH"
			if _, err := exec.LookPath(c.name); err == nil {
				mark, hint = "✓", "add it: /agents add-tool "+c.name
			}
			out = append(out, fmt.Sprintf("  %s %-10s %s", mark, c.name, hint))
		}
		return append(out,
			"any other tool: /agents add-tool <name> <command> [args…]   — {task} marks where the task goes",
			"  e.g. /agents add-tool mytool mytool --headless {task}")
	}
	// One word and it's a known CLI → catalog flags; otherwise the full form.
	name, command, args := fields[0], fields[0], []string(nil)
	if len(fields) >= 2 {
		command, args = fields[1], fields[2:]
	} else if args = catalogArgs(name); args == nil {
		return []string{
			fmt.Sprintf("don't know %q — give its full launch: /agents add-tool %s <command> [args…]  ({task} = where the task goes)", name, name),
			"one-word adds work for: " + strings.Join(catalogNames(), ", "),
		}
	}
	if len(args) == 0 { // full form without args, but a catalog command → its known flags
		args = catalogArgs(command)
	}
	if _, err := exec.LookPath(command); err != nil {
		return []string{fmt.Sprintf("%q not found in PATH — install it first, or give a full path", command)}
	}
	if existing, ok := a.cfg.Agents[name]; ok && existing.Kind != "external" {
		// Don't silently clobber an LLM profile with a CLI tool of the same name.
		return []string{fmt.Sprintf("%q is an LLM profile (%s · %s) — /agents rm %s first, or pick another name", name, existing.Provider, existing.Model, name)}
	}
	if a.cfg.Agents == nil {
		a.cfg.Agents = map[string]config.AgentProfile{}
	}
	a.cfg.Agents[name] = config.AgentProfile{Kind: "external", Command: command, Args: args}
	if err := config.SaveAgents(a.cfg.Agents); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	_ = a.wire() // the agent tool appears / the roster updates
	return []string{
		fmt.Sprintf("external profile %q → %s — saved", name, strings.Join(append([]string{command}, expandTaskArgs(args, "{task}")...), " ")),
		"runs outside the sandbox (own tools & permissions, /rewind can't see its edits) — every launch asks",
	}
}

// agentsLines lists configured agent profiles (for /agents).
func (a *app) agentsLines() []string {
	exec := "off"
	if a.cfg.Spawn.Exec {
		exec = "on"
	}
	if len(a.cfg.Agents) == 0 {
		return []string{
			"no sub-agent profiles yet. A profile is a named model the assistant can delegate a task to.",
			"  add one: /agents add <name> <provider> [model]   (omit the model to list them)",
			"  e.g.:    /agents add grok openrouter x-ai/grok-4.3",
		}
	}
	out := []string{fmt.Sprintf("sub-agent profiles (spawn: %s · shell exec: %s):", a.cfg.Spawn.Default, exec)}
	for _, n := range agentProfileNames(a.cfg) {
		p := a.cfg.Agents[n]
		if p.Kind == "external" {
			out = append(out, fmt.Sprintf("  %-14s external · %s", n, oneLine(strings.Join(append([]string{p.Command}, p.Args...), " "), 50)))
			continue
		}
		model := p.Model
		if model == "" {
			model = "(provider default)"
		}
		line := fmt.Sprintf("  %-14s %s · %s", n, p.Provider, model)
		if p.Prompt != "" {
			line += "  — " + oneLine(p.Prompt, 40)
		}
		out = append(out, line)
	}
	out = append(out, "  /agents add <name> <provider> [model] · add-tool <name> <command> [args…] · rm <name> · exec on|off")
	return out
}

// agentsAdd creates (or replaces) a profile. With no model it lists the chosen
// provider's models so the user can re-run with one; with a model it resolves the
// query to a single id (exact or unique substring) and saves.
func (a *app) agentsAdd(ctx context.Context, arg string) []string {
	fields := strings.Fields(arg)
	if len(fields) < 2 {
		return []string{"usage: /agents add <name> <provider> [model]", "  e.g. /agents add grok openrouter x-ai/grok-4.3"}
	}
	name, provider := fields[0], fields[1]
	conn, errLine := a.providerConn(provider)
	if errLine != "" {
		return []string{errLine}
	}
	cctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	ids := listModelIDs(cctx, conn)
	if len(fields) < 3 { // no model → list them to pick from
		if len(ids) == 0 {
			return []string{fmt.Sprintf("could not list %s models — pass one: /agents add %s %s <model>", provider, name, provider)}
		}
		out := []string{fmt.Sprintf("%s models — re-run: /agents add %s %s <model>", provider, name, provider)}
		for i, id := range ids {
			if i >= 50 {
				out = append(out, fmt.Sprintf("  …and %d more (narrow by typing part of the id)", len(ids)-50))
				break
			}
			out = append(out, "  "+id)
		}
		return out
	}
	setTo, lines := resolveModelArg(ids, strings.Join(fields[2:], " "))
	if setTo == "" {
		return lines // ambiguous — the matches are listed
	}
	if a.cfg.Agents == nil {
		a.cfg.Agents = map[string]config.AgentProfile{}
	}
	a.cfg.Agents[name] = config.AgentProfile{Provider: provider, Model: setTo}
	if err := config.SaveAgents(a.cfg.Agents); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	_ = a.wire() // the agent tool may now exist (or its roster changed)
	return []string{fmt.Sprintf("profile %q → %s · %s — saved", name, provider, setTo)}
}

// agentsRemove deletes a profile by name.
func (a *app) agentsRemove(name string) []string {
	name = strings.TrimSpace(name)
	if _, ok := a.cfg.Agents[name]; !ok {
		return []string{fmt.Sprintf("no profile %q (have: %s)", name, a.profilesOrHint())}
	}
	delete(a.cfg.Agents, name)
	if err := config.SaveAgents(a.cfg.Agents); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	_ = a.wire()
	return []string{fmt.Sprintf("removed profile %q", name)}
}

// agentsExec toggles whether sub-agents get the run (shell) tool.
func (a *app) agentsExec(arg string) []string {
	switch strings.TrimSpace(arg) {
	case "on", "yes", "true":
		a.cfg.Spawn.Exec = true
	case "off", "no", "false", "":
		a.cfg.Spawn.Exec = false
	default:
		return []string{"usage: /agents exec on|off"}
	}
	if err := config.SaveSpawn(a.cfg.Spawn); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	_ = a.wire()
	state := "off"
	if a.cfg.Spawn.Exec {
		state = "on"
	}
	return []string{"sub-agent shell exec → " + state + " — saved"}
}

// providerConn resolves a provider name (local or external-with-key) to its
// connection, or returns a one-line error for the caller to show.
func (a *app) providerConn(provider string) (config.LLM, string) {
	if provider == "local" {
		return a.cfg.LLM, ""
	}
	rp, ok := config.ResolveProvider(a.cfg, provider)
	if !ok {
		return config.LLM{}, fmt.Sprintf("unknown provider %q — configured: %s", provider, strings.Join(a.configuredProviderNames(), ", "))
	}
	if config.IsCustomProvider(a.cfg, provider) {
		if rp.BaseURL == "" { // a custom provider must at least name where to connect
			return config.LLM{}, fmt.Sprintf("%s has no base_url — set providers.%s.base_url in config", provider, provider)
		}
		return rp, "" // keyless is fine (local Ollama/vLLM/etc.)
	}
	if rp.APIKey == "" { // built-in cloud templates need a key
		return config.LLM{}, fmt.Sprintf("%s has no API key — add one with /ai key %s <token> first", provider, provider)
	}
	return rp, ""
}

// configuredProviderNames is local plus every external provider with a key —
// built-in templates and user-added custom providers alike.
func (a *app) configuredProviderNames() []string {
	out := []string{"local"}
	seen := map[string]bool{"local": true}
	add := func(n string) {
		if seen[n] {
			return
		}
		if l, ok := config.ResolveProvider(a.cfg, n); ok && (l.APIKey != "" || config.IsCustomProvider(a.cfg, n)) {
			out = append(out, n)
			seen[n] = true
		}
	}
	for _, n := range config.KnownProviders() {
		add(n)
	}
	for n := range a.cfg.Providers { // custom providers added with /ai add
		add(n)
	}
	sort.Strings(out[1:]) // local stays first
	return out
}

// agentProfileNames lists configured agent-profile names (sorted), for errors.
func agentProfileNames(cfg config.Config) []string {
	names := make([]string, 0, len(cfg.Agents))
	for n := range cfg.Agents {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// oneLine clips s to a single short line for prompts/labels.
func oneLine(s string, max int) string { return textutil.OneLine(s, max) }

// providerModel is the "provider · model" label shown in the status line.
func (a *app) providerModel() string { return a.providerName() + " · " + a.activeLLM().Model }

// effectiveDir is the session's current working directory: the /cd target, or the
// workspace root if none was set.
func (a *app) effectiveDir() string {
	if a.workdir != "" {
		return a.workdir
	}
	return a.workspace
}

// cdCommand sets the session working directory (the base for relative file/run/git
// paths and the default for sub-agents), confined to the workspace jail. So you
// point it at your project once instead of repeating the path everywhere.
func (a *app) cdCommand(arg string) []string {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		return []string{"working directory: " + a.effectiveDir(), "  /cd <dir> to change it (within the workspace jail)"}
	}
	abs, err := a.pol.Resolve(arg) // expands ~, resolves, enforces the jail
	if err != nil {
		return []string{"cd: " + err.Error()}
	}
	if fi, e := os.Stat(abs); e != nil || !fi.IsDir() {
		return []string{"cd: not a directory: " + arg}
	}
	if _, err := a.pol.SetWorkdir(arg); err != nil {
		return []string{"cd: " + err.Error()}
	}
	a.workdir = abs
	a.ag.SetSystem(a.systemPrompt()) // let the model see the new working dir
	return []string{"working directory → " + abs}
}

// offlineCommand toggles offline mode (no internet egress) and persists it. The
// web tool refuses, startup/`/update` skip GitHub — local model calls still work.
func (a *app) offlineCommand(arg string) []string {
	switch strings.TrimSpace(arg) {
	case "on", "yes", "true":
		a.cfg.Offline = true
	case "off", "no", "false":
		a.cfg.Offline = false
	case "":
		return []string{"offline mode is " + onOff(a.cfg.Offline) + " — /offline on|off",
			"  on = no internet: web tool + update checks off (your local model still works)"}
	default:
		return []string{"usage: /offline on|off"}
	}
	if err := config.SaveOffline(a.cfg.Offline); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	if err := a.wire(); err != nil { // rebuild the web tool with the new flag
		return []string{"error: " + err.Error()}
	}
	if a.cfg.Offline {
		return []string{"offline mode → on (web + update checks disabled; local model still works)"}
	}
	return []string{"offline mode → off (internet re-enabled)"}
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// reflectCommand controls the post-task reflection: on/off, or run it on a
// configured profile (a capable model) instead of the current one.
func (a *app) reflectCommand(arg string) []string {
	arg = strings.TrimSpace(arg)
	switch arg {
	case "":
		return []string{"reflection: " + onOff(!a.cfg.ReflectDisabled) + " · using " + a.reflectUsing(),
			"  /reflect on|off · /reflect <profile> (distill on another model) · /reflect self"}
	case "on", "yes", "true":
		a.cfg.ReflectDisabled = false
	case "off", "no", "false":
		a.cfg.ReflectDisabled = true
	case "self", "main", "local":
		a.cfg.ReflectProfile, a.cfg.ReflectDisabled = "", false
	default:
		if _, ok := a.cfg.Agents[arg]; !ok {
			return []string{fmt.Sprintf("unknown profile %q — have: %s; or use on|off|self", arg, a.profilesOrHint())}
		}
		a.cfg.ReflectProfile, a.cfg.ReflectDisabled = arg, false
	}
	if err := config.SaveReflectCfg(a.cfg.ReflectDisabled, a.cfg.ReflectProfile); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	return []string{"reflection → " + onOff(!a.cfg.ReflectDisabled) + " · using " + a.reflectUsing()}
}

func (a *app) reflectUsing() string {
	if a.cfg.ReflectProfile != "" {
		return "profile " + a.cfg.ReflectProfile
	}
	return "current model"
}

// goalState is the standing goal — an explicit, persisted objective the agent
// pursues across the judge-driven loop. Stored at <workspace>/.agent/goal.json so
// it survives a restart and can be resumed.
type goalState struct {
	Text   string `json:"text"`
	Status string `json:"status"` // active | done | incomplete
}

func (a *app) goalPath() string { return filepath.Join(a.workspace, ".agent", "goal.json") }

// loadGoal reads the persisted goal (best-effort; a missing/garbled file = none).
func (a *app) loadGoal() {
	data, err := os.ReadFile(a.goalPath())
	if err != nil {
		return
	}
	var g goalState
	if json.Unmarshal(data, &g) == nil {
		a.goal = g
	}
}

func (a *app) saveGoal() error {
	if err := os.MkdirAll(filepath.Dir(a.goalPath()), 0o755); err != nil {
		return err
	}
	data, _ := json.MarshalIndent(a.goal, "", "  ")
	return os.WriteFile(a.goalPath(), data, 0o644)
}

// launchGoalText reports the goal to set-and-pursue, or ("", false) when the
// argument is a sub-command (blank / status / clear / a TTL knob) instead.
func (a *app) launchGoalText(rest string) (string, bool) {
	rest = strings.TrimSpace(rest)
	if rest == "" {
		return "", false
	}
	switch strings.Fields(rest)[0] {
	case "go", "resume": // resume the standing goal
		return a.goal.Text, a.goal.Text != ""
	case "clear", "drop", "done", "ttl", "off", "on":
		return "", false
	}
	return rest, true
}

// setGoal records a new standing goal (active) and persists it.
func (a *app) setGoal(text string) {
	a.goal = goalState{Text: strings.TrimSpace(text), Status: "active"}
	if err := a.saveGoal(); err != nil {
		slog.Warn("goal not persisted", "err", err)
	}
}

// finishGoal updates the standing goal's status from a finished run, but only when
// that run was actually pursuing it (same text, still active).
func (a *app) finishGoal(goal string, tr agent.Transcript) {
	if a.goal.Status != "active" || strings.TrimSpace(goal) != a.goal.Text {
		return
	}
	// With the judge loop off (/goal off · ttl 0) GoalMet is never set — the model
	// decides when it's done, so a clean uninterrupted finish counts as done
	// (otherwise the goal stays "incomplete" and the resume prompt nags forever).
	if (tr.GoalMet || a.cfg.GoalMaxReturns == 0) && !tr.Stopped && !tr.Cancelled {
		a.goal.Status = "done"
	} else {
		a.goal.Status = "incomplete"
	}
	if err := a.saveGoal(); err != nil {
		slog.Warn("goal status not persisted", "err", err)
	}
}

// goalCommand handles the non-launching forms: show status, clear the goal, or set
// the return TTL. Setting a new goal to pursue goes through the run path instead.
func (a *app) goalCommand(arg string) []string {
	fields := strings.Fields(strings.TrimSpace(arg))
	verb := ""
	if len(fields) > 0 {
		verb = fields[0]
	}
	switch verb {
	case "clear", "drop", "done":
		had := a.goal.Text
		a.goal = goalState{}
		os.Remove(a.goalPath())
		if had == "" {
			return []string{"no standing goal to clear"}
		}
		return []string{"goal cleared"}
	case "ttl", "off", "on":
		return a.goalTTL(verb, fields)
	default:
		return a.goalStatus()
	}
}

// goalTTL sets the return budget: /goal ttl <n>, /goal off (0), /goal on (default).
func (a *app) goalTTL(verb string, fields []string) []string {
	n := a.cfg.GoalMaxReturns
	switch verb {
	case "off":
		n = 0
	case "on":
		n = config.Default().GoalMaxReturns
	case "ttl":
		if len(fields) < 2 {
			return []string{"usage: /goal ttl <n>  (re-feed budget; 0 = off)"}
		}
		v, err := strconv.Atoi(fields[1])
		if err != nil || v < 0 {
			return []string{"usage: /goal ttl <n>  (a non-negative count)"}
		}
		n = v
	}
	a.cfg.GoalMaxReturns = n // applied per goal run via goalTTLFor
	if err := config.SaveGoalMaxReturns(n); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	if n == 0 {
		return []string{"goal loop → off (one run; the model decides when it's done)"}
	}
	return []string{fmt.Sprintf("goal loop → up to %d re-feed(s) before giving up", n)}
}

func (a *app) goalStatus() []string {
	ttl := fmt.Sprintf("TTL %d re-feed(s)", a.cfg.GoalMaxReturns)
	if a.cfg.GoalMaxReturns == 0 {
		ttl = "loop off"
	}
	if a.goal.Text == "" {
		return []string{
			"no standing goal · " + ttl,
			"  /goal <text> sets one and pursues it · a judge re-feeds it until met · /goal ttl <n>",
		}
	}
	out := []string{
		fmt.Sprintf("goal [%s]: %s", a.goal.Status, a.goal.Text),
		"  " + ttl + " · /goal go to resume · /goal clear to drop · /goal ttl <n>",
	}
	return out
}

// goalTTLFor returns the judge re-feed budget for a run: the configured TTL only
// when the run is pursuing the active standing goal, else 0 (a plain task is one
// run, no judge overhead).
func (a *app) goalTTLFor(goal string) int {
	if a.goal.Status == "active" && strings.TrimSpace(goal) == a.goal.Text {
		return a.cfg.GoalMaxReturns
	}
	return 0
}

const maxPromptHist = 200

func (a *app) promptHistPath() string { return filepath.Join(a.workspace, ".agent", "history") }

// loadPromptHist reads the persisted input history (best-effort).
func (a *app) loadPromptHist() {
	data, err := os.ReadFile(a.promptHistPath())
	if err != nil {
		return
	}
	var h []string
	if json.Unmarshal(data, &h) == nil {
		a.promptHist = h
	}
}

// addPromptHist appends a submitted line (skipping a consecutive duplicate), caps
// the ring, and persists it so ↑ recall survives a restart.
func (a *app) addPromptHist(line string) {
	if n := len(a.promptHist); n > 0 && a.promptHist[n-1] == line {
		return
	}
	a.promptHist = append(a.promptHist, line)
	if len(a.promptHist) > maxPromptHist {
		a.promptHist = a.promptHist[len(a.promptHist)-maxPromptHist:]
	}
	data, _ := json.Marshal(a.promptHist)
	if err := atomicfile.Write(a.promptHistPath(), data, 0o644); err != nil {
		slog.Warn("prompt history not saved", "err", err)
	}
}

// historyCommand lists recent prompts (newest first), optionally filtered by a
// substring — a way to dig through what you've asked across sessions.
func (a *app) historyCommand(arg string) []string {
	filter := strings.ToLower(strings.TrimSpace(arg))
	var out []string
	const show = 30
	for i := len(a.promptHist) - 1; i >= 0 && len(out) < show; i-- {
		p := a.promptHist[i]
		if filter != "" && !strings.Contains(strings.ToLower(p), filter) {
			continue
		}
		out = append(out, "  "+oneLine(p, 100))
	}
	if len(out) == 0 {
		if filter != "" {
			return []string{"no prompts match " + arg}
		}
		return []string{"no prompt history yet"}
	}
	head := fmt.Sprintf("prompt history (newest first, %d shown) — ↑/↓ recalls into the input:", len(out))
	if filter != "" {
		head = fmt.Sprintf("prompts matching %q (newest first):", arg)
	}
	return append([]string{head}, out...)
}

// completePath returns the longest-common-prefix completion and the matching
// workspace files (relative to the path base the model resolves against) for an
// @file prefix — powers Tab-completion of @path references in the input.
func (a *app) completePath(prefix string) (string, []string) {
	base := a.pol.Workdir()
	skip := map[string]bool{".git": true, "node_modules": true, "vendor": true, "dist": true, ".agent": true}
	var matches []string
	_ = filepath.WalkDir(base, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if p != base && (skip[d.Name()] || strings.HasPrefix(d.Name(), ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if rel, e := filepath.Rel(base, p); e == nil && strings.HasPrefix(rel, prefix) {
			matches = append(matches, rel)
			if len(matches) >= 200 {
				return filepath.SkipAll
			}
		}
		return nil
	})
	sort.Strings(matches)
	return longestCommonPrefix(matches), matches
}

// completeDir Tab-completes a directory path for /cd, one segment at a time like
// a shell: it lists the sub-directories of the already-typed parent whose name
// starts with the segment under the cursor. Candidates keep the parent the user
// typed (incl. ~, .., an absolute root) and end in "/" so completing can descend.
// Resolution goes through the policy engine, so only dirs inside the jail show.
func (a *app) completeDir(prefix string) (string, []string) {
	dirPart, seg := "", prefix
	if i := strings.LastIndexByte(prefix, '/'); i >= 0 {
		dirPart, seg = prefix[:i+1], prefix[i+1:]
	}
	listBase := dirPart
	if listBase == "" {
		listBase = "."
	}
	abs, err := a.pol.Resolve(listBase) // ~, relative→workdir, and the jail
	if err != nil {
		return "", nil
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", nil
	}
	var matches []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") && !strings.HasPrefix(seg, ".") {
			continue // hide dot-dirs unless the user is explicitly typing one
		}
		if strings.HasPrefix(name, seg) {
			matches = append(matches, dirPart+name+"/")
		}
	}
	sort.Strings(matches)
	return longestCommonPrefix(matches), matches
}

// gitOut runs a read-only git command in dir and returns its stdout.
func gitOut(dir string, args ...string) (string, error) {
	out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).Output()
	return string(out), err
}

// mustGit is gitOut with the error dropped — for best-effort reads (diff/stat)
// where an empty string is an acceptable answer in a non-repo dir.
func mustGit(dir string, args ...string) string { s, _ := gitOut(dir, args...); return s }

// diffCommand shows the uncommitted working-tree changes in the current dir — a
// quick "what did the agent change" review without leaving the TUI. The caller
// colorizes the lines. Caps very large diffs so the log doesn't flood.
func (a *app) diffCommand() []string {
	dir := a.effectiveDir()
	if out, err := gitOut(dir, "rev-parse", "--is-inside-work-tree"); err != nil || strings.TrimSpace(out) != "true" {
		return []string{"/diff needs a git repo here — nothing to compare against"}
	}
	stat := strings.TrimSpace(mustGit(dir, "diff", "--stat"))
	patch := strings.TrimRight(mustGit(dir, "diff"), "\n")
	untracked := strings.TrimSpace(mustGit(dir, "ls-files", "--others", "--exclude-standard"))
	if stat == "" && untracked == "" {
		return []string{"no uncommitted changes"}
	}
	var out []string
	if stat != "" {
		out = append(out, strings.Split(stat, "\n")...)
	}
	if patch != "" {
		out = append(out, "")
		out = append(out, strings.Split(patch, "\n")...)
	}
	if untracked != "" {
		out = append(out, "", "untracked (new) files:")
		for _, f := range strings.Split(untracked, "\n") {
			out = append(out, "  + "+f)
		}
	}
	const maxLines = 400
	if len(out) > maxLines {
		out = append(out[:maxLines], fmt.Sprintf("… (%d more lines — run `git diff` for the rest)", len(out)-maxLines))
	}
	return out
}

// goalSteps is the hard step backstop for one goal pursuit, falling back to the
// model's own per-task cap if it isn't configured.
func (a *app) goalSteps() int {
	if a.cfg.GoalMaxSteps > 0 {
		return a.cfg.GoalMaxSteps
	}
	return a.activeLLM().MaxSteps
}

// reasoningParams resolves the merge-params for (provider, model). scope ""
// checks "<provider>/<model>" then "<provider>"; scope "reflect" checks the
// reflect-prefixed keys first (a separate setting for the learning pass), then
// falls back to the normal ones.
func (a *app) reasoningParams(provider, model, scope string) map[string]any {
	keys := []string{provider + "/" + model, provider}
	if scope != "" {
		keys = append([]string{scope + ":" + provider + "/" + model, scope + ":" + provider}, keys...)
	}
	for _, k := range keys {
		if raw, ok := a.cfg.Reasoning[k]; ok {
			var m map[string]any
			if json.Unmarshal(raw, &m) == nil {
				return m
			}
		}
	}
	return nil
}

// withReasoning attaches the resolved reasoning params (Extra) to a connection
// before a client is built.
func (a *app) withReasoning(l config.LLM, provider, scope string) config.LLM {
	l.Extra = a.reasoningParams(provider, l.Model, scope)
	return l
}

// reasoningShape returns the request-body params that set reasoning to level
// (low|medium|high|minimal) or turn it off, in the given provider's own format.
// ok=false for a provider whose shape we don't know — the user sets it raw.
func reasoningShape(provider, level string) (json.RawMessage, bool) {
	off := level == "off"
	switch provider {
	case "local", "openai", "grok", "groq":
		// OpenAI-style reasoning_effort. "off" has no portable value → caller deletes.
		if off {
			return nil, true
		}
		return json.RawMessage(fmt.Sprintf(`{"reasoning_effort":%q}`, level)), true
	case "openrouter":
		if off {
			return json.RawMessage(`{"reasoning":{"enabled":false}}`), true
		}
		return json.RawMessage(fmt.Sprintf(`{"reasoning":{"effort":%q}}`, level)), true
	case "anthropic":
		if off {
			return json.RawMessage(`{}`), true // omit thinking = off
		}
		return nil, false // budget_tokens is model-specific — set raw
	case "zai":
		// GLM thinking is binary — any level enables it, off disables.
		if off {
			return json.RawMessage(`{"thinking":{"type":"disabled"}}`), true
		}
		return json.RawMessage(`{"thinking":{"type":"enabled"}}`), true
	}
	return nil, false
}

// reasoningCommand sets reasoning for the CURRENT provider+model in that
// provider's own shape (low|medium|high|minimal|off), stored as a per-model
// override. Unknown providers / custom shapes: edit config.reasoning directly.
func (a *app) reasoningCommand(arg string) []string {
	arg = strings.ToLower(strings.TrimSpace(arg))
	// "/reasoning reflect <level>" targets the learning pass's model with a
	// reflect-scoped override; otherwise the current task model.
	scope, provider, model := "", a.providerName(), a.activeLLM().Model
	if rest, ok := strings.CutPrefix(arg, "reflect"); ok && (rest == "" || rest[0] == ' ') {
		scope, arg = "reflect:", strings.TrimSpace(rest)
		_, _, _, provider, model = a.reflectTarget()
	}
	key := scope + provider + "/" + model
	label := provider + " · " + model
	if scope != "" {
		label = "reflection · " + label
	}
	if arg == "" {
		cur := "default"
		if raw, ok := a.cfg.Reasoning[key]; ok {
			cur = string(raw)
		} else if raw, ok := a.cfg.Reasoning[scope+provider]; ok {
			cur = string(raw) + " (provider default)"
		}
		return []string{
			fmt.Sprintf("reasoning for %s: %s", label, cur),
			"  /reasoning off|minimal|low|medium|high — set it for this model",
			"  /reasoning reflect <level> — a separate setting for the learning pass",
			"  custom shapes: edit \"reasoning\" in config.json (key " + key + ")",
		}
	}
	switch arg {
	case "off", "minimal", "low", "medium", "high":
	default:
		return []string{"usage: /reasoning off|minimal|low|medium|high"}
	}
	shape, ok := a.applyReasoning(key, provider, arg)
	if !ok {
		return []string{
			fmt.Sprintf("don't know %s's reasoning param — set it raw in config.json under", provider),
			fmt.Sprintf("  \"reasoning\": { %q: { …provider's params… } }", key),
		}
	}
	if shape == nil {
		return []string{fmt.Sprintf("reasoning → default for %s · %s (cleared)", provider, model)}
	}
	return []string{fmt.Sprintf("reasoning → %s for %s · %s  %s", arg, provider, model, string(shape))}
}

// reasoningLevels is the cycle order shared by /reasoning and the /config row.
var reasoningLevels = []string{"off", "minimal", "low", "medium", "high"}

// applyReasoning sets (or clears, for a portable "off") the reasoning override at
// key to provider's shape for level, persists it, and re-wires. ok=false means the
// provider has no known portable shape (set it raw in config.json instead).
func (a *app) applyReasoning(key, provider, level string) (json.RawMessage, bool) {
	shape, known := reasoningShape(provider, level)
	if !known {
		return nil, false
	}
	if a.cfg.Reasoning == nil {
		a.cfg.Reasoning = map[string]json.RawMessage{}
	}
	if shape == nil { // "off" with no portable value → clear the override (server default)
		delete(a.cfg.Reasoning, key)
	} else {
		a.cfg.Reasoning[key] = shape
	}
	if err := config.SaveReasoning(a.cfg.Reasoning); err != nil {
		slog.Warn("reasoning not persisted", "err", err)
	}
	_ = a.wire() // rebuild the client so the change takes effect
	return shape, true
}

// reasoningLevel reverse-maps the stored raw param back to a level name for display
// ("default" if none is set, "custom" if it doesn't match a known level).
func (a *app) reasoningLevel(provider, model string) string {
	raw, ok := a.cfg.Reasoning[provider+"/"+model]
	if !ok {
		raw, ok = a.cfg.Reasoning[provider] // provider default
	}
	if !ok {
		return "default"
	}
	for _, lvl := range reasoningLevels {
		if shape, known := reasoningShape(provider, lvl); known && shape != nil && string(shape) == string(raw) {
			return lvl
		}
	}
	return "custom"
}

// nextReasoning returns the next level in the cycle after cur (wrapping); a
// non-level (default/custom) starts the cycle at its head.
func nextReasoning(cur string) string {
	for i, l := range reasoningLevels {
		if l == cur {
			return reasoningLevels[(i+1)%len(reasoningLevels)]
		}
	}
	return reasoningLevels[0]
}

// mcpServerNames lists configured MCP server names, sorted.
func (a *app) mcpServerNames() []string {
	names := make([]string, 0, len(a.cfg.McpServers))
	for n := range a.cfg.McpServers {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// mcpClient returns a connected client for a server, launching + caching it on
// first use (lazy — servers aren't spawned until something needs them).
func (a *app) mcpClient(ctx context.Context, name string) (*mcp.Client, error) {
	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	if c, ok := a.mcpClients[name]; ok {
		return c, nil
	}
	srv, ok := a.cfg.McpServers[name]
	if !ok {
		return nil, fmt.Errorf("unknown MCP server %q (configured: %s)", name, strings.Join(a.mcpServerNames(), ", "))
	}
	cctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	c, err := mcp.Connect(cctx, name, srv)
	if err != nil {
		return nil, err
	}
	if a.mcpClients == nil {
		a.mcpClients = map[string]*mcp.Client{}
	}
	a.mcpClients[name] = c
	return c, nil
}

// closeMCP shuts down every launched MCP server (called on exit).
func (a *app) closeMCP() {
	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	for _, c := range a.mcpClients {
		c.Close()
	}
	a.mcpClients = nil
}

// mcpList is the catalog the `mcp` tool's list action returns.
func (a *app) mcpList(ctx context.Context) string {
	if len(a.cfg.McpServers) == 0 {
		return "no MCP servers configured — add them under \"mcp_servers\" in config.json"
	}
	var b strings.Builder
	for _, name := range a.mcpServerNames() {
		c, err := a.mcpClient(ctx, name)
		if err != nil {
			fmt.Fprintf(&b, "%s: (unavailable: %s)\n", name, oneLine(err.Error(), 70))
			continue
		}
		fmt.Fprintf(&b, "%s:\n", name)
		for _, t := range c.Tools() {
			fmt.Fprintf(&b, "  %s — %s\n", t.Name, oneLine(t.Description, 70))
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// mcpSchema returns one tool's input schema (for the tool's schema action).
func (a *app) mcpSchema(ctx context.Context, server, tool string) string {
	c, err := a.mcpClient(ctx, server)
	if err != nil {
		return "error: " + err.Error()
	}
	for _, t := range c.Tools() {
		if t.Name == tool {
			if len(t.InputSchema) == 0 {
				return "(no input schema)"
			}
			return string(t.InputSchema)
		}
	}
	return fmt.Sprintf("no tool %q on %q", tool, server)
}

// mcpCall runs an MCP tool, asking approval first (it's external code that can do
// anything). The approval prompt is serialized like sub-agent spawns.
func (a *app) mcpCall(ctx context.Context, server, tool string, args map[string]any) (string, error) {
	c, err := a.mcpClient(ctx, server)
	if err != nil {
		return "", err
	}
	detail := server + "." + tool
	if len(args) > 0 {
		if b, e := json.Marshal(args); e == nil {
			detail += " " + oneLine(string(b), 60)
		}
	}
	a.spawnMu.Lock()
	approved := a.approveGated("mcp call", detail)
	a.spawnMu.Unlock()
	if !approved {
		return "", fmt.Errorf("mcp call denied by user")
	}
	cctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	return c.Call(cctx, tool, args)
}

// maybeDetectWindowSync detects the window inline for the REPL/one-shot paths;
// the TUI skips this and re-detects off the UI thread (detectWindowCmd) so a slow
// provider can't freeze the screen.
func (a *app) maybeDetectWindowSync() {
	if !a.tui {
		a.detectContextWindow()
	}
}

// applyWindow records a detected context window for a provider (UI thread, so it
// never races View/auto-compact). "local" sets the LM Studio connection; any
// other name sets that provider's preset.
func (a *app) applyWindow(provider string, tokens int) {
	if tokens <= 0 {
		return
	}
	if provider == "local" {
		a.cfg.LLM.ContextWindow = tokens
		return
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[provider]
	p.ContextWindow = tokens
	a.cfg.Providers[provider] = p
}

// detectContextWindow learns the active model's context window so the status bar
// and auto-compact size against the real limit. LM Studio reports the loaded
// model's window via its native API; external providers that list a
// context_length (OpenRouter) report it via /models. Reset on every model/
// provider switch (windowDetected=false) so it re-detects. Best-effort.
func (a *app) detectContextWindow() {
	if a.windowDetected {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if a.isLocal() {
		if !a.cfg.LLM.LMStudio() {
			return
		}
		if w := llm.DetectContextWindow(ctx, a.cfg.LLM.BaseURL, a.cfg.LLM.Model, http.DefaultClient); w > 0 {
			a.cfg.LLM.ContextWindow = w
			a.windowDetected = true
			slog.Info("detected context window", "tokens", w, "model", a.cfg.LLM.Model)
		}
		return
	}
	act := a.activeLLM()
	if w := llm.DetectModelContext(ctx, act.BaseURL, act.APIKey, act.Model, http.DefaultClient); w > 0 {
		if a.cfg.Providers == nil {
			a.cfg.Providers = map[string]config.LLM{}
		}
		p := a.cfg.Providers[a.cfg.Provider]
		p.ContextWindow = w
		a.cfg.Providers[a.cfg.Provider] = p
		a.windowDetected = true
		slog.Info("detected context window", "tokens", w, "model", act.Model, "provider", a.providerName())
	}
}

// wire (re)builds the policy-gated tools, LLM client, agent, and reflector from
// the current config and the current approver/tracers. Called at startup, after
// /login, and when the TUI installs its bridge.
// wire (re)builds the full runtime stack from a.cfg — policy engine, tool
// registries (main + sub-agent), LLM client (usage carried over), agent loop and
// system prompt. Called at startup and after any setting that changes what the
// agent is allowed to do or talk to. NOT safe while a task goroutine is running
// (the running agent's history would race) — callers gate on that.
func (a *app) wire() error {
	pol, err := policy.New(a.cfg)
	if err != nil {
		return fmt.Errorf("policy: %w", err)
	}
	a.pol = pol // host jail; a sub-agent pointed at a dir gets its own
	if a.workdir != "" {
		_, _ = a.pol.SetWorkdir(a.workdir) // re-apply /cd across a re-wire (best-effort)
	}
	var reg *tool.Registry
	tools := []tool.Tool{
		tool.NewFile(pol, gatedApprover{a}, a.snapFile),
		tool.NewRun(pol, gatedApprover{a}, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second),
		tool.NewGit(pol, gatedApprover{a}),
		tool.NewWeb(http.DefaultClient, a.cfg.Offline),
		tool.NewHelp(a.kb, func(d string) string { return reg.Usage(d) }),
		tool.NewCalc(),
	}
	// The skill tool only exists when a skill is enabled, so it costs nothing in
	// the catalog until the user opts in.
	if a.skills != nil && a.skills.HasEnabled() {
		tools = append(tools, tool.NewSkill(a.skills))
	}
	// Sub-agents get their own registry: no `agent` tool (no recursion) and no run
	// tool unless spawn.exec is on. buildSubReg is the single source of truth.
	a.subReg = a.buildSubReg(pol)
	// The `agent` tool is only worth its catalog space when there is a profile to
	// delegate to; with no profiles configured, the tool is hidden entirely.
	if a.hasSubagentTargets() {
		tools = append(tools, tool.NewAgent(a.spawnAgent, a.spawnAgentBackground))
	}
	// One proxy tool fronts every configured MCP server, so the catalog grows by a
	// single tool — not by every server's schemas. Only when servers are set.
	if len(a.cfg.McpServers) > 0 {
		tools = append(tools, tool.NewMCP(a.mcpList, a.mcpCall, a.mcpSchema))
	}
	reg = tool.NewRegistry(tools...)
	a.tracer = trace.Multi(a.fileTracer, a.uiTracer)
	// Carry the session's running token total into the rebuilt client so a
	// /skills, /permissions or /login re-wire doesn't zero the counter.
	var seedP, seedC int
	if a.client != nil {
		seedP, seedC = a.client.Usage()
	}
	a.client = llm.NewOpenAIClient(a.withReasoning(a.activeLLM(), a.providerName(), ""))
	a.client.SeedUsage(seedP, seedC)
	a.client.OnRetry = func(attempt int, wait time.Duration, reason string) {
		slog.Warn("llm retry", "attempt", attempt, "wait", wait, "reason", reason)
		a.emit("retry", map[string]any{"attempt": attempt, "wait_ms": wait.Milliseconds(), "reason": reason})
	}
	// Carry the live session into the new agent. wire() is called again to install
	// the TUI bridge and on /login to reload config; without this hand-off the
	// restored conversation would be dropped and every launch would start blank.
	var prior []llm.Message
	if a.ag != nil {
		prior = a.ag.History()
	}
	a.ag = agent.New(a.client, reg, a.kb, a.tracer, a.systemPrompt(), a.goalSteps())
	a.ag.SetHistory(prior)
	a.ag.SetPlanMode(a.planMode)   // carry the mode into the rebuilt agent
	a.ag.SetBeforeTurn(a.drainBtw) // /btw notes fold in between turns of a running task
	return nil
}

// setMode switches between plan (investigate + propose) and auto (execute) and
// returns a one-line confirmation.
func (a *app) setMode(plan bool) string {
	a.planMode = plan
	if a.ag != nil {
		a.ag.SetPlanMode(plan)
	}
	if plan {
		return "plan mode on — investigates and proposes a plan; changes nothing"
	}
	return "auto mode on — executes the task"
}

func (a *app) reconfigure() error {
	cfg, err := config.Load(a.workspace)
	if err != nil {
		return err
	}
	a.cfg = cfg
	if err := a.wire(); err != nil {
		return err
	}
	a.loadSession()          // a fresh agent — restore the persisted session
	a.windowDetected = false // model may have changed (/login) — re-detect
	a.maybeDetectWindowSync()
	return nil
}

// autoCompactRatio is the fraction of the context window at which the session is
// auto-compacted, leaving headroom for the next task.
const autoCompactRatio = 0.75

// autoCompactNeeded decides whether to fold the session into a summary: the last
// prompt is past the threshold and there's enough history to be worth it. A zero
// window disables it.
func autoCompactNeeded(ctxTokens, window, sessionLen int) bool {
	if window <= 0 || sessionLen < 4 {
		return false
	}
	return ctxTokens >= int(float64(window)*autoCompactRatio)
}

func (a *app) shouldAutoCompact() bool {
	return autoCompactNeeded(a.client.Context(), a.activeLLM().ContextWindow, a.ag.SessionLen())
}

// Session memory persists per workspace AND per agent name, so each named agent
// (see /rename) keeps its own thread of context across restarts. /new and /clear
// wipe the active one.
func (a *app) sessionPath() string {
	return filepath.Join(a.workspace, ".agent", "sessions", slugName(a.cfg.Name)+".json")
}

// legacySessionPath is the pre-naming location, read as a fallback for the
// default name so existing sessions still restore after upgrading.
func (a *app) legacySessionPath() string {
	return filepath.Join(a.workspace, ".agent", "session.json")
}

// slugName turns a display name into a safe filename stem.
func slugName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return "ipsupport-code"
	}
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			b.WriteRune(r)
			lastDash = false
		default: // anything else becomes a single dash (no runs)
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	if out := strings.Trim(b.String(), "-"); out != "" {
		return out
	}
	return "agent"
}

// existingSessionPath returns the path of the saved session for the current name
// (or the legacy path for the default name), or "" if none exists.
func (a *app) existingSessionPath() string {
	if _, err := os.Stat(a.sessionPath()); err == nil {
		return a.sessionPath()
	}
	if slugName(a.cfg.Name) == "ipsupport-code" {
		if _, err := os.Stat(a.legacySessionPath()); err == nil {
			return a.legacySessionPath()
		}
	}
	return ""
}

func (a *app) loadSession() {
	path := a.existingSessionPath()
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var h []llm.Message
	if json.Unmarshal(data, &h) == nil {
		a.ag.SetHistory(h)
	}
}

// newNamedSession saves the current thread (so it stays returnable via /sessions),
// adopts name for this run, re-wires, and starts a FRESH empty thread. persist
// writes name as the default identity (explicit /new <name>); a bare /new's
// auto-named scratch thread doesn't persist, so the default doesn't drift.
func (a *app) newNamedSession(name string, persist bool) error {
	a.saveSession()
	a.cfg.Name = name
	if persist {
		if err := config.SaveGlobal(name, a.cfg.LLM); err != nil {
			return err
		}
	}
	if err := a.wire(); err != nil {
		return err
	}
	a.ag.Reset()          // fresh — don't load name's prior thread
	a.resetSessionAllow() // a new session shouldn't inherit "allow all this session"
	return nil
}

// autoSessionName picks the next free "<base>-N" so a bare /new gets a fresh
// scratch thread without clobbering an existing one.
func (a *app) autoSessionName() string {
	base := slugName(a.cfg.Name)
	taken := map[string]bool{}
	for _, s := range a.listSessions() {
		taken[s.name] = true
	}
	for i := 2; ; i++ {
		if n := fmt.Sprintf("%s-%d", base, i); !taken[n] {
			return n
		}
	}
}

// humanizeAgo renders how long ago t was, compactly.
func humanizeAgo(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	}
}

func (a *app) saveSession() {
	if err := os.MkdirAll(filepath.Dir(a.sessionPath()), 0o755); err != nil {
		return
	}
	if data, err := json.Marshal(a.ag.History()); err == nil {
		_ = os.WriteFile(a.sessionPath(), data, 0o644)
	}
}

// sessionMeta describes one saved session for the /sessions list.
type sessionMeta struct {
	name   string
	count  int
	mod    time.Time
	active bool
}

// readSessionMeta reads a session file's message count and mod time (0 on error).
func readSessionMeta(path string) (count int, mod time.Time) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, time.Time{}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, time.Time{}
	}
	var h []llm.Message
	if json.Unmarshal(data, &h) != nil {
		return 0, time.Time{}
	}
	return len(h), fi.ModTime()
}

// listSessions returns every saved session in this workspace (the per-name files
// plus the legacy one for the default name), most recently used first.
func (a *app) listSessions() []sessionMeta {
	active := slugName(a.cfg.Name)
	seen := map[string]bool{}
	var out []sessionMeta
	dir := filepath.Join(a.workspace, ".agent", "sessions")
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		slug := strings.TrimSuffix(e.Name(), ".json")
		count, mod := readSessionMeta(filepath.Join(dir, e.Name()))
		if count == 0 {
			continue
		}
		seen[slug] = true
		out = append(out, sessionMeta{name: slug, count: count, mod: mod, active: slug == active})
	}
	if !seen["ipsupport-code"] { // legacy file counts as the default name
		if count, mod := readSessionMeta(a.legacySessionPath()); count > 0 {
			out = append(out, sessionMeta{name: "ipsupport-code", count: count, mod: mod, active: active == "ipsupport-code"})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].mod.After(out[j].mod) })
	return out
}

// sessionsCommand handles /sessions: list (no arg), delete <name>, or switch to a
// name (any other arg). Switching adopts that name (like /rename) and loads its
// thread. Returns lines to show plus whether a switch happened (the TUI then
// replays a recap).
func (a *app) sessionsCommand(rest string) (lines []string, switched bool) {
	rest = strings.TrimSpace(rest)
	if rest == "" {
		return a.listSessionsLines(), false
	}
	sub, arg := splitCommand(rest)
	if sub == "delete" || sub == "rm" {
		return a.deleteSessionNamed(arg), false
	}
	// otherwise the whole arg is the target session name to switch to
	if err := a.switchSession(rest); err != nil {
		return []string{"switch failed: " + err.Error()}, false
	}
	return []string{"switched to session " + a.cfg.Name}, true
}

func (a *app) listSessionsLines() []string {
	ss := a.listSessions()
	if len(ss) == 0 {
		return []string{"no saved sessions yet — the current one saves after each task"}
	}
	out := []string{"saved sessions (● active) — /sessions <name> to switch · /sessions delete <name>:"}
	for _, s := range ss {
		mark := "  "
		if s.active {
			mark = "● "
		}
		out = append(out, fmt.Sprintf("  %s%-20s %d exchange(s) · %s", mark, s.name, s.count/2, humanizeAgo(s.mod)))
	}
	return out
}

// switchSession persists the current thread, adopts name as the agent's identity
// (saved, like /rename), re-wires so the prompt reflects it, and loads that name's
// thread (empty if it's a brand-new name).
func (a *app) switchSession(name string) error {
	a.saveSession()
	a.cfg.Name = name
	if err := config.SaveGlobal(name, a.cfg.LLM); err != nil {
		return err
	}
	if err := a.wire(); err != nil { // new-name prompt; carries (old) history
		return err
	}
	a.ag.Reset()
	a.loadSession() // replace with the target name's thread
	return nil
}

// deleteSessionNamed removes a named session's file (and the legacy file for the
// default name). If it's the active thread, memory is cleared too.
func (a *app) deleteSessionNamed(name string) []string {
	if strings.TrimSpace(name) == "" {
		return []string{"usage: /sessions delete <name>"}
	}
	slug := slugName(name)
	removed := false
	if os.Remove(filepath.Join(a.workspace, ".agent", "sessions", slug+".json")) == nil {
		removed = true
	}
	if slug == "ipsupport-code" && os.Remove(a.legacySessionPath()) == nil {
		removed = true
	}
	if !removed {
		return []string{"no saved session named " + slug}
	}
	if slug == slugName(a.cfg.Name) && a.ag != nil {
		a.ag.Reset()
	}
	return []string{"deleted session " + slug}
}

// maxFacts caps how many learned project facts we keep (most recent win).
const maxFacts = 30

func (a *app) factsPath() string { return filepath.Join(a.workspace, ".agent", "facts.json") }

func (a *app) loadFacts() {
	if data, err := os.ReadFile(a.factsPath()); err == nil {
		var f []string
		if json.Unmarshal(data, &f) == nil {
			a.facts = f
		}
	}
}

// addFacts dedupe-appends learned facts (most recent maxFacts kept), persists,
// and returns the genuinely new ones.
func (a *app) addFacts(facts []string) []string {
	seen := map[string]bool{}
	for _, f := range a.facts {
		seen[strings.ToLower(f)] = true
	}
	var added []string
	for _, f := range facts {
		f = strings.TrimSpace(f)
		if f == "" || seen[strings.ToLower(f)] {
			continue
		}
		seen[strings.ToLower(f)] = true
		a.facts = append(a.facts, f)
		added = append(added, f)
	}
	if len(a.facts) > maxFacts {
		a.facts = append([]string(nil), a.facts[len(a.facts)-maxFacts:]...)
	}
	if len(added) > 0 {
		_ = os.MkdirAll(filepath.Dir(a.factsPath()), 0o755)
		if data, err := json.Marshal(a.facts); err == nil {
			_ = os.WriteFile(a.factsPath(), data, 0o644)
		}
	}
	return added
}

const maxInstructions = 6000

// loadInstructions reads a project instructions file from the workspace (the
// agent's CLAUDE.md, à la Claude Code). Returns its content and the file it came
// from, or empty when none exists.
func loadInstructions(workspace string) (text, source string) {
	for _, name := range []string{"CLAUDE.md", "AGENTS.md", ".agent/instructions.md"} {
		data, err := os.ReadFile(filepath.Join(workspace, name))
		if err != nil || strings.TrimSpace(string(data)) == "" {
			continue
		}
		clipped, _ := textutil.Clip(string(data), maxInstructions)
		return clipped, name
	}
	return "", ""
}

// loadSystemOverride reads a system-prompt override that REPLACES the built-in
// base — workspace .agent/system.md wins, then the global one. Empty when none.
func loadSystemOverride(workspace string) (text, source string) {
	for _, p := range []string{filepath.Join(workspace, ".agent", "system.md"), config.SystemPromptPath()} {
		data, err := os.ReadFile(p)
		if err != nil || strings.TrimSpace(string(data)) == "" {
			continue
		}
		clipped, _ := textutil.Clip(string(data), maxInstructions)
		return clipped, p
	}
	return "", ""
}

// systemPrompt is the base prompt plus the real environment (OS + workspace) and
// any project instructions. The base is the built-in default unless a system.md
// override replaces it. Records the instructions and prompt sources for /status.
func (a *app) systemPrompt() string {
	text, src := loadInstructions(a.workspace)
	a.instrSrc = src

	base, psrc := agent.DefaultSystemPrompt(), "built-in"
	if override, osrc := loadSystemOverride(a.workspace); override != "" {
		base, psrc = override, osrc
	} else if a.cfg.Name != "" && a.cfg.Name != "ipsupport-code" { // honor /rename (default only)
		base = strings.ReplaceAll(base, "ipsupport-code", a.cfg.Name)
	}
	a.promptSrc = psrc
	out := base + fmt.Sprintf(
		"\n\nEnvironment: you are running on %s; your working directory is %s. Relative paths resolve there. Use commands that exist on this OS — on darwin prefer vm_stat/top/sw_vers over Linux-only tools like free.",
		runtime.GOOS, a.effectiveDir())
	if text != "" {
		out += "\n\n## Project instructions (from " + src + ") — follow these:\n" + text
	}
	if a.skills != nil {
		if idx := a.skills.Index(); idx != "" {
			out += "\n\n## Skills (load full instructions with the skill tool when the topic fits):\n" + idx
		}
	}
	out += a.subagentTargetsPrompt() // dynamic roster of delegate targets (empty if none)
	if len(a.facts) > 0 {            // learned project facts — keep the injected set small
		facts := a.facts
		if len(facts) > 15 {
			facts = facts[len(facts)-15:]
		}
		out += "\n\n## Known facts about this project (learned on past runs):\n- " + strings.Join(facts, "\n- ")
	}
	return out
}

func (a *app) emit(kind string, fields map[string]any) {
	if a.tracer != nil {
		a.tracer.Emit(kind, fields)
	}
}

func (a *app) recordRun(tr agent.Transcript) {
	a.tasks++
	a.steps += tr.Steps
	for _, m := range tr.Messages {
		if m.Role == "tool" {
			a.toolCalls++
		}
	}
}

// recordUsage attributes the tokens spent since the last call to today's
// provider/model bucket in the persistent ledger. Best-effort; called once a
// task (and its reflection) has finished. The client's cumulative count carries
// across re-wires, so the delta is always the work done since the prior task.
func (a *app) recordUsage() {
	if a.usage == nil {
		return
	}
	p, c := a.client.Usage()
	dp, dc := p-a.lastPrompt, c-a.lastCompl
	a.lastPrompt, a.lastCompl = p, c
	if dp <= 0 && dc <= 0 {
		return
	}
	a.usage.Add(today(), a.providerName(), a.activeLLM().Model, dp, dc)
	a.addSessionCost(a.activeLLM().Model, dp, dc) // for the budget guard
	if err := a.usage.Save(); err != nil {
		slog.Warn("usage ledger save failed", "err", err)
	}
}

// addSessionCost accrues estimated spend for the budget guard. Mutex-guarded:
// parallel sub-agent spawns record their spend from their own goroutines.
func (a *app) addSessionCost(model string, prompt, completion int) {
	a.costMu.Lock()
	a.sessionCostUSD += usage.CostUSD(model, prompt, completion, a.priceOverrides())
	a.costMu.Unlock()
}

// sessionCost reads the accrued estimated spend (guarded, see addSessionCost).
func (a *app) sessionCost() float64 {
	a.costMu.Lock()
	defer a.costMu.Unlock()
	return a.sessionCostUSD
}

// budgetExceeded reports whether this session's estimated spend has hit the cap.
func (a *app) budgetExceeded() bool {
	return a.cfg.SessionBudgetUSD > 0 && a.sessionCost() >= a.cfg.SessionBudgetUSD
}

// budgetMsg is the refusal shown when a task would run over the session budget.
func (a *app) budgetMsg() string {
	return fmt.Sprintf("budget reached — spent ~$%.2f of the $%.2f session cap. Raise it with /budget <n>, or /budget off.",
		a.sessionCost(), a.cfg.SessionBudgetUSD)
}

// budgetCommand shows or sets the per-session spend cap (USD).
func (a *app) budgetCommand(arg string) []string {
	arg = strings.TrimSpace(arg)
	switch arg {
	case "":
		if a.cfg.SessionBudgetUSD <= 0 {
			return []string{fmt.Sprintf("no session budget · spent ~$%.2f this run", a.sessionCost()),
				"  /budget <usd> caps spend per run · /budget off disables"}
		}
		return []string{fmt.Sprintf("session budget: $%.2f · spent ~$%.2f this run", a.cfg.SessionBudgetUSD, a.sessionCost()),
			"  /budget <usd> to change · /budget off to disable"}
	case "off", "no", "0":
		a.cfg.SessionBudgetUSD = 0
	default:
		v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "$"), 64)
		if err != nil || v < 0 {
			return []string{"usage: /budget <usd>  (e.g. /budget 5), or /budget off"}
		}
		a.cfg.SessionBudgetUSD = v
	}
	if err := config.SaveSessionBudget(a.cfg.SessionBudgetUSD); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	if a.cfg.SessionBudgetUSD == 0 {
		return []string{"session budget → off"}
	}
	return []string{fmt.Sprintf("session budget → $%.2f (spent ~$%.2f so far this run)", a.cfg.SessionBudgetUSD, a.sessionCost())}
}

func today() string { return time.Now().Format("2006-01-02") }

// reflectTarget picks the model for the reflection pass: a configured profile
// (reflect_profile) on a capable model, else the current model. Returns the
// client, whether to use the lite (local) prompt, whether it's a separate client
// (so its token spend must be recorded), and its provider/model for the ledger.
func (a *app) reflectTarget() (client *llm.OpenAIClient, lite, separate bool, provider, model string) {
	prov, cfg, usingProfile := a.providerName(), a.activeLLM(), false
	if name := strings.TrimSpace(a.cfg.ReflectProfile); name != "" {
		if p, ok := a.cfg.Agents[name]; ok {
			pp := p.Provider
			if pp == "" {
				pp = "local"
			}
			c, usable := a.cfg.LLM, true
			if pp != "local" {
				if rp, rok := config.ResolveProvider(a.cfg, pp); rok && rp.APIKey != "" {
					c = rp
				} else {
					usable = false
				}
			}
			if usable {
				if p.Model != "" {
					c.Model = p.Model
				}
				prov, cfg, usingProfile = pp, c, true
			}
		}
	}
	model, lite = cfg.Model, prov == "local"
	// reflect-scope reasoning lets the learning pass differ from the main run.
	_, ovM := a.cfg.Reasoning["reflect:"+prov+"/"+model]
	_, ovP := a.cfg.Reasoning["reflect:"+prov]
	if !usingProfile && !ovM && !ovP { // same model, no reflect override → reuse main client
		return a.client, lite, false, prov, model
	}
	cfg.Extra = a.reasoningParams(prov, model, "reflect")
	return llm.NewOpenAIClient(cfg), lite, true, prov, model
}

// reflectAndStore runs the post-task reflection and persists new lessons,
// emitting a "lesson" event for each. Returns how many were new.
func (a *app) reflectAndStore(ctx context.Context, tr agent.Transcript) int {
	if a.cfg.ReflectDisabled { // /reflect off — skip the lesson-distillation pass
		return 0
	}
	client, lite, separate, provider, model := a.reflectTarget()
	// Signal the learning phase: the task is already DONE; anything slow/looping
	// from here is the reflection pass, not the task (so it's clear where a hang is).
	a.emit("reflecting", map[string]any{"model": model})
	refl := reflect.New(client)
	refl.Lite = lite // facts-only, terse — for a small local model that loops
	lessons, err := refl.Reflect(ctx, tr)
	if err != nil {
		slog.Warn("reflection failed", "err", err)
		return 0
	}
	if separate && a.usage != nil { // a dedicated reflect model's spend isn't in the main client
		if p, c := client.Usage(); p > 0 || c > 0 {
			a.usage.Add(today(), provider, model, p, c)
			_ = a.usage.Save()
		}
	}
	learned := 0
	for _, p := range lessons.Pitfalls {
		if a.kb.Add(p) {
			learned++
			a.emit("lesson", map[string]any{"domain": p.Domain, "proven_fix": p.ProvenFix})
		}
	}
	// Persist whenever any lesson was processed, not only on a brand-new one: a
	// duplicate still bumps Hits/LastSeen, and dropping that means a recurring
	// lesson can silently age out and be purged.
	if len(lessons.Pitfalls) > 0 {
		if err := a.kb.Save(); err != nil {
			slog.Warn("knowledge save failed", "err", err)
		}
	}
	if added := a.addFacts(lessons.Facts); len(added) > 0 {
		a.ag.SetSystem(a.systemPrompt()) // fold new facts into the prompt for the next task
		for _, f := range added {
			a.emit("fact", map[string]any{"text": f})
		}
	}
	return learned
}

// runOne is the plain (printing) path used in one-shot and piped modes.
func (a *app) runOne(ctx context.Context, goal string) {
	if a.budgetExceeded() {
		fmt.Println(a.budgetMsg())
		return
	}
	a.injectJobResults() // finished background jobs land before the model thinks
	a.beginCheckpoint(goal)
	defer a.endCheckpoint()
	a.ag.SetGoalLoop(a.goalTTLFor(goal)) // judge-loop only when pursuing an explicit goal
	tr, err := a.ag.Run(ctx, goal)
	if err != nil {
		slog.Error("run failed", "err", err)
		fmt.Fprintln(os.Stderr, "error:", err)
		return
	}
	a.recordRun(tr)
	a.finishGoal(goal, tr)
	if strings.TrimSpace(tr.Final) != "" {
		fmt.Println(tr.Final)
	} else {
		fmt.Println("(no final answer — step budget exhausted)")
	}
	if !tr.Stopped {
		if learned := a.reflectAndStore(ctx, tr); learned > 0 {
			fmt.Fprintf(os.Stderr, "(learned %d new lesson(s))\n", learned)
		}
	}
	a.recordUsage()
	a.saveSession()
	a.detectContextWindow() // the model is loaded now — confirm the real window
	if a.shouldAutoCompact() {
		if n, err := a.ag.Compact(ctx); err == nil && n > 0 {
			a.saveSession()
			fmt.Fprintf(os.Stderr, "(auto-compacted %d messages to free context)\n", n)
		}
	}
}

// runTaskStreaming is the TUI path: no printing — progress reaches the screen via
// the UI tracer. Errors surface as an "error" event.
func (a *app) runTaskStreaming(ctx context.Context, goal string) {
	if a.budgetExceeded() {
		a.emit("error", map[string]any{"text": a.budgetMsg()})
		return
	}
	a.injectJobResults() // finished background jobs land before the model thinks
	a.beginCheckpoint(goal)
	defer a.endCheckpoint()
	a.ag.SetGoalLoop(a.goalTTLFor(goal)) // judge-loop only when pursuing an explicit goal
	tr, err := a.ag.Run(ctx, goal)
	if err != nil {
		a.emit("error", map[string]any{"text": err.Error()})
		return
	}
	a.recordRun(tr)
	a.finishGoal(goal, tr)
	if !tr.Stopped { // reflect only on a clean finish, not on any premature stop
		a.reflectAndStore(ctx, tr)
	}
	a.recordUsage()
	a.saveSession() // persist the partial work so a follow-up can continue
}

func (a *app) repl(ctx context.Context) {
	fmt.Printf("ipsupport-code %s — type a task, or /help for commands.\n", version)
	if n := a.startupNotice(ctx); n != "" {
		fmt.Fprintln(os.Stderr, n)
	}
	for {
		fmt.Print("\n> ")
		line, err := a.reader.ReadString('\n')
		if err != nil {
			fmt.Println()
			return
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		a.addPromptHist(line) // persist for /history + the TUI's ↑ recall
		switch {
		case line == "!":
			a.runShell(ctx)
		case strings.HasPrefix(line, "!"):
			a.runShellLine(ctx, strings.TrimPrefix(line, "!"))
		case strings.HasPrefix(line, "/"):
			if a.command(ctx, line) {
				return
			}
		default:
			a.runOne(ctx, line)
		}
	}
}

// command handles a /slash line in plain mode. Returns true when the REPL should
// exit.
// printLines prints a command handler's output lines (the REPL dispatch shim).
func printLines(lines []string) {
	for _, l := range lines {
		fmt.Println(l)
	}
}

func (a *app) command(ctx context.Context, line string) (quit bool) {
	cmd, rest := splitCommand(line)
	switch cmd {
	case "/help", "/?":
		fmt.Print(helpText())
	case "/status":
		fmt.Print(a.statusText())
	case "/usage":
		if lines, handled := a.usageManage(rest); handled {
			for _, l := range lines {
				fmt.Println(l)
			}
		} else {
			fmt.Print(a.usageText())
		}
	case "/sessions":
		lines, _ := a.sessionsCommand(rest)
		for _, l := range lines {
			fmt.Println(l)
		}
	case "/agents", "/agent":
		printLines(a.agentsCommand(ctx, rest))
	case "/login", "/init":
		maybeInit(a.reader, true)
		if err := a.reconfigure(); err != nil {
			fmt.Println("reconfigure failed:", err)
		} else {
			fmt.Println("config reloaded.")
		}
	case "/new": // branch to a NEW session; the current one stays in /sessions
		name, persist := strings.TrimSpace(rest), true
		if name == "" {
			name, persist = a.autoSessionName(), false
		}
		if err := a.newNamedSession(name, persist); err != nil {
			fmt.Println("could not start session:", err)
		} else {
			fmt.Printf("started a new session %q — the previous one is in /sessions\n", a.cfg.Name)
		}
	case "/reset", "/clear": // wipe THIS thread
		a.ag.Reset()
		a.saveSession()
		fmt.Println("session cleared.")
	case "/compact":
		n, err := a.ag.Compact(ctx)
		if err != nil {
			fmt.Println("compact failed:", err)
		} else {
			a.saveSession()
			fmt.Printf("compacted %d messages → summary.\n", n)
		}
	case "/plan":
		fmt.Println(a.setMode(true))
	case "/auto":
		fmt.Println(a.setMode(false))
	case "/update":
		if a.cfg.Offline {
			fmt.Println("offline mode is on — /update needs the internet. Run /offline off first.")
		} else {
			runUpdate(strings.Fields(rest))
		}
	case "/offline":
		printLines(a.offlineCommand(rest))
	case "/cd":
		printLines(a.cdCommand(rest))
	case "/knowledge", "/kb":
		printLines(a.knowledgeCommand(rest))
	case "/mcp":
		fmt.Println(a.mcpList(ctx))
	case "/rewind":
		printLines(a.rewindCommand(rest))
	case "/reflect":
		printLines(a.reflectCommand(rest))
	case "/goal":
		if text, ok := a.launchGoalText(rest); ok {
			a.setGoal(text)
			a.runOne(ctx, text)
		} else {
			printLines(a.goalCommand(rest))
		}
	case "/history":
		printLines(a.historyCommand(rest))
	case "/budget":
		printLines(a.budgetCommand(rest))
	case "/jobs":
		printLines(a.jobsCommand(rest))
	case "/btw": // in plain mode a task runs synchronously, so this always steers the NEXT run
		printLines(a.btwCommand(rest, false))
	case "/diff":
		printLines(a.diffCommand())
	case "/reasoning":
		printLines(a.reasoningCommand(rest))
	case "/ai":
		printLines(a.aiCommand(rest))
	case "/model":
		printLines(a.modelCommand(ctx, rest))
	case "/config":
		printLines(a.configOverview())
	case "/shell", "/sh":
		a.runShell(ctx)
	case "/skills":
		printLines(a.skillsCommand(ctx, rest))
	case "/permissions", "/perms":
		printLines(a.permissionsCommand(rest))
	case "/color":
		fmt.Println("/color changes the TUI frame color — interactive mode only.")
	case "/rename":
		if name := strings.TrimSpace(rest); name == "" {
			fmt.Println("usage: /rename <new name>")
		} else {
			a.cfg.Name = name
			if err := config.SaveGlobal(name, a.cfg.LLM); err != nil {
				fmt.Println("rename failed:", err)
			} else {
				fmt.Println("renamed →", name)
			}
		}
	case "/loop":
		interval, max, goal, ok := parseLoop(rest)
		if !ok {
			fmt.Println(loopUsage)
			break
		}
		for i := 0; max == 0 || i < max; i++ {
			if i > 0 {
				select {
				case <-ctx.Done():
				case <-time.After(interval):
				}
			}
			if ctx.Err() != nil {
				break
			}
			if max > 0 {
				fmt.Printf("— loop %d/%d · every %s —\n", i+1, max, interval)
			} else {
				fmt.Printf("— loop %d · every %s —\n", i+1, interval)
			}
			a.runOne(ctx, goal)
		}
	case "/exit", "/quit":
		return true
	default:
		fmt.Printf("unknown command %q — try /help\n", cmd)
	}
	return false
}

// --- providers: switch the AI between LM Studio and external OpenAI-compatible
// providers (OpenAI, Grok/xAI, Groq, OpenRouter). The agent and tools are
// model-agnostic, so this only swaps the connection and re-wires. ---

func (a *app) providerName() string {
	if a.cfg.Provider == "" {
		return "local"
	}
	return a.cfg.Provider
}

func (a *app) aiCommand(rest string) []string {
	if strings.TrimSpace(rest) == "" {
		return a.providerList()
	}
	sub, arg := splitCommand(rest)
	switch sub {
	case "key":
		name, tok := splitCommand(arg)
		return a.setProviderKey(name, tok)
	case "add":
		return a.addProvider(arg)
	}
	return a.setProvider(sub)
}

// addProvider registers a custom OpenAI-compatible provider (any base URL), so
// you're not limited to the built-in templates: /ai add <name> <base_url> [model],
// then /ai key <name> <token>, then /ai <name>.
func (a *app) addProvider(arg string) []string {
	f := strings.Fields(arg)
	if len(f) < 2 {
		return []string{"usage: /ai add <name> <base_url> [model]", "  e.g. /ai add mylab http://localhost:8080/v1 llama-3.1-8b"}
	}
	name, url := f[0], f[1]
	if name == "local" {
		return []string{`"local" is reserved — it's your LM Studio endpoint (/config to change it)`}
	}
	if _, ok := config.ProviderTemplates[name]; ok {
		return []string{fmt.Sprintf("%q is a built-in provider — just /ai key %s <token>", name, name)}
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return []string{"base_url must start with http:// or https://"}
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[name]
	p.BaseURL = strings.TrimRight(url, "/")
	if len(f) >= 3 {
		p.Model = f[2]
	}
	a.cfg.Providers[name] = p
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		return []string{"error: " + err.Error()}
	}
	return []string{fmt.Sprintf("added %q → %s — next: /ai key %s <token>, then /ai %s", name, p.BaseURL, name, name)}
}

func (a *app) providerList() []string {
	out := []string{"providers (active ●):"}
	row := func(name, base, note string) {
		mark := "  "
		if name == a.providerName() {
			mark = "● "
		}
		out = append(out, fmt.Sprintf("  %s%-11s %s  %s", mark, name, base, note))
	}
	row("local", a.cfg.LLM.BaseURL, "(LM Studio)")
	seen := map[string]bool{}
	names := append([]string{}, config.KnownProviders()...)
	for _, n := range names {
		seen[n] = true
	}
	for n := range a.cfg.Providers { // custom providers (added via /ai add)
		if !seen[n] {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	for _, n := range names {
		l, _ := config.ResolveProvider(a.cfg, n)
		note := "no key"
		if l.APIKey != "" {
			note = "key set"
		}
		if _, tmpl := config.ProviderTemplates[n]; !tmpl {
			note += " · custom"
		}
		row(n, l.BaseURL, note)
	}
	return append(out, "  /ai <name> switch · /ai add <name> <url> · /ai key <name> <token>")
}

func (a *app) setProvider(name string) []string {
	if name != "local" {
		l, ok := config.ResolveProvider(a.cfg, name)
		if !ok {
			return []string{fmt.Sprintf("unknown provider %q — try: local, %s", name, strings.Join(config.KnownProviders(), ", "))}
		}
		if config.IsCustomProvider(a.cfg, name) {
			if l.BaseURL == "" {
				return []string{fmt.Sprintf("%s has no base_url — set providers.%s.base_url in config", name, name)}
			}
		} else if l.APIKey == "" { // built-in cloud template
			return []string{fmt.Sprintf("%s needs an API key — run: /ai key %s <token>  (or set the env var)", name, name)}
		}
	}
	a.cfg.Provider = name
	a.windowDetected = false
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		return []string{"error: " + err.Error()}
	}
	if err := a.wire(); err != nil {
		return []string{"error: " + err.Error()}
	}
	a.maybeDetectWindowSync() // REPL only — the TUI re-detects off-thread
	act := a.activeLLM()
	return []string{fmt.Sprintf("→ %s · %s · model %s", a.providerName(), act.BaseURL, act.Model)}
}

func (a *app) setProviderKey(name, token string) []string {
	_, isTmpl := config.ProviderTemplates[name]
	_, isCustom := a.cfg.Providers[name]
	if !isTmpl && !isCustom {
		return []string{fmt.Sprintf("unknown provider %q — add it first: /ai add %s <base_url>  (built-ins: %s)", name, name, strings.Join(config.KnownProviders(), ", "))}
	}
	if strings.TrimSpace(token) == "" {
		return []string{"usage: /ai key " + name + " <token>"}
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[name]
	p.APIKey = strings.TrimSpace(token)
	a.cfg.Providers[name] = p
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		return []string{"error: " + err.Error()}
	}
	if a.cfg.Provider == name {
		_ = a.wire()
	}
	return []string{"key saved for " + name}
}

// modelCommand lists the active provider's models (no arg), or resolves an arg:
// an exact id or unique substring switches the model, an ambiguous substring
// lists the matches (handy for OpenRouter's hundreds of models).
func (a *app) modelCommand(ctx context.Context, rest string) []string {
	cctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	arg := strings.TrimSpace(rest)
	if arg == "" {
		return modelLines(cctx, a.activeLLM(), a.providerName())
	}
	setTo, lines := resolveModelArg(listModelIDs(cctx, a.activeLLM()), arg)
	if setTo != "" {
		return a.setModel(setTo)
	}
	return lines
}

// listModelIDs returns just the model ids a provider advertises (LM Studio's
// native list or the OpenAI /models list), nil on failure.
func listModelIDs(ctx context.Context, act config.LLM) []string {
	if act.LMStudio() {
		ms, err := llm.ListLMStudioModels(ctx, act.BaseURL, http.DefaultClient)
		if err != nil {
			return nil
		}
		ids := make([]string, len(ms))
		for i, m := range ms {
			ids[i] = m.ID
		}
		return ids
	}
	ids, _ := llm.ListModels(ctx, act.BaseURL, act.APIKey, http.DefaultClient)
	return ids
}

// resolveModelArg decides what `/model <arg>` does against the advertised ids:
// an exact id or a unique substring match returns setTo (switch to it); an
// ambiguous substring returns the matching list; no list or no match falls back
// to setTo=arg (trust the user — offline, or a model not yet listed).
func resolveModelArg(ids []string, arg string) (setTo string, lines []string) {
	for _, id := range ids {
		if id == arg {
			return arg, nil
		}
	}
	var matches []string
	lower := strings.ToLower(arg)
	for _, id := range ids {
		if strings.Contains(strings.ToLower(id), lower) {
			matches = append(matches, id)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return arg, nil
	default:
		lines = []string{fmt.Sprintf("%d models match %q — /model <id> to pick:", len(matches), arg)}
		for _, id := range matches {
			lines = append(lines, "  "+id)
		}
		return "", lines
	}
}

// modelLines lists a provider's models — rich (state · context · quant) for LM
// Studio via its native API, plain ids otherwise. Pure (no app state), so the
// REPL and the TUI's async path share it.
// modelListCap bounds the no-arg model list so a provider with hundreds of
// models (OpenRouter) doesn't flood the screen; /model <substr> filters instead.
const modelListCap = 50

func modelLines(ctx context.Context, act config.LLM, provider string) []string {
	head := fmt.Sprintf("models on %s (current %s) — /model <id> to pick · /model <text> to filter:", provider, act.Model)
	if act.LMStudio() {
		ms, err := llm.ListLMStudioModels(ctx, act.BaseURL, http.DefaultClient)
		if err != nil {
			return []string{"couldn't list models: " + err.Error()}
		}
		out := []string{head}
		for _, m := range ms {
			tag := "·"
			if m.State == "loaded" {
				tag = "●"
			}
			win := ""
			switch {
			case m.LoadedContextLength > 0:
				win = "ctx " + humanK(m.LoadedContextLength)
			case m.MaxContextLength > 0:
				win = "max " + humanK(m.MaxContextLength)
			}
			out = append(out, fmt.Sprintf("  %s %-36s %-9s %s", tag, m.ID, win, m.Quantization))
		}
		if len(ms) == 0 {
			out = append(out, "  (none reported)")
		}
		return out
	}
	ids, err := llm.ListModels(ctx, act.BaseURL, act.APIKey, http.DefaultClient)
	if err != nil {
		return []string{"couldn't list models: " + err.Error()}
	}
	out := []string{head}
	for i, id := range ids {
		if i >= modelListCap {
			out = append(out, fmt.Sprintf("  …and %d more — /model <text> to filter", len(ids)-modelListCap))
			break
		}
		out = append(out, "  "+id)
	}
	if len(ids) == 0 {
		out = append(out, "  (none reported)")
	}
	return out
}

func (a *app) setModel(name string) []string {
	if a.isLocal() {
		a.cfg.LLM.Model = name
		_ = config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	} else {
		if a.cfg.Providers == nil {
			a.cfg.Providers = map[string]config.LLM{}
		}
		p := a.cfg.Providers[a.cfg.Provider]
		p.Model = name
		a.cfg.Providers[a.cfg.Provider] = p
		_ = config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
	}
	a.windowDetected = false
	if err := a.wire(); err != nil {
		return []string{"error: " + err.Error()}
	}
	a.maybeDetectWindowSync() // REPL only — the TUI re-detects off-thread
	return []string{"model → " + name}
}

// configOverview is the control panel: current settings + the command to change
// each, so the config file never needs hand-editing.
func (a *app) configOverview() []string {
	act := a.activeLLM()
	key := "—"
	if act.APIKey != "" {
		key = "set"
	}
	return []string{
		"config — change with the command on the right:",
		fmt.Sprintf("  provider     %-22s /ai <name> · /ai key <name> <tok>", a.providerName()),
		fmt.Sprintf("  server       %s", act.BaseURL),
		fmt.Sprintf("  model        %-22s /model", act.Model),
		fmt.Sprintf("  api key      %s", key),
		fmt.Sprintf("  context      %-22s (auto-compact ~75%%)", ctxLabel(act.ContextWindow)),
		fmt.Sprintf("  channel      %-22s update stable|nightly", channelOf(a.cfg)),
		fmt.Sprintf("  reasoning    %-22s /reasoning off|low|…", a.reasoningLevel(a.providerName(), act.Model)),
		fmt.Sprintf("  budget       %-22s /budget <usd>|off", a.budgetStatusLine()),
		fmt.Sprintf("  offline      %-22s /offline on|off", onOff(a.cfg.Offline)),
		fmt.Sprintf("  permissions  files=%-5s run=%-5s    /permissions", a.cfg.File.Default, a.cfg.Run.Default),
		fmt.Sprintf("  sub-agents   %-22s /agents", fmt.Sprintf("%d profile(s)", len(a.cfg.Agents))),
		fmt.Sprintf("  name         %-22s /rename <name>", a.cfg.Name),
		fmt.Sprintf("  prompt       %-22s -dump-prompt > .agent/system.md", promptOrDefault(a.promptSrc)),
		"  file: ~/.config/ipsupport-code/config.json (chmod 600)",
	}
}

func ctxLabel(w int) string {
	if w <= 0 {
		return "(provider default)"
	}
	return humanK(w)
}

// --- skills: surface-agnostic handlers, returning plain lines the REPL prints
// and the TUI styles. Install touches the network, so callers run it off the UI
// thread; the rest are local filesystem ops. ---

func (a *app) skillsCommand(ctx context.Context, rest string) []string {
	if a.skills == nil {
		return []string{"skills unavailable"}
	}
	if strings.TrimSpace(rest) == "" {
		return a.skillsStatus()
	}
	sub, arg := splitCommand(rest)
	switch sub {
	case "on", "enable":
		return a.skillsToggle(arg, true)
	case "off", "disable":
		return a.skillsToggle(arg, false)
	case "install", "add":
		return a.skillsInstall(ctx, arg)
	case "remove", "rm":
		return a.skillsRemove(arg)
	case "list":
		return a.skillsStatus()
	default:
		return []string{"usage: /skills [on|off|remove <name>] [install <url|git>]"}
	}
}

func (a *app) skillsStatus() []string {
	list := a.skills.List()
	if len(list) == 0 {
		return []string{"no skills — add one with /skills install <url|git>"}
	}
	out := []string{"skills:"}
	for _, sk := range list {
		mark := "off"
		if sk.Enabled {
			mark = "on "
		}
		out = append(out, fmt.Sprintf("  [%s] %-20s %s", mark, sk.Name, sk.Description))
	}
	return append(out, "  /skills on|off <name> · install <url|git> · remove <name>")
}

func (a *app) skillsToggle(name string, on bool) []string {
	if err := a.skills.SetEnabled(name, on); err != nil {
		return []string{"error: " + err.Error()}
	}
	_ = a.wire() // (de)register the skill tool + refresh the prompt index; session preserved
	if on {
		return []string{"enabled " + name}
	}
	return []string{"disabled " + name}
}

func (a *app) skillsRemove(name string) []string {
	if err := a.skills.Remove(name); err != nil {
		return []string{"error: " + err.Error()}
	}
	_ = a.wire()
	return []string{"removed " + name}
}

func (a *app) skillsInstall(ctx context.Context, src string) []string {
	if strings.TrimSpace(src) == "" {
		return []string{"usage: /skills install <url|git>"}
	}
	names, err := a.skills.Install(ctx, src)
	if err != nil {
		return []string{"install failed: " + err.Error()}
	}
	_ = a.wire() // installed skills are enabled, so register the tool + index
	return []string{"installed & enabled: " + strings.Join(names, ", ")}
}

// --- permissions: relax the policy so non-destructive actions stop prompting.
// The deny floor (secrets, .git, .env, rm -rf, …) is never affected. ---

func (a *app) permissionsCommand(rest string) []string {
	if strings.TrimSpace(rest) == "" {
		return a.permissionsStatus()
	}
	sub, arg := splitCommand(rest)
	switch sub {
	case "files", "file":
		return a.permissionsSet(&a.cfg.File.Default, arg, "file writes")
	case "run", "shell":
		return a.permissionsSet(&a.cfg.Run.Default, arg, "shell commands")
	case "agents", "agent", "spawn":
		return a.permissionsSetSpawn(arg)
	case "reset":
		a.resetSessionAllow()
		return []string{"session allowances revoked — approvals ask again"}
	default:
		return []string{"usage: /permissions [files on|off] [run on|off] [agents on|off] [reset]"}
	}
}

// permissionsSetSpawn relaxes (on) or restores (off) the spawn-approval prompt
// for the agent tool, and persists it globally (profiles live there too).
func (a *app) permissionsSetSpawn(arg string) []string {
	switch strings.TrimSpace(arg) {
	case "on", "allow", "yes":
		a.cfg.Spawn.Default = "allow"
	case "off", "ask", "no", "":
		a.cfg.Spawn.Default = "ask"
	default:
		return []string{"usage: on (spawn without asking) | off (ask each spawn)"}
	}
	if err := config.SaveSpawn(a.cfg.Spawn); err != nil {
		return []string{"warning: not persisted: " + err.Error()}
	}
	return []string{fmt.Sprintf("sub-agent spawns → %s — saved", a.cfg.Spawn.Default)}
}

func (a *app) permissionsStatus() []string {
	exec := "off"
	if a.cfg.Spawn.Exec {
		exec = "on"
	}
	out := []string{
		"permissions:",
		fmt.Sprintf("  files   %s   (jail %q)", a.cfg.File.Default, a.cfg.File.Jail),
		fmt.Sprintf("  run     %s", a.cfg.Run.Default),
		fmt.Sprintf("  agents  %s   (sub-agent shell exec: %s — set in /config)", a.cfg.Spawn.Default, exec),
		"  deny floor (always on): secrets, .git, .env, rm -rf, sudo, …",
	}
	if allowed := a.sessionAllows(); len(allowed) > 0 {
		out = append(out, "  session-allowed ('a' on a prompt): "+strings.Join(allowed, ", ")+"  — /permissions reset revokes")
	}
	return append(out,
		"  /permissions files on  → stop asking before file writes in the workspace",
		"  /permissions agents on → spawn sub-agents without asking each time",
	)
}

// sessionAllows lists the categories currently allowed for this session (sorted).
func (a *app) sessionAllows() []string {
	a.sessionMu.Lock()
	defer a.sessionMu.Unlock()
	out := make([]string, 0, len(a.sessionAllow))
	for cat := range a.sessionAllow {
		out = append(out, categoryLabel(cat))
	}
	sort.Strings(out)
	return out
}

func (a *app) permissionsSet(field *string, arg, label string) []string {
	switch strings.TrimSpace(arg) {
	case "on", "allow", "yes":
		*field = "allow"
	case "off", "ask", "no", "":
		*field = "ask"
	default:
		return []string{"usage: on (auto-allow) | off (ask)"}
	}
	if err := a.wire(); err != nil {
		return []string{"error: " + err.Error()}
	}
	msg := fmt.Sprintf("%s → %s (deny floor still enforced)", label, *field)
	if err := config.SaveWorkspacePolicy(a.workspace, a.cfg.Run, a.cfg.File); err != nil {
		return []string{msg, "warning: not persisted: " + err.Error()}
	}
	return []string{msg + " — saved to .agent/config.json"}
}

// shellPath is the user's interactive shell, falling back to /bin/sh.
func shellPath() string {
	if s := os.Getenv("SHELL"); s != "" {
		return s
	}
	return "/bin/sh"
}

// runShell drops to an interactive shell in the workspace so the user can do
// things by hand; control returns when they exit it.
func (a *app) runShell(ctx context.Context) {
	sh := shellPath()
	fmt.Printf("— %s (exit to return to ipsupport-code) —\n", sh)
	cmd := exec.CommandContext(ctx, sh)
	cmd.Dir = a.workspace
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	_ = cmd.Run() // a non-zero shell exit is normal; nothing to report
	fmt.Println("— back in ipsupport-code —")
}

// runShellLine runs a single shell command (the !cmd shortcut) in the workspace.
func (a *app) runShellLine(ctx context.Context, cmdline string) {
	if strings.TrimSpace(cmdline) == "" {
		return
	}
	cmd := exec.CommandContext(ctx, shellPath(), "-c", cmdline)
	cmd.Dir = a.workspace
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	_ = cmd.Run()
}

func splitCommand(line string) (cmd, rest string) {
	f := strings.Fields(line)
	if len(f) == 0 {
		return "", ""
	}
	cmd = f[0]
	return cmd, strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), cmd))
}

const loopUsage = "usage: /loop <interval> [xN] <task>   e.g. /loop 5m check the build   ·   /loop 30s x10 tail the log   (esc stops it)"

// parseLoop parses "/loop <interval> [xN] <task>": an interval (a Go duration
// like 30s/5m/1h), an optional max-iteration cap written xN (e.g. x10; 0 = run
// until stopped), and the task. ok=false (caller prints loopUsage) on a missing/
// bad interval or empty task.
func parseLoop(rest string) (interval time.Duration, max int, goal string, ok bool) {
	parts := strings.Fields(rest)
	if len(parts) < 2 {
		return 0, 0, "", false
	}
	d, err := time.ParseDuration(parts[0])
	if err != nil || d <= 0 {
		return 0, 0, "", false
	}
	parts = parts[1:]
	if n, isCount := parseLoopCount(parts[0]); isCount {
		max = n
		parts = parts[1:]
	}
	goal = strings.TrimSpace(strings.Join(parts, " "))
	if goal == "" {
		return 0, 0, "", false
	}
	return d, max, goal, true
}

// parseLoopCount reads an "xN" max-iteration token (x10, X10). false if not one.
func parseLoopCount(s string) (int, bool) {
	if len(s) < 2 || (s[0] != 'x' && s[0] != 'X') {
		return 0, false
	}
	n, err := strconv.Atoi(s[1:])
	if err != nil || n < 1 {
		return 0, false
	}
	return n, true
}

// loopLabel renders the interval (and count cap, if any) for the echo line.
func loopLabel(interval time.Duration, max int) string {
	if max > 0 {
		return fmt.Sprintf("%s x%d", interval, max)
	}
	return interval.String()
}

func helpText() string {
	return `commands:
  /status          show config, knowledge base, and trace paths
  /usage           session counters + token usage
  /budget [usd]    cap estimated spend per run (refuses new tasks once hit; off to disable)
  /jobs            background sub-agent jobs: list · result <id> · kill <id>
  /btw <note>      steer a RUNNING task without stopping it (esc cancels; /btw nudges)
  /diff            show uncommitted changes in the workspace (what the agent changed)
  /login           (re)configure the server URL / model / key, then reload
  /new [name]      start a NEW session (the old one stays in /sessions)
  /clear           wipe this session's context (same session)
  /compact         summarize the session so far to free up context
  /plan, /auto     plan mode (propose only) vs auto mode (execute)
  /ai [name]       switch AI provider; /ai key <name> <tok>; /ai add <name> <url> [model] (custom)
  /model [name]    list the provider's models, or pick one
  /config          control panel: all settings + how to change them
  /update [chan]   self-update from GitHub (chan = stable|nightly, saved)
  /offline [on|off] work without internet — disables web + update checks
  /cd [dir]        set the working directory (relative paths + sub-agents use it)
  /knowledge       learned-lessons store: report · clear · purge <days> · retain <days>
  /mcp             list configured MCP servers and their tools (mcp_servers in config.json)
  /rewind [n]      roll back to a previous step (restores files + trims the chat)
  /reflect [on|off|<profile>] post-task learning; run it on a stronger model
  /goal <text>     set & pursue a multi-turn goal (go · clear · ttl <n> · off)
  /reasoning [off|low|…] trim a thinking model's reasoning (minimal|low|medium|high)
  /shell, /sh      drop to a shell in the workspace (exit to return)
  /skills          list/toggle/install on-demand instruction packs
  /agents          sub-agent profiles: add (LLM) · add-tool (external CLI: codex/claude/…) · rm · exec
  /permissions     relax approval for file / shell / sub-agent-spawn actions
  /color [name]    change the TUI frame color (cycles if no name)
  /rename <name>   rename the agent (saved in settings)
  /sessions        list / switch / delete saved sessions (per agent name)
  /history [text]  recent prompts across sessions (↑/↓ recalls in the TUI); filter by text
  /loop <ival> <task>  re-run a task on an interval (e.g. /loop 5m <task>, /loop 30s x10 <task>; esc stops)
  /help            this list
  /exit, /quit     leave

keys: Enter send · alt+enter newline · Tab complete /commands + @file paths ·
  ctrl+r history search · shift+tab plan/auto · ctrl+u clear input · esc cancel

sub-agents: define profiles in /config → Sub-agents (provider → model → name),
  then ask e.g. "review internal/tool across grok and claude, then merge". The
  assistant fans out in parallel; /agents add|rm|exec, /permissions agents on.

Anything not starting with '/' is run as a task.
`
}

func (a *app) statusText() string {
	instr := a.instrSrc
	if instr == "" {
		instr = "(none)"
	}
	act := a.activeLLM()
	return fmt.Sprintf(`status:
  version      %s (%s channel)
  provider     %s
  server       %s
  model        %s
  max_steps    %d
  reasoning    %s
  offline      %s
  workspace    %s
  jail         %q
  defaults     run=%s  file=%s
  prompt       %s
  instructions %s
  session      %d messages
  goal         %s
  budget       %s
  jobs         %d running (/jobs)
  knowledge    %s (%d lessons)
  facts        %d learned
  trace        %s
`,
		version, channelOf(a.cfg), a.providerName(),
		act.BaseURL, act.Model, act.MaxSteps,
		a.reasoningLevel(a.providerName(), act.Model), onOff(a.cfg.Offline),
		a.cfg.Workspace, a.cfg.File.Jail, a.cfg.Run.Default, a.cfg.File.Default,
		promptOrDefault(a.promptSrc), instr, a.ag.SessionLen(),
		a.goalStatusLine(), a.budgetStatusLine(), a.jobsPending(),
		a.cfg.KBPath, len(a.kb.All()), len(a.facts), a.cfg.TracePath)
}

// budgetStatusLine is the one-line spend summary shown in /status.
func (a *app) budgetStatusLine() string {
	if a.cfg.SessionBudgetUSD <= 0 {
		return fmt.Sprintf("none · spent ~$%.2f this run", a.sessionCost())
	}
	return fmt.Sprintf("$%.2f cap · spent ~$%.2f this run", a.cfg.SessionBudgetUSD, a.sessionCost())
}

// goalStatusLine is the one-line goal summary shown in /status.
func (a *app) goalStatusLine() string {
	ttl := fmt.Sprintf("TTL %d", a.cfg.GoalMaxReturns)
	if a.cfg.GoalMaxReturns == 0 {
		ttl = "loop off"
	}
	if a.goal.Text == "" {
		return "(none) · " + ttl
	}
	return fmt.Sprintf("%s [%s] · %s", oneLine(a.goal.Text, 50), a.goal.Status, ttl)
}

// promptOrDefault labels the system-prompt source for /status.
func promptOrDefault(src string) string {
	if src == "" {
		return "built-in"
	}
	return src
}

// channelOf returns the configured update channel, defaulting to stable.
func channelOf(cfg config.Config) string {
	if cfg.Channel == "" {
		return selfupdate.Stable
	}
	return cfg.Channel
}

func (a *app) usageText() string {
	p, c := a.client.Usage()
	var b strings.Builder
	fmt.Fprintf(&b, `usage (this session):
  tasks       %d
  steps       %d
  tool calls  %d
  tokens      %d prompt + %d completion = %d
  lessons     %d in knowledge base
`, a.tasks, a.steps, a.toolCalls, p, c, p+c, len(a.kb.All()))
	if roll := a.usageRollups(); len(roll) > 0 {
		b.WriteString("\ntokens (cumulative, saved · $ estimated):\n")
		for _, r := range roll {
			fmt.Fprintf(&b, "  %-12s %s\n", r[0], r[1])
		}
	}
	days, models := a.usageLedger()
	if len(days) > 0 {
		b.WriteString("\ntokens by day:\n")
		for _, r := range days {
			fmt.Fprintf(&b, "  %-12s %s\n", r[0], r[1])
		}
	}
	if len(models) > 0 {
		b.WriteString("\ntokens by provider/model:\n")
		for _, r := range models {
			fmt.Fprintf(&b, "  %-28s %s\n", r[0], r[1])
		}
	}
	b.WriteString("\nmanage: /usage clear · /usage purge <days> · /usage retain <days>\n")
	return b.String()
}

// usageRollups summarizes cumulative token spend (and estimated $ cost) over
// common windows from the saved ledger (today / 7d / 30d / all time).
func (a *app) usageRollups() [][2]string {
	if a.usage == nil {
		return nil
	}
	now := time.Now()
	ov := a.priceOverrides()
	row := func(label, cutoff string) [2]string {
		v := humanK(a.usage.TotalSince(cutoff).Tokens()) + " tok"
		if c := fmtCost(a.usage.CostSince(cutoff, ov)); c != "" {
			v += "  " + c
		}
		return [2]string{label, v}
	}
	return [][2]string{
		row("today", now.Format("2006-01-02")),
		row("last 7 days", now.AddDate(0, 0, -6).Format("2006-01-02")),
		row("last 30 days", now.AddDate(0, 0, -29).Format("2006-01-02")),
		row("all time", ""),
	}
}

// priceOverrides converts the config price table to the usage package's form.
func (a *app) priceOverrides() map[string]usage.Price {
	if len(a.cfg.Prices) == 0 {
		return nil
	}
	m := make(map[string]usage.Price, len(a.cfg.Prices))
	for k, v := range a.cfg.Prices {
		m[k] = usage.Price{In: v[0], Out: v[1]}
	}
	return m
}

// fmtCost renders an estimated dollar cost ("" for ~0 so free models show no $).
func fmtCost(c float64) string {
	switch {
	case c <= 0:
		return ""
	case c < 0.01:
		return "<$0.01"
	default:
		return fmt.Sprintf("~$%.2f", c)
	}
}

// cutoffDays is the ISO date N days ago — entries older than it are "older than N
// days" for purge/retention.
func cutoffDays(days int) string { return time.Now().AddDate(0, 0, -days).Format("2006-01-02") }

// applyUsageRetention drops ledger entries older than the configured window, on
// startup. No-op when retention is off (0).
func (a *app) applyUsageRetention() {
	if a.usage == nil || a.cfg.UsageRetentionDays <= 0 {
		return
	}
	if n := a.usage.Purge(cutoffDays(a.cfg.UsageRetentionDays)); n > 0 {
		_ = a.usage.Save()
	}
}

// applyKnowledgeRetention drops learned lessons older than the configured window
// on startup. No-op when retention is off (0).
func (a *app) applyKnowledgeRetention() {
	if a.kb == nil || a.cfg.KnowledgeRetentionDays <= 0 {
		return
	}
	if n := a.kb.Purge(a.cfg.KnowledgeRetentionDays); n > 0 {
		_ = a.kb.Save()
	}
}

// knowledgeCommand handles /knowledge: a report, or clear / purge <days> /
// retain <days> to manage the learned-lessons store.
func (a *app) knowledgeCommand(rest string) []string {
	sub, arg := splitCommand(rest)
	switch sub {
	case "":
		return a.knowledgeReport()
	case "clear":
		n := a.kb.Clear()
		if err := a.kb.Save(); err != nil {
			return []string{"error: " + err.Error()}
		}
		return []string{fmt.Sprintf("cleared %d learned lessons", n)}
	case "purge":
		days, err := strconv.Atoi(strings.TrimSpace(arg))
		if err != nil || days < 0 {
			return []string{"usage: /knowledge purge <days>  (drop lessons last seen over N days ago)"}
		}
		n := a.kb.Purge(days)
		if err := a.kb.Save(); err != nil {
			return []string{"error: " + err.Error()}
		}
		return []string{fmt.Sprintf("purged %d lessons older than %d days", n, days)}
	case "retain":
		days, err := strconv.Atoi(strings.TrimSpace(arg))
		if err != nil || days < 0 {
			return []string{"usage: /knowledge retain <days>  (0 = keep forever)"}
		}
		a.cfg.KnowledgeRetentionDays = days
		if err := config.SaveKnowledgeRetention(days); err != nil {
			return []string{"warning: not persisted: " + err.Error()}
		}
		n := a.kb.Purge(days)
		_ = a.kb.Save()
		msg := fmt.Sprintf("retention → %d days (auto-purge on startup)", days)
		if days == 0 {
			msg = "retention → off (lessons kept forever)"
		}
		if n > 0 {
			msg += fmt.Sprintf("; dropped %d now", n)
		}
		return []string{msg}
	default:
		return []string{"usage: /knowledge [clear] [purge <days>] [retain <days>]"}
	}
}

// knowledgeReport summarizes the lesson store: total, per-domain counts, retention.
func (a *app) knowledgeReport() []string {
	all := a.kb.All()
	if len(all) == 0 {
		return []string{"no learned lessons yet — they accrue from task reflections"}
	}
	byDomain := map[string]int{}
	for _, p := range all {
		byDomain[p.Domain]++
	}
	doms := make([]string, 0, len(byDomain))
	for d := range byDomain {
		doms = append(doms, d)
	}
	sort.Slice(doms, func(i, j int) bool { return byDomain[doms[i]] > byDomain[doms[j]] })
	out := []string{fmt.Sprintf("learned lessons: %d", len(all))}
	for _, d := range doms {
		out = append(out, fmt.Sprintf("  %-10s %d", d, byDomain[d]))
	}
	ret := "off (kept forever)"
	if a.cfg.KnowledgeRetentionDays > 0 {
		ret = fmt.Sprintf("%d days", a.cfg.KnowledgeRetentionDays)
	}
	out = append(out, "retention: "+ret, "  /knowledge clear · purge <days> · retain <days>")
	return out
}

// usageManage handles /usage subcommands. handled=false (for no/unknown args)
// tells the caller to show the report instead.
func (a *app) usageManage(rest string) ([]string, bool) {
	sub, arg := splitCommand(rest)
	switch sub {
	case "":
		return nil, false
	case "clear":
		if a.usage != nil {
			a.usage.Clear()
			_ = a.usage.Save()
		}
		return []string{"usage history cleared"}, true
	case "purge":
		days, err := strconv.Atoi(strings.TrimSpace(arg))
		if err != nil || days <= 0 {
			return []string{"usage: /usage purge <days>   (drop saved entries older than N days)"}, true
		}
		n := 0
		if a.usage != nil {
			n = a.usage.Purge(cutoffDays(days))
			_ = a.usage.Save()
		}
		return []string{fmt.Sprintf("purged %d entries older than %d day(s)", n, days)}, true
	case "retain":
		days, err := strconv.Atoi(strings.TrimSpace(arg))
		if err != nil || days < 0 {
			return []string{"usage: /usage retain <days>   (auto-drop older than N days on startup; 0 = keep forever)"}, true
		}
		a.cfg.UsageRetentionDays = days
		if err := config.SaveUsageRetention(days); err != nil {
			return []string{"error: " + err.Error()}, true
		}
		if days == 0 {
			return []string{"retention off — keeping usage history forever"}, true
		}
		n := 0
		if a.usage != nil {
			n = a.usage.Purge(cutoffDays(days))
			_ = a.usage.Save()
		}
		return []string{fmt.Sprintf("retention set to %d day(s) — purged %d older entries", days, n)}, true
	default:
		return []string{"usage: /usage [clear | purge <days> | retain <days>]"}, true
	}
}

// usageLedger returns the per-day and per-provider/model token rows (capped) from
// the persistent ledger, formatted "label" → "Nk tok" for display.
func (a *app) usageLedger() (days, models [][2]string) {
	if a.usage == nil {
		return nil, nil
	}
	for i, t := range a.usage.ByDay() {
		if i >= 14 {
			break
		}
		days = append(days, [2]string{t.Key, humanK(t.Tokens()) + " tok"})
	}
	ov := a.priceOverrides()
	for i, t := range a.usage.ByModel() {
		if i >= 8 {
			break
		}
		model := t.Key
		if idx := strings.IndexByte(t.Key, '/'); idx >= 0 {
			model = t.Key[idx+1:] // strip the "provider/" prefix for price matching
		}
		v := humanK(t.Tokens()) + " tok"
		if c := fmtCost(usage.CostUSD(model, t.Prompt, t.Completion, ov)); c != "" {
			v += "  " + c
		}
		models = append(models, [2]string{t.Key, v})
	}
	return days, models
}

// maybeInit runs the interactive first-time setup, writing the LM Studio
// connection to the user config. It triggers when forced (-init / /login) or on a
// real first run (no user config yet and an interactive terminal).
func maybeInit(reader *bufio.Reader, force bool) {
	if !force && (config.GlobalExists() || !isTTY()) {
		return
	}
	def := config.Default()
	if cur, err := config.Load("."); err == nil {
		def = cur
	}
	fmt.Println("Setup — connect your model (press Enter to keep the current value).")
	fmt.Println("  In LM Studio: load a tool-calling model and start the local server (Developer tab).")
	url := ask(reader, "Server URL", def.LLM.BaseURL)
	key := ask(reader, "API key (blank for LM Studio)", def.LLM.APIKey)

	// Best-effort probe: show what's loaded there so the model name isn't a blind
	// guess, and so we can confirm the connection at the end. (max_steps and the
	// context window are left at defaults — the latter is auto-detected on launch.)
	probeCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	models, probeErr := llm.ListModels(probeCtx, url, key, http.DefaultClient)
	cancel()
	if probeErr == nil && len(models) > 0 {
		shown := models
		if len(shown) > 6 {
			shown = shown[:6]
		}
		fmt.Printf("  ✓ reached %s — models: %s\n", url, strings.Join(shown, ", "))
	}

	l := config.LLM{
		BaseURL:       url,
		Model:         ask(reader, "Model name", def.LLM.Model),
		APIKey:        key,
		Temperature:   def.LLM.Temperature,
		MaxSteps:      def.LLM.MaxSteps,
		ContextWindow: def.LLM.ContextWindow,
	}
	if err := config.SaveGlobal(def.Name, l); err != nil { // preserve any custom name
		slog.Warn("could not save config", "err", err)
		return
	}
	fmt.Printf("Saved to %s\n", config.GlobalPath())
	if probeErr != nil {
		fmt.Printf("  ⚠ couldn't reach %s yet — start LM Studio's local server (Developer tab); it'll connect on your first task.\n", url)
	} else {
		fmt.Printf("  ✓ connected — using %s\n", l.Model)
	}
	fmt.Println("Setup done — type a task to begin, or /help for commands.")
	fmt.Println()
}

func ask(r *bufio.Reader, label, def string) string {
	if def != "" {
		fmt.Printf("  %s [%s]: ", label, def)
	} else {
		fmt.Printf("  %s: ", label)
	}
	line, err := r.ReadString('\n')
	if err != nil {
		return def
	}
	if v := strings.TrimSpace(line); v != "" {
		return v
	}
	return def
}

func isTTY() bool { return term.IsTerminal(int(os.Stdin.Fd())) }

// approvalCategory groups the per-action approval kinds into the coarse buckets a
// session-allow applies to, so "allow all file edits this session" covers
// write/edit/append/mkdir, not just the one action you happened to approve.
func approvalCategory(kind string) string {
	switch kind {
	case "write", "edit", "append", "mkdir":
		return "file"
	case "run":
		return "run"
	case "git":
		return "git"
	default:
		if strings.HasPrefix(kind, "spawn") {
			return "spawn"
		}
		if strings.HasPrefix(kind, "mcp") {
			return "mcp"
		}
		return kind
	}
}

// categoryLabel is the human phrase for the "allow all … this session" hint.
func categoryLabel(cat string) string {
	switch cat {
	case "file":
		return "file changes"
	case "run":
		return "shell commands"
	case "git":
		return "git actions"
	case "spawn":
		return "sub-agent spawns"
	case "external agent":
		return "external CLI agents"
	case "mcp":
		return "MCP calls"
	}
	return cat
}

func (a *app) sessionAllowed(kind string) bool {
	a.sessionMu.Lock()
	defer a.sessionMu.Unlock()
	return a.sessionAllow[approvalCategory(kind)]
}

// allowSession grants "don't ask again this session" for the kind's whole category.
func (a *app) allowSession(kind string) {
	a.sessionMu.Lock()
	defer a.sessionMu.Unlock()
	if a.sessionAllow == nil {
		a.sessionAllow = map[string]bool{}
	}
	a.sessionAllow[approvalCategory(kind)] = true
}

// resetSessionAllow clears every session-allow (called on /new and /clear).
func (a *app) resetSessionAllow() {
	a.sessionMu.Lock()
	defer a.sessionMu.Unlock()
	a.sessionAllow = nil
}

// approveGated is the approval path the tools use: a session-allow for the kind's
// category short-circuits the prompt, otherwise it asks the real approver.
func (a *app) approveGated(kind, detail string) bool {
	if a.sessionAllowed(kind) {
		return true
	}
	return a.approver.Approve(kind, detail)
}

// gatedApprover adapts approveGated to the tool.Approver interface.
type gatedApprover struct{ app *app }

func (g gatedApprover) Approve(kind, detail string) bool { return g.app.approveGated(kind, detail) }

// stdinApprover prompts the operator on stderr for a policy "ask" decision. The
// mutex serializes prompts so concurrent tool-call approvals never read the
// shared stdin reader at the same time. `a` grants the kind's category for the
// whole session (via the app's session-allow set).
type stdinApprover struct {
	mu  sync.Mutex
	r   *bufio.Reader
	app *app
}

func (s *stdinApprover) Approve(kind, detail string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(os.Stderr, "\n[approve %s] %s\n  allow? [y/N/a=all %s this session] ", kind, detail, categoryLabel(approvalCategory(kind)))
	line, err := s.r.ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.TrimSpace(strings.ToLower(line)) {
	case "y", "yes":
		return true
	case "a", "all", "always":
		if s.app != nil {
			s.app.allowSession(kind)
		}
		return true
	}
	return false
}

func setupLogging() {
	level := slog.LevelWarn
	switch strings.ToLower(os.Getenv("IPS_LOG")) {
	case "debug":
		level = slog.LevelDebug
	case "info":
		level = slog.LevelInfo
	case "error":
		level = slog.LevelError
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))
}

func newRunID() string {
	return fmt.Sprintf("%s-%d", time.Now().UTC().Format("20060102T150405"), os.Getpid())
}
