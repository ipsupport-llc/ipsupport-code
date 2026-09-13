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
	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
	"github.com/ipsupport-llc/ipsupport-code/internal/reflect"
	"github.com/ipsupport-llc/ipsupport-code/internal/sandbox"
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

// overrideFlags collects repeated -override key=value flags (flag.Value, not
// flag.StringVar — the stdlib flag package has no built-in repeatable string
// flag, and this is the standard way to add one: Set is called once per
// occurrence instead of just replacing a single value).
type overrideFlags []string

func (o *overrideFlags) String() string { return strings.Join(*o, ",") }
func (o *overrideFlags) Set(v string) error {
	*o = append(*o, v)
	return nil
}

func main() {
	// Landlock re-exec leg (Linux): when the run tool wraps a command in the
	// sandbox, it re-runs THIS binary, which self-restricts and execs the real
	// command. Must run first; a normal launch falls straight through.
	sandbox.MaybeExecConfined()
	var (
		workspace       string
		doInit          bool
		showVersion     bool
		dumpPrompt      bool
		newSession      bool
		sessionName     string
		skipPermissions bool
		overrides       overrideFlags
	)
	flag.StringVar(&workspace, "C", ".", "workspace directory")
	flag.BoolVar(&doInit, "init", false, "re-run first-time setup (server URL, API key, model)")
	flag.BoolVar(&showVersion, "version", false, "print version and exit")
	flag.BoolVar(&dumpPrompt, "dump-prompt", false, "print the built-in system prompt and exit (e.g. > .agent/system.md to start editing)")
	flag.BoolVar(&newSession, "new", false, "start a fresh session (don't restore the saved one)")
	flag.StringVar(&sessionName, "session", "", "use a named session (a separate saved thread)")
	flag.BoolVar(&skipPermissions, "skip-permissions", false, "don't ask before file writes or shell commands this run (equivalent to -override run.default=allow -override file.default=allow); not persisted")
	flag.Var(&overrides, "override", "override a config key for this run only, key=value (repeatable, e.g. -override llm.temperature=0.7); same dotted keys as `config set`, never persisted")
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

	// -skip-permissions is just sugar for the two -override key=values it's
	// documented as — expanding it here, before build(), means there's only
	// ONE application path (build()'s override loop) to keep correct, not two.
	if skipPermissions {
		overrides = append(overrides, "run.default=allow", "file.default=allow")
	}

	// -session selects a named thread for this run (its own saved file); passed
	// into build() so it's applied to cfg.Name BEFORE wire() runs — wire() bakes
	// the session name into the archiver/history tool paths once, at
	// construction time, so applying it after wire() (as a prior version of
	// this code did) left those two pointed at the previous name's archive for
	// the whole run even though the session file itself picked up the change.
	app, cleanup, err := build(workspace, sessionName, []string(overrides), reader)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer cleanup()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	switch {
	case strings.TrimSpace(strings.Join(flag.Args(), " ")) != "":
		if !newSession {
			app.loadSession() // one-shot: silently continue the saved session
		}
		if err := app.runOne(ctx, strings.TrimSpace(strings.Join(flag.Args(), " "))); err != nil {
			// The task never ran at all — exit nonzero so scripts/CI checking $?
			// see the failure instead of falling through to the implicit exit-0
			// below. os.Exit skips every registered defer, so run them by hand.
			stop()
			cleanup()
			os.Exit(1)
		}
	case isTTY():
		// The TUI owns the alt-screen — routing logs to stderr would bleed raw
		// "level=WARN …" lines over the interface (retries are shown in-UI anyway).
		if closeLog := redirectLogToFile(); closeLog != nil {
			defer closeLog()
		}
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
	stdin     *stdinOwner // the sole owner of reader's actual reads (see stdinOwner)

	fileTracer trace.Tracer  // JSONL dataset
	uiTracer   trace.Tracer  // live TUI (nil in plain mode)
	tracer     trace.Tracer  // composite, set in wire()
	approver   tool.Approver // stdin (plain) or the TUI bridge (the actual prompt)

	// sessionAllow is the in-memory "don't ask again this session" set, keyed by
	// approval category (file/run/git/spawn/mcp). Granted from an approval prompt,
	// never persisted; cleared on /new and /clear. gatedApprover consults it.
	sessionMu    sync.Mutex
	sessionAllow map[string]bool

	promptHist []string          // submitted inputs (tasks + /commands), oldest→newest, persisted per workspace for ↑ recall across restarts
	snippets   map[string]string // named prompt templates (/snip save+recall), persisted globally

	jobMu  sync.Mutex // guards jobs/jobSeq — background sub-agent runs (see jobs.go)
	jobs   []*job
	jobSeq int

	btwMu      sync.Mutex // guards pendingBtw — /steer notes folded into a running task
	pendingBtw []string   // steering notes awaiting the running task's next turn (see jobs.go drainBtw)

	asideMu      sync.Mutex // guards pendingAside — /btw side questions answered in one no-tools turn
	pendingAside []string

	// taskEpoch bumps on each task start and on a force-detach. A run's outputs
	// (its taskDoneMsg, its post-Run side effects) are honoured only while the
	// epoch it captured is still current — a force-detached run is orphaned.
	taskEpoch atomic.Int64

	// modelEpoch bumps whenever the active model/provider changes (/login,
	// /ai <provider>, /model <name>). An in-flight windowMsg probe is honoured
	// only while the epoch it captured is still current — a stale probe for a
	// model that's no longer active is discarded instead of clobbering the
	// newly active model's context-window state.
	modelEpoch atomic.Int64

	// approvalWaitNS accumulates nanoseconds spent BLOCKED in an approval
	// prompt (approveGated), across every tool call and every concurrent
	// sub-agent — never decremented. A caller measuring one run's own
	// duration snapshots this before and after and subtracts the delta, so a
	// human taking 30s to approve a file write doesn't get counted as 30s of
	// "the model was thinking/generating" in tok/s (internal/usage).
	approvalWaitNS atomic.Int64

	costMu         sync.Mutex // guards sessionCostUSD (parallel sub-agent spawns accrue too)
	sessionCostUSD float64    // estimated spend this process run, for the SessionBudgetUSD guard

	client          *llm.OpenAIClient
	ag              *agent.Agent
	pol             *policy.Engine // host policy/jail; sub-agents in a dir get their own
	workdir         string         // absolute session working dir (set by /cd); "" = workspace
	subReg          *tool.Registry // tools for sub-agents (no `agent` tool → no recursion)
	spawnSeq        atomic.Int64   // unique id per sub-agent spawn (for grouping its UI events)
	mcpMu           sync.Mutex     // guards the lazy MCP client cache, in-flight connect attempts, and mcpShuttingDown
	mcpClients      map[string]*mcp.Client
	mcpInFlight     map[int]context.CancelFunc // connect attempts not yet cached or discarded — see mcpClient/closeMCP
	mcpInFlightSeq  int                        // next key into mcpInFlight
	mcpShuttingDown bool                       // set by closeMCP; once true, mcpClient discards instead of caching
	ckptMu          sync.Mutex                 // guards checkpoints / the in-progress one
	checkpoints     []*checkpoint              // per-turn file+history snapshots for /rewind (session lifetime)
	curCkpt         *checkpoint                // the checkpoint being filled during the running turn
	instrSrc        string                     // project instructions file in effect, "" if none
	promptSrc       string                     // "built-in" or the system.md override path
	facts           []string                   // durable project facts learned over time (per workspace)
	planMode        bool                       // plan (propose) vs auto (execute); survives re-wire
	goal            goalState                  // standing goal pursued by the judge loop (per workspace)
	windowDetected  bool                       // got the real loaded context window (vs a default/guess)
	sessionRestored bool                       // a saved session was restored at startup (TUI renders a recap)
	historyToolOn   bool                       // history tool is in the current tool list (see hasArchivedHistory, maybeRewireHistoryTool)
	tui             bool                       // running the TUI (detect the context window off-thread, not inline)
	startNew        bool                       // -new: skip the startup chooser, begin a fresh session

	// statusMu guards tasks/steps/toolCalls below plus facts (above) and goal
	// (above): recordRun/finishGoal/reflection's addFacts write them from the
	// running task's own goroutine, while /status, /usage, and a bare /goal
	// read them from the UI goroutine while that task is still running (or
	// reflecting) — same brief-critical-section style as costMu/sessionMu.
	statusMu                sync.Mutex
	tasks, steps, toolCalls int
	lastPrompt, lastCompl   int // client usage snapshot for per-task ledger deltas

	// lastRealContext is the real conversation's prompt-token fullness (from the
	// last Agent.Run's Transcript.PromptTokens), snapshotted right after Run
	// succeeds and BEFORE reflectAndStore/judgeGoal get a chance to clobber the
	// shared client's own Context() reading with their own (much shorter) Chat
	// calls. shouldAutoCompact reads this instead of a.client.Context() fresh.
	lastRealContext int
}

func build(workspace, sessionName string, overrides []string, reader *bufio.Reader) (*app, func(), error) {
	cfg, err := config.Load(workspace)
	if err != nil {
		return nil, nil, fmt.Errorf("load config: %w", err)
	}
	// -override/-skip-permissions apply in memory ONLY, right after loading and
	// before anything else reads cfg (wire(), the session-name override just
	// below, etc.) — same key=value shape as `config set`, but never written to
	// disk: a later plain launch with no override flags must see exactly what
	// was there before.
	for _, kv := range overrides {
		key, val, ok := strings.Cut(kv, "=")
		if !ok {
			return nil, nil, fmt.Errorf("invalid -override %q: want key=value", kv)
		}
		if err := config.ApplyOverride(&cfg, key, val); err != nil {
			return nil, nil, fmt.Errorf("-override %s: %w", key, err)
		}
	}
	if sessionName != "" { // must land before wire() — see the call site in main()
		cfg.Name = sessionName
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
	a.stdin = newStdinOwner(reader)                      // sole reader of `reader`'s actual bytes from here on (see stdinOwner)
	a.approver = &stdinApprover{stdin: a.stdin, app: a}  // set after: it references the app for session-allow
	cleanup := func() { a.shutdownJobs(); a.closeMCP() } // cancel background jobs (killing external-agent subprocesses too) and shut down any launched MCP servers on exit
	a.applyUsageRetention()                              // honor usage_retention_days on startup
	a.applyKnowledgeRetention()                          // honor knowledge_retention_days on startup
	if ft, err := trace.NewFileTracer(cfg.TracePath, newRunID()); err != nil {
		slog.Warn("trace disabled", "err", err)
	} else {
		a.fileTracer = ft
		cleanup = func() { a.shutdownJobs(); a.closeMCP(); _ = ft.Close() }
	}
	a.loadFacts()      // learned project facts → folded into the prompt by wire()
	a.loadGoal()       // standing goal (if any) → resumable across restarts
	a.loadPromptHist() // ↑ recall spans past runs (persisted per workspace)
	a.loadSnippets()   // /snip prompt templates (persisted globally)
	if err := a.wire(); err != nil {
		return nil, nil, err
	}
	// The prior session is restored by the caller: interactively via chooseSession
	// (restore/new/delete), or auto-loaded in non-interactive modes.
	// windowDetected always starts at its Go zero-value (false) on a fresh
	// process — seed it from the persisted ContextWindowManual flag so a
	// deliberate /config override survives a restart instead of getting
	// silently overwritten by the auto-detect pass below (see setContextWindow).
	a.windowDetected = a.activeLLM().ContextWindowManual
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
// spawnAgent is the SpawnFunc handed to the `agent` tool (no output tap).
func (a *app) spawnAgent(ctx context.Context, profile, task, dir string) (string, error) {
	return a.spawnAgentTapped(ctx, profile, task, dir, nil)
}

// spawnAgentTapped resolves profile/dir against the CURRENT app config and
// runs it. Must be called synchronously — from the goroutine that owns a's
// config — never from a background job's own goroutine, which races the
// foreground's /cd, /ai add, /permissions, and wire(). spawnAgentBackground
// resolves via resolveSpawn on the safe side of that boundary instead; see
// spawnPlan.
func (a *app) spawnAgentTapped(ctx context.Context, profile, task, dir string, onLine func(string)) (string, error) {
	if strings.TrimSpace(task) == "" {
		return "", fmt.Errorf("task is required")
	}
	plan, external, extP, err := a.resolveSpawn(profile, dir)
	if err != nil {
		return "", err
	}
	if external { // a local CLI agent, not one of our LLM sub-agents
		return a.spawnExternalAgent(ctx, plan.profile, extP, task, plan.subWorkspace, plan.tracer, onLine)
	}
	return a.runSpawnPlan(ctx, plan, task, onLine)
}

// spawnPlan is everything runSpawnPlan needs, resolved up front against the
// live app config — so a background job's goroutine (launched well after
// resolveSpawn returns) only ever touches these captured values afterward,
// never a.cfg/a.subReg/a.workdir/a.planMode/a.tracer live. a.cfg.Agents in
// particular is a map: a concurrent read (here) racing a concurrent write
// (/ai add from the foreground) is a Go runtime panic, not just a stale
// value.
type spawnPlan struct {
	profile        string
	provider       string
	llmCfg         config.LLM
	rolePrompt     string
	subReg         *tool.Registry
	subWorkspace   string
	planMode       bool
	spawnDefault   string
	tracer         trace.Tracer
	priceOverrides map[string]usage.Price
	goalMaxSteps   int
}

// resolveSpawn resolves profile/dir into a spawnPlan (or, for an external CLI
// agent profile, its own config.AgentProfile — spawnExternalAgent takes that
// directly rather than re-reading a.cfg.Agents itself). Must be called
// synchronously; see spawnPlan.
func (a *app) resolveSpawn(profile, dir string) (spawnPlan, bool, config.AgentProfile, error) {
	profile = strings.TrimSpace(profile)
	if profile == "" {
		return spawnPlan{}, false, config.AgentProfile{}, fmt.Errorf("profile is required — configured: %s", a.profilesOrHint())
	}
	resolved, ok := a.resolveProfileName(profile)
	if !ok {
		return spawnPlan{}, false, config.AgentProfile{}, fmt.Errorf("unknown profile %q — configured: %s", profile, a.profilesOrHint())
	}
	profile = resolved
	p := a.cfg.Agents[profile]

	// Capture a.tracer HERE, synchronously, before EITHER branch below — wire()
	// reassigns it (a fresh trace.Multi(a.fileTracer, a.uiTracer)) with no lock on
	// every call, and both the external-agent branch (spawnExternalAgent) and the
	// LLM branch (runSpawnPlan) run inside a background job's own goroutine, so
	// neither must ever read a.tracer live again — see spawnPlan.
	tracer := a.tracer

	if p.Kind == "external" {
		// External CLIs run outside our sandbox with their own permissions — there's
		// no read-only mode to hand them the way SetPlanMode gives an LLM sub-agent
		// (see runSpawnPlan), so plan mode can only keep its "stays read-only"
		// guarantee (see the subagents skill) by refusing the launch outright.
		if a.planMode {
			return spawnPlan{}, true, p, fmt.Errorf("plan mode is ON — external agent %q was NOT launched (it runs outside the sandbox with no read-only mode); list it as a step in your plan, then finish", profile)
		}
		// Resolve dir HERE, synchronously, same as the LLM branch below —
		// spawnExternalAgent runs from inside a background job's own goroutine,
		// which must never call a.effectiveDir()/resolveSpawnDir() itself
		// (a.workdir has no lock and /cd mutates it live).
		root := a.effectiveDir()
		if d := strings.TrimSpace(dir); d != "" {
			resolved, err := a.resolveSpawnDir(d)
			if err != nil {
				return spawnPlan{}, true, p, err
			}
			root = resolved
		}
		if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
			return spawnPlan{}, true, p, fmt.Errorf("dir %q is not a directory", root)
		}
		return spawnPlan{profile: profile, subWorkspace: root, tracer: tracer}, true, p, nil
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
			return spawnPlan{}, false, p, fmt.Errorf("profile %q: unknown provider %q", profile, provider)
		}
		if config.IsCustomProvider(a.cfg, provider) {
			if rp.BaseURL == "" { // a custom provider must at least name where to connect
				return spawnPlan{}, false, p, fmt.Errorf("profile %q: %s has no base_url — set providers.%s.base_url in config", profile, provider, provider)
			}
		} else if rp.APIKey == "" { // built-in cloud templates need a key
			return spawnPlan{}, false, p, fmt.Errorf("profile %q: %s has no API key — add one with /ai key %s <token>", profile, provider, provider)
		}
		llmCfg = rp
	}
	if p.Model != "" {
		llmCfg.Model = p.Model
	}
	// Resolve reasoning params HERE too, synchronously — a.cfg.Reasoning is a
	// live map that /reasoning (applyReasoning) mutates from the foreground with
	// no lock, so runSpawnPlan's own goroutine must never read it again itself.
	llmCfg = a.withReasoning(llmCfg, provider, "")
	// Resolve price overrides HERE too, synchronously — a.cfg is reassigned
	// wholesale (a.cfg = cfg) by reconfigure() (/login) with no lock, so
	// runSpawnPlan's own goroutine must never read a.cfg.Prices (via
	// a.priceOverrides()) again itself — see addSessionCost.
	priceOverrides := a.priceOverrides()

	// Resolve the working directory (default: the session workspace). The path may
	// point anywhere — ~ is expanded, relatives resolve against the session — but
	// the sub-agent gets its OWN jail rooted there, so it still can't escape it.
	subWorkspace := a.pol.Workdir()
	var subReg *tool.Registry
	if d := strings.TrimSpace(dir); d != "" {
		root, err := a.resolveSpawnDir(d)
		if err != nil {
			return spawnPlan{}, false, p, err
		}
		if fi, statErr := os.Stat(root); statErr != nil || !fi.IsDir() {
			return spawnPlan{}, false, p, fmt.Errorf("dir %q is not a directory", dir)
		}
		subCfg := a.cfg
		subCfg.Workspace = root
		subCfg.File.Jail = "." // keep the jail — confine the sub-agent to its own dir
		subPol, pErr := policy.New(subCfg)
		if pErr != nil {
			return spawnPlan{}, false, p, pErr
		}
		subReg, subWorkspace = a.buildSubReg(subPol, root), root
	} else {
		// No explicit dir: still give the sub-agent its OWN *policy.Engine, not
		// a.pol. a.pol.workdir has no lock and /cd mutates it live — sharing the
		// pointer (as a.subReg does at wire() time) would let a foreground /cd
		// bleed into an already-running background delegate's relative path
		// resolution. Snapshot the host's current dir now as the delegate's fixed
		// starting workdir.
		subPol, pErr := policy.New(a.cfg)
		if pErr != nil {
			return spawnPlan{}, false, p, pErr
		}
		if _, err := subPol.SetWorkdir(subWorkspace); err != nil {
			return spawnPlan{}, false, p, err
		}
		subReg = a.buildSubReg(subPol, a.hostSandboxRoot())
	}
	return spawnPlan{
		profile: profile, provider: provider, llmCfg: llmCfg, rolePrompt: p.Prompt,
		subReg: subReg, subWorkspace: subWorkspace, planMode: a.planMode, spawnDefault: a.cfg.Spawn.Default,
		tracer: tracer, priceOverrides: priceOverrides, goalMaxSteps: a.cfg.GoalMaxSteps,
	}, false, p, nil
}

// runSpawnPlan runs an already-resolved plan — safe to call from a background
// job's own goroutine, since it only touches plan (captured up front by
// resolveSpawn) and state that's already concurrency-safe on its own
// (a.usage has its own synchronization; a.kb is a stable pointer once wire()
// has run). a.tracer is NOT stable — wire() reassigns it with no lock on
// every call, and wire() runs repeatedly throughout a live process, not just
// at startup — so this uses plan.tracer (captured synchronously by
// resolveSpawn) instead of a.emit/a.tracer. Likewise a.cfg is NOT stable —
// reconfigure() (/login) reassigns it wholesale (a.cfg = cfg) with no lock —
// so the sub-agent's spend is added via plan.priceOverrides (also captured
// synchronously by resolveSpawn), never by having addSessionCost re-read
// a.cfg.Prices live.
func (a *app) runSpawnPlan(ctx context.Context, plan spawnPlan, task string, onLine func(string)) (string, error) {
	// Ask before spawning unless the policy is relaxed. "ask" (default) guards
	// every spawn — even local ones still cost compute, and a runaway main model
	// could fan out endlessly. This wait is human-paced and must NOT be held
	// behind a shared lock — the approver (TUI bridge / stdin prompt) already
	// serializes concurrent prompts on its own, and a mutex held here would
	// block an unrelated job's own spawn/mcp approval behind this one's.
	if plan.spawnDefault != "allow" {
		approved := a.approveGated(ctx, "spawn agent", fmt.Sprintf("%s · %s · %s\n  task: %s", plan.profile, plan.llmCfg.Model, plan.subWorkspace, task))
		if !approved {
			return "", fmt.Errorf("spawn denied by user")
		}
	}

	id := fmt.Sprintf("sub%d", a.spawnSeq.Add(1)) // groups this sub-agent's UI events
	client := llm.NewOpenAIClient(plan.llmCfg)    // reasoning params already resolved in resolveSpawn
	sub := agent.New(client, plan.subReg, a.kb, plan.tracer, a.subAgentPrompt(plan.subWorkspace, plan.rolePrompt), resolveStepBudget(plan.goalMaxSteps, plan.llmCfg))
	sub.SetPlanMode(plan.planMode)
	sub.SetLabel(id)
	sub.SetContextWindow(plan.llmCfg.ContextWindow)
	if plan.tracer != nil {
		plan.tracer.Emit("subagent", map[string]any{"agent": id, "profile": plan.profile, "provider": plan.provider, "model": plan.llmCfg.Model, "dir": plan.subWorkspace, "task": oneLine(task, 80)})
	}

	waitSnapshot := a.approvalWaitNS.Load()
	start := time.Now()
	tr, err := sub.Run(ctx, task)
	dur := a.runDuration(start, waitSnapshot)
	if a.usage != nil { // the sub-agent's spend counts too
		pt, ct := client.Usage()
		a.usage.Add(today(), plan.provider, plan.llmCfg.Model, pt, ct, dur)
		a.addSessionCost(plan.llmCfg.Model, pt, ct, plan.priceOverrides) // sub-agent spend counts toward /budget too
		_ = a.usage.Save()
	}
	done := map[string]any{"agent": id, "profile": plan.profile, "ok": err == nil}
	if err != nil {
		done["error"] = oneLine(err.Error(), 60)
	}
	if plan.tracer != nil {
		plan.tracer.Emit("subagent_done", done)
	}
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

// buildSubReg builds a sub-agent's tool registry against a given policy (and
// its jail root, for the sandbox wrapper — see sandboxWrapperFor): the host
// tools except `agent` (so sub-agents can't recurse) and except help (a usage
// helper they don't need). The run (shell) tool is included only when spawn.exec
// is on — the sharpest capability to hand an autonomous sub-agent.
func (a *app) buildSubReg(pol *policy.Engine, root string) *tool.Registry {
	tools := []tool.Tool{tool.NewFile(pol, gatedApprover{a}, a.snapFile)}
	if a.cfg.Spawn.Exec {
		tools = append(tools, tool.NewRun(pol, gatedApprover{a}, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second, a.sandboxWrapperFor(root)))
	}
	tools = append(tools, tool.NewGit(pol, gatedApprover{a}), tool.NewWeb(nil, a.cfg.Offline), tool.NewCalc())
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
		"\n\nToday is %s. Environment: you are running on %s; your working directory is %s. Use commands that exist on this OS. All file/run/git paths resolve in that directory.",
		time.Now().Format("2006-01-02"), runtime.GOOS, workspace)
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
// web tool refuses, startup/`/update` skip GitHub — the model connection itself
// (whether that's actually localhost or a remote endpoint you configured) is
// untouched either way, so "offline" doesn't guarantee no network traffic at all.
func (a *app) offlineCommand(arg string) []string {
	switch strings.TrimSpace(arg) {
	case "on", "yes", "true":
		a.cfg.Offline = true
	case "off", "no", "false":
		a.cfg.Offline = false
	case "":
		return []string{"offline mode is " + onOff(a.cfg.Offline) + " — /offline on|off",
			"  on = no internet: web tool + update checks off (your model connection is untouched — local or remote)"}
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
		return []string{"offline mode → on (web + update checks disabled; your model connection is untouched)"}
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
	Text    string `json:"text"`
	Status  string `json:"status"`            // active | done | incomplete
	Offered bool   `json:"offered,omitempty"` // already surfaced once after a restart — don't auto-nag again
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
		a.statusMu.Lock()
		a.goal = g
		a.statusMu.Unlock()
	}
}

// goalSnapshot returns a copy of the current standing goal (guarded, see
// statusMu) — safe to read while a task's own goroutine may be writing it.
func (a *app) goalSnapshot() goalState {
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	return a.goal
}

func (a *app) saveGoal() error {
	data, _ := json.MarshalIndent(a.goalSnapshot(), "", "  ")
	return atomicfile.Write(a.goalPath(), data, 0o644)
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
		text := a.goalSnapshot().Text
		return text, text != ""
	case "clear", "drop", "done", "ttl", "off", "on":
		return "", false
	}
	return rest, true
}

// clearGoal drops the standing goal and its persisted file. Goal state is
// workspace-scoped, not session-scoped (goalPath has no session/agent-name
// component) — so anything that starts a genuinely NEW session must call this,
// or the old session's goal silently attaches to the fresh, unrelated thread.
func (a *app) clearGoal() {
	a.statusMu.Lock()
	a.goal = goalState{}
	a.statusMu.Unlock()
	os.Remove(a.goalPath())
}

// setGoal records a new standing goal (active) and persists it.
func (a *app) setGoal(text string) {
	a.statusMu.Lock()
	a.goal = goalState{Text: strings.TrimSpace(text), Status: "active"}
	a.statusMu.Unlock()
	if err := a.saveGoal(); err != nil {
		slog.Warn("goal not persisted", "err", err)
	}
}

// finishGoal updates the standing goal's status from a finished run, but only when
// that run was actually pursuing it (same text, still active).
func (a *app) finishGoal(goal string, tr agent.Transcript) {
	a.statusMu.Lock()
	if a.goal.Status != "active" || strings.TrimSpace(goal) != a.goal.Text {
		a.statusMu.Unlock()
		return
	}
	// With the judge loop off (/goal off · ttl 0) GoalMet is never set — the model
	// decides when it's done, so a clean uninterrupted finish counts as done
	// (otherwise the goal stays "incomplete" and the resume prompt nags forever).
	done := (tr.GoalMet || a.cfg.GoalMaxReturns == 0) && !tr.Stopped && !tr.Cancelled
	if done {
		// Done: clear it entirely so it never resurfaces after a restart.
		a.goal = goalState{}
	} else {
		// Still unfinished: re-arm one resume offer for the next restart (you
		// just engaged it).
		a.goal.Status, a.goal.Offered = "incomplete", false
	}
	a.statusMu.Unlock()
	if done {
		os.Remove(a.goalPath())
		return
	}
	if err := a.saveGoal(); err != nil {
		slog.Warn("goal status not persisted", "err", err)
	}
}

// markGoalOffered records that the standing goal has been surfaced once after a
// restart, so it isn't re-offered on every subsequent start (until you engage it
// again or set a new goal).
func (a *app) markGoalOffered() {
	a.statusMu.Lock()
	if a.goal.Offered || a.goal.Text == "" {
		a.statusMu.Unlock()
		return
	}
	a.goal.Offered = true
	a.statusMu.Unlock()
	if err := a.saveGoal(); err != nil {
		slog.Warn("goal offer-state not persisted", "err", err)
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
		had := a.goalSnapshot().Text
		a.clearGoal()
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
	g := a.goalSnapshot()
	if g.Text == "" {
		return []string{
			"no standing goal · " + ttl,
			"  /goal <text> sets one and pursues it · a judge re-feeds it until met · /goal ttl <n>",
		}
	}
	out := []string{
		fmt.Sprintf("goal [%s]: %s", g.Status, g.Text),
		"  " + ttl + " · /goal go to resume · /goal clear to drop · /goal ttl <n>",
	}
	return out
}

// goalTTLFor returns the judge re-feed budget for a run: the configured TTL only
// when the run is pursuing the active standing goal, else 0 (a plain task is one
// run, no judge overhead).
func (a *app) goalTTLFor(goal string) int {
	g := a.goalSnapshot()
	if g.Status == "active" && strings.TrimSpace(goal) == g.Text {
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
// the ring, and persists it so ↑ recall survives a restart. What's PERSISTED
// (and what ↑ recall shows back) is redacted (see redactSecrets); the line
// actually dispatched to run the command is untouched — this only affects
// what lands on disk.
func (a *app) addPromptHist(line string) {
	line = redactSecrets(line)
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

// redactSecrets masks a raw credential out of a line before it's persisted to
// prompt history: `/ai key <name> <token>` and `/ai add ... key=<token>` are
// the two shapes that carry one in plain text. The history file (0644, not a
// designated secret store) isn't covered by the file tool's secret-read
// exclusions, so an unredacted key there is readable back out via file.read.
func redactSecrets(line string) string {
	fields := strings.Fields(line)
	if len(fields) < 2 || fields[0] != "/ai" {
		return line
	}
	switch fields[1] {
	case "key":
		if len(fields) >= 4 { // /ai key <name> <token>
			fields[len(fields)-1] = "***"
			return strings.Join(fields, " ")
		}
	case "add":
		redacted := false
		for i, f := range fields {
			if strings.HasPrefix(f, "key=") {
				fields[i] = "key=***"
				redacted = true
			}
		}
		if redacted {
			return strings.Join(fields, " ")
		}
	}
	return line
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

// gitOutTimeout bounds every gitOut call. A var (not const) so tests can
// shrink it to keep a hung-subprocess test fast; production never overrides
// it. A few seconds is generous for what's always a quick, read-only query
// (rev-parse/diff/ls-files) — and unlike internal/tool's git tool (which runs
// under the caller's own cancellable context), gitOut is called straight from
// the bubbletea UI goroutine with no timeout of its own, so this is the only
// thing standing between a wedged git subprocess and an indefinite hang.
var gitOutTimeout = 5 * time.Second

// gitOut runs a read-only git command in dir and returns its stdout. It
// applies the same guards internal/tool/git.go's git tool applies to every
// invocation against a workspace: "-c core.fsmonitor=" (a local .git/config
// can bind core.fsmonitor to an arbitrary executable that git would
// otherwise run as a subprocess on commands like status/diff — with no
// approval gate at all here) and, for a "diff" subcommand, "--no-textconv
// --no-ext-diff" (a .gitattributes textconv/ext-diff driver is the same kind
// of unapproved-subprocess risk). It also bounds the call with gitOutTimeout
// and kills the whole process group on expiry (internal/procgroup), so a
// stuck hook/driver/pager can't hang the caller.
func gitOut(dir string, args ...string) (string, error) {
	if len(args) > 0 && args[0] == "diff" {
		extended := append([]string{"diff", "--no-textconv", "--no-ext-diff"}, args[1:]...)
		args = extended
	}
	ctx, cancel := context.WithTimeout(context.Background(), gitOutTimeout)
	defer cancel()
	gitArgs := append([]string{"-C", dir, "-c", "core.fsmonitor="}, args...)
	cmd := exec.CommandContext(ctx, "git", gitArgs...)
	procgroup.Set(cmd)
	out, err := cmd.Output()
	return string(out), err
}

// mustGit is gitOut with the error dropped — for best-effort reads (diff/stat)
// where an empty string is an acceptable answer in a non-repo dir.
func mustGit(dir string, args ...string) string { s, _ := gitOut(dir, args...); return s }

// diffCommand shows the uncommitted working-tree changes in the current dir — a
// quick "what did the agent change" review without leaving the TUI. The caller
// colorizes the lines. Caps very large diffs so the log doesn't flood.
func (a *app) diffCommand() []string { return diffLinesForDir(a.effectiveDir()) }

// diffLinesForDir is diffCommand's logic, parameterized on dir. The async
// /diff dispatch (runDiffCmd in tui.go) snapshots a.effectiveDir() on the UI
// goroutine and runs this in a background goroutine — reading a.effectiveDir()
// again from that goroutine would race a concurrent /cd — so this must not
// touch any other app state.
func diffLinesForDir(dir string) []string {
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

// goalSteps is the hard tool-call-round backstop for one goal pursuit: an
// explicit GoalMaxSteps override wins, else the active connection's own
// per-connection budget (stepBudget — an explicit MaxSteps, or auto-scaled
// from its context window).
func (a *app) goalSteps() int {
	return resolveStepBudget(a.cfg.GoalMaxSteps, a.activeLLM())
}

// resolveStepBudget applies the full override-precedence chain for the
// tool-call-round budget: an explicit top-level GoalMaxSteps wins, else the
// connection's own stepBudget (its own explicit MaxSteps, or auto-scaled from
// its context window). Both the main agent (goalSteps) and sub-agent
// construction (runSpawnPlan) must go through this — a sub-agent is still
// bound by the same top-level GoalMaxSteps override as the main agent.
func resolveStepBudget(goalMaxSteps int, l config.LLM) int {
	if goalMaxSteps > 0 {
		return goalMaxSteps
	}
	return stepBudget(l)
}

// stepBudget resolves the tool-call-round budget for one connection: an
// explicit l.MaxSteps wins, else auto-scale from its context window.
func stepBudget(l config.LLM) int {
	if l.MaxSteps > 0 {
		return l.MaxSteps
	}
	return autoStepBudget(l.ContextWindow)
}

// refContextWindow/refStepBudget anchor autoStepBudget to today's long-standing
// behavior at the historical default context window (8192 → 80 steps), so a
// typical/default setup sees no change; a bigger window scales the budget up
// proportionally (a bigger KV cache affords more tool-call rounds before a
// complex task's own accumulated back-and-forth threatens to fill it), and a
// smaller one scales it down, within floor/ceiling. The ceiling exists because
// a task genuinely needing hundreds of rounds is almost certainly stuck in a
// loop, not legitimately working, regardless of how much room is available.
const (
	refContextWindow = 8192
	refStepBudget    = 80
	minStepBudget    = 40
	maxStepBudget    = 400
)

// autoStepBudget derives the per-goal tool-call-round budget from a
// connection's real context window when no explicit MaxSteps/GoalMaxSteps
// override is set.
func autoStepBudget(contextWindow int) int {
	if contextWindow <= 0 {
		return refStepBudget
	}
	n := refStepBudget * contextWindow / refContextWindow
	if n < minStepBudget {
		n = minStepBudget
	}
	if n > maxStepBudget {
		n = maxStepBudget
	}
	return n
}

// refMaxHistory anchors autoMaxHistory the same way autoStepBudget anchors the
// step budget: unchanged at the historical default window, scaling up for a
// bigger one. remember()'s FIFO trim (unlike Compact) has no summarization
// step, so it's meant as a rare backstop, not everyday routine — scaling it
// with the window means a big-context session naturally keeps proportionally
// more real cross-task memory before that backstop ever has to fire.
const (
	refMaxHistory     = 16
	minMaxHistory     = 16
	maxAutoMaxHistory = rawMemoryMaxHistory // same ceiling memory=raw already uses
)

// autoMaxHistory derives Agent.remember's cross-task message cap from a
// connection's real context window when no explicit override (Config.
// MaxHistory) is set.
func autoMaxHistory(contextWindow int) int {
	if contextWindow <= 0 {
		return refMaxHistory
	}
	n := refMaxHistory * contextWindow / refContextWindow
	if n < minMaxHistory {
		n = minMaxHistory
	}
	if n > maxAutoMaxHistory {
		n = maxAutoMaxHistory
	}
	return n
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

// nextReasoning returns the next level in the cycle after cur (wrapping) for
// provider; a non-level (default/custom) starts the cycle at its head. For a
// provider whose "off" has no distinct wire shape (reasoningShape returns a
// nil shape — local/openai/grok/groq), "off" and "default" are the exact same
// request on the wire, so including "off" in the cycle is a permanent no-op:
// applying it just deletes an already-absent key, the display never leaves
// "default", and minimal/low/medium/high become unreachable. Those providers
// cycle minimal→high only; "off" stays reachable (and meaningful) wherever it
// has a real shape of its own.
func nextReasoning(provider, cur string) string {
	levels := reasoningLevels
	if shape, known := reasoningShape(provider, "off"); known && shape == nil {
		levels = reasoningLevels[1:]
	}
	for i, l := range levels {
		if l == cur {
			return levels[(i+1)%len(levels)]
		}
	}
	return levels[0]
}

// mcpServerNames lists configured MCP server names, sorted. servers is the
// server set to use — see mcpList for why this isn't read from a.cfg here.
func (a *app) mcpServerNames(servers map[string]mcp.Server) []string {
	names := make([]string, 0, len(servers))
	for n := range servers {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// mcpClient returns a connected client for a server, launching + caching it on
// first use (lazy — servers aren't spawned until something needs them). The
// launch itself is approval-gated, separately from mcpCall's per-invocation
// approval: list/schema carry no Mutates flag and look like safe reads, but the
// first call to either one would otherwise launch the server (a workspace
// config can name an arbitrary command) with no approval at all. Once
// approved, the cached client short-circuits both the connect and the prompt
// for the rest of the session.
func (a *app) mcpClient(ctx context.Context, servers map[string]mcp.Server, name string) (*mcp.Client, error) {
	a.mcpMu.Lock()
	if c, ok := a.mcpClients[name]; ok {
		a.mcpMu.Unlock()
		return c, nil
	}
	if a.mcpShuttingDown {
		a.mcpMu.Unlock()
		return nil, fmt.Errorf("mcp server %q: shutting down", name)
	}
	// Register this attempt as in-flight BEFORE releasing mcpMu for the
	// approval wait and dial below, so closeMCP can find and actively cancel
	// it even though it isn't in mcpClients yet — see closeMCP.
	attemptCtx, attemptCancel := context.WithCancel(ctx)
	id := a.mcpInFlightSeq
	a.mcpInFlightSeq++
	if a.mcpInFlight == nil {
		a.mcpInFlight = map[int]context.CancelFunc{}
	}
	a.mcpInFlight[id] = attemptCancel
	a.mcpMu.Unlock()
	defer func() {
		a.mcpMu.Lock()
		delete(a.mcpInFlight, id)
		a.mcpMu.Unlock()
		attemptCancel()
	}()

	srv, ok := servers[name]
	if !ok {
		return nil, fmt.Errorf("unknown MCP server %q (configured: %s)", name, strings.Join(a.mcpServerNames(servers), ", "))
	}
	detail := name
	switch {
	case srv.Command != "":
		detail = fmt.Sprintf("%s: %s %s", name, srv.Command, strings.Join(srv.Args, " "))
	case srv.URL != "":
		detail = fmt.Sprintf("%s: %s", name, srv.URL)
	}
	// Approval and the dial itself run WITHOUT mcpMu held: both can block
	// indefinitely (approveGated waits on the user; Connect waits on the
	// network/subprocess), and holding the cache lock across either would let
	// any concurrent mcpMu.Lock() block right along with it — including
	// invalidateStaleMCP, which /login and /init call synchronously from
	// bubbletea's single event-loop goroutine. If THAT lock stalls on the
	// event-loop goroutine, Update() never returns; and since the approval
	// pending here can only be answered by Update() processing the next
	// keystroke, the whole TUI deadlocks permanently. See
	// TestMCPClientLockNotHeldDuringApproval. attemptCtx (rather than ctx)
	// carries the wait so closeMCP can cut it short on shutdown instead of
	// leaving the eventual connection an orphan with nothing left to close it.
	approved := a.approveGated(attemptCtx, "mcp launch", detail)
	if !approved {
		return nil, fmt.Errorf("mcp server %q launch denied by user", name)
	}
	cctx, cancel := context.WithTimeout(attemptCtx, 20*time.Second)
	defer cancel()
	c, err := mcp.Connect(cctx, name, srv)
	if err != nil {
		return nil, err
	}

	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	if a.mcpShuttingDown {
		// Shutdown began while this connect was in flight and closeMCP has
		// already run (or is running) — it can't see this client since it was
		// never in the cache. Close it ourselves rather than leave it running
		// with nothing left to ever close it.
		c.Close()
		return nil, fmt.Errorf("mcp server %q: shutting down", name)
	}
	// Another goroutine may have raced in and cached a client for this same
	// server while we were unlocked approving/dialing — reuse it and close
	// the redundant one we just built, rather than leaking a duplicate
	// connection or leaving two different Client objects in use for the same
	// server. We don't re-check here whether the server's config changed
	// underneath us during that window: invalidateStaleMCP already evicts a
	// changed server's cached client on reconfigure, and racing a config edit
	// against this rare first-connect window isn't worth reconciling further.
	if existing, ok := a.mcpClients[name]; ok {
		c.Close()
		return existing, nil
	}
	if a.mcpClients == nil {
		a.mcpClients = map[string]*mcp.Client{}
	}
	a.mcpClients[name] = c
	return c, nil
}

// invalidateStaleMCP evicts (and closes) any cached MCP client whose server
// was removed from config, or whose spec changed, comparing against old — the
// server map as of just before this reconfigure. mcpClient only ever checks
// the cache, never the current config, once a client exists — so without this
// a config reload (/login, /init) would leave calls going to a since-edited
// server's stale URL/command/auth until the process restarted.
func (a *app) invalidateStaleMCP(old map[string]mcp.Server) {
	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	for name, c := range a.mcpClients {
		if spec, ok := a.cfg.McpServers[name]; ok && spec.Equal(old[name]) {
			continue // unchanged — keep the live connection
		}
		c.Close()
		delete(a.mcpClients, name)
	}
}

// mcpCloseTimeout bounds how long closeMCP waits for an in-flight connect
// attempt to actually unwind after being cancelled — long enough for a
// stdio subprocess to notice the cancellation and exit, short enough not to
// hang process exit on a wedged one.
const mcpCloseTimeout = 2 * time.Second

// closeMCP shuts down every launched MCP server (called on exit). It also
// cancels any connect attempt still in flight (approval wait or dial) and
// marks shutdown started. Without this, closeMCP only ever saw connections
// already cached in a.mcpClients — a connect still inside its approval-wait
// or dial window (mcpClient releases mcpMu across both; see mcpClient) isn't
// there yet, so its subprocess would survive as an orphan once this process
// exits with nothing left to close it.
func (a *app) closeMCP() {
	a.mcpMu.Lock()
	a.mcpShuttingDown = true
	for _, c := range a.mcpClients {
		c.Close()
	}
	a.mcpClients = nil
	for _, cancel := range a.mcpInFlight {
		cancel()
	}
	a.mcpMu.Unlock()

	deadline := time.Now().Add(mcpCloseTimeout)
	for a.mcpInFlightCount() > 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
}

// mcpInFlightCount reports how many connect attempts are still in flight
// (for closeMCP's shutdown wait).
func (a *app) mcpInFlightCount() int {
	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	return len(a.mcpInFlight)
}

// mcpList is the catalog the `mcp` tool's list action returns. servers is the
// server set to use, passed in rather than read from a.cfg here: mcpList runs
// in a background goroutine dispatched by the TUI's /mcp command (see that
// case in tui.go), and reconfigure() (/login, /init) reassigns a.cfg wholesale
// with no lock — a live a.cfg.McpServers read from that goroutine would race
// it. Callers that aren't racing a.cfg (the REPL, and the mcp tool wiring,
// both of which can only run while no reconfigure is concurrently possible)
// pass a.cfg.McpServers directly.
func (a *app) mcpList(ctx context.Context, servers map[string]mcp.Server) string {
	if len(servers) == 0 {
		return "no MCP servers configured — add them under \"mcp_servers\" in config.json"
	}
	var b strings.Builder
	for _, name := range a.mcpServerNames(servers) {
		c, err := a.mcpClient(ctx, servers, name)
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
	c, err := a.mcpClient(ctx, a.cfg.McpServers, server)
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
// anything).
func (a *app) mcpCall(ctx context.Context, server, tool string, args map[string]any) (string, error) {
	c, err := a.mcpClient(ctx, a.cfg.McpServers, server)
	if err != nil {
		return "", err
	}
	detail := server + "." + tool
	if len(args) > 0 {
		if b, e := json.Marshal(args); e == nil {
			detail += " " + oneLine(string(b), 60)
		}
	}
	approved := a.approveGated(ctx, "mcp call", detail)
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
// other name sets that provider's preset. Returns whether the value actually
// changed from what was already in effect, so a caller can decide whether a
// re-wire is worth it.
func (a *app) applyWindow(provider string, tokens int) bool {
	if tokens <= 0 {
		return false
	}
	if provider == "local" {
		changed := a.cfg.LLM.ContextWindow != tokens
		a.cfg.LLM.ContextWindow = tokens
		return changed
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[provider]
	changed := p.ContextWindow != tokens
	p.ContextWindow = tokens
	a.cfg.Providers[provider] = p
	return changed
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
			changed := a.cfg.LLM.ContextWindow != w
			a.cfg.LLM.ContextWindow = w
			a.windowDetected = true
			slog.Info("detected context window", "tokens", w, "model", a.cfg.LLM.Model)
			a.rewireAfterWindowChange(changed)
		}
		return
	}
	act := a.activeLLM()
	if w := llm.DetectModelContext(ctx, act.BaseURL, act.APIKey, act.Model, http.DefaultClient); w > 0 {
		if a.cfg.Providers == nil {
			a.cfg.Providers = map[string]config.LLM{}
		}
		p := a.cfg.Providers[a.cfg.Provider]
		changed := p.ContextWindow != w
		p.ContextWindow = w
		a.cfg.Providers[a.cfg.Provider] = p
		a.windowDetected = true
		slog.Info("detected context window", "tokens", w, "model", act.Model, "provider", a.providerName())
		a.rewireAfterWindowChange(changed)
	}
}

// rewireAfterWindowChange re-wires the live agent/client after detection
// changes ContextWindow — otherwise wire() (which bakes ContextWindow into the
// Agent's step budget/history cap/mid-task trim threshold and the client's
// per-turn generation cap) stays frozen at whatever value was in effect at the
// LAST wire() call until some unrelated later trigger (a /model switch,
// /login) happens to call it again. Best-effort: a re-wire failure here is
// logged, not fatal — detection already updated a.cfg either way.
func (a *app) rewireAfterWindowChange(changed bool) {
	if !changed {
		return
	}
	if err := a.wire(); err != nil {
		slog.Warn("re-wire after context window detection failed", "err", err)
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
// sandboxWrapper builds the OS-sandbox command wrapper for the HOST's run
// tool from the current config (mode + jail + offline). See sandboxWrapperFor
// for the root-parameterized core a sub-agent with its own jail also uses.
func (a *app) sandboxWrapper() tool.CmdWrapper {
	return a.sandboxWrapperFor(a.hostSandboxRoot())
}

// hostSandboxRoot is the host's own jail root: the workspace, or its
// configured Jail subdirectory if one is set.
func (a *app) hostSandboxRoot() string {
	root := a.cfg.Workspace
	if j := strings.TrimSpace(a.cfg.File.Jail); j != "" && j != "." {
		if filepath.IsAbs(j) {
			root = j
		} else {
			root = filepath.Join(a.cfg.Workspace, j)
		}
	}
	return root
}

// sandboxWrapperFor builds the OS-sandbox command wrapper confined to root, or
// nil when sandboxing is off or the platform can't confine (then commands run
// as before). root is parameterized rather than always the host's own jail so
// a sub-agent spawned with its OWN jail (a different directory entirely) gets
// a wrapper confined to THAT root — otherwise a sandboxed host could still
// hand an unconfined sub-agent the run tool. The network follows offline mode
// either way, since that's a whole-process setting, not per-jail.
func (a *app) sandboxWrapperFor(root string) tool.CmdWrapper {
	mode := a.cfg.Sandbox
	if mode == "" || mode == sandbox.Off {
		return nil
	}
	if !sandbox.Available(mode) {
		slog.Warn("sandbox not available on this platform — running commands unconfined", "mode", mode)
		return nil
	}
	roots := []string{root}
	// Build tooling writes caches outside the workspace (go → ~/.cache/go-build,
	// linters, etc.). Allow the user cache dir so `go build` / `go test` work
	// under the sandbox; it's tool scratch, not user data.
	if cache, err := os.UserCacheDir(); err == nil {
		roots = append(roots, cache)
	}
	spec := sandbox.Spec{WritableRoots: roots, AllowNetwork: !a.cfg.Offline}
	return func(name string, args []string) (string, []string) {
		n, out, _ := sandbox.Wrap(mode, spec, name, args)
		return n, out
	}
}

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
		tool.NewRun(pol, gatedApprover{a}, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second, a.sandboxWrapper()),
		tool.NewGit(pol, gatedApprover{a}),
		tool.NewWeb(nil, a.cfg.Offline), // nil → NewWeb's own 30s-timeout client; task ctx has no deadline of its own
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
	a.subReg = a.buildSubReg(pol, a.hostSandboxRoot())
	// The `agent` tool is only worth its catalog space when there is a profile to
	// delegate to; with no profiles configured, the tool is hidden entirely.
	if a.hasSubagentTargets() {
		tools = append(tools, tool.NewAgent(a.spawnAgent, a.spawnAgentBackground))
	}
	// One proxy tool fronts every configured MCP server, so the catalog grows by a
	// single tool — not by every server's schemas. Only when servers are set.
	if len(a.cfg.McpServers) > 0 {
		tools = append(tools, tool.NewMCP(func(ctx context.Context) string {
			return a.mcpList(ctx, a.cfg.McpServers)
		}, a.mcpCall, a.mcpSchema))
	}
	// The history tool only earns its catalog space once there's something
	// durably archived to recall (see hasArchivedHistory) — a brand-new session
	// has nothing to look back on yet. Recorded in historyToolOn so
	// maybeRewireHistoryTool can tell, at the next task, whether the archive's
	// first write since happened and the tool now needs adding.
	a.historyToolOn = a.hasArchivedHistory()
	if a.historyToolOn {
		tools = append(tools, tool.NewHistory(historySource{path: a.archivePath()}))
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
	var priorGen int64
	if a.ag != nil {
		prior = a.ag.History()
		priorGen = a.ag.HistoryGen()
	}
	a.ag = agent.New(a.client, reg, a.kb, a.tracer, a.systemPrompt(), a.goalSteps())
	// Carry historyGen forward too: a fresh Agent starts at 0, and SetHistory
	// below always bumps by exactly 1 — without seeding, every rebuild would
	// land back at gen=1 regardless of how many rebuilds (or a /clear) came
	// before, letting a checkpoint invalidated pre-rebuild become spuriously
	// valid again against the new Agent instance.
	a.ag.SeedHistoryGen(priorGen)
	a.ag.SetHistory(prior)
	a.ag.SetPlanMode(a.planMode)     // carry the mode into the rebuilt agent
	a.ag.SetBeforeTurn(a.beforeTurn) // /steer notes + finished background jobs fold in between steps of a running task
	a.ag.SetAsides(a.drainAsides)    // /btw side questions answered between steps, one no-tools turn each
	a.ag.SetArchiver(&sessionArchiver{path: a.archivePath()})
	a.ag.SetContextWindow(a.activeLLM().ContextWindow) // so a single long task can watch its OWN growing trail mid-run
	a.ag.SetMaxStuckTurns(a.cfg.MaxStuckTurns)         // 0 = internal/agent's own default
	// A local server's KV-cache only helps while the prompt PREFIX stays
	// identical between requests; remember()'s trim cuts from the front, which
	// breaks that just like a summary compact would — so the cap is meant as a
	// rare backstop, not everyday routine. An explicit Config.MaxHistory wins;
	// otherwise it auto-scales from the context window (a bigger window
	// affords keeping proportionally more real cross-task memory before that
	// backstop ever needs to fire), floored at raw mode's own minimum since raw
	// opts out of auto-compact entirely and leans on this cap alone.
	hist := a.cfg.MaxHistory
	if hist <= 0 {
		hist = autoMaxHistory(a.activeLLM().ContextWindow)
		if a.cfg.Memory == "raw" && hist < rawMemoryMaxHistory {
			hist = rawMemoryMaxHistory
		}
	}
	a.ag.SetMaxHistory(hist)
	return nil
}

// rawMemoryMaxHistory is memory "raw"'s message cap — high enough that the
// silent FIFO trim in Agent.remember is a safety backstop against a truly
// runaway session, not something a normal session ever reaches.
const rawMemoryMaxHistory = 500

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
	oldServers := a.cfg.McpServers
	a.cfg = cfg
	a.invalidateStaleMCP(oldServers)
	if err := a.wire(); err != nil {
		return err
	}
	a.loadSession()          // a fresh agent — restore the persisted session
	a.windowDetected = false // model may have changed (/login) — re-detect
	a.modelEpoch.Add(1)      // orphan any in-flight probe for the old model
	a.maybeDetectWindowSync()
	return nil
}

// autoCompactRatio is the default fraction of the context window at which the
// session is auto-compacted, leaving headroom for the next task. Overridable
// per config via compactThreshold.
const autoCompactRatio = 0.75

// compactThreshold resolves the configured CompactThreshold override, falling
// back to the built-in default when unset (0).
func compactThreshold(cfgVal float64) float64 {
	if cfgVal > 0 {
		return cfgVal
	}
	return autoCompactRatio
}

// minCompactHeadroom is the smallest headroom, in tokens, auto-compact will
// ever leave regardless of the configured ratio. A flat percentage scales
// badly across wildly different context windows: 15% headroom is ~19k tokens
// on a 128k window (plenty), but only ~600 tokens on a tiny 4k-token local
// model's window — barely enough room for one more tool result before the
// next request risks silently overflowing it. Caught live: a compact_threshold
// of 0.85 (reachable via the /config panel's cycle-on-enter) on a 4.1k-window
// model left so little headroom that auto-compact effectively never fired
// again, and dozens of near-duplicate turns piled up uncompacted.
const minCompactHeadroom = 1500

// autoCompactNeeded decides whether to fold the session into a summary: the last
// prompt is past ratio (or would leave less than minCompactHeadroom tokens of
// room, whichever triggers earlier) and there's enough history to be worth it.
// A zero window disables it.
func autoCompactNeeded(ctxTokens, window, sessionLen int, ratio float64) bool {
	if window <= 0 || sessionLen < 4 {
		return false
	}
	trigger := int(float64(window) * ratio)
	if floor := window - minCompactHeadroom; floor < trigger {
		trigger = floor
	}
	if trigger < 0 {
		trigger = 0
	}
	return ctxTokens >= trigger
}

// shouldAutoCompact reports whether the running session should fold into a
// summary right now. Memory "raw" (see Config.Memory) opts out entirely — the
// session stays verbatim and only trims via the plain FIFO cap in
// Agent.remember, never through an LLM-written recap.
func (a *app) shouldAutoCompact() bool {
	if a.cfg.Memory == "raw" {
		return false
	}
	return autoCompactNeeded(a.lastRealContext, a.activeLLM().ContextWindow, a.ag.SessionLen(), compactThreshold(a.cfg.CompactThreshold))
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

// sessionFile is the on-disk shape of a saved session: the message history
// plus the one bit of app-level state worth resuming alongside it — an active
// /cd, so switching back to a session lands you back in the directory you
// were working in, not the workspace root. Older files are a bare JSON array
// of messages (the pre-Workdir format); decodeSessionFile falls back to that.
type sessionFile struct {
	History []llm.Message `json:"history"`
	Workdir string        `json:"workdir,omitempty"`
}

// decodeSessionFile parses a saved session file. A legacy bare-array file
// fails to unmarshal into the (object-shaped) sessionFile — that failure is
// the reliable signal to fall back to the old format, not a heuristic on the
// content.
func decodeSessionFile(data []byte) (sessionFile, bool) {
	var sf sessionFile
	if json.Unmarshal(data, &sf) == nil {
		return sf, true
	}
	var h []llm.Message
	if json.Unmarshal(data, &h) == nil {
		return sessionFile{History: h}, true
	}
	return sessionFile{}, false
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
	sf, ok := decodeSessionFile(data)
	if !ok {
		return
	}
	a.ag.SetHistory(sf.History)
	a.restoreWorkdir(sf.Workdir)
}

// restoreWorkdir re-applies a saved /cd — best-effort, silently keeping the
// workspace root if the directory no longer exists or now falls outside the
// jail (e.g. the project moved, or the workspace's own jail policy changed).
func (a *app) restoreWorkdir(dir string) {
	if dir == "" {
		return
	}
	abs, err := a.pol.Resolve(dir)
	if err != nil {
		return
	}
	if fi, err := os.Stat(abs); err != nil || !fi.IsDir() {
		return
	}
	if _, err := a.pol.SetWorkdir(dir); err != nil {
		return
	}
	a.workdir = abs
	a.ag.SetSystem(a.systemPrompt()) // let the model see the restored working dir
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
	a.clearGoal()         // a new session shouldn't inherit the old one's standing goal either
	return nil
}

// autoSessionName picks the next free "<base>-N" so a bare /new gets a fresh
// scratch thread without clobbering an existing one. base is the caller's
// choice of starting point: a bare /new branches off the CURRENT session name
// (continuity — you're mid-session, taking a scratch offshoot of it), while
// the startup chooser's "start new" row wants a name independent of whatever
// session was last active, not "<old session>-2" (see its call site).
func (a *app) autoSessionName(base string) string {
	base = slugName(base)
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
	sf := sessionFile{History: a.ag.History(), Workdir: a.workdirRel()}
	if data, err := json.Marshal(sf); err == nil {
		_ = atomicfile.Write(a.sessionPath(), data, 0o644)
	}
}

// workdirRel is the active /cd, relative to the workspace root, for
// persistence — portable across a workspace that gets moved or checked out at
// a different absolute path (a bare absolute a.workdir would not survive
// that). "" if no /cd is active or it can't be made relative.
func (a *app) workdirRel() string {
	if a.workdir == "" {
		return ""
	}
	rel, err := filepath.Rel(a.workspace, a.workdir)
	if err != nil {
		return ""
	}
	return rel
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
	sf, ok := decodeSessionFile(data)
	if !ok {
		return 0, time.Time{}
	}
	return len(sf.History), fi.ModTime()
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

// deleteSessionNamed removes a named session's file and its companion archive
// (and the legacy file for the default name). If it's the active thread,
// memory is cleared too.
func (a *app) deleteSessionNamed(name string) []string {
	if strings.TrimSpace(name) == "" {
		return []string{"usage: /sessions delete <name>"}
	}
	slug := slugName(name)
	removed := false
	if os.Remove(filepath.Join(a.workspace, ".agent", "sessions", slug+".json")) == nil {
		removed = true
	}
	os.Remove(filepath.Join(a.workspace, ".agent", "sessions", slug+".archive.jsonl")) // best-effort: not every session has one
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
			a.statusMu.Lock()
			a.facts = f
			a.statusMu.Unlock()
		}
	}
}

// factsCount reads how many learned facts are stored (guarded, see statusMu).
func (a *app) factsCount() int {
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	return len(a.facts)
}

// factsSnapshot returns a copy of the learned facts (guarded, see statusMu) —
// safe to read while a task's own goroutine may be appending to them
// (reflectAndStore's addFacts).
func (a *app) factsSnapshot() []string {
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	return append([]string(nil), a.facts...)
}

// clearFacts drops every learned project fact for this workspace. Facts are
// meant to be durable across sessions, but /clear is the user's explicit
// "start fresh" signal — without this, a fact learned about an abandoned
// line of work (e.g. a build command for a subdirectory that no longer
// exists) keeps leaking into the system prompt of every task after /clear.
func (a *app) clearFacts() {
	a.statusMu.Lock()
	a.facts = nil
	a.statusMu.Unlock()
	_ = os.Remove(a.factsPath())
}

// addFacts dedupe-appends learned facts (most recent maxFacts kept), persists,
// and returns the genuinely new ones.
func (a *app) addFacts(facts []string) []string {
	a.statusMu.Lock()
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
	var snapshot []string
	if len(added) > 0 {
		snapshot = append([]string(nil), a.facts...)
	}
	a.statusMu.Unlock()
	if snapshot != nil {
		if data, err := json.Marshal(snapshot); err == nil {
			_ = atomicfile.Write(a.factsPath(), data, 0o644)
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
		"\n\nToday is %s. Environment: you are running on %s; your working directory is %s. Relative paths resolve there — and by default this is a HARD JAIL: no tool (file, run's cwd, git) can reach a path outside it, an absolute path elsewhere is rejected, not silently redirected. If a task genuinely needs a different directory, say so — don't keep retrying different absolute paths or cwd values, they'll all fail the same way. Use commands that exist on this OS — on darwin prefer vm_stat/top/sw_vers over Linux-only tools like free.",
		time.Now().Format("2006-01-02"), runtime.GOOS, a.effectiveDir())
	if text != "" {
		out += "\n\n## Project instructions (from " + src + ") — follow these:\n" + text
	}
	if a.skills != nil {
		if idx := a.skills.Index(); idx != "" {
			out += "\n\n## Skills (load full instructions with the skill tool when the topic fits):\n" + idx
		}
	}
	out += a.subagentTargetsPrompt()                // dynamic roster of delegate targets (empty if none)
	if facts := a.factsSnapshot(); len(facts) > 0 { // learned project facts — keep the injected set small
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
	a.statusMu.Lock()
	a.tasks++
	a.steps += tr.Steps
	for _, m := range tr.Messages {
		if m.Role == "tool" {
			a.toolCalls++
		}
	}
	a.statusMu.Unlock()
}

// usageCounts reads the session's task/step/tool-call counters (guarded, see
// statusMu — written by recordRun from the task's own goroutine while /usage
// may read them from the UI goroutine).
func (a *app) usageCounts() (tasks, steps, toolCalls int) {
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	return a.tasks, a.steps, a.toolCalls
}

// runDuration is elapsed time since start, minus any time spent BLOCKED on an
// approval prompt in that span (waitSnapshot is a.approvalWaitNS.Load() taken
// right before start) — so a human's approval delay isn't folded into the
// run's duration and counted as model "thinking"/generating time, which would
// understate tok/s (see approveGated, internal/usage). Never negative.
func (a *app) runDuration(start time.Time, waitSnapshot int64) time.Duration {
	dur := time.Since(start)
	dur -= time.Duration(a.approvalWaitNS.Load() - waitSnapshot)
	if dur < 0 {
		dur = 0
	}
	return dur
}

// recordUsage attributes the tokens spent since the last call to today's
// provider/model bucket in the persistent ledger. Best-effort; called once a
// task (and its reflection) has finished. The client's cumulative count carries
// across re-wires, so the delta is always the work done since the prior task.
// dur is the wall-clock time the run took (0 if unknown) — folded into the
// ledger so /usage can show an approximate tokens/sec (see usage.Entry).
func (a *app) recordUsage(dur time.Duration) {
	if a.usage == nil {
		return
	}
	p, c := a.client.Usage()
	dp, dc := p-a.lastPrompt, c-a.lastCompl
	a.lastPrompt, a.lastCompl = p, c
	if dp <= 0 && dc <= 0 {
		return
	}
	a.usage.Add(today(), a.providerName(), a.activeLLM().Model, dp, dc, dur)
	a.addSessionCost(a.activeLLM().Model, dp, dc, a.priceOverrides()) // for the budget guard
	if err := a.usage.Save(); err != nil {
		slog.Warn("usage ledger save failed", "err", err)
	}
}

// addSessionCost accrues estimated spend for the budget guard. Mutex-guarded:
// parallel sub-agent spawns record their spend from their own goroutines.
// overrides must be the caller's own resolved price table, not read live from
// a.cfg here — a.cfg is reassigned wholesale by reconfigure() (/login) with
// no lock, so a background job's own goroutine (runSpawnPlan) passes its
// plan.priceOverrides, captured synchronously by resolveSpawn; the
// foreground-only callers (recordUsage, reflectAndStore) pass a.priceOverrides()
// directly since they never race reconfigure() (both are on the goroutine
// that owns a.cfg).
func (a *app) addSessionCost(model string, prompt, completion int, overrides map[string]usage.Price) {
	a.costMu.Lock()
	a.sessionCostUSD += usage.CostUSD(model, prompt, completion, overrides)
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
				if rp, rok := config.ResolveProvider(a.cfg, pp); rok && (rp.APIKey != "" || (config.IsCustomProvider(a.cfg, pp) && rp.BaseURL != "")) {
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
	start := time.Now()
	lessons, err := refl.Reflect(ctx, tr)
	dur := time.Since(start)
	if separate && a.usage != nil { // a dedicated reflect model's spend isn't in the main client —
		// recorded even on a failed call below: tokens already streamed/billed
		// before the error still cost real money and must not vanish from the
		// ledger/budget guard just because the call ultimately errored.
		if p, c := client.Usage(); p > 0 || c > 0 {
			a.usage.Add(today(), provider, model, p, c, dur)
			a.addSessionCost(model, p, c, a.priceOverrides()) // for the budget guard
			_ = a.usage.Save()
		}
	}
	if err != nil {
		slog.Warn("reflection failed", "err", err)
		return 0
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

// runOne is the plain (printing) path used in one-shot and piped modes. It
// returns a non-nil error when the task never ran at all (e.g. the initial
// model request failed) so the caller can signal a nonzero exit status
// instead of silently exiting 0.
func (a *app) runOne(ctx context.Context, goal string) error {
	if a.budgetExceeded() {
		fmt.Println(a.budgetMsg())
		return nil
	}
	a.injectJobResults()       // finished background jobs land before the model thinks
	a.maybeRewireHistoryTool() // the archive may have gained its first entry since wire()
	cp := a.beginCheckpoint(goal)
	defer a.endCheckpoint(cp)
	a.ag.SetGoalLoop(a.goalTTLFor(goal), a.cfg.GoalNudge) // judge-loop only when pursuing an explicit goal
	waitSnapshot := a.approvalWaitNS.Load()
	start := time.Now()
	tr, err := a.ag.Run(ctx, goal)
	dur := a.runDuration(start, waitSnapshot)
	if err != nil {
		a.recordUsage(dur) // the failed attempt may have burned real tokens — don't drop them
		slog.Error("run failed", "err", err)
		fmt.Fprintln(os.Stderr, "error:", err)
		return err
	}
	a.lastRealContext = tr.PromptTokens // snapshot the real fullness before reflectAndStore (below) can clobber the shared client's own Context()
	a.recordRun(tr)
	a.finishGoal(goal, tr)
	a.recordUsage(dur)
	a.saveSession() // the conversation is decided now — save it before the slower,
	// best-effort reflection pass below, which the process could be interrupted
	// during (e.g. a signal mid-reflection) without losing this turn's real output
	if strings.TrimSpace(tr.Final) != "" {
		fmt.Println(tr.Final)
	} else {
		fmt.Println("(no final answer — step budget exhausted)")
	}
	if !tr.Stopped {
		reflStart := time.Now()
		learned := a.reflectAndStore(ctx, tr)
		// Reflection is itself a real LLM call (or two, on a dedicated client) —
		// flush its tokens into the ledger/budget guard right now, before control
		// returns anywhere a next task (which may never come — one-shot mode, or
		// /exit right after) would otherwise be the only thing left to pick them up.
		a.recordUsage(time.Since(reflStart))
		if learned > 0 {
			fmt.Fprintf(os.Stderr, "(learned %d new lesson(s))\n", learned)
		}
	}
	a.detectContextWindow() // the model is loaded now — confirm the real window
	if a.shouldAutoCompact() {
		compactStart := time.Now()
		n, err := a.ag.Compact(ctx)
		a.recordUsage(time.Since(compactStart)) // compaction is a real LLM call too — flush it now, even on failure
		if err == nil && n > 0 {
			a.saveSession()
			fmt.Fprintf(os.Stderr, "(auto-compacted %d messages to free context)\n", n)
		}
	}
	return nil
}

// runTaskStreaming is the TUI path: no printing — progress reaches the screen via
// the UI tracer. Errors surface as an "error" event.
func (a *app) runTaskStreaming(ctx context.Context, goal string, epoch int64) {
	if a.budgetExceeded() {
		a.emit("error", map[string]any{"text": a.budgetMsg()})
		return
	}
	// maybeRewireHistoryTool is NOT called here: runTaskStreaming runs on the
	// tea.Cmd's own goroutine, and wire() (which it can trigger) reassigns
	// a.client/a.ag with no lock while the UI reads them live — see
	// maybeRewireHistoryTool's doc. runTask/runLoop (tui.go) call it
	// synchronously before this goroutine even starts.
	a.injectJobResults() // finished background jobs land before the model thinks
	cp := a.beginCheckpoint(goal)
	defer a.endCheckpoint(cp)
	a.ag.SetGoalLoop(a.goalTTLFor(goal), a.cfg.GoalNudge) // judge-loop only when pursuing an explicit goal
	waitSnapshot := a.approvalWaitNS.Load()
	start := time.Now()
	tr, err := a.ag.Run(ctx, goal)
	dur := a.runDuration(start, waitSnapshot)
	if a.taskEpoch.Load() != epoch {
		return // force-detached mid-run — its results belong to a run the UI abandoned
	}
	if err != nil {
		a.recordUsage(dur) // the failed attempt may have burned real tokens — don't drop them
		a.emit("error", map[string]any{"text": err.Error()})
		return
	}
	a.lastRealContext = tr.PromptTokens // snapshot the real fullness before reflectAndStore (below) can clobber the shared client's own Context()
	a.recordRun(tr)
	a.finishGoal(goal, tr)
	a.recordUsage(dur)
	a.saveSession() // the conversation is decided now — save it before the slower,
	// best-effort reflection below. That matters because /exit quits immediately even
	// while state is still "busy" (reflecting) rather than making the user wait for it
	// (see commandWhileBusy) — saving here first means the just-finished exchange (the
	// part the user actually sees and cares about) reaches disk before that race even
	// becomes possible, instead of depending on reflection finishing first.
	if !tr.Stopped { // reflect only on a clean finish, not on any premature stop
		reflStart := time.Now()
		a.reflectAndStore(ctx, tr)
		// Flush reflection's own tokens now — same reasoning as runOne's matching
		// flush (see there): a next task's recordUsage delta may never come (the
		// process could exit right after), so this can't wait for that.
		a.recordUsage(time.Since(reflStart))
	}
}

func (a *app) repl(ctx context.Context) {
	fmt.Printf("ipsupport-code %s — type a task, or /help for commands.\n", version)
	if n := a.startupNotice(ctx); n != "" {
		fmt.Fprintln(os.Stderr, n)
	}
	for {
		fmt.Print("\n> ")
		line, err := a.stdin.readCmdLine()
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
			name, persist = a.autoSessionName(a.cfg.Name), false
		}
		hadContent := a.ag.SessionLen() > 0 // an empty session isn't saved — don't claim it's in /sessions
		if err := a.newNamedSession(name, persist); err != nil {
			fmt.Println("could not start session:", err)
		} else if hadContent {
			fmt.Printf("started a new session %q — the previous one is in /sessions\n", a.cfg.Name)
		} else {
			fmt.Printf("started a new session %q\n", a.cfg.Name)
		}
	case "/reset", "/clear": // wipe THIS thread
		a.ag.Reset()
		a.resetSessionAllow()
		a.clearFacts()
		a.ag.SetSystem(a.systemPrompt())
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
		fmt.Println(a.mcpList(ctx, a.cfg.McpServers))
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
	case "/snip":
		if act := a.snip(rest); act.recall != "" {
			fmt.Println(act.recall)
		} else {
			printLines(act.lines)
		}
	case "/steer": // in plain mode a task runs synchronously, so this always steers the NEXT run
		printLines(a.steerCommand(rest, false))
	case "/btw": // side question — plain mode has no running task, so answer it now
		if strings.TrimSpace(rest) == "" {
			printLines([]string{"usage: /btw <question> — a quick answer from the conversation, no tools"})
		} else {
			base := append([]llm.Message{llm.System(a.ag.System())}, a.ag.History()...)
			printLines([]string{"✦ by the way:", a.ag.AnswerAside(ctx, base, rest)})
		}
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
// you're not limited to the built-in templates: /ai add <name> <base_url> [model]
// [key=<token>]. Give key= to add and key it in one step; otherwise follow with
// /ai key <name> <token>. Then /ai <name> to switch.
func (a *app) addProvider(arg string) []string {
	// Pull an optional key=<token> out of the args first, so a key can be given in
	// the SAME command — and so a token can't be silently mistaken for the model.
	var key string
	var f []string
	for _, w := range strings.Fields(arg) {
		if v, ok := strings.CutPrefix(w, "key="); ok {
			key = v
			continue
		}
		f = append(f, w)
	}
	if len(f) < 2 {
		return []string{"usage: /ai add <name> <base_url> [model] [key=<token>]",
			"  e.g. /ai add mylab https://api.lab.co/v1 llama-3.1-8b key=sk-...",
			"  (a built-in provider just needs a key: /ai key <name> <token>)"}
	}
	name, url := f[0], f[1]
	if name == "local" {
		return []string{`"local" is reserved — it's your LM Studio endpoint (/config to change it)`}
	}
	if _, ok := config.ProviderTemplates[name]; ok {
		return []string{fmt.Sprintf("%q is a built-in provider — set its key in one step: /ai key %s <token>", name, name)}
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return []string{"base_url must start with http:// or https:// — got " + url}
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[name]
	p.BaseURL = strings.TrimRight(url, "/")
	if len(f) >= 3 {
		p.Model = f[2]
	}
	if key != "" {
		p.APIKey = key
	}
	a.cfg.Providers[name] = p
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		return []string{"error: " + err.Error()}
	}
	if key != "" {
		if a.cfg.Provider == name {
			_ = a.wire()
		}
		return []string{fmt.Sprintf("added %q → %s with a key — /ai %s to use it", name, p.BaseURL, name)}
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
	a.modelEpoch.Add(1) // orphan any in-flight probe for the old provider/model
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
	a.modelEpoch.Add(1) // orphan any in-flight probe for the old model
	if err := a.wire(); err != nil {
		return []string{"error: " + err.Error()}
	}
	a.maybeDetectWindowSync() // REPL only — the TUI re-detects off-thread
	return []string{"model → " + name}
}

// toggleLoopDetection flips DisableLoopDetection for the CURRENTLY ACTIVE
// provider's connection and persists it — same local-vs-named-provider
// branching as setModel. On by default everywhere; lets a trusted, capable
// hosted provider (Claude, OpenAI) opt out of the local-model-oriented
// repetition detectors per-connection.
func (a *app) toggleLoopDetection() error {
	if a.isLocal() {
		a.cfg.LLM.DisableLoopDetection = !a.cfg.LLM.DisableLoopDetection
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.DisableLoopDetection = !p.DisableLoopDetection
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
}

// setTemperature sets the sampling temperature for the CURRENTLY ACTIVE
// provider's connection and persists it — same local-vs-named-provider
// branching as setModel. 0 means "unset" (server default — see Chat's
// c.temp > 0 gate).
func (a *app) setTemperature(v float64) error {
	if a.isLocal() {
		a.cfg.LLM.Temperature = v
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.Temperature = v
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
}

// setTopP sets nucleus-sampling top_p for the CURRENTLY ACTIVE provider's
// connection and persists it — same branching as setTemperature. 0 means
// "unset" (server default).
func (a *app) setTopP(v float64) error {
	if a.isLocal() {
		a.cfg.LLM.TopP = v
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.TopP = v
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
}

// setMaxOutputTokens sets the request's max_tokens for the CURRENTLY ACTIVE
// provider's connection and persists it — same local-vs-named-provider
// branching as setTemperature/setTopP. 0 means "unset" (the server's own
// default, which can be too small for a reasoning model's thinking phase —
// see Config.LLM.MaxOutputTokens).
func (a *app) setMaxOutputTokens(v int) error {
	if a.isLocal() {
		a.cfg.LLM.MaxOutputTokens = v
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.MaxOutputTokens = v
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
}

// setIdleTimeout sets the idle watchdog (seconds with NO response/stream data
// before a request is treated as a hiccup and retried) for the CURRENTLY
// ACTIVE provider's connection and persists it — same local-vs-named-provider
// branching as setTemperature/setTopP. 0 means "unset" (the client's built-in
// 90s default).
func (a *app) setIdleTimeout(v int) error {
	if a.isLocal() {
		a.cfg.LLM.IdleTimeoutSeconds = v
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.IdleTimeoutSeconds = v
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
}

// setContextWindow sets the model's context size (tokens) for the CURRENTLY
// ACTIVE provider's connection and persists it — same local-vs-named-provider
// branching as setTemperature/setTopP/setIdleTimeout. A nonzero value also
// marks the window as already "detected", so the next task's best-effort
// auto-detect (detectContextWindow) doesn't silently overwrite a deliberate
// manual override; setting it back to 0 clears that, letting auto-detect run
// again.
func (a *app) setContextWindow(v int) error {
	a.windowDetected = v > 0
	if a.isLocal() {
		a.cfg.LLM.ContextWindow = v
		a.cfg.LLM.ContextWindowManual = v > 0 // persisted twin of windowDetected — survives a restart, see ContextWindowManual
		return config.SaveGlobal(a.cfg.Name, a.cfg.LLM)
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	p := a.cfg.Providers[a.cfg.Provider]
	p.ContextWindow = v
	p.ContextWindowManual = v > 0
	a.cfg.Providers[a.cfg.Provider] = p
	return config.SaveProviders(a.cfg.Provider, a.cfg.Providers)
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
  /steer <note>    fold a note into a RUNNING task without stopping it (esc cancels)
  /btw <question>  ask a quick side question mid-task — one-turn answer, no tools, task keeps going
  /snip [name]     prompt templates: /snip <name> recalls into the input · save <name> [text] · list · rm <name>
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
  ctrl+r history search · shift+tab plan/auto · ctrl+u clear input · esc cancel (twice = force-detach a stuck task)

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
		a.cfg.KBPath, len(a.kb.All()), a.factsCount(), a.cfg.TracePath)
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
	g := a.goalSnapshot()
	if g.Text == "" {
		return "(none) · " + ttl
	}
	return fmt.Sprintf("%s [%s] · %s", oneLine(g.Text, 50), g.Status, ttl)
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
	tasks, steps, toolCalls := a.usageCounts()
	var b strings.Builder
	fmt.Fprintf(&b, `usage (this session):
  tasks       %d
  steps       %d
  tool calls  %d
  tokens      %d prompt + %d completion = %d
  lessons     %d in knowledge base
`, tasks, steps, toolCalls, p, c, p+c, len(a.kb.All()))
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
		v := humanK(t.Tokens()) + " tok"
		if rate := t.TokensPerSec(); rate > 0 {
			v += fmt.Sprintf("  (%.1f tok/s)", rate)
		}
		days = append(days, [2]string{t.Key, v})
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
		if rate := t.TokensPerSec(); rate > 0 {
			v += fmt.Sprintf("  (%.1f tok/s)", rate)
		}
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
	hasLocal := def.Provider == "" || def.Provider == "local" // an already-configured cloud provider flips the default
	if askYN(reader, "Local model server running (LM Studio / Ollama / vLLM)?", hasLocal) {
		initLocalModel(reader, def)
	} else {
		initCloudProvider(reader, def)
	}
}

// initLocalModel configures the built-in "local" provider — LM Studio by
// default, but any OpenAI-compatible local server (Ollama, vLLM…) works the
// same way by pointing the URL at it.
func initLocalModel(reader *bufio.Reader, def config.Config) {
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
	if err := config.SaveLocalModel(def.Name, l); err != nil { // preserve any custom name; also (re)activates "local"
		slog.Warn("could not save config", "err", err)
		return
	}
	fmt.Printf("Saved to %s\n", config.GlobalPath())
	if probeErr != nil {
		fmt.Printf("  ⚠ couldn't reach %s yet — start LM Studio's local server (Developer tab); it'll connect on your first task.\n", url)
	} else {
		fmt.Printf("  ✓ connected — using %s\n", l.Model)
	}
	finishInit()
}

// initCloudProvider configures a cloud provider — the path for a machine with
// no local model server at all (the common case on a plain VPS or a fresh
// laptop without LM Studio installed). A name matching a built-in template
// (openai, anthropic, …) uses that vendor's real endpoint; any other name is a
// CUSTOM OpenAI-compatible endpoint (your own gateway, LiteLLM, a proxy…) and
// additionally asks for its Base URL — mirroring `/ai add`'s existing runtime
// rule exactly, so picking "openai" always means the real api.openai.com and
// anything else is never silently bound to it.
func initCloudProvider(reader *bufio.Reader, def config.Config) {
	names := config.KnownProviders()
	defName := def.Provider
	if defName == "" || defName == "local" {
		defName = "openai"
	}
	fmt.Printf("  Built-in providers: %s — or type any other name for your OWN OpenAI-compatible endpoint (a self-hosted gateway, LiteLLM, a proxy…).\n", strings.Join(names, ", "))
	name := ask(reader, "Provider", defName)
	if name == "local" { // reserved for the local-model slot — that's the OTHER branch of setup
		fmt.Println(`  "local" is reserved for a local model server — restart setup and answer "Y" to the first question instead. Using ` + defName + " for now.")
		name = defName
	}

	_, isTemplate := config.ProviderTemplates[name]
	existing, _ := config.ResolveProvider(def, name) // any already-saved key/model, or the env var
	l := config.LLM{}
	if !isTemplate {
		fmt.Printf("  %q isn't a built-in — setting it up as a custom OpenAI-compatible endpoint (built-ins: %s).\n", name, strings.Join(names, ", "))
		url := ask(reader, "Base URL", existing.BaseURL)
		if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
			fmt.Printf("  ⚠ that doesn't look like a URL (want http:// or https://) — fix it later with: /ai add %s <base_url>\n", name)
		}
		l.BaseURL = strings.TrimRight(url, "/")
	}
	key := ask(reader, "API key", existing.APIKey)
	if key == "" {
		fmt.Printf("  ⚠ no key set — add one later with: /ai key %s <token>\n", name)
	}
	l.APIKey = key
	l.Model = ask(reader, "Model", existing.Model)

	providers := def.Providers
	if providers == nil {
		providers = map[string]config.LLM{}
	}
	providers[name] = l
	if err := config.SaveProviders(name, providers); err != nil {
		slog.Warn("could not save config", "err", err)
		return
	}
	fmt.Printf("Saved to %s\n", config.GlobalPath())
	resolved, _ := config.ResolveProvider(config.Config{Provider: name, Providers: providers}, name)
	fmt.Printf("  → using %s · %s · model %s\n", name, resolved.BaseURL, resolved.Model)
	finishInit()
}

func finishInit() {
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

// askYN asks a yes/no question; Enter (or unreadable input, e.g. piped EOF)
// keeps defYes.
func askYN(r *bufio.Reader, label string, defYes bool) bool {
	hint := "y/N"
	if defYes {
		hint = "Y/n"
	}
	fmt.Printf("  %s [%s]: ", label, hint)
	line, err := r.ReadString('\n')
	if err != nil {
		return defYes
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "":
		return defYes
	case "y", "yes":
		return true
	case "n", "no":
		return false
	default:
		return defYes
	}
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
// category short-circuits the prompt, otherwise it asks the real approver. ctx is
// the caller's own context (a job's, for a background sub-agent).
func (a *app) approveGated(ctx context.Context, kind, detail string) bool {
	if a.sessionAllowed(kind) {
		return true
	}
	start := time.Now()
	ok := a.approver.Approve(ctx, kind, detail)
	a.approvalWaitNS.Add(int64(time.Since(start)))
	return ok
}

// gatedApprover adapts approveGated to the tool.Approver interface.
type gatedApprover struct{ app *app }

func (g gatedApprover) Approve(ctx context.Context, kind, detail string) bool {
	return g.app.approveGated(ctx, kind, detail)
}

// stdinApprover prompts the operator on stderr for a policy "ask" decision. The
// mutex serializes prompts so concurrent tool-call approvals never read a line
// meant for each other; `stdin` (a *stdinOwner, shared with the plain REPL's
// command loop) additionally keeps this from ever racing the loop's OWN read
// of the same underlying reader — see stdinOwner. `a` grants the kind's
// category for the whole session (via the app's session-allow set). The
// underlying stdin read can't be interrupted (there's no way to cancel a
// blocking ReadString), so Approve runs it on a helper goroutine and races it
// against ctx.Done(), denying and returning promptly on cancellation — see
// below.
type stdinApprover struct {
	mu    sync.Mutex
	stdin *stdinOwner
	app   *app
}

// Approve blocks until the operator answers or ctx is cancelled (e.g.
// Ctrl-C), whichever comes first. The actual stdin read happens on a
// detached goroutine: if ctx is cancelled first, Approve denies and returns
// immediately WITHOUT waiting for that goroutine, which may then sit blocked
// on readApproveLine forever — same as any real interactive terminal read
// nobody's listening for the result of anymore. That's fine: this process is
// exiting shortly anyway (SIGINT), and forcibly closing stdin here would break
// the REPL's own later reads.
func (s *stdinApprover) Approve(ctx context.Context, kind, detail string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(os.Stderr, "\n[approve %s] %s\n  allow? [y/N/a=all %s this session] ", kind, detail, categoryLabel(approvalCategory(kind)))

	result := make(chan lineResult, 1) // buffered: the read goroutine must never block sending here, even if abandoned
	go func() {
		line, err := s.stdin.readApproveLine()
		result <- lineResult{line, err}
	}()

	var res lineResult
	select {
	case res = <-result:
	case <-ctx.Done():
		return false
	}
	if res.err != nil {
		return false
	}
	switch strings.TrimSpace(strings.ToLower(res.line)) {
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

// stdinOwner is the ONLY goroutine ever allowed to call ReadString on the
// process's shared plain-mode stdin reader. Two independent call sites need a
// line from it: the plain REPL's command loop (readCmdLine, ~repl()'s main
// loop) and stdinApprover's y/n prompt (readApproveLine), which a detached
// background job's own goroutine can trigger at any time — including the exact
// moment the REPL loop is itself blocked waiting for the next command. Two
// goroutines calling ReadString on the same *bufio.Reader concurrently is a
// data race on its internal buffer (undefined behavior — the read meant for
// one caller can be silently handed to the other, or worse); routing both
// through this owner means at most one goroutine ever touches the reader.
//
// run reads one line, THEN picks a recipient — so a request that only shows up
// mid-read still gets first crack at that line once it completes. Between the
// two, an approval always wins over a merely-queued command read (see
// pickRecipient), mirroring the TUI's own modal "approval takes the keys"
// behavior: a background job asking the operator something right now takes
// priority over whatever the next typed command would have been.
// The reader goroutine starts lazily, on the first actual request (see start):
// build() constructs a stdinOwner unconditionally, before the caller knows
// whether this run will end up in the TUI or the plain REPL. In TUI mode
// a.approver is replaced with the TUI's own channel-based approver (see
// tui.go) before any request can reach this one, so readCmdLine/readApproveLine
// are never called there — but an eagerly-started run() doesn't know that: it
// would sit forever in ReadString on the process's real os.Stdin, racing
// bubbletea's own raw-mode reader for the same bytes and silently stealing
// keystrokes the TUI never sees.
type stdinOwner struct {
	r          *bufio.Reader
	cmdReq     chan chan lineResult
	approveReq chan chan lineResult
	once       sync.Once
}

// lineResult is one ReadString('\n') outcome, delivered to whichever request
// (command or approval) wins that read.
type lineResult struct {
	line string
	err  error
}

func newStdinOwner(r *bufio.Reader) *stdinOwner {
	return &stdinOwner{r: r, cmdReq: make(chan chan lineResult, 1), approveReq: make(chan chan lineResult, 1)}
}

// start launches the sole reader goroutine on first use; a no-op thereafter.
func (o *stdinOwner) start() {
	o.once.Do(func() { go o.run() })
}

// run is the sole goroutine that touches r, for the life of the process.
func (o *stdinOwner) run() {
	for {
		line, err := o.r.ReadString('\n')
		o.pickRecipient() <- lineResult{line, err}
	}
}

// pickRecipient blocks until the command loop or a pending approval wants the
// line just read, favoring an approval whenever both are waiting at once.
func (o *stdinOwner) pickRecipient() chan lineResult {
	select {
	case reply := <-o.approveReq:
		return reply
	default:
	}
	select {
	case reply := <-o.approveReq:
		return reply
	case reply := <-o.cmdReq:
		return reply
	}
}

// readCmdLine reads the plain REPL's next command line.
func (o *stdinOwner) readCmdLine() (string, error) { return o.readVia(o.cmdReq) }

// readApproveLine reads a background job's approval answer, taking priority
// over a command line the REPL loop may be waiting on (see pickRecipient).
func (o *stdinOwner) readApproveLine() (string, error) { return o.readVia(o.approveReq) }

func (o *stdinOwner) readVia(req chan chan lineResult) (string, error) {
	o.start()
	reply := make(chan lineResult, 1)
	req <- reply
	res := <-reply
	return res.line, res.err
}

// logLevel is the log threshold: warnings and up by default (so retries/errors
// are visible), tuned by IPS_LOG=debug|info|warn|error.
func logLevel() slog.Level {
	switch strings.ToLower(os.Getenv("IPS_LOG")) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "error":
		return slog.LevelError
	}
	return slog.LevelWarn
}

func setupLogging() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel()})))
}

// redirectLogToFile points the logger at config.LogPath() instead of stderr, so
// TUI runs don't corrupt the alt-screen with raw log lines. Returns a closer, or
// nil (leaving stderr logging) if the file can't be opened. Tail the file to
// watch warnings/retries live.
func redirectLogToFile() func() {
	_ = os.MkdirAll(filepath.Dir(config.LogPath()), 0o755) // config dir may not exist on first run
	f, err := os.OpenFile(config.LogPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		slog.Warn("could not open log file — logging to stderr", "path", config.LogPath(), "err", err)
		return nil
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(f, &slog.HandlerOptions{Level: logLevel()})))
	return func() { f.Close() }
}

func newRunID() string {
	return fmt.Sprintf("%s-%d", time.Now().UTC().Format("20060102T150405"), os.Getpid())
}
