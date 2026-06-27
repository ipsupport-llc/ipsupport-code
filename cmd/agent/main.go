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
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
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
	flag.BoolVar(&doInit, "init", false, "re-run first-time setup (server URL, model)")
	flag.BoolVar(&showVersion, "version", false, "print version and exit")
	flag.BoolVar(&dumpPrompt, "dump-prompt", false, "print the built-in system prompt and exit (e.g. > .agent/system.md to start editing)")
	flag.BoolVar(&newSession, "new", false, "start a fresh session (don't restore the saved one)")
	flag.StringVar(&sessionName, "session", "", "use a named session (a separate saved thread)")
	flag.Parse()
	if showVersion {
		fmt.Println("ipsupport-code", version)
		return
	}
	if dumpPrompt {
		fmt.Println(agent.DefaultSystemPrompt())
		return
	}
	if args := flag.Args(); len(args) >= 1 && args[0] == "update" {
		runUpdate(args[1:])
		return
	}
	setupLogging()

	reader := bufio.NewReader(os.Stdin)
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
	approver   tool.Approver // stdin (plain) or the TUI bridge

	client          *llm.OpenAIClient
	ag              *agent.Agent
	pol             *policy.Engine // host policy/jail; sub-agents in a dir get their own
	subReg          *tool.Registry // tools for sub-agents (no `agent` tool → no recursion)
	spawnMu         sync.Mutex     // serializes spawn-approval prompts during parallel fan-out
	spawnSeq        atomic.Int64   // unique id per sub-agent spawn (for grouping its UI events)
	refl            *reflect.Reflector
	instrSrc        string   // project instructions file in effect, "" if none
	promptSrc       string   // "built-in" or the system.md override path
	facts           []string // durable project facts learned over time (per workspace)
	planMode        bool     // plan (propose) vs auto (execute); survives re-wire
	windowDetected  bool     // got the real loaded context window (vs a default/guess)
	sessionRestored bool     // a saved session was restored at startup (TUI renders a recap)
	tui             bool     // running the TUI (detect the context window off-thread, not inline)
	startNew        bool     // -new: skip the startup chooser, begin a fresh session

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

	cleanup := func() {}
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, usage: usageStore, skills: skills, reader: reader, approver: &stdinApprover{r: reader}}
	a.applyUsageRetention() // honor usage_retention_days on startup
	if ft, err := trace.NewFileTracer(cfg.TracePath, newRunID()); err != nil {
		slog.Warn("trace disabled", "err", err)
	} else {
		a.fileTracer = ft
		cleanup = func() { _ = ft.Close() }
	}
	a.loadFacts() // learned project facts → folded into the prompt by wire()
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
	p, ok := a.cfg.Agents[profile]
	if !ok {
		return "", fmt.Errorf("unknown profile %q — configured: %s", profile, a.profilesOrHint())
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
	subReg, subWorkspace := a.subReg, a.workspace
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
		approved := a.approver.Approve("spawn agent", fmt.Sprintf("%s · %s · %s — %s", profile, llmCfg.Model, subWorkspace, oneLine(task, 60)))
		a.spawnMu.Unlock()
		if !approved {
			return "", fmt.Errorf("spawn denied by user")
		}
	}

	id := fmt.Sprintf("sub%d", a.spawnSeq.Add(1)) // groups this sub-agent's UI events
	client := llm.NewOpenAIClient(llmCfg)
	sub := agent.New(client, subReg, a.kb, a.tracer, a.subAgentPrompt(subWorkspace, p.Prompt), llmCfg.MaxSteps)
	sub.SetPlanMode(a.planMode) // inherit the current mode
	sub.SetLabel(id)
	a.emit("subagent", map[string]any{"agent": id, "profile": profile, "provider": provider, "model": llmCfg.Model, "dir": subWorkspace, "task": oneLine(task, 80)})

	tr, err := sub.Run(ctx, task)
	if a.usage != nil { // the sub-agent's spend counts too
		pt, ct := client.Usage()
		a.usage.Add(today(), provider, llmCfg.Model, pt, ct)
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
		dir = filepath.Join(a.workspace, dir)
	}
	return filepath.Clean(dir), nil
}

// buildSubReg builds a sub-agent's tool registry against a given policy: the host
// tools except `agent` (so sub-agents can't recurse) and except help (a usage
// helper they don't need). The run (shell) tool is included only when spawn.exec
// is on — the sharpest capability to hand an autonomous sub-agent.
func (a *app) buildSubReg(pol *policy.Engine) *tool.Registry {
	tools := []tool.Tool{tool.NewFile(pol, a.approver)}
	if a.cfg.Spawn.Exec {
		tools = append(tools, tool.NewRun(pol, a.approver, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second))
	}
	tools = append(tools, tool.NewGit(pol, a.approver), tool.NewWeb(http.DefaultClient), tool.NewCalc())
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
	for _, n := range names {
		p := a.cfg.Agents[n]
		m := p.Model
		if m == "" {
			m = "default"
		}
		parts = append(parts, fmt.Sprintf("%s→%s·%s", n, p.Provider, m))
	}
	return "\n\n## Sub-agents you can delegate to (the agent tool)\n" +
		"Profiles (call agent with profile=<name>): " + strings.Join(parts, ", ") + "\n" +
		"Pass dir=<path> to point one at another project. Fan a task out across several profiles in one turn — they run in parallel — then merge their findings into one answer."
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
	case "rm", "remove", "del", "delete":
		return a.agentsRemove(arg)
	case "exec":
		return a.agentsExec(arg)
	default:
		return []string{"usage: /agents [add <name> <provider> [model]] [rm <name>] [exec on|off]"}
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
	out = append(out, "  /agents add <name> <provider> [model] · /agents rm <name> · /agents exec on|off")
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
	if rp.APIKey == "" {
		return config.LLM{}, fmt.Sprintf("%s has no API key — add one with /ai key %s <token> first", provider, provider)
	}
	return rp, ""
}

// configuredProviderNames is local plus external providers that have a key.
func (a *app) configuredProviderNames() []string {
	out := []string{"local"}
	for _, n := range config.KnownProviders() {
		if l, ok := config.ResolveProvider(a.cfg, n); ok && l.APIKey != "" {
			out = append(out, n)
		}
	}
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
func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	out, _ := textutil.Clip(s, max)
	return out
}

// providerModel is the "provider · model" label shown in the status line.
func (a *app) providerModel() string { return a.providerName() + " · " + a.activeLLM().Model }

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
func (a *app) wire() error {
	pol, err := policy.New(a.cfg)
	if err != nil {
		return fmt.Errorf("policy: %w", err)
	}
	a.pol = pol // host jail; a sub-agent pointed at a dir gets its own
	var reg *tool.Registry
	tools := []tool.Tool{
		tool.NewFile(pol, a.approver),
		tool.NewRun(pol, a.approver, time.Duration(a.cfg.Run.TimeoutSeconds)*time.Second),
		tool.NewGit(pol, a.approver),
		tool.NewWeb(http.DefaultClient),
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
		tools = append(tools, tool.NewAgent(a.spawnAgent))
	}
	reg = tool.NewRegistry(tools...)
	a.tracer = trace.Multi(a.fileTracer, a.uiTracer)
	// Carry the session's running token total into the rebuilt client so a
	// /skills, /permissions or /login re-wire doesn't zero the counter.
	var seedP, seedC int
	if a.client != nil {
		seedP, seedC = a.client.Usage()
	}
	a.client = llm.NewOpenAIClient(a.activeLLM())
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
	a.ag = agent.New(a.client, reg, a.kb, a.tracer, a.systemPrompt(), a.activeLLM().MaxSteps)
	a.ag.SetHistory(prior)
	a.ag.SetPlanMode(a.planMode) // carry the mode into the rebuilt agent
	a.refl = reflect.New(a.client)
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
	a.ag.Reset() // fresh — don't load name's prior thread
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
		"\n\nEnvironment: you are running on %s; the workspace is %s. Use commands that exist on this OS — on darwin prefer vm_stat/top/sw_vers over Linux-only tools like free.",
		runtime.GOOS, a.workspace)
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
	if err := a.usage.Save(); err != nil {
		slog.Warn("usage ledger save failed", "err", err)
	}
}

func today() string { return time.Now().Format("2006-01-02") }

// reflectAndStore runs the post-task reflection and persists new lessons,
// emitting a "lesson" event for each. Returns how many were new.
func (a *app) reflectAndStore(ctx context.Context, tr agent.Transcript) int {
	lessons, err := a.refl.Reflect(ctx, tr)
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
	if learned > 0 {
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
	tr, err := a.ag.Run(ctx, goal)
	if err != nil {
		slog.Error("run failed", "err", err)
		fmt.Fprintln(os.Stderr, "error:", err)
		return
	}
	a.recordRun(tr)
	if strings.TrimSpace(tr.Final) != "" {
		fmt.Println(tr.Final)
	} else {
		fmt.Println("(no final answer — step budget exhausted)")
	}
	if learned := a.reflectAndStore(ctx, tr); learned > 0 {
		fmt.Fprintf(os.Stderr, "(learned %d new lesson(s))\n", learned)
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
	tr, err := a.ag.Run(ctx, goal)
	if err != nil {
		a.emit("error", map[string]any{"text": err.Error()})
		return
	}
	a.recordRun(tr)
	a.reflectAndStore(ctx, tr)
	a.recordUsage()
	a.saveSession()
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
		switch {
		case line == "":
			continue
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
		for _, l := range a.agentsCommand(ctx, rest) {
			fmt.Println(l)
		}
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
		runUpdate(strings.Fields(rest))
	case "/ai":
		for _, l := range a.aiCommand(rest) {
			fmt.Println(l)
		}
	case "/model":
		for _, l := range a.modelCommand(ctx, rest) {
			fmt.Println(l)
		}
	case "/config":
		for _, l := range a.configOverview() {
			fmt.Println(l)
		}
	case "/shell", "/sh":
		a.runShell(ctx)
	case "/skills":
		for _, l := range a.skillsCommand(ctx, rest) {
			fmt.Println(l)
		}
	case "/permissions", "/perms":
		for _, l := range a.permissionsCommand(rest) {
			fmt.Println(l)
		}
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
	if sub == "key" {
		name, tok := splitCommand(arg)
		return a.setProviderKey(name, tok)
	}
	return a.setProvider(sub)
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
	for _, n := range config.KnownProviders() {
		l, _ := config.ResolveProvider(a.cfg, n)
		note := "no key"
		if l.APIKey != "" {
			note = "key set"
		}
		row(n, l.BaseURL, note)
	}
	return append(out, "  /ai <name> switch · /ai key <name> <token> · /model pick model")
}

func (a *app) setProvider(name string) []string {
	if name != "local" {
		l, ok := config.ResolveProvider(a.cfg, name)
		if !ok {
			return []string{fmt.Sprintf("unknown provider %q — try: local, %s", name, strings.Join(config.KnownProviders(), ", "))}
		}
		if l.APIKey == "" {
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
	if _, ok := config.ProviderTemplates[name]; !ok {
		return []string{fmt.Sprintf("unknown provider %q — keys are for: %s", name, strings.Join(config.KnownProviders(), ", "))}
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
	default:
		return []string{"usage: /permissions [files on|off] [run on|off] [agents on|off]"}
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
	return []string{
		"permissions:",
		fmt.Sprintf("  files   %s   (jail %q)", a.cfg.File.Default, a.cfg.File.Jail),
		fmt.Sprintf("  run     %s", a.cfg.Run.Default),
		fmt.Sprintf("  agents  %s   (sub-agent shell exec: %s — set in /config)", a.cfg.Spawn.Default, exec),
		"  deny floor (always on): secrets, .git, .env, rm -rf, sudo, …",
		"  /permissions files on  → stop asking before file writes in the workspace",
		"  /permissions agents on → spawn sub-agents without asking each time",
	}
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
  /login           (re)configure the server URL / model / key, then reload
  /new [name]      start a NEW session (the old one stays in /sessions)
  /clear           wipe this session's context (same session)
  /compact         summarize the session so far to free up context
  /plan, /auto     plan mode (propose only) vs auto mode (execute)
  /ai [name]       switch AI provider (local|openai|grok|groq|openrouter); /ai key <name> <tok>
  /model [name]    list the provider's models, or pick one
  /config          control panel: all settings + how to change them
  /update [chan]   self-update from GitHub (chan = stable|nightly, saved)
  /shell, /sh      drop to a shell in the workspace (exit to return)
  /skills          list/toggle/install on-demand instruction packs
  /agents          sub-agent profiles: /agents add|rm|exec (models the agent tool delegates to)
  /permissions     relax approval for file / shell / sub-agent-spawn actions
  /color [name]    change the TUI frame color (cycles if no name)
  /rename <name>   rename the agent (saved in settings)
  /sessions        list / switch / delete saved sessions (per agent name)
  /loop <ival> <task>  re-run a task on an interval (e.g. /loop 5m <task>, /loop 30s x10 <task>; esc stops)
  /help            this list
  /exit, /quit     leave
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
  workspace    %s
  jail         %q
  defaults     run=%s  file=%s
  prompt       %s
  instructions %s
  session      %d messages
  knowledge    %s (%d lessons)
  trace        %s
`,
		version, channelOf(a.cfg), a.providerName(),
		act.BaseURL, act.Model, act.MaxSteps,
		a.cfg.Workspace, a.cfg.File.Jail, a.cfg.Run.Default, a.cfg.File.Default,
		promptOrDefault(a.promptSrc), instr, a.ag.SessionLen(),
		a.cfg.KBPath, len(a.kb.All()), a.cfg.TracePath)
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
	fmt.Println("Setup — configure your model connection (press Enter to keep current).")
	l := config.LLM{
		BaseURL:       ask(reader, "LM Studio server URL", def.LLM.BaseURL),
		Model:         ask(reader, "Model name", def.LLM.Model),
		Temperature:   def.LLM.Temperature,
		MaxSteps:      atoiOr(ask(reader, "Max steps per task", strconv.Itoa(def.LLM.MaxSteps)), def.LLM.MaxSteps),
		ContextWindow: atoiOr(ask(reader, "Context window in tokens (0 = no auto-compact)", strconv.Itoa(def.LLM.ContextWindow)), def.LLM.ContextWindow),
		APIKey:        ask(reader, "API key (blank for LM Studio)", def.LLM.APIKey),
	}
	if err := config.SaveGlobal(def.Name, l); err != nil { // preserve any custom name
		slog.Warn("could not save config", "err", err)
		return
	}
	fmt.Printf("Saved to %s\n\n", config.GlobalPath())
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

func atoiOr(s string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return def
}

// stdinApprover prompts the operator on stderr for a policy "ask" decision. The
// mutex serializes prompts so concurrent tool-call approvals never read the
// shared stdin reader at the same time.
type stdinApprover struct {
	mu sync.Mutex
	r  *bufio.Reader
}

func (a *stdinApprover) Approve(kind, detail string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	fmt.Fprintf(os.Stderr, "\n[approve %s] %s\n  allow? [y/N] ", kind, detail)
	line, err := a.r.ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.TrimSpace(strings.ToLower(line)) {
	case "y", "yes":
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
