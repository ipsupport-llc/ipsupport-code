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
	"strconv"
	"strings"
	"sync"
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
	)
	flag.StringVar(&workspace, "C", ".", "workspace directory")
	flag.BoolVar(&doInit, "init", false, "re-run first-time setup (server URL, model)")
	flag.BoolVar(&showVersion, "version", false, "print version and exit")
	flag.BoolVar(&dumpPrompt, "dump-prompt", false, "print the built-in system prompt and exit (e.g. > .agent/system.md to start editing)")
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

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	switch {
	case strings.TrimSpace(strings.Join(flag.Args(), " ")) != "":
		app.runOne(ctx, strings.TrimSpace(strings.Join(flag.Args(), " ")))
	case isTTY():
		if err := app.runTUI(ctx); err != nil {
			fmt.Fprintln(os.Stderr, "tui:", err)
		}
	default:
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

	client         *llm.OpenAIClient
	ag             *agent.Agent
	refl           *reflect.Reflector
	instrSrc       string   // project instructions file in effect, "" if none
	promptSrc      string   // "built-in" or the system.md override path
	facts          []string // durable project facts learned over time (per workspace)
	planMode       bool     // plan (propose) vs auto (execute); survives re-wire
	windowDetected bool     // got the real loaded context window (vs a default/guess)

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
	a.loadSession()         // restore the prior conversation for this workspace
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

// detectContextWindow asks LM Studio for the loaded model's context length and,
// once it answers, uses it instead of the configured default. Only for the local
// provider — external providers don't expose /api/v0/models, and their window
// comes from the preset. No-op after it succeeds once; best-effort, never blocks.
func (a *app) detectContextWindow() {
	if a.windowDetected || !a.isLocal() || !a.cfg.LLM.LMStudio() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if w := llm.DetectContextWindow(ctx, a.cfg.LLM.BaseURL, a.cfg.LLM.Model, http.DefaultClient); w > 0 {
		a.cfg.LLM.ContextWindow = w
		a.windowDetected = true
		slog.Info("detected context window", "tokens", w, "model", a.cfg.LLM.Model)
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
	a.detectContextWindow()
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

// Session memory persists per workspace so "the prior context" survives a
// restart. /new and /clear wipe it.
func (a *app) sessionPath() string { return filepath.Join(a.workspace, ".agent", "session.json") }

func (a *app) loadSession() {
	data, err := os.ReadFile(a.sessionPath())
	if err != nil {
		return
	}
	var h []llm.Message
	if json.Unmarshal(data, &h) == nil {
		a.ag.SetHistory(h)
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
	if len(a.facts) > 0 { // learned project facts — keep the injected set small
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
		fmt.Print(a.usageText())
	case "/login", "/init":
		maybeInit(a.reader, true)
		if err := a.reconfigure(); err != nil {
			fmt.Println("reconfigure failed:", err)
		} else {
			fmt.Println("config reloaded.")
		}
	case "/new", "/reset", "/clear":
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
	a.detectContextWindow()
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

// modelCommand lists the active provider's models (no arg) or sets one.
func (a *app) modelCommand(ctx context.Context, rest string) []string {
	if name := strings.TrimSpace(rest); name != "" {
		return a.setModel(name)
	}
	cctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	return modelLines(cctx, a.activeLLM(), a.providerName())
}

// modelLines lists a provider's models — rich (state · context · quant) for LM
// Studio via its native API, plain ids otherwise. Pure (no app state), so the
// REPL and the TUI's async path share it.
func modelLines(ctx context.Context, act config.LLM, provider string) []string {
	head := fmt.Sprintf("models on %s (current %s) — /model <name> to pick:", provider, act.Model)
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
	for _, id := range ids {
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
	a.detectContextWindow()
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
	default:
		return []string{"usage: /permissions [files on|off] [run on|off]"}
	}
}

func (a *app) permissionsStatus() []string {
	return []string{
		"permissions:",
		fmt.Sprintf("  files  %s   (jail %q)", a.cfg.File.Default, a.cfg.File.Jail),
		fmt.Sprintf("  run    %s", a.cfg.Run.Default),
		"  deny floor (always on): secrets, .git, .env, rm -rf, sudo, …",
		"  /permissions files on  → stop asking before file writes in the workspace",
		"  /permissions files off → ask again",
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
	cmd = strings.Fields(line)[0]
	return cmd, strings.TrimSpace(strings.TrimPrefix(line, cmd))
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
  /new             clear the session conversation memory
  /clear           fresh start — clear the screen and the session
  /compact         summarize the session so far to free up context
  /plan, /auto     plan mode (propose only) vs auto mode (execute)
  /ai [name]       switch AI provider (local|openai|grok|groq|openrouter); /ai key <name> <tok>
  /model [name]    list the provider's models, or pick one
  /config          control panel: all settings + how to change them
  /update [chan]   self-update from GitHub (chan = stable|nightly, saved)
  /shell, /sh      drop to a shell in the workspace (exit to return)
  /skills          list/toggle/install on-demand instruction packs
  /permissions     relax approval for non-destructive file/shell actions
  /color [name]    change the TUI frame color (cycles if no name)
  /rename <name>   rename the agent (saved in settings)
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
	return b.String()
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
	for i, t := range a.usage.ByModel() {
		if i >= 8 {
			break
		}
		models = append(models, [2]string{t.Key, humanK(t.Tokens()) + " tok"})
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
