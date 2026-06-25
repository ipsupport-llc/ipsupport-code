// Command ipsupport-code is a self-learning local agent for LM Studio. With a
// goal argument it runs one task; with none on a terminal it opens a Bubble Tea
// TUI (plain line REPL when piped). After each task it reflects and persists
// what it learned.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
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
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
	"github.com/ipsupport-llc/ipsupport-code/internal/trace"
)

func main() {
	var (
		workspace string
		doInit    bool
	)
	flag.StringVar(&workspace, "C", ".", "workspace directory")
	flag.BoolVar(&doInit, "init", false, "re-run first-time setup (server URL, model)")
	flag.Parse()
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

// app bundles everything one process needs across tasks, plus session counters.
type app struct {
	cfg       config.Config
	workspace string
	kb        *knowledge.KB
	reader    *bufio.Reader

	fileTracer trace.Tracer  // JSONL dataset
	uiTracer   trace.Tracer  // live TUI (nil in plain mode)
	tracer     trace.Tracer  // composite, set in wire()
	approver   tool.Approver // stdin (plain) or the TUI bridge

	client   *llm.OpenAIClient
	ag       *agent.Agent
	refl     *reflect.Reflector
	instrSrc string // project instructions file in effect, "" if none

	tasks, steps, toolCalls int
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

	cleanup := func() {}
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: reader, approver: &stdinApprover{r: reader}}
	if ft, err := trace.NewFileTracer(cfg.TracePath, newRunID()); err != nil {
		slog.Warn("trace disabled", "err", err)
	} else {
		a.fileTracer = ft
		cleanup = func() { _ = ft.Close() }
	}
	if err := a.wire(); err != nil {
		return nil, nil, err
	}
	return a, cleanup, nil
}

// wire (re)builds the policy-gated tools, LLM client, agent, and reflector from
// the current config and the current approver/tracers. Called at startup, after
// /login, and when the TUI installs its bridge.
func (a *app) wire() error {
	pol, err := policy.New(a.cfg)
	if err != nil {
		return fmt.Errorf("policy: %w", err)
	}
	reg := tool.NewRegistry(
		tool.NewFile(pol, a.approver),
		tool.NewRun(pol, a.approver),
		tool.NewGit(pol, a.approver),
		tool.NewWeb(http.DefaultClient),
		tool.NewHelp(a.kb),
		tool.NewCalc(),
	)
	a.tracer = trace.Multi(a.fileTracer, a.uiTracer)
	a.client = llm.NewOpenAIClient(a.cfg.LLM)
	a.ag = agent.New(a.client, reg, a.kb, a.tracer, a.systemPrompt(), a.cfg.LLM.MaxSteps)
	a.refl = reflect.New(a.client)
	return nil
}

func (a *app) reconfigure() error {
	cfg, err := config.Load(a.workspace)
	if err != nil {
		return err
	}
	a.cfg = cfg
	return a.wire()
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

// systemPrompt is the base prompt plus any project instructions; records the
// source in instrSrc for /status.
func (a *app) systemPrompt() string {
	text, src := loadInstructions(a.workspace)
	a.instrSrc = src
	if text == "" {
		return agent.DefaultSystemPrompt()
	}
	return agent.DefaultSystemPrompt() +
		"\n\n## Project instructions (from " + src + ") — follow these:\n" + text
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

// reflectAndStore runs the post-task reflection and persists new lessons,
// emitting a "lesson" event for each. Returns how many were new.
func (a *app) reflectAndStore(ctx context.Context, tr agent.Transcript) int {
	lessons, err := a.refl.Reflect(ctx, tr)
	if err != nil {
		slog.Warn("reflection failed", "err", err)
		return 0
	}
	learned := 0
	for _, p := range lessons {
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
}

func (a *app) repl(ctx context.Context) {
	fmt.Println("ipsupport-code — type a task, or /help for commands.")
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
	case "/new", "/reset":
		a.ag.Reset()
		fmt.Println("session cleared.")
	case "/goal":
		if rest == "" {
			fmt.Println("usage: /goal <task>")
		} else {
			a.runOne(ctx, rest)
		}
	case "/loop":
		n, goal := parseLoop(rest)
		if goal == "" {
			fmt.Println("usage: /loop [count] <task>")
			break
		}
		for i := 0; i < n; i++ {
			if ctx.Err() != nil {
				break
			}
			fmt.Printf("— loop %d/%d —\n", i+1, n)
			a.runOne(ctx, goal)
		}
	case "/exit", "/quit":
		return true
	default:
		fmt.Printf("unknown command %q — try /help\n", cmd)
	}
	return false
}

func splitCommand(line string) (cmd, rest string) {
	cmd = strings.Fields(line)[0]
	return cmd, strings.TrimSpace(strings.TrimPrefix(line, cmd))
}

func parseLoop(rest string) (n int, goal string) {
	n, goal = 3, rest
	parts := strings.Fields(rest)
	if len(parts) == 0 {
		return 0, ""
	}
	if v, err := strconv.Atoi(parts[0]); err == nil {
		n = v
		goal = strings.TrimSpace(strings.TrimPrefix(rest, parts[0]))
	}
	if n < 1 {
		n = 1
	}
	return n, strings.TrimSpace(goal)
}

func helpText() string {
	return `commands:
  /status          show config, knowledge base, and trace paths
  /usage           session counters + token usage
  /login           (re)configure the server URL / model / key, then reload
  /new             clear the session conversation memory
  /goal <task>     run a task explicitly
  /loop [n] <task> run a task n times (default 3) so lessons compound
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
	return fmt.Sprintf(`status:
  server       %s
  model        %s
  max_steps    %d
  workspace    %s
  jail         %q
  defaults     run=%s  file=%s
  instructions %s
  session      %d messages
  knowledge    %s (%d lessons)
  trace        %s
`,
		a.cfg.LLM.BaseURL, a.cfg.LLM.Model, a.cfg.LLM.MaxSteps,
		a.cfg.Workspace, a.cfg.File.Jail, a.cfg.Run.Default, a.cfg.File.Default,
		instr, a.ag.SessionLen(),
		a.cfg.KBPath, len(a.kb.All()), a.cfg.TracePath)
}

func (a *app) usageText() string {
	p, c := a.client.Usage()
	return fmt.Sprintf(`usage (this session):
  tasks       %d
  steps       %d
  tool calls  %d
  tokens      %d prompt + %d completion = %d
  lessons     %d in knowledge base
`, a.tasks, a.steps, a.toolCalls, p, c, p+c, len(a.kb.All()))
}

// maybeInit runs the interactive first-time setup, writing the LM Studio
// connection to the user config. It triggers when forced (-init / /login) or on a
// real first run (no user config yet and an interactive terminal).
func maybeInit(reader *bufio.Reader, force bool) {
	if !force && (config.GlobalExists() || !isTTY()) {
		return
	}
	def := config.Default().LLM
	if cur, err := config.Load("."); err == nil {
		def = cur.LLM
	}
	fmt.Println("Setup — configure your model connection (press Enter to keep current).")
	l := config.LLM{
		BaseURL:     ask(reader, "LM Studio server URL", def.BaseURL),
		Model:       ask(reader, "Model name", def.Model),
		Temperature: def.Temperature,
		MaxSteps:    atoiOr(ask(reader, "Max steps per task", strconv.Itoa(def.MaxSteps)), def.MaxSteps),
		APIKey:      ask(reader, "API key (blank for LM Studio)", def.APIKey),
	}
	if err := config.SaveGlobalLLM(l); err != nil {
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
