// Command ipsupport-code is a self-learning local agent for LM Studio. With a
// goal argument it runs one task; with none it opens a REPL. After each task it
// reflects and persists what it learned.
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
	"strconv"
	"strings"
	"time"

	"golang.org/x/term"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/reflect"
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

	if goal := strings.TrimSpace(strings.Join(flag.Args(), " ")); goal != "" {
		app.runOne(ctx, goal)
		return
	}
	app.repl(ctx)
}

// maybeInit runs the interactive first-time setup, writing the LM Studio
// connection to the user config. It triggers when forced (-init) or on a real
// first run (no user config yet and an interactive terminal). Piped/non-TTY
// first runs silently use defaults so scripts don't block.
func maybeInit(reader *bufio.Reader, force bool) {
	if !force && (config.GlobalExists() || !isTTY()) {
		return
	}
	def := config.Default().LLM
	fmt.Println("First-time setup — configure your model connection (press Enter for defaults).")
	llm := config.LLM{
		BaseURL:     ask(reader, "LM Studio server URL", def.BaseURL),
		Model:       ask(reader, "Model name", def.Model),
		Temperature: def.Temperature,
		MaxSteps:    atoiOr(ask(reader, "Max steps per task", strconv.Itoa(def.MaxSteps)), def.MaxSteps),
		APIKey:      ask(reader, "API key (blank for LM Studio)", ""),
	}
	if err := config.SaveGlobalLLM(llm); err != nil {
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

func isTTY() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

func atoiOr(s string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return def
}

// app bundles everything one process needs across tasks.
type app struct {
	ag     *agent.Agent
	refl   *reflect.Reflector
	kb     *knowledge.KB
	tracer trace.Tracer
	reader *bufio.Reader
}

func build(workspace string, reader *bufio.Reader) (*app, func(), error) {
	cfg, err := config.Load(workspace)
	if err != nil {
		return nil, nil, fmt.Errorf("load config: %w", err)
	}
	kb, err := knowledge.Open(cfg.KBPath)
	if err != nil {
		slog.Warn("knowledge base unreadable; starting empty", "err", err)
		kb, _ = knowledge.Open("") // empty in-memory store
	}
	pol, err := policy.New(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("policy: %w", err)
	}

	approver := &stdinApprover{r: reader}

	reg := tool.NewRegistry(
		tool.NewFile(pol, approver),
		tool.NewRun(pol, approver),
		tool.NewWeb(http.DefaultClient),
		tool.NewHelp(kb),
		tool.NewCalc(),
	)
	client := llm.NewOpenAIClient(cfg.LLM)

	var tracer trace.Tracer
	cleanup := func() {}
	if ft, err := trace.NewFileTracer(cfg.TracePath, newRunID()); err != nil {
		slog.Warn("trace disabled", "err", err)
	} else {
		tracer = ft
		cleanup = func() { _ = ft.Close() }
	}

	return &app{
		ag:     agent.New(client, reg, kb, tracer, "", cfg.LLM.MaxSteps),
		refl:   reflect.New(client),
		kb:     kb,
		tracer: tracer,
		reader: reader,
	}, cleanup, nil
}

// runOne executes a single goal, then reflects and persists new lessons.
func (a *app) runOne(ctx context.Context, goal string) {
	tr, err := a.ag.Run(ctx, goal)
	if err != nil {
		slog.Error("run failed", "err", err)
		fmt.Fprintln(os.Stderr, "error:", err)
		return
	}
	if strings.TrimSpace(tr.Final) != "" {
		fmt.Println(tr.Final)
	} else {
		fmt.Println("(no final answer — step budget exhausted)")
	}

	lessons, err := a.refl.Reflect(ctx, tr)
	if err != nil {
		slog.Warn("reflection failed", "err", err)
		return
	}
	learned := 0
	for _, p := range lessons {
		if a.kb.Add(p) {
			learned++
			if a.tracer != nil {
				a.tracer.Emit("lesson", map[string]any{"domain": p.Domain, "proven_fix": p.ProvenFix})
			}
		}
	}
	if learned > 0 {
		if err := a.kb.Save(); err != nil {
			slog.Warn("knowledge save failed", "err", err)
		}
		fmt.Fprintf(os.Stderr, "(learned %d new lesson(s))\n", learned)
	}
}

func (a *app) repl(ctx context.Context) {
	fmt.Println("ipsupport-code — type a task, or 'exit' to quit.")
	for {
		fmt.Print("\n> ")
		line, err := a.reader.ReadString('\n')
		if err != nil {
			fmt.Println()
			return
		}
		goal := strings.TrimSpace(line)
		switch goal {
		case "":
			continue
		case "exit", "quit":
			return
		}
		a.runOne(ctx, goal)
	}
}

// stdinApprover prompts the operator for a policy "ask" decision.
type stdinApprover struct{ r *bufio.Reader }

func (a *stdinApprover) Approve(kind, detail string) bool {
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
