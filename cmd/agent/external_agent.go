package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// External CLI agents (codex, claude, aider…) run OUTSIDE our sandbox: their own
// tools, their own permissions, no policy jail, and their edits are invisible to
// /rewind. They are a different trust class from LLM sub-agents, so every launch
// asks its own approval under the "external agent" category — deliberately NOT a
// "spawn …" kind, which approvalCategory would fold into the ordinary spawn
// bucket and an earlier allow-session for spawns would silently cover these too.

const (
	defaultExternalTimeout = 15 * time.Minute
	maxExternalStdout      = 8_000 // bytes of stdout tail handed back to the parent
	maxExternalStderr      = 2_000
)

// externalCatalog holds known CLI coding agents with their NON-INTERACTIVE launch
// shape, so `/agents add-tool codex` works without remembering flags (and a bare
// `/agents add-tool` scans PATH to show what's installed). Best-effort defaults —
// the full add-tool form overrides them for a custom setup.
var externalCatalog = []struct {
	name string
	args []string
}{
	{"codex", []string{"exec", "{task}"}},               // OpenAI Codex CLI
	{"claude", []string{"-p", "{task}"}},                // Claude Code (print mode)
	{"gemini", []string{"-p", "{task}"}},                // Gemini CLI
	{"qwen", []string{"-p", "{task}"}},                  // Qwen Code
	{"aider", []string{"--yes", "--message", "{task}"}}, // aider (one-shot)
	{"goose", []string{"run", "-t", "{task}"}},          // Goose
	{"opencode", []string{"run", "{task}"}},             // OpenCode
}

// catalogArgs returns the known launch args for a catalog CLI (nil = unknown).
func catalogArgs(command string) []string {
	for _, c := range externalCatalog {
		if c.name == command {
			return c.args
		}
	}
	return nil
}

// catalogNames lists the catalog CLI names in display order.
func catalogNames() []string {
	names := make([]string, len(externalCatalog))
	for i, c := range externalCatalog {
		names[i] = c.name
	}
	return names
}

// spawnExternalAgent runs an external-CLI profile as a sub-agent: exec Command in
// the target dir with the task substituted into Args, and hand the tail of its
// output (plus a change summary) back to the delegating model.
func (a *app) spawnExternalAgent(ctx context.Context, profile string, p config.AgentProfile, task, dir string) (string, error) {
	command := strings.TrimSpace(p.Command)
	if command == "" {
		return "", fmt.Errorf("profile %q: external agent has no command", profile)
	}
	root := a.effectiveDir()
	if d := strings.TrimSpace(dir); d != "" {
		resolved, err := a.resolveSpawnDir(d)
		if err != nil {
			return "", err
		}
		root = resolved
	}
	if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
		return "", fmt.Errorf("dir %q is not a directory", root)
	}

	// Always gated, even when ordinary spawns are relaxed (spawn allow) — an
	// unsandboxed agent asks separately. 'a' on the prompt relaxes just this
	// category for the session. Serialized like spawns so fan-outs ask one at a
	// time. The FULL task goes on its own detail line — you're approving an
	// autonomous agent, so you get to read exactly what it was told to do.
	head := command + " · " + root
	if profile != command {
		head = profile + " · " + head
	}
	a.spawnMu.Lock()
	approved := a.approveGated("external agent", head+"\n  task: "+task)
	a.spawnMu.Unlock()
	if !approved {
		return "", fmt.Errorf("external agent denied by user")
	}

	timeout := defaultExternalTimeout
	if p.Timeout > 0 {
		timeout = time.Duration(p.Timeout) * time.Second
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	id := fmt.Sprintf("sub%d", a.spawnSeq.Add(1)) // groups this sub-agent's UI events
	a.emit("subagent", map[string]any{"agent": id, "profile": profile, "provider": "external", "model": command, "dir": root, "task": oneLine(task, 80)})

	cmd := exec.CommandContext(cctx, command, expandTaskArgs(p.Args, task)...)
	cmd.Dir = root
	setProcGroup(cmd) // cancel/timeout kills the whole process tree, not just the child
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	runErr := cmd.Run()
	if cctx.Err() == context.DeadlineExceeded {
		runErr = fmt.Errorf("timed out after %s — use the CLI's non-interactive mode (exec / -p / --message) so it can't sit waiting for input", timeout)
	}

	body := externalResult(command, stdout.String(), stderr.String(), strings.TrimSpace(mustGit(root, "diff", "--stat")), runErr)
	done := map[string]any{"agent": id, "profile": profile, "ok": runErr == nil}
	if runErr != nil {
		done["error"] = oneLine(runErr.Error(), 60)
	}
	a.emit("subagent_done", done)
	if runErr != nil {
		return "", fmt.Errorf("%s failed: %w\n%s", command, runErr, body)
	}
	return body, nil
}

// expandTaskArgs substitutes the task into {task} / {{task}} placeholders; when no
// arg carries a placeholder the task is appended as the final argument. {{task}}
// is replaced first — it contains {task}, and the reverse order would leave braces.
func expandTaskArgs(args []string, task string) []string {
	out := make([]string, 0, len(args)+1)
	hit := false
	for _, arg := range args {
		if strings.Contains(arg, "{task}") || strings.Contains(arg, "{{task}}") {
			hit = true
			arg = strings.ReplaceAll(arg, "{{task}}", task)
			arg = strings.ReplaceAll(arg, "{task}", task)
		}
		out = append(out, arg)
	}
	if !hit {
		out = append(out, task)
	}
	return out
}

// externalResult assembles what the parent model gets back: the TAIL of stdout (a
// CLI agent's final answer is at the end; the start is progress noise), a stderr
// tail on failure, and a git diff --stat summary — never the full patch, which can
// be thousands of tokens (the user reviews it with /diff).
func externalResult(command, stdout, stderr, diffStat string, runErr error) string {
	var b strings.Builder
	fmt.Fprintf(&b, "external agent `%s` finished.\n", command)
	if out := textutil.Tail(strings.TrimSpace(stdout), maxExternalStdout); out != "" {
		b.WriteString("\n--- output (tail) ---\n" + out + "\n")
	}
	if runErr != nil {
		if errOut := textutil.Tail(strings.TrimSpace(stderr), maxExternalStderr); errOut != "" {
			b.WriteString("\n--- stderr (tail) ---\n" + errOut + "\n")
		}
	}
	if diffStat != "" {
		b.WriteString("\n--- changes (git diff --stat) ---\n" + diffStat + "\n(full patch: /diff)\n")
	}
	return strings.TrimSpace(b.String())
}
