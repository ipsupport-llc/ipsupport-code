package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// Background jobs — fire-and-forget sub-agents. `agent.run(..., background=true)`
// returns an acknowledgement immediately; the sub-agent (LLM or external CLI)
// runs detached from the calling task, and its result is folded into the
// conversation at the start of the NEXT turn (injectJobResults). /jobs lists,
// reads and kills them. Jobs deliberately use their own context: the parent
// finishing — or esc cancelling it — must not kill a job.

// job is one background sub-agent run.
type job struct {
	id        int
	profile   string
	task      string
	dir       string
	started   time.Time
	finished  time.Time
	cancel    context.CancelFunc
	done      bool
	ok        bool
	result    string // final answer, or the error text
	delivered bool   // already folded into the model's conversation
}

// spawnAgentBackground is the SpawnFunc for background=true: validate up front
// (a typo must fail NOW, not silently inside the job), register the job, run the
// normal spawn path — approval included — in a goroutine, and return at once.
func (a *app) spawnAgentBackground(_ context.Context, profile, task, dir string) (string, error) {
	if strings.TrimSpace(task) == "" {
		return "", fmt.Errorf("task is required")
	}
	resolved, ok := a.resolveProfileName(strings.TrimSpace(profile))
	if !ok {
		return "", fmt.Errorf("unknown profile %q — configured: %s", profile, a.profilesOrHint())
	}
	// Detached lifetime: the job survives its parent task (and esc). Cancelled
	// only via /jobs kill — or process exit.
	jctx, cancel := context.WithCancel(context.Background())
	a.jobMu.Lock()
	a.jobSeq++
	j := &job{id: a.jobSeq, profile: resolved, task: task, dir: dir, started: time.Now(), cancel: cancel}
	a.jobs = append(a.jobs, j)
	id := j.id
	a.jobMu.Unlock()
	a.emit("job_started", map[string]any{"job": id, "profile": resolved, "task": oneLine(task, 80)})

	go func() {
		defer cancel()
		out, err := a.spawnAgent(jctx, resolved, task, dir) // approval still gates inside
		a.jobMu.Lock()
		j.done, j.finished = true, time.Now()
		if err != nil {
			j.result = "error: " + err.Error()
		} else {
			j.ok, j.result = true, out
		}
		a.jobMu.Unlock()
		a.emit("job_done", map[string]any{"job": id, "profile": resolved, "ok": err == nil})
	}()
	return fmt.Sprintf("background job #%d started (%s). Its result will be delivered to you at the start of a later turn — continue with other work; don't wait or poll.", id, resolved), nil
}

// injectJobResults folds finished, undelivered background-job results into the
// conversation as user-role notes, so the model sees them on its next turn.
// Called at the start of a task, before the agent runs (same goroutine — safe).
func (a *app) injectJobResults() {
	a.jobMu.Lock()
	var notes []string
	for _, j := range a.jobs {
		if j.done && !j.delivered {
			j.delivered = true
			status := "finished"
			if !j.ok {
				status = "FAILED"
			}
			notes = append(notes, fmt.Sprintf("[background job #%d %s — %s · %s]\n%s",
				j.id, status, j.profile, oneLine(j.task, 80), j.result))
		}
	}
	a.jobMu.Unlock()
	if len(notes) == 0 {
		return
	}
	h := a.ag.History()
	for _, n := range notes {
		h = append(h, llm.User(n))
	}
	a.ag.SetHistory(h)
}

// jobsPending reports how many jobs are still running (for /status).
func (a *app) jobsPending() int {
	a.jobMu.Lock()
	defer a.jobMu.Unlock()
	n := 0
	for _, j := range a.jobs {
		if !j.done {
			n++
		}
	}
	return n
}

// jobsCommand lists background jobs, prints a job's full result, or kills one.
func (a *app) jobsCommand(arg string) []string {
	sub, rest := splitCommand(arg)
	a.jobMu.Lock()
	defer a.jobMu.Unlock()
	switch sub {
	case "":
		if len(a.jobs) == 0 {
			return []string{"no background jobs this session", "  the assistant starts one via agent.run(..., background=true)"}
		}
		out := []string{"background jobs:"}
		for _, j := range a.jobs {
			switch {
			case !j.done:
				out = append(out, fmt.Sprintf("  ⚙ #%d %-10s running %s · %s", j.id, j.profile,
					time.Since(j.started).Truncate(time.Second), oneLine(j.task, 60)))
			case j.ok:
				note := "result delivered"
				if !j.delivered {
					note = "result lands next turn · /jobs result " + strconv.Itoa(j.id)
				}
				out = append(out, fmt.Sprintf("  ✓ #%d %-10s done in %s · %s", j.id, j.profile,
					j.finished.Sub(j.started).Truncate(time.Second), note))
			default:
				out = append(out, fmt.Sprintf("  ✖ #%d %-10s failed · %s", j.id, j.profile, oneLine(j.result, 60)))
			}
		}
		return append(out, "  /jobs result <id> · /jobs kill <id>")
	case "result", "show":
		j := findJob(a.jobs, rest)
		if j == nil {
			return []string{"no such job — /jobs lists them"}
		}
		if !j.done {
			return []string{fmt.Sprintf("job #%d is still running (%s)", j.id, time.Since(j.started).Truncate(time.Second))}
		}
		return append([]string{fmt.Sprintf("job #%d — %s:", j.id, j.profile)}, strings.Split(j.result, "\n")...)
	case "kill", "stop", "cancel":
		j := findJob(a.jobs, rest)
		if j == nil {
			return []string{"no such job — /jobs lists them"}
		}
		if j.done {
			return []string{fmt.Sprintf("job #%d already finished", j.id)}
		}
		j.cancel()
		return []string{fmt.Sprintf("cancelling job #%d — it will record its outcome shortly", j.id)}
	default:
		return []string{"usage: /jobs [result <id>] [kill <id>]"}
	}
}

// findJob resolves a job by its numeric id (callers hold jobMu).
func findJob(jobs []*job, arg string) *job {
	id, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(arg), "#")))
	if err != nil {
		return nil
	}
	for _, j := range jobs {
		if j.id == id {
			return j
		}
	}
	return nil
}
