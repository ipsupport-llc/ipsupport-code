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
	result    string    // final answer, or the error text
	delivered bool      // already folded into the model's conversation
	lastLine  string    // most recent output line from an external agent (liveness)
	lastAt    time.Time // when lastLine arrived — a long gap hints the job is stuck
}

// setJobProgress records the latest output line of a running job (called from the
// external agent's output tap; safe from that goroutine).
func (a *app) setJobProgress(id int, line string) {
	a.jobMu.Lock()
	for _, j := range a.jobs {
		if j.id == id {
			j.lastLine, j.lastAt = line, time.Now()
			break
		}
	}
	a.jobMu.Unlock()
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
		// Tap the external agent's output so /jobs shows a live "last line" — a
		// stuck job stops updating while a working one keeps ticking.
		out, err := a.spawnAgentTapped(jctx, resolved, task, dir, func(line string) { a.setJobProgress(id, line) })
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

// jobNote formats a finished job as the user-role note the model reads.
func jobNote(j *job) string {
	status := "finished"
	if !j.ok {
		status = "FAILED"
	}
	return fmt.Sprintf("[background job #%d %s — %s · %s]\n%s",
		j.id, status, j.profile, oneLine(j.task, 80), j.result)
}

// takeFinishedJobs marks every done-but-undelivered job delivered and returns
// their notes as user-role messages. Shared by both delivery paths; the
// `delivered` flag keeps them from double-delivering.
func (a *app) takeFinishedJobs() []llm.Message {
	a.jobMu.Lock()
	defer a.jobMu.Unlock()
	var out []llm.Message
	for _, j := range a.jobs {
		if j.done && !j.delivered {
			j.delivered = true
			out = append(out, llm.User(jobNote(j)))
		}
	}
	return out
}

// injectJobResults folds finished, undelivered background-job results into the
// durable history, so the model sees them at the start of its next task. Called
// before the agent runs (same goroutine — safe). The between-steps path is
// drainJobResults, via the beforeTurn hook.
func (a *app) injectJobResults() {
	if notes := a.takeFinishedJobs(); len(notes) > 0 {
		a.ag.SetHistory(append(a.ag.History(), notes...))
	}
}

// beforeTurn is the agent's between-steps hook (SetBeforeTurn): it folds both
// /btw asides and any just-finished background-job results into the model's
// working set, so a job that completes MID-task lands on the next
// reason→act→observe step instead of waiting for the next task boundary.
func (a *app) beforeTurn() []llm.Message {
	return append(a.drainBtw(), a.drainJobResults()...)
}

// drainJobResults is the between-steps counterpart of injectJobResults: it
// delivers a job that finished while a task is running onto the model's next
// step, and folds the same notes into durable history so the report survives the
// run (matching the task-boundary path). Returns nil when nothing is pending.
func (a *app) drainJobResults() []llm.Message {
	notes := a.takeFinishedJobs()
	if len(notes) > 0 {
		a.ag.SetHistory(append(a.ag.History(), notes...))
	}
	return notes
}

// --- /btw side-channel steering --------------------------------------------
//
// /btw drops a note into a RUNNING task without stopping it (esc cancels; /btw
// steers). The note is buffered under btwMu and the agent's beforeTurn hook
// (drainBtw) folds it into the model's working set between turns, so it lands on
// the next reason→act→observe step instead of interrupting the stream. Typed
// while idle, it simply steers the next run.

// addBtw queues a user aside. Safe to call from the UI goroutine while the run
// goroutine drains it.
func (a *app) addBtw(note string) bool {
	note = strings.TrimSpace(note)
	if note == "" {
		return false
	}
	a.btwMu.Lock()
	a.pendingBtw = append(a.pendingBtw, note)
	a.btwMu.Unlock()
	return true
}

// drainBtw is the agent's beforeTurn hook: it returns queued /btw notes as
// user-role messages (prefixed so the model reads them as an aside, not a new
// task) and clears the buffer.
func (a *app) drainBtw() []llm.Message {
	a.btwMu.Lock()
	notes := a.pendingBtw
	a.pendingBtw = nil
	a.btwMu.Unlock()
	if len(notes) == 0 {
		return nil
	}
	msgs := make([]llm.Message, 0, len(notes))
	for _, n := range notes {
		msgs = append(msgs, llm.User("[by the way] "+n))
	}
	return msgs
}

// btwPending reports how many /btw notes are queued (for /status).
func (a *app) btwPending() int {
	a.btwMu.Lock()
	defer a.btwMu.Unlock()
	return len(a.pendingBtw)
}

// btwCommand handles /btw. running distinguishes steering a live task from
// leaving a note for the next run (the wording, not the mechanism, differs).
func (a *app) btwCommand(rest string, running bool) []string {
	if strings.TrimSpace(rest) == "" {
		if n := a.btwPending(); n > 0 {
			return []string{fmt.Sprintf("%d aside(s) queued for the next turn — add more with /btw <note>", n)}
		}
		return []string{"usage: /btw <note> — drop a side-note into the running task without stopping it"}
	}
	a.addBtw(rest)
	if running {
		return []string{"✦ noted — steering the running task; it lands on the next step"}
	}
	return []string{"✦ noted — will steer your next run"}
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
				prog := ""
				if j.lastLine != "" {
					prog = fmt.Sprintf(" · ↳ %s (%s ago)", oneLine(j.lastLine, 40),
						time.Since(j.lastAt).Truncate(time.Second))
				}
				out = append(out, fmt.Sprintf("  ⚙ #%d %-10s running %s%s · %s", j.id, j.profile,
					time.Since(j.started).Truncate(time.Second), prog, oneLine(j.task, 60)))
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
