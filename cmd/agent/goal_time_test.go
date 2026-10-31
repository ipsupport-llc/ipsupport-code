package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// A met goal says how long it took: the working time of every run at it,
// across a restart, and how many runs that was.
func TestAMetGoalSaysHowLongItTook(t *testing.T) {
	a := &app{workspace: t.TempDir(), cfg: config.Default()}
	a.setGoal("ship it")
	a.addGoalWork(40 * time.Minute)
	a.finishGoal(agent.Transcript{}) // not met yet

	b := &app{workspace: a.workspace, cfg: config.Default()} // a restart
	b.loadGoal()
	b.setGoal("ship it") // /goal go resumes the same goal
	b.addGoalWork(32*time.Minute + 20*time.Second)
	met, worked, runs := b.finishGoal(agent.Transcript{GoalMet: true})
	if !met || worked != 72*time.Minute+20*time.Second || runs != 2 {
		t.Fatalf("finishGoal = (%v, %v, %d), want met after 1h12m20s over 2 runs", met, worked, runs)
	}
	if got := goalDoneLine(worked, runs); got != "goal met — worked 1h12m over 2 runs" {
		t.Errorf("line = %q", got)
	}

	// A different goal starts its own count; a run with no goal counts nowhere.
	b.setGoal("ship it")
	b.addGoalWork(time.Hour)
	b.setGoal("something else")
	b.addGoalWork(5 * time.Second)
	if _, worked, runs := b.finishGoal(agent.Transcript{GoalMet: true}); worked != 5*time.Second || runs != 1 {
		t.Errorf("a replaced goal kept the old one's time: %v over %d", worked, runs)
	}
	b.addGoalWork(time.Minute)
	if b.goal.Worked != 0 {
		t.Error("time was counted with no goal standing")
	}
	// Not met: nothing to say yet.
	b.setGoal("next")
	if met, _, _ := b.finishGoal(agent.Transcript{}); met {
		t.Error("an unmet goal reported as met")
	}
}

func TestWorkTimeReadsLikeAPerson(t *testing.T) {
	for d, want := range map[time.Duration]string{
		42 * time.Second:                "42s",
		4*time.Minute + 5*time.Second:   "4m05s",
		72*time.Minute + 29*time.Second: "1h12m",
		1500 * time.Millisecond:         "2s",
		26*time.Hour + 3*time.Minute:    "26h03m",
	} {
		if got := workTime(d); got != want {
			t.Errorf("workTime(%v) = %q, want %q", d, got, want)
		}
	}
	if got := goalDoneLine(30*time.Second, 1); got != "goal met — worked 30s" {
		t.Errorf("one run: %q", got)
	}
}

// The TUI says it after the answer, from the goal_done event.
func TestTheGoalDoneEventIsShown(t *testing.T) {
	m := &tuiModel{app: &app{cfg: config.Default()}}
	lines := m.renderEvent(uiEvent{kind: "goal_done", fields: map[string]any{"worked": 4 * time.Minute, "runs": 3}})
	if len(lines) != 1 || !strings.Contains(lines[0], "goal met — worked 4m00s over 3 runs") {
		t.Errorf("rendered %q", lines)
	}
}

type goalEvents struct {
	kinds []string
	done  map[string]any
}

func (g *goalEvents) Emit(kind string, f map[string]any) {
	g.kinds = append(g.kinds, kind)
	if kind == "goal_done" {
		g.done = f
	}
}

// End to end: a run the judge confirms meets the goal ends with goal_done, after
// the answer, its time counted.
func TestARunThatMeetsTheGoalSaysSo(t *testing.T) {
	setHome(t, t.TempDir())
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if notFoundOffPOST(w, r) {
			return
		}
		n++
		io.Copy(io.Discard, r.Body)
		switch n {
		case 1:
			io.WriteString(w, toolCallRespWithUsage("calc", `{"expression":"1+1"}`, 10, 5))
		case 2:
			io.WriteString(w, contentRespWithUsage("the answer", 10, 5))
		default: // the judge, then anything after
			io.WriteString(w, contentRespWithUsage("DONE", 10, 5))
		}
	}))
	defer srv.Close()
	a, cleanup, err := build(t.TempDir(), "", nil, false, bufio.NewReader(strings.NewReader("")))
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	a.cfg.LLM.BaseURL, a.cfg.LLM.Model = srv.URL, "fake"
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.setGoal("do the thing")
	ev := &goalEvents{}
	a.tracer = ev
	a.runTaskStreaming(context.Background(), "do the thing", a.taskEpoch.Load())
	if !slices.Contains(ev.kinds, "goal_done") {
		t.Fatalf("events %v: no goal_done for a run the judge said met the goal", ev.kinds)
	}
	if worked, _ := ev.done["worked"].(time.Duration); worked <= 0 || ev.done["runs"] != 1 {
		t.Errorf("goal_done = %v, want this run's time counted, one run", ev.done)
	}
}
