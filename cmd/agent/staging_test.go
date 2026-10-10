package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func busyPanel(t *testing.T) *tuiModel {
	t.Helper()
	m := panelModel(t)
	if err := m.app.wire(); err != nil {
		t.Fatal(err)
	}
	m.cancel = func() {}
	return m
}

func enterOn(m *tuiModel, key string) {
	cursorOn(m, key)
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
}

// Reported: "knowledge retention off (kept forever) — staged ×34". Presses
// during a task stage one change — where the row lands — shown on the row.
func TestCyclePressesDuringATaskStageWhereTheyLand(t *testing.T) {
	m := busyPanel(t)
	for range 34 { // 0 → 7 → 30 → 90 → 180 → 0 …: 34 presses land on 180
		enterOn(m, "knowledge_retention")
	}
	if len(m.cfgPending) != 1 || m.cfgPending[0].value != "180" {
		t.Fatalf("pending %+v, want one change landing on 180", m.cfgPending)
	}
	p := panelText(m)
	if !strings.Contains(p, "180 days") || strings.Contains(p, "×") {
		t.Fatalf("the row doesn't show where it lands:\n%s", p)
	}
	if m.app.cfg.KnowledgeRetentionDays != 0 {
		t.Fatal("changed under the running task")
	}
	m.cancel = nil
	m.applyPendingConfig()
	if m.app.cfg.KnowledgeRetentionDays != 180 || reload(t).KnowledgeRetentionDays != 180 {
		t.Fatalf("retention %d after the task, want 180", m.app.cfg.KnowledgeRetentionDays)
	}
}

// Pressing a toggle back to where it is stages nothing.
func TestATogglePressedBackStagesNothing(t *testing.T) {
	m := busyPanel(t)
	enterOn(m, "goal_nudge")
	enterOn(m, "goal_nudge")
	if len(m.cfgPending) != 0 {
		t.Fatalf("pending %+v after toggling back", m.cfgPending)
	}
}

// A tuning change and a model pick made for provider A stay A's when a switch
// to B is staged after them — they used to be saved to B.
func TestStagedChangesKeepTheirConnection(t *testing.T) {
	m := busyPanel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k", Model: "lab-model"}}
	localModel := m.app.cfg.LLM.Model
	enterOn(m, "provider") // the switch staged FIRST, so it is saved first
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // mylab
	editRow(t, m, "temperature", "0.6")         // still local's, which the task runs on
	cursorOn(m, "model")
	m.configActivate()
	m.Update(pickListMsg{items: manyItems(3), epoch: m.app.modelEpoch.Load()})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // model-00, for local
	m.cancel = nil
	m.applyPendingConfig()
	c := reload(t)
	if c.Provider != "mylab" {
		t.Fatalf("provider %q, want mylab", c.Provider)
	}
	if c.LLM.Temperature != 0.6 || c.LLM.Model != "model-00" {
		t.Fatalf("local = %+v, want temperature 0.6 and model-00 (was %s)", c.LLM, localModel)
	}
	if p := c.Providers["mylab"]; p.Temperature != 0 || p.Model != "lab-model" {
		t.Fatalf("mylab got local's changes: %+v", p)
	}
}

// Closing /config drains a /clear queued behind the task; the panel's note
// used to slice the log by its old length and panic.
func TestClosingConfigAfterAQueuedClearDoesNotPanic(t *testing.T) {
	m := panelModel(t)
	m.app.kb, _ = knowledge.Open("")
	if err := m.app.wire(); err != nil {
		t.Fatal(err)
	}
	m.ready = true
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	m.state = stConfig
	for range 50 {
		m.push("a line of earlier output")
	}
	m.taskDoneAway, m.queued = true, []string{"/clear"}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
}

// On a 24-line terminal the open model list leaves the panel on the screen.
func TestTheModelListFitsAShortTerminal(t *testing.T) {
	m := panelModel(t)
	m.ready = true
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	m.state = stConfig
	enterOn(m, "model")
	m.Update(pickListMsg{items: manyItems(40), epoch: m.app.modelEpoch.Load()})
	if n := strings.Count(ansi.Strip(m.renderConfigPanel()), "\n") + 1; n > m.viewportHeight() {
		t.Fatalf("panel is %d lines in a %d-line log area", n, m.viewportHeight())
	}
}

// A server can accept an id it doesn't list: typed, it is offered as typed.
func TestATypedModelIdCanBeUsedAsTyped(t *testing.T) {
	m := panelModel(t)
	enterOn(m, "model")
	m.Update(pickListMsg{items: []pickItem{{value: "foo-large"}}, epoch: m.app.modelEpoch.Load()})
	typeKeys(m, "foo")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnd})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.app.cfg.LLM.Model != "foo" {
		t.Fatalf("model %q, want foo as typed", m.app.cfg.LLM.Model)
	}
}

// Picking a provider that can't be used keeps the list open with the reason.
func TestAProviderThatCantBeUsedKeepsTheListOpen(t *testing.T) {
	m := panelModel(t)
	t.Setenv("OPENAI_API_KEY", "")
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	enterOn(m, "provider")
	m.cfgPick.p.items = append(m.cfgPick.p.items, pickItem{value: "openai"}) // listed without a key
	typeKeys(m, "openai")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfgPick == nil || m.cfgPick.err == "" || m.app.cfg.Provider == "openai" {
		t.Fatalf("list %v, provider %q: want it open with the reason", m.cfgPick, m.app.cfg.Provider)
	}
}

// A provider staged for removal can't also be staged as the one to switch to.
func TestConflictingProviderChangesAreRefused(t *testing.T) {
	m := busyPanel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1", APIKey: "k"}}
	enterOn(m, "removeprovider")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // removal of mylab staged
	enterOn(m, "provider")
	typeKeys(m, "mylab")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfgPick == nil || !strings.Contains(m.cfgPick.err, "removal") || len(m.cfgPending) != 1 {
		t.Fatalf("pending %+v: the switch should be refused", m.cfgPending)
	}
}

// What fails when the task ends is said in the panel, not only behind it.
func TestAStagedChangeThatFailsSaysSoInThePanel(t *testing.T) {
	m := busyPanel(t)
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.app.cfg.Provider = "mylab"
	enterOn(m, "removeprovider")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	dir := filepath.Join(m.app.workspace, ".agent") // the project pins it meanwhile
	os.MkdirAll(dir, 0o700)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"provider":"mylab"}`), 0o600)
	m.cancel = nil
	m.applyPendingConfig()
	if p := panelText(m); !strings.Contains(p, "mylab not removed") {
		t.Fatalf("the failure isn't in the panel:\n%s", p)
	}
}

// A /model lookup that resolves after the connection changed doesn't switch
// the new connection to a model found on the old one.
func TestAStaleModelLookupIsDropped(t *testing.T) {
	m := panelModel(t)
	m.state = stRunning
	was := m.app.cfg.LLM.Model
	m.Update(modelsMsg{setTo: "old-connection-model", epoch: m.app.modelEpoch.Load() - 1})
	if m.app.cfg.LLM.Model != was {
		t.Fatalf("model %q: a stale lookup switched it", m.app.cfg.LLM.Model)
	}
}

// Bare /ai behind other work (a /model lookup has no cancel) stays a listing.
func TestBareAIBehindOtherWorkStaysAListing(t *testing.T) {
	m := panelModel(t)
	m.state = stRunning // a /model lookup in flight: no m.cancel
	m.commandWhileBusy("/ai")
	if m.state == stPick {
		t.Fatal("/ai opened a list behind the lookup")
	}
}

// A failed save leaves the setting as it was.
func TestOfflineRollsBackOnAFailedSave(t *testing.T) {
	m := panelModel(t)
	blockSaves(t)
	m.app.offlineCommand("on")
	if m.app.cfg.Offline {
		t.Fatal("offline is on in memory after a failed save")
	}
}

// A provider staged for removal can't have its settings or model staged too
// (saved after the removal, they brought back a broken stub), and the reverse.
func TestRemovalAndChangesToTheSameProviderAreRefused(t *testing.T) {
	m := busyPanel(t)
	enterOn(m, "removeprovider") // nothing saved: the row says so and stages nothing
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.app.cfg.Provider = "mylab"
	_ = m.app.wire()
	editRow(t, m, "temperature", "0.5") // mylab's
	enterOn(m, "removeprovider")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfgPick == nil || !strings.Contains(m.cfgPick.err, "staged") {
		t.Fatalf("removal over a staged change was not refused: %+v", m.cfgPick)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})

	m2 := busyPanel(t)
	m2.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m2.app.cfg.Provider = "mylab"
	_ = m2.app.wire()
	enterOn(m2, "removeprovider")
	m2.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	m2.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // removal staged
	editRow(t, m2, "temperature", "0.5")
	if len(m2.cfgPending) != 1 {
		t.Fatalf("pending %+v: tuning a provider staged for removal must be refused", m2.cfgPending)
	}
	m2.cancel = nil
	m2.applyPendingConfig()
	if _, back := m2.app.cfg.Providers["mylab"]; back {
		t.Fatal("the removed provider came back")
	}
}

// A failed save during replay is counted as not saved, and the row's value
// stays what is on disk.
func TestAFailedReplayIsNotCountedAsSaved(t *testing.T) {
	m := busyPanel(t)
	enterOn(m, "spawn")
	blockSaves(t)
	was := m.app.cfg.Spawn.Default
	m.cancel = nil
	m.applyPendingConfig()
	if m.app.cfg.Spawn.Default != was {
		t.Fatalf("spawn %q in memory after a failed save, want %q", m.app.cfg.Spawn.Default, was)
	}
	if h := strings.Join(m.history, "\n"); !strings.Contains(h, "saved 0 staged") || !strings.Contains(h, "1 not saved") {
		t.Fatalf("the failure isn't counted:\n%s", h)
	}
}

// A list left open over the task stays open when nothing about the
// connection changed.
func TestAnOpenListSurvivesAnUnrelatedReplay(t *testing.T) {
	m := busyPanel(t)
	enterOn(m, "goal_nudge")
	enterOn(m, "provider")
	m.cancel = nil
	m.applyPendingConfig()
	if m.cfgPick == nil {
		t.Fatal("the provider list was closed by a goal-nudge change")
	}
}

// A stale reply to an earlier /model doesn't spoil the list now open.
func TestAStaleReplyDoesNotSpoilTheModelList(t *testing.T) {
	m := panelModel(t)
	m.state = stIdle
	m.runCommand("/model")
	m.Update(pickListMsg{err: "old", epoch: m.app.modelEpoch.Load() - 1})
	if !m.pick.loading || m.pick.err != "" {
		t.Fatalf("list %+v: a stale reply loaded into it", m.pick)
	}
}
