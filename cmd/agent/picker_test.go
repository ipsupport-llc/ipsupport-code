package main

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

func manyItems(n int) []pickItem {
	var out []pickItem
	for i := range n {
		out = append(out, pickItem{value: fmt.Sprintf("model-%02d", i)})
	}
	return out
}

func TestPickerScrollsFiltersAndTakesTypedText(t *testing.T) {
	p := newPicker(manyItems(40), "model-25")
	if v, _ := p.choice(); v != "model-25" {
		t.Fatalf("starts on %q, want the current one", v)
	}
	view := strings.Join(p.view(lipgloss.NewStyle(), "", ""), "\n")
	if !strings.Contains(view, "more") || !strings.Contains(view, "model-25") || strings.Contains(view, "model-00") {
		t.Fatalf("not a scrolled window around the cursor:\n%s", view)
	}
	p.key(tea.KeyMsg{Type: tea.KeyEnd})
	if v, _ := p.choice(); v != "model-39" {
		t.Fatalf("end → %q", v)
	}
	p.key(tea.KeyMsg{Type: tea.KeyDown}) // wraps
	if v, _ := p.choice(); v != "model-00" {
		t.Fatalf("down from the last → %q", v)
	}
	p.key(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
	if n := len(p.visible()); n != 13 { // 03, 13, 23, 30–39
		t.Fatalf("filter 3 → %d items", n)
	}
	p.key(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if _, ok := p.choice(); ok {
		t.Fatal("a fixed list took text it doesn't have")
	}
	p.free = true
	if v, ok := p.choice(); !ok || v != "3x" {
		t.Fatalf("a free list should take the typed text, got %q", v)
	}
}

// The complaint: picking a model in /config meant typing its name. Now the
// server's models are listed on the row, and one it doesn't list can still be typed.
func TestConfigModelRowIsAList(t *testing.T) {
	m := panelModel(t)
	cursorOn(m, "model")
	_, cmd := m.configActivate()
	if m.cfgPick == nil || !m.cfgPick.p.loading || cmd == nil {
		t.Fatal("the model row did not open a list and fetch it")
	}
	m.Update(pickListMsg{items: manyItems(30), epoch: m.app.modelEpoch.Load()})
	if m.cfgPick.p.loading || len(m.cfgPick.p.visible()) != 30 {
		t.Fatalf("list not loaded: %+v", m.cfgPick.p)
	}
	m.width, m.height = 400, 200
	panel := m.renderConfigPanel()
	if !strings.Contains(panel, "model-00") || !strings.Contains(panel, "↓ 20 more") {
		t.Fatalf("the list is not shown on the row:\n%s", panel)
	}
	typeKeys(m, "17")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfgPick != nil || m.app.cfg.LLM.Model != "model-17" || reload(t).LLM.Model != "model-17" {
		t.Fatalf("model = %q, want model-17 picked and saved", m.app.cfg.LLM.Model)
	}
	// Not listed: what was typed is used.
	m.configActivate()
	m.Update(pickListMsg{items: manyItems(3), epoch: m.app.modelEpoch.Load()})
	typeKeys(m, "my-own-model")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.app.cfg.LLM.Model != "my-own-model" {
		t.Fatalf("model = %q, want the typed name", m.app.cfg.LLM.Model)
	}
}

// A list fetched for another connection is not shown for this one.
func TestAStaleModelListIsDropped(t *testing.T) {
	m := panelModel(t)
	cursorOn(m, "model")
	m.configActivate()
	m.Update(pickListMsg{items: manyItems(3), epoch: m.app.modelEpoch.Load() - 1})
	if !m.cfgPick.p.loading {
		t.Fatal("a list for an older connection was loaded")
	}
}

func TestConfigProviderRowIsAList(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"airllm": {BaseURL: "https://a.example.com/v1"}, "mylab": {BaseURL: "https://lab.example.com/v1"}}
	cursorOn(m, "provider")
	m.configActivate()
	if m.cfgPick == nil || len(m.cfgPick.p.visible()) != 3 {
		t.Fatalf("want local, airllm, mylab listed: %+v", m.cfgPick)
	}
	typeKeys(m, "lab")
	if _, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}); cmd == nil {
		t.Fatal("switching did not probe the window")
	}
	if m.app.cfg.Provider != "mylab" || m.cfgPick != nil {
		t.Fatalf("provider %q, list open %v", m.app.cfg.Provider, m.cfgPick != nil)
	}
}

func TestModelCommandOpensAList(t *testing.T) {
	m := panelModel(t)
	m.state = stIdle
	_, cmd := m.runCommand("/model")
	if m.state != stPick || m.pickKind != "model" || cmd == nil {
		t.Fatalf("state %v kind %q: /model did not open a list", m.state, m.pickKind)
	}
	m.Update(pickListMsg{items: manyItems(5), epoch: m.app.modelEpoch.Load()})
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.state != stIdle || m.app.cfg.LLM.Model != "model-02" {
		t.Fatalf("state %v, model %q", m.state, m.app.cfg.LLM.Model)
	}
	m.runCommand("/model")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.state != stIdle || m.app.cfg.LLM.Model != "model-02" {
		t.Fatal("esc did not leave everything as it was")
	}
}

func TestAICommandOpensAList(t *testing.T) {
	m := panelModel(t)
	m.state = stIdle
	m.app.cfg.Providers = map[string]config.LLM{"mylab": {BaseURL: "https://lab.example.com/v1"}}
	m.runCommand("/ai")
	if m.state != stPick || m.pickKind != "provider" {
		t.Fatalf("state %v kind %q", m.state, m.pickKind)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.app.cfg.Provider != "mylab" || m.state != stIdle {
		t.Fatalf("provider %q, state %v", m.app.cfg.Provider, m.state)
	}
}

func TestSessionsCommandOpensAList(t *testing.T) {
	m := panelModel(t)
	m.state = stIdle
	if err := m.app.wire(); err != nil {
		t.Fatal(err)
	}
	m.app.ag.SetHistory([]llm.Message{llm.User("g0"), {Role: "assistant", Content: "a0"}})
	m.app.saveSession()
	m.app.sessionsCommand("alice")
	m.app.ag.SetHistory([]llm.Message{llm.User("g1"), {Role: "assistant", Content: "a1"}})
	m.app.saveSession()
	m.runCommand("/sessions")
	if m.state != stPick || m.pickKind != "session" || len(m.pick.visible()) != 2 {
		t.Fatalf("state %v kind %q", m.state, m.pickKind)
	}
	typeKeys(m, "ipsupport")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.app.cfg.Name != "ipsupport-code" || m.state != stIdle {
		t.Fatalf("session %q, state %v", m.app.cfg.Name, m.state)
	}
}

// Over a running task the bare forms only list: a pick there would switch the
// provider or the thread under it.
func TestListsOverARunningTaskDoNotSwitch(t *testing.T) {
	m := panelModel(t)
	m.state = stRunning
	m.cancel = func() {}
	for _, c := range []string{"/ai", "/sessions"} {
		m.commandWhileBusy(c)
		if m.state != stRunning || m.pick != nil {
			t.Fatalf("%s opened a list over the task: state %v", c, m.state)
		}
	}
}

// Bare /reasoning says what each part uses now, not only how to change it.
func TestReasoningShowsTheCurrentSettings(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.LLM.Model = "m"
	m.app.reasoningCommand("low")
	m.app.reasoningCommand("judge minimal")
	out := strings.Join(m.app.reasoningCommand(""), "\n")
	for _, want := range []string{"task:          low", "learning pass: same as the task", "goal judge:    minimal"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

// The judge row showed "custom" for any level set, and its cycle never left minimal.
func TestJudgeReasoningRowShowsAndCyclesItsLevel(t *testing.T) {
	m := panelModel(t)
	m.app.reasoningCommand("judge minimal")
	if _, v, _ := m.configRowView("judge_reasoning"); v != "minimal" {
		t.Fatalf("row shows %q, want minimal", v)
	}
	cursorOn(m, "judge_reasoning")
	m.configActivate()
	if _, v, _ := m.configRowView("judge_reasoning"); v != "low" {
		t.Fatalf("after enter the row shows %q, want low", v)
	}
}
