package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

func panelText(m *tuiModel) string {
	m.width, m.height = 400, 200
	return ansi.Strip(m.renderConfigPanel())
}

// Reported: enter on "remove provider" did nothing. During a task the row
// said "can't change under a running task" into the log, which the panel covers. Now
// the list opens anyway, the pick is staged, and the panel says so.
func TestRemovingAProviderDuringATaskIsStagedAndSaid(t *testing.T) {
	m := panelModel(t)
	m.app.cfg.Providers = map[string]config.LLM{"airllm": {BaseURL: "https://a.example.com/v1"}, "openrouter": {APIKey: "k"}}
	m.cancel = func() {} // a task is running
	cursorOn(m, "removeprovider")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.cfgPick == nil {
		t.Fatal("no list opened during the task")
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // airllm: confirm?
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // yes
	if _, still := m.app.cfg.Providers["airllm"]; !still {
		t.Fatal("removed under the running task")
	}
	if p := panelText(m); !strings.Contains(p, "airllm is removed when the task finishes") {
		t.Fatalf("the panel doesn't say what happened:\n%s", p)
	}
	m.cancel = nil
	m.applyPendingConfig()
	if _, still := m.app.cfg.Providers["airllm"]; still {
		t.Fatal("not removed when the task ended")
	}
}

func TestPickingAModelDuringATaskIsStaged(t *testing.T) {
	m := panelModel(t)
	m.cancel = func() {}
	cursorOn(m, "model")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	m.Update(pickListMsg{items: manyItems(3), epoch: m.app.modelEpoch.Load()})
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter}) // model-00
	if m.app.cfg.LLM.Model == "model-00" || !strings.Contains(panelText(m), "staged — model model-00") {
		t.Fatalf("model %q; panel:\n%s", m.app.cfg.LLM.Model, panelText(m))
	}
	m.cancel = nil
	m.applyPendingConfig()
	if m.app.cfg.LLM.Model != "model-00" {
		t.Fatalf("model %q after the task, want model-00", m.app.cfg.LLM.Model)
	}
}

// A refusal, and what a pick answered, show under the rows.
func TestConfigSaysWhatHappenedInThePanel(t *testing.T) {
	m := panelModel(t)
	m.cancel = func() {}
	cursorOn(m, "apikey")
	m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if p := panelText(m); !strings.Contains(p, "can't change under a running task") {
		t.Fatalf("the refusal is not in the panel:\n%s", p)
	}
	m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if p := panelText(m); strings.Contains(p, "can't change under a running task") {
		t.Fatal("the note outlived the next key")
	}
}
