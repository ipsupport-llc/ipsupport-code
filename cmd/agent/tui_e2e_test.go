package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/exp/teatest"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

func tuiFakeServer(t *testing.T, responses ...string) string {
	t.Helper()
	i := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		resp := tuiContent("done")
		if i < len(responses) {
			resp = responses[i]
			i++
		}
		io.WriteString(w, resp)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func tuiContent(text string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": text}}},
	})
	return string(b)
}

func tuiTestApp(t *testing.T, url string) *app {
	t.Helper()
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "kb.json"))
	c := config.Default()
	c.Workspace = t.TempDir()
	c.LLM.BaseURL = url
	c.LLM.Model = "fake"
	c.Run.Default = "allow"
	c.File.Default = "allow"
	c.File.Jail = "."
	return &app{cfg: c, workspace: c.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
}

// Drive the real TUI model end to end: type a task, press Enter, and confirm the
// streamed final answer reaches the screen.
func TestTUI_E2E_StreamsAnswer(t *testing.T) {
	url := tuiFakeServer(t, tuiContent("hello from the model"), tuiContent("[]"))
	a := tuiTestApp(t, url)
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("say hi")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "hello from the model")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}

// /help renders into the work area.
func TestTUI_E2E_HelpCommand(t *testing.T) {
	a := tuiTestApp(t, tuiFakeServer(t))
	m, err := a.newTUIModel(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	tm := teatest.NewTestModel(t, m, teatest.WithInitialTermSize(120, 40))
	tm.Type("/help")
	tm.Send(tea.KeyMsg{Type: tea.KeyEnter})

	teatest.WaitFor(t, tm.Output(), func(b []byte) bool {
		return strings.Contains(string(b), "/loop")
	}, teatest.WithDuration(5*time.Second))

	tm.Send(tea.KeyMsg{Type: tea.KeyCtrlC})
	tm.WaitFinished(t, teatest.WithFinalTimeout(3*time.Second))
}
