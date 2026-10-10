package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// bodyServer answers every request and keeps the last request body.
func bodyServer(t *testing.T) (string, func() map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		json.Unmarshal(b, &last)
		mu.Unlock()
		io.WriteString(w, tuiContent("ok"))
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return last
	}
}

// The question: a reasoning level is a setting — why wait for the task to end?
// It, and the rest of what a request is sent with, now reaches the running
// task's next request; the config is written once the task is over.
func TestTuningReachesTheRunningTask(t *testing.T) {
	m := panelModel(t)
	url, last := bodyServer(t)
	m.app.cfg.LLM.BaseURL, m.app.cfg.LLM.Type = url, "openai"
	if err := m.app.wire(); err != nil {
		t.Fatal(err)
	}
	m.cancel = func() {} // a task is running
	editRow(t, m, "temperature", "0.4")
	cursorOn(m, "reasoning")
	m.configActivate() // default → minimal
	cursorOn(m, "loop_detection")
	m.configActivate() // on → off
	if _, err := m.app.client.Chat(context.Background(), []llm.Message{llm.User("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	if b := last(); b["temperature"] != 0.4 || b["reasoning_effort"] != "minimal" {
		t.Fatalf("the next request was sent with %v", b)
	}
	if l := m.app.cfg.LLM; l.Temperature == 0.4 || len(m.app.cfg.Reasoning) != 0 || l.DisableLoopDetection {
		t.Fatalf("the config was written while the task ran: %+v %s", l, m.app.cfg.Reasoning)
	}
	m.width, m.height = 400, 200
	if p := m.renderConfigPanel(); !strings.Contains(p, "in use now") {
		t.Fatalf("the panel doesn't say the change is in use:\n%s", p)
	}

	m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	m.state = stRunning
	m.commandWhileBusy("/reasoning high")
	if len(m.queued) != 0 {
		t.Fatalf("/reasoning high was queued: %v", m.queued)
	}
	m.app.client.Chat(context.Background(), []llm.Message{llm.User("hi")}, nil)
	if b := last(); b["reasoning_effort"] != "high" {
		t.Fatalf("/reasoning high did not reach the next request: %v", b)
	}

	m.cancel = nil // the task ends
	m.applyPendingConfig()
	c := reload(t)
	if c.LLM.Temperature != 0.4 || !c.LLM.DisableLoopDetection || m.app.reasoningLevel("local", c.LLM.Model) != "high" {
		t.Fatalf("not saved when the task ended: %+v, reasoning %s", c.LLM, c.Reasoning)
	}
}

// What can't change under a running task still waits: the learning pass's
// reasoning, and a level the provider has no shape for.
func TestReasoningThatCannotApplyLiveIsQueued(t *testing.T) {
	m := panelModel(t)
	m.state = stRunning
	m.cancel = func() {}
	for _, c := range []string{"/reasoning reflect low", "/reasoning bogus"} {
		m.commandWhileBusy(c)
	}
	if len(m.queued) != 2 {
		t.Fatalf("queued %v, want both waiting for the task", m.queued)
	}
}
