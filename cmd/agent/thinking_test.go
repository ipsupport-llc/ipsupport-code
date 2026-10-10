package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// Reported: a long reasoning line slid sideways as it streamed, hard to
// read. It wraps instead: every word shown, no line past the terminal.
func TestThinkingPanelWrapsLongLines(t *testing.T) {
	long := "first " + strings.Repeat("the model keeps reasoning about the change ", 12) + "last"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"delta":{"reasoning_content":"`+long+`"}}]}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	m := &tuiModel{state: stRunning, width: 60, showThinking: true, accent: lipgloss.Color("13"), input: textarea.New(),
		app: &app{cfg: config.Default()}}
	m.app.client = llm.NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "fake"})
	if _, err := m.app.client.Chat(context.Background(), []llm.Message{llm.User("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	view := m.thinkingView()
	body := strings.Join(view, "\n")
	if !strings.Contains(body, "last") || len(view) < 4 {
		t.Fatalf("the line was not wrapped into the panel:\n%s", body)
	}
	for _, l := range view {
		if w := ansi.StringWidth(l); w > m.width {
			t.Fatalf("a line is %d wide on a %d-wide terminal: %q", w, m.width, ansi.Strip(l))
		}
	}
	if len(view) > 13 { // the header and the last 12 lines
		t.Fatalf("%d lines, want the last 12 under the header", len(view))
	}
}
