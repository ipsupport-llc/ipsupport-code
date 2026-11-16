package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// stderrOf runs f and returns what it wrote to os.Stderr.
func stderrOf(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = old }()
	f()
	w.Close()
	out, _ := io.ReadAll(r)
	return string(out)
}

func wiredApp(t *testing.T) *app {
	t.Helper()
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	cfg.LLM.ContextWindow = 4096
	kb, _ := knowledge.Open("")
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader(""))}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	return a
}

// LM Studio has the configured id neither in its list nor loaded, and another
// model is: the plain path says both things once, sizes against the loaded
// model for now, and keeps probing so its own window replaces that later.
func TestPlainRunSaysTheModelIsMissingAndTheWindowBorrowed(t *testing.T) {
	setHome(t, t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"data":[{"id":"google/gemma-4-e2b","state":"loaded","loaded_context_length":8192}]}`)
	}))
	defer srv.Close()
	if err := config.SaveGlobal("", config.LLM{BaseURL: srv.URL + "/v1", Type: "lmstudio", Model: "gemma-4-26b@q8_0", ContextWindow: 4096}); err != nil {
		t.Fatal(err)
	}
	var a *app
	quiet := stderrOf(t, func() {
		var cleanup func()
		var err error
		a, cleanup, err = build(t.TempDir(), "", nil, false, bufio.NewReader(strings.NewReader("")))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(cleanup)
	})
	if strings.Contains(quiet, "⚠") {
		t.Errorf("startup printed %q — it can't tell plain from TUI yet, so it must stay quiet", quiet)
	}
	if a.cfg.LLM.ContextWindow != 8192 || a.windowDetected {
		t.Errorf("window %d, detected %v — want 8192 borrowed for now, still probing", a.cfg.LLM.ContextWindow, a.windowDetected)
	}

	out := stderrOf(t, a.maybeDetectWindowSync)
	for _, want := range []string{
		`"gemma-4-26b@q8_0" isn't in LM Studio's model list`,
		`context window 8192 taken from the loaded model "google/gemma-4-e2b"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stderr %q lacks %q", out, want)
		}
	}
	if again := stderrOf(t, a.maybeDetectWindowSync); again != "" {
		t.Errorf("said again: %q", again)
	}
}

// The TUI's probe lands the same notes in the log, once, and leaves a borrowed
// window provisional.
func TestTUIProbeSaysTheWindowIsBorrowed(t *testing.T) {
	a := wiredApp(t)
	m := &tuiModel{app: a, state: stIdle}
	model := a.cfg.LLM.Model
	msg := windowMsg{provider: "local", tokens: 8192, epoch: a.modelEpoch.Load(), model: model, borrowed: true,
		lm: llm.LMWindow{Tokens: 8192, From: "other-model", Known: true, Listed: true}}
	m.Update(msg)
	m.Update(msg)

	if a.windowDetected {
		t.Error("a borrowed window marked the window detected — the model's own would never replace it")
	}
	if a.cfg.LLM.ContextWindow != 8192 {
		t.Errorf("window %d, want the borrowed 8192 meanwhile", a.cfg.LLM.ContextWindow)
	}
	n := 0
	for _, l := range m.history {
		if strings.Contains(l, `taken from the loaded model "other-model"`) {
			n++
		}
	}
	if n != 1 {
		t.Errorf("borrowed-window note shown %d times, want 1; log: %q", n, m.history)
	}
	for _, l := range m.history {
		if strings.Contains(l, "model list") {
			t.Errorf("a listed model was called missing: %q", l)
		}
	}
}

// A reply from another model reaches the TUI as one warning event, however
// many replies repeat it; the plain path prints it to stderr, also once.
func TestAReplyFromAnotherModelIsSaidOnce(t *testing.T) {
	a := wiredApp(t)
	rec := &goalEvents{}
	a.uiTracer, a.tracer = rec, rec
	a.modelMismatch("wanted", "google/gemma-4-e2b")
	a.modelMismatch("wanted", "google/gemma-4-e2b")
	warnings := 0
	for _, k := range rec.kinds {
		if k == "warning" {
			warnings++
		}
	}
	if warnings != 1 {
		t.Errorf("warning events = %d, want 1 (kinds %v)", warnings, rec.kinds)
	}
	m := &tuiModel{app: a}
	lines := m.renderEvent(uiEvent{kind: "warning", fields: map[string]any{"text": "x answered"}})
	if len(lines) != 1 || !strings.Contains(lines[0], "⚠ x answered") {
		t.Errorf("warning rendered as %q", lines)
	}

	b := wiredApp(t)
	out := stderrOf(t, func() {
		b.modelMismatch("wanted", "other")
		b.modelMismatch("wanted", "other")
	})
	if strings.Count(out, `asked for model "wanted" but "other" answered`) != 1 {
		t.Errorf("plain stderr %q, want the warning once", out)
	}
}

// End to end on the plain path: a reply naming another model ends up on
// stderr through the client wire() built.
func TestAMismatchedReplyReachesTheUser(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"model\":\"google/gemma-4-e2b\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	a := wiredApp(t)
	a.cfg.LLM.BaseURL, a.cfg.LLM.Model = srv.URL+"/v1", "gemma-4-26b@q8_0"
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	out := stderrOf(t, func() {
		if _, err := a.client.Chat(context.Background(), []llm.Message{llm.User("hi")}, nil); err != nil {
			t.Error(err)
		}
	})
	if !strings.Contains(out, `asked for model "gemma-4-26b@q8_0" but "google/gemma-4-e2b" answered`) {
		t.Errorf("stderr %q", out)
	}
}

// Nothing is said when LM Studio gave no list, or listed and loaded the model.
func TestNoNotesWithoutReason(t *testing.T) {
	if n := lmNotes(llm.LMWindow{}, "m"); len(n) != 0 {
		t.Errorf("no list: %q", n)
	}
	if n := lmNotes(llm.LMWindow{Known: true, Listed: true, Tokens: 8192, From: "pub/m"}, "m"); len(n) != 0 {
		t.Errorf("our model, loaded: %q", n)
	}
}
