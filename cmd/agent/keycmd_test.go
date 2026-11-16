package main

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
)

// A built-in provider whose key comes from api_key_cmd counts as configured,
// with no api_key or env var behind it.
func TestAKeyCommandCountsAsAKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	cfg := config.Default()
	cfg.Provider = "openai"
	cfg.Providers = map[string]config.LLM{"openai": {APIKeyCmd: []string{"get-key"}}}
	if err := checkActiveProvider(cfg); err != nil {
		t.Errorf("checkActiveProvider: %v", err)
	}
	a := &app{cfg: cfg}
	if _, why := a.providerConn("openai"); why != "" {
		t.Errorf("providerConn: %s", why)
	}
	found := false
	for _, n := range a.configuredProviderNames() {
		found = found || n == "openai"
	}
	if !found {
		t.Errorf("configured providers %v lack openai", a.configuredProviderNames())
	}
	cfg.Providers = nil
	a = &app{cfg: cfg}
	if _, why := a.providerConn("openai"); why == "" {
		t.Error("no key and no command still counted as configured")
	}
}

// TestKeyCmdMarker is an api_key_cmd for the tests below: it leaves a file
// behind, so a test can tell whether it ran.
func TestKeyCmdMarker(t *testing.T) {
	if f := os.Getenv("KEYCMD_MARKER"); f != "" && len(os.Args) > 0 && os.Getenv("KEYCMD_RUN") == "1" {
		os.WriteFile(f, []byte("ran"), 0o600)
		fmt.Fprintln(os.Stderr, "signing you in")
		fmt.Println("k")
		os.Exit(0)
	}
}

// Context detection after a provider switch is a background probe: it must
// not run api_key_cmd, which may open a sign-in nobody asked for.
func TestTheWindowProbeDoesNotRunTheKeyCommand(t *testing.T) {
	setHome(t, t.TempDir())
	marker := filepath.Join(t.TempDir(), "ran")
	t.Setenv("KEYCMD_MARKER", marker)
	t.Setenv("KEYCMD_RUN", "1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"data":[{"id":"m","context_length":65536}]}`)
	}))
	defer srv.Close()
	cfg := config.Default()
	cfg.Provider = "gateway"
	cfg.Providers = map[string]config.LLM{"gateway": {BaseURL: srv.URL, Model: "m",
		APIKeyCmd: []string{os.Args[0], "-test.run=^TestKeyCmdMarker$", "--", t.Name()}}}
	a := &app{cfg: cfg}
	a.detectContextWindow()
	if _, err := os.Stat(marker); err == nil {
		t.Error("the probe ran api_key_cmd")
	}
}

// A helper's stderr reaches the screen as it runs: a line in the TUI log, or
// stderr on a plain run.
func TestKeyCmdLinesAreShown(t *testing.T) {
	m := &tuiModel{app: &app{cfg: config.Default()}}
	if l := m.renderEvent(uiEvent{kind: "keycmd", fields: map[string]any{"text": "enter ABCD"}}); len(l) != 1 || !strings.Contains(l[0], "api_key_cmd: enter ABCD") {
		t.Errorf("TUI line %q", l)
	}
	rec := &keyEvents{}
	a := &app{cfg: config.Default(), uiTracer: rec, tracer: rec}
	a.keyCmdLine("enter ABCD")
	if len(rec.kinds) != 1 || rec.kinds[0] != "keycmd" {
		t.Errorf("events %v", rec.kinds)
	}
}

type keyEvents struct{ kinds []string }

func (k *keyEvents) Emit(kind string, _ map[string]any) { k.kinds = append(k.kinds, kind) }

// wire() routes a helper's stderr to the screen.
func TestWireShowsKeyCmdLines(t *testing.T) {
	setHome(t, t.TempDir())
	t.Setenv("KEYCMD_MARKER", filepath.Join(t.TempDir(), "ran"))
	t.Setenv("KEYCMD_RUN", "1")
	cfg := config.Default()
	cfg.Workspace = t.TempDir()
	kb, _ := knowledge.Open("")
	rec := &keyEvents{}
	a := &app{cfg: cfg, workspace: cfg.Workspace, kb: kb, reader: bufio.NewReader(strings.NewReader("")), uiTracer: rec}
	if err := a.wire(); err != nil {
		t.Fatal(err)
	}
	a.tracer = rec
	conn := config.LLM{APIKeyCmd: []string{os.Args[0], "-test.run=^TestKeyCmdMarker$", "--", t.Name()}}
	if _, err := llm.APIKey(context.Background(), conn); err != nil {
		t.Fatal(err)
	}
	if len(rec.kinds) != 1 || rec.kinds[0] != "keycmd" {
		t.Errorf("events %v, want the helper's line", rec.kinds)
	}
}
