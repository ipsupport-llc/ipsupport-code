package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func keys(f *textField, ks ...tea.KeyMsg) {
	for _, k := range ks {
		f.key(k)
	}
}

func typed(s string) tea.KeyMsg { return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)} }

// A typo in the middle of a long URL is fixed in place, not by deleting
// everything after it.
func TestTextFieldEditsInTheMiddle(t *testing.T) {
	f := newTextField("http://localhost:1234/v1", false)
	for i := 0; i < len("/v1"); i++ {
		f.key(tea.KeyMsg{Type: tea.KeyLeft})
	}
	for i := 0; i < 4; i++ {
		f.key(tea.KeyMsg{Type: tea.KeyBackspace})
	}
	keys(&f, typed("8080"))
	if got := f.value(); got != "http://localhost:8080/v1" {
		t.Fatalf("value = %q", got)
	}
	f.key(tea.KeyMsg{Type: tea.KeyCtrlU})
	if got := f.value(); got != "/v1" {
		t.Fatalf("ctrl+u kept %q, want what was after the cursor", got)
	}
	f.key(tea.KeyMsg{Type: tea.KeyCtrlE})
	keys(&f, typed("\nx\r\n"))
	if got := f.value(); got != "/v1x" {
		t.Fatalf("a pasted line break went in: %q", got)
	}
}

func TestTextFieldMasksAKey(t *testing.T) {
	f := newTextField("", true)
	keys(&f, typed("sk-secret"))
	if v := f.view(); strings.Contains(v, "secret") || !strings.Contains(v, "●●●") {
		t.Fatalf("view = %q, want the key masked", v)
	}
	if f.value() != "sk-secret" {
		t.Fatalf("value = %q", f.value())
	}
}
