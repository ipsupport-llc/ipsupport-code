package main

import (
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

func TestSplitCommand(t *testing.T) {
	cmd, rest := splitCommand("/loop 3 do the thing")
	if cmd != "/loop" || rest != "3 do the thing" {
		t.Errorf("splitCommand = %q,%q", cmd, rest)
	}
}

func TestParseLoop(t *testing.T) {
	if n, g := parseLoop("5 build it"); n != 5 || g != "build it" {
		t.Errorf("parseLoop(5 build it) = %d,%q", n, g)
	}
	if n, g := parseLoop("just a task"); n != 3 || g != "just a task" {
		t.Errorf("parseLoop default count = %d,%q", n, g)
	}
	if n, g := parseLoop(""); n != 0 || g != "" {
		t.Errorf("parseLoop empty = %d,%q", n, g)
	}
}

// The pipe-through-script smoke test can't reliably confirm quit semantics, so
// verify the exit path directly: /exit must yield tea.Quit.
func TestExitCommandQuits(t *testing.T) {
	m := &tuiModel{input: textinput.New()}
	_, cmd := m.runCommand("/exit")
	if cmd == nil {
		t.Fatal("/exit returned a nil command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("/exit did not produce tea.Quit")
	}
}
