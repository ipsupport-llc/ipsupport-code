package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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

func TestLongestCommonPrefix(t *testing.T) {
	if got := longestCommonPrefix([]string{"/color", "/compact"}); got != "/co" {
		t.Errorf("lcp = %q, want /co", got)
	}
	if got := longestCommonPrefix([]string{"/help"}); got != "/help" {
		t.Errorf("lcp single = %q", got)
	}
}

func TestTabCompletesSingleMatch(t *testing.T) {
	m := &tuiModel{input: textinput.New()}
	m.input.SetValue("/comp")
	m.completeCommand()
	if m.input.Value() != "/compact " {
		t.Errorf("completed to %q, want '/compact '", m.input.Value())
	}
}

func TestRenderDiff(t *testing.T) {
	m := &tuiModel{width: 80, accent: lipgloss.Color("13")}
	diff := "--- a.txt\n+++ a.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+LINE2\n line3\n"
	out := strings.Join(m.renderDiff("a.txt", diff), "\n")

	if !strings.Contains(out, "Update(a.txt)") {
		t.Error("missing header Update(a.txt)")
	}
	if !strings.Contains(out, "Added 1 line, removed 1 line") {
		t.Errorf("summary not as expected in:\n%s", out)
	}
	if !strings.Contains(out, bgGreen) || !strings.Contains(out, bgRed) {
		t.Error("missing green/red row backgrounds")
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
