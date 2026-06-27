package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// The interactive /config panel: a sectioned settings screen navigated with
// ↑↓, where Enter cycles a value in place (mode, color, channel, provider,
// permissions, run timeout) or hands off to the matching flow (model list, key
// entry, rename). esc closes it. Cycling applies and persists immediately.

// cfgRow is one line: a section header (header != "") or a selectable setting
// (key != "").
type cfgRow struct {
	header string
	key    string
}

var configRows = []cfgRow{
	{header: "Model & provider"},
	{key: "provider"},
	{key: "model"},
	{key: "apikey"},
	{header: "Behavior"},
	{key: "mode"},
	{key: "permissions"},
	{key: "timeout"},
	{header: "Appearance & updates"},
	{key: "color"},
	{key: "channel"},
	{key: "name"},
}

// cfgKeys is the selectable keys in order (cfgCursor indexes into this).
func cfgKeys() []string {
	var ks []string
	for _, r := range configRows {
		if r.key != "" {
			ks = append(ks, r.key)
		}
	}
	return ks
}

// openConfig enters the panel.
func (m *tuiModel) openConfig() {
	m.cfgCursor = 0
	m.state = stConfig
}

// configMove moves the cursor by delta, wrapping.
func (m *tuiModel) configMove(delta int) {
	n := len(cfgKeys())
	m.cfgCursor = (m.cfgCursor + delta + n) % n
}

// configKey returns the currently selected setting key.
func (m *tuiModel) configKey() string { return cfgKeys()[m.cfgCursor] }

// configRowView returns the display label, current value, and a hint for a key.
func (m *tuiModel) configRowView(key string) (label, value, hint string) {
	act := m.app.activeLLM()
	switch key {
	case "provider":
		extra := "enter: cycle"
		if len(m.configuredProviders()) < 2 {
			extra = "/ai key <name> <tok> to add one"
		}
		return "provider", m.app.providerName(), extra
	case "model":
		return "model", act.Model, "enter: choose"
	case "apikey":
		v := "— none"
		if act.APIKey != "" {
			v = "● set"
		}
		return "api key", v, "enter: set"
	case "mode":
		v := "⏵⏵ auto"
		if m.app.planMode {
			v = "⏸ plan"
		}
		return "mode", v, "enter: toggle"
	case "permissions":
		return "permissions", fmt.Sprintf("files %s · run %s", m.app.cfg.File.Default, m.app.cfg.Run.Default), "enter: cycle"
	case "timeout":
		return "run timeout", runTimeoutLabel(m.app.cfg.Run.TimeoutSeconds), "enter: cycle"
	case "color":
		return "color", colorLabel(m.accent), "enter: cycle"
	case "channel":
		return "channel", channelOf(m.app.cfg), "enter: toggle"
	case "name":
		return "name", m.app.cfg.Name, "enter: rename"
	}
	return key, "", ""
}

// configActivate handles Enter on the selected row.
func (m *tuiModel) configActivate() (tea.Model, tea.Cmd) {
	switch m.configKey() {
	case "provider":
		m.cycleProvider()
	case "mode":
		m.app.setMode(!m.app.planMode)
	case "permissions":
		m.cyclePermissions()
	case "timeout":
		m.cycleTimeout()
	case "color":
		m.setColor("") // cycle accent
	case "channel":
		m.toggleChannel()
	case "model": // needs the live model list — hand off to /model
		m.state = stIdle
		return m.runCommand("/model")
	case "apikey": // secret entry — prefill the command, let them type the token
		m.state = stIdle
		prov := m.app.providerName()
		if prov == "local" {
			prov = ""
		}
		m.input.SetValue(strings.TrimRight("/ai key "+prov, " ") + " ")
		m.input.CursorEnd()
	case "name":
		m.state = stIdle
		m.input.SetValue("/rename ")
		m.input.CursorEnd()
	}
	return m, nil
}

// configuredProviders is local plus the external providers that have a key set
// (preset or env) — the ones the panel can actually switch between.
func (m *tuiModel) configuredProviders() []string {
	out := []string{"local"}
	for _, n := range config.KnownProviders() {
		if l, ok := config.ResolveProvider(m.app.cfg, n); ok && l.APIKey != "" {
			out = append(out, n)
		}
	}
	return out
}

// cycleProvider switches to the next configured provider and re-wires.
func (m *tuiModel) cycleProvider() {
	provs := m.configuredProviders()
	if len(provs) < 2 {
		return
	}
	cur := m.app.providerName()
	idx := 0
	for i, p := range provs {
		if p == cur {
			idx = i
			break
		}
	}
	_ = m.app.setProvider(provs[(idx+1)%len(provs)]) // saves + re-wires + re-detects
}

var permCycle = []string{"ask", "allow", "deny"}

// cyclePermissions advances both the file and run defaults together through
// ask → allow → deny, persists the workspace policy, and re-wires.
func (m *tuiModel) cyclePermissions() {
	next := nextStr(m.app.cfg.File.Default, permCycle)
	m.app.cfg.File.Default = next
	m.app.cfg.Run.Default = next
	_ = config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File)
	_ = m.app.wire()
}

var timeoutCycle = []int{60, 120, 300, 600}

// cycleTimeout advances the run timeout through the preset values, persists, and
// re-wires so the run tool picks up the new default.
func (m *tuiModel) cycleTimeout() {
	m.app.cfg.Run.TimeoutSeconds = nextInt(m.app.cfg.Run.TimeoutSeconds, timeoutCycle)
	_ = config.SaveWorkspacePolicy(m.app.workspace, m.app.cfg.Run, m.app.cfg.File)
	_ = m.app.wire()
}

// nextStr returns the element after cur in cycle (wrapping); cycle[0] if cur
// isn't found or is the last.
func nextStr(cur string, cycle []string) string {
	for i, v := range cycle {
		if v == cur {
			return cycle[(i+1)%len(cycle)]
		}
	}
	return cycle[0]
}

// nextInt is nextStr for ints.
func nextInt(cur int, cycle []int) int {
	for i, v := range cycle {
		if v == cur {
			return cycle[(i+1)%len(cycle)]
		}
	}
	return cycle[0]
}

// toggleChannel flips stable ⇄ nightly and persists it.
func (m *tuiModel) toggleChannel() {
	next := "nightly"
	if channelOf(m.app.cfg) == "nightly" {
		next = "stable"
	}
	m.app.cfg.Channel = next
	_ = config.SaveChannel(next)
}

// renderConfigPanel draws the boxed, sectioned settings screen.
func (m *tuiModel) renderConfigPanel() string {
	accent := lipgloss.NewStyle().Foreground(m.accent)
	cur := m.configKey()

	lines := []string{accent.Bold(true).Render("config")}
	for _, r := range configRows {
		if r.header != "" {
			lines = append(lines, "", accent.Render("  "+r.header))
			continue
		}
		label, value, hint := m.configRowView(r.key)
		labelCol := fmt.Sprintf("%-12s", label)
		valCol := fmt.Sprintf("%-22s", value)
		if r.key == cur {
			lines = append(lines, accent.Render(" ▸ ")+accent.Bold(true).Render(labelCol+" "+valCol)+" "+cDim.Render(hint))
		} else {
			lines = append(lines, "   "+cDim.Render(labelCol)+" "+valCol+" "+cDim.Render(hint))
		}
	}
	lines = append(lines, "", cDim.Render("  ↑↓ move · enter change · esc close"))
	lines = append(lines, cDim.Render("  ~/.config/ipsupport-code/config.json (chmod 600)"))

	box := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(m.accent).Padding(0, 1)
	return box.Render(strings.Join(lines, "\n"))
}

// runTimeoutLabel renders the run timeout for the panel (0 = the built-in 60s).
func runTimeoutLabel(sec int) string {
	if sec <= 0 {
		return "60s (default)"
	}
	return (time.Duration(sec) * time.Second).String()
}

// colorLabel maps the current accent color code back to its name (or the raw
// code if it isn't one of the named colors).
func colorLabel(c lipgloss.Color) string {
	code := string(c)
	for name, v := range colorNames {
		if v == code {
			return "▮ " + name
		}
	}
	return "▮ " + code
}
