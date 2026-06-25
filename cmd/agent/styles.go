package main

import "github.com/charmbracelet/lipgloss"

// Palette uses ANSI 16-color codes so it adapts to the user's terminal theme.
var (
	cTitle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("13"))
	cYou      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("14"))
	cBot      = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
	cFinal    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15"))
	cToolCall = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	cOk       = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
	cErr      = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	cLesson   = lipgloss.NewStyle().Foreground(lipgloss.Color("13"))
	cDim      = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))

	// diffCtx styles unchanged context lines in a diff. Added/removed rows are
	// built with raw ANSI in tui.go so a full-row background can coexist with
	// chroma syntax colors (added) or stay plain white (removed).
	diffCtx = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
)

// Named colors for /color, plus a cycle order when no name is given.
var (
	colorNames = map[string]string{
		"red": "9", "green": "10", "yellow": "11", "blue": "12",
		"magenta": "13", "cyan": "14", "white": "15", "orange": "208", "pink": "213",
	}
	colorCycle = []string{"13", "14", "10", "11", "9", "12", "213", "208"}
)
