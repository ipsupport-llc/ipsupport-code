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
)
