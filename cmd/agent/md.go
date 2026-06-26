package main

import (
	"strings"

	"github.com/charmbracelet/glamour"
)

// renderMarkdown turns the model's final answer into terminal markdown — bold,
// inline code, lists, headings, and fenced code blocks with syntax colors — the
// way Claude Code (or a code host) shows it, instead of a flat string. It is
// best-effort: on any failure, or empty input, it returns the text unchanged.
//
// The renderer bakes word-wrap in at build time, so it is rebuilt when the
// viewport width changes. renderMarkdown is only ever called from the Bubble Tea
// update loop (one goroutine), so the cache needs no lock.
var (
	mdRenderer *glamour.TermRenderer
	mdWidth    int
)

func renderMarkdown(s string, width int) string {
	if strings.TrimSpace(s) == "" {
		return s
	}
	if width < 20 {
		width = 80
	}
	if mdRenderer == nil || width != mdWidth {
		r, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(width),
		)
		if err != nil {
			return s
		}
		mdRenderer, mdWidth = r, width
	}
	out, err := mdRenderer.Render(s)
	if err != nil {
		return s
	}
	// glamour pads with leading/trailing blank lines; trim so the answer sits
	// flush in the log.
	return strings.Trim(out, "\n")
}
