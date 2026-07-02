// Package textutil holds small string helpers shared across packages.
package textutil

import (
	"strings"
	"unicode/utf8"
)

// OneLine collapses all runs of whitespace (incl. newlines) to single spaces and
// rune-safely clips the result to at most max bytes — the shared "flatten to one
// tidy line, capped" helper used across the packages for log/error rendering.
func OneLine(s string, max int) string {
	out, _ := Clip(strings.Join(strings.Fields(s), " "), max)
	return out
}

// Clip truncates s to at most max bytes without splitting a multibyte UTF-8 rune,
// returning the (possibly shorter) string and whether it was truncated. Backing
// off to a rune boundary avoids feeding the model — and the JSONL trace — a
// broken trailing rune (the byte-cap-halves-Cyrillic bug class).
func Clip(s string, max int) (string, bool) {
	if max < 0 || len(s) <= max {
		return s, false
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut], true
}
