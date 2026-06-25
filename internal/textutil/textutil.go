// Package textutil holds small string helpers shared across packages.
package textutil

import "unicode/utf8"

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
