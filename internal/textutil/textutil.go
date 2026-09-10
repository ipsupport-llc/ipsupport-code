// Package textutil holds small string helpers shared across packages.
package textutil

import (
	"bytes"
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

// Tail keeps the last max bytes of s, backing off to a rune boundary so a
// multibyte character is never split; a clipped result is prefixed with "…".
// The counterpart of Clip for when the END of the text is the valuable part
// (e.g. a CLI agent's final answer after pages of progress noise).
func Tail(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := len(s) - max
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return "…" + s[cut:]
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

// BoundedWriter is an io.Writer that keeps only the first max bytes ever
// written to it, discarding the rest instead of buffering it — so wiring a
// subprocess's stdout/stderr to one caps this process's memory use at max
// regardless of how much output a runaway command produces. Truncated
// reports whether any bytes were dropped. Write always reports success
// (returns len(p), nil) since dropping the tail is by design, not an error.
type BoundedWriter struct {
	buf       bytes.Buffer
	max       int
	Truncated bool
}

// NewBoundedWriter returns a BoundedWriter that retains at most max bytes.
func NewBoundedWriter(max int) *BoundedWriter {
	return &BoundedWriter{max: max}
}

func (w *BoundedWriter) Write(p []byte) (int, error) {
	n := len(p)
	remaining := w.max - w.buf.Len()
	if remaining <= 0 {
		if n > 0 {
			w.Truncated = true
		}
		return n, nil
	}
	if len(p) > remaining {
		w.Truncated = true
		p = p[:remaining]
	}
	w.buf.Write(p)
	return n, nil
}

// String returns the retained bytes.
func (w *BoundedWriter) String() string { return w.buf.String() }

// BoundedTailWriter is an io.Writer that keeps only the most recent max
// bytes written to it — the tail counterpart of BoundedWriter — without ever
// holding more than a small bounded multiple of max bytes at once, so tailing
// a runaway command's output doesn't require buffering all of it first.
// Truncated reports whether any bytes were ever dropped.
type BoundedTailWriter struct {
	max       int
	buf       []byte
	Truncated bool
}

// NewBoundedTailWriter returns a BoundedTailWriter that retains at most the
// last max bytes written.
func NewBoundedTailWriter(max int) *BoundedTailWriter {
	if max < 0 {
		max = 0
	}
	return &BoundedTailWriter{max: max}
}

func (w *BoundedTailWriter) Write(p []byte) (int, error) {
	n := len(p)
	if w.max == 0 {
		if n > 0 {
			w.Truncated = true
		}
		return n, nil
	}
	if len(p) > w.max {
		w.Truncated = true
		p = p[len(p)-w.max:]
	}
	w.buf = append(w.buf, p...)
	if len(w.buf) > w.max {
		w.Truncated = true
		w.buf = w.buf[len(w.buf)-w.max:]
	}
	return n, nil
}

// String returns the retained tail, backed off to a UTF-8 rune boundary at
// the start since a dropped prefix can land mid-rune.
func (w *BoundedTailWriter) String() string {
	i := 0
	for i < len(w.buf) && !utf8.RuneStart(w.buf[i]) {
		i++
	}
	return string(w.buf[i:])
}
