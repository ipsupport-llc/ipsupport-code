package main

import (
	"bytes"
	"testing"
)

// generousLineTapCap is a sanity ceiling far above any reasonable per-line
// buffer cap — chosen independently of lineTap's own constant so this test
// still fails meaningfully (rather than merely failing to compile) against
// the pre-fix lineTap, which had no cap at all.
const generousLineTapCap = 64 * 1024

func TestLineTapBufBoundedAndCarriageReturnFlushes(t *testing.T) {
	var lines []string
	tap := &lineTap{onLine: func(s string) { lines = append(lines, s) }}

	// A CLI emitting one giant line with no '\n' (a huge single-line JSON blob,
	// or '\r'-driven progress with no final newline) must not grow buf
	// unboundedly, independent of the caps already applied to the same stream
	// via BoundedTailWriter.
	chunk := bytes.Repeat([]byte("x"), 1<<20) // 1MiB per write, no '\r' or '\n' at all
	for i := 0; i < 20; i++ {                 // ~20MiB total, well beyond any reasonable cap
		if _, err := tap.Write(chunk); err != nil {
			t.Fatalf("Write returned error: %v", err)
		}
	}
	if len(tap.buf) > generousLineTapCap {
		t.Fatalf("buf grew to %d bytes after ~20MiB with no terminator, want <= %d (bounded)", len(tap.buf), generousLineTapCap)
	}

	// '\r' must trigger the same emission/reset behavior as '\n' does, so
	// '\r'-driven progress output flushes regularly instead of defeating the cap.
	lines = nil
	tap.buf = nil
	tap.Write([]byte("50%\r60%\r70%"))
	want := []string{"50%", "60%"}
	if len(lines) != len(want) || lines[0] != want[0] || lines[1] != want[1] {
		t.Fatalf("lines = %v, want %v (\\r should flush a line like \\n does)", lines, want)
	}
	if string(tap.buf) != "70%" {
		t.Fatalf("buf after \\r-terminated chunk = %q, want %q (remainder after the last \\r)", tap.buf, "70%")
	}
}
