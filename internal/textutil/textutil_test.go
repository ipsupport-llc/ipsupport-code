package textutil

import (
	"testing"
	"unicode/utf8"
)

func TestOneLine(t *testing.T) {
	if got := OneLine("  a\n\tb   c  ", 100); got != "a b c" {
		t.Errorf("collapse = %q, want 'a b c'", got)
	}
	if got := OneLine("abcdef", 3); got != "abc" {
		t.Errorf("clip = %q, want abc", got)
	}
	if got := OneLine("é", 1); got != "" { // must not split the 2-byte rune
		t.Errorf("rune-safe clip = %q, want empty", got)
	}
}

func TestClipRuneSafe(t *testing.T) {
	s := "héllo" // 'é' is 2 bytes → cutting at byte 2 would split it
	got, truncated := Clip(s, 2)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if !utf8.ValidString(got) {
		t.Errorf("clip produced invalid UTF-8: %q", got)
	}
	if got != "h" {
		t.Errorf("got %q, want %q", got, "h")
	}
}

func TestClipNoTruncation(t *testing.T) {
	if got, tr := Clip("abc", 10); tr || got != "abc" {
		t.Errorf("Clip(abc,10) = %q,%v want abc,false", got, tr)
	}
}

func TestBoundedWriterCapsMemoryNotJustOutput(t *testing.T) {
	w := NewBoundedWriter(10)
	chunk := make([]byte, 1<<20) // 1MiB per write
	for i := range chunk {
		chunk[i] = 'x'
	}
	for i := 0; i < 100; i++ { // ~100MiB total written
		n, err := w.Write(chunk)
		if err != nil || n != len(chunk) {
			t.Fatalf("Write = %d,%v, want %d,nil (must report full write)", n, err, len(chunk))
		}
	}
	if got := w.String(); len(got) != 10 {
		t.Errorf("retained %d bytes, want exactly the 10-byte cap", len(got))
	}
	if !w.Truncated {
		t.Error("Truncated should be true after writing far past the cap")
	}
}

func TestBoundedWriterNoTruncationWithinCap(t *testing.T) {
	w := NewBoundedWriter(100)
	w.Write([]byte("hello"))
	w.Write([]byte(" world"))
	if got := w.String(); got != "hello world" {
		t.Errorf("String() = %q, want %q", got, "hello world")
	}
	if w.Truncated {
		t.Error("Truncated should be false when total writes stay under the cap")
	}
}

func TestBoundedTailWriterKeepsOnlyTheTail(t *testing.T) {
	w := NewBoundedTailWriter(5)
	chunk := make([]byte, 1<<20)
	for i := range chunk {
		chunk[i] = 'a'
	}
	for i := 0; i < 50; i++ { // ~50MiB total, only the last 5 bytes should survive
		w.Write(chunk)
	}
	w.Write([]byte("tail!"))
	if got := w.String(); got != "tail!" {
		t.Errorf("String() = %q, want %q", got, "tail!")
	}
	if !w.Truncated {
		t.Error("Truncated should be true after writing far past the cap")
	}
}

func TestBoundedTailWriterRuneSafeAtDroppedPrefix(t *testing.T) {
	w := NewBoundedTailWriter(3)
	w.Write([]byte("héllo")) // 'é' is 2 bytes: h(1) é(2) l(1) l(1) o(1) = 6 bytes total
	// dropping to the last 3 bytes would land mid-'é' at a naive byte cut
	if got := w.String(); !utf8.ValidString(got) {
		t.Errorf("tail produced invalid UTF-8: %q", got)
	}
}

func TestBoundedTailWriterNoTruncationWithinCap(t *testing.T) {
	w := NewBoundedTailWriter(100)
	w.Write([]byte("hello"))
	if got := w.String(); got != "hello" || w.Truncated {
		t.Errorf("String()=%q Truncated=%v, want %q,false", got, w.Truncated, "hello")
	}
}
