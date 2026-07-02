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
