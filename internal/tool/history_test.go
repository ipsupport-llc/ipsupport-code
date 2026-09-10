package tool

import (
	"context"
	"strings"
	"testing"
)

type fakeHistory struct {
	recent []string
	found  []string
}

func (f fakeHistory) Recent(n int) []string    { return f.recent }
func (f fakeHistory) Search(q string) []string { return f.found }

func TestHistoryRecent(t *testing.T) {
	src := fakeHistory{recent: []string{"goal: a\nanswer: did a", "goal: b\nanswer: did b"}}
	r := NewHistory(src).Call(context.Background(), "recent", map[string]any{"n": 2})
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "did a") || !strings.Contains(r.Content, "did b") {
		t.Errorf("recent = %q, want both entries", r.Content)
	}
}

func TestHistorySearchNoMatch(t *testing.T) {
	src := fakeHistory{}
	r := NewHistory(src).Call(context.Background(), "search", map[string]any{"query": "world.go"})
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "no match") {
		t.Errorf("empty search = %q, want a 'no match' message", r.Content)
	}
}

func TestHistorySearchRequiresQuery(t *testing.T) {
	r := NewHistory(fakeHistory{}).Call(context.Background(), "search", map[string]any{})
	if !r.IsError {
		t.Error("search with no query should fail validation, not silently succeed")
	}
}
