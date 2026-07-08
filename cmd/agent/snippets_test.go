package main

import (
	"os"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// snipHome points HOME at a temp dir so SnippetsPath() writes there, not the real
// user config.
func snipHome(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
}

func TestSnipSaveInlineRecallPersist(t *testing.T) {
	snipHome(t)
	a := &app{}

	if got := a.snip("save greet hello there"); got.recall != "" || !strings.Contains(got.lines[0], "saved snippet: greet") {
		t.Fatalf("save = %+v", got)
	}
	// recall pulls the text back (into the input / stdout)
	if got := a.snip("greet"); got.recall != "hello there" {
		t.Fatalf("recall = %+v, want recall %q", got, "hello there")
	}
	// persisted: a fresh app loads it from disk
	if _, err := os.Stat(config.SnippetsPath()); err != nil {
		t.Fatalf("snippets file not written: %v", err)
	}
	b := &app{}
	b.loadSnippets()
	if b.snippets["greet"] != "hello there" {
		t.Errorf("reloaded snippet = %q", b.snippets["greet"])
	}
}

func TestSnipSaveLastPromptSkipsCommands(t *testing.T) {
	snipHome(t)
	a := &app{promptHist: []string{"review the diff for races", "/snip save x", "!ls"}}

	if got := a.snip("save rev"); !strings.Contains(got.lines[0], "saved snippet: rev") {
		t.Fatalf("save-from-last = %+v", got)
	}
	if a.snippets["rev"] != "review the diff for races" {
		t.Errorf("saved %q, want the last real task (not the /command or !shell)", a.snippets["rev"])
	}
}

func TestSnipSaveNoPriorPrompt(t *testing.T) {
	snipHome(t)
	a := &app{}
	if got := a.snip("save x"); got.recall != "" || !strings.Contains(got.lines[0], "no previous prompt") {
		t.Errorf("expected 'no previous prompt' error, got %+v", got)
	}
}

func TestSnipReservedName(t *testing.T) {
	snipHome(t)
	a := &app{promptHist: []string{"a task"}}
	if got := a.snip("save save some text"); !strings.Contains(got.lines[0], "reserved") {
		t.Errorf("expected reserved-word rejection, got %+v", got)
	}
	if _, ok := a.snippets["save"]; ok {
		t.Error("reserved name was stored")
	}
}

func TestSnipRecallMissAndList(t *testing.T) {
	snipHome(t)
	a := &app{}
	if got := a.snip("nope"); got.recall != "" || !strings.Contains(got.lines[0], "no such snippet") {
		t.Errorf("recall miss = %+v", got)
	}
	if got := a.snip("list"); !strings.Contains(got.lines[0], "no snippets") {
		t.Errorf("empty list = %+v", got)
	}
	a.snip("save one first text")
	a.snip("save two second text")
	list := strings.Join(a.snip("list").lines, "\n")
	if !strings.Contains(list, "one") || !strings.Contains(list, "two") {
		t.Errorf("list missing snippets:\n%s", list)
	}
}

func TestSnipRemove(t *testing.T) {
	snipHome(t)
	a := &app{}
	a.snip("save gone bye")
	if got := a.snip("rm gone"); !strings.Contains(got.lines[0], "removed snippet: gone") {
		t.Fatalf("rm = %+v", got)
	}
	if got := a.snip("gone"); !strings.Contains(got.lines[0], "no such snippet") {
		t.Errorf("recall after rm = %+v", got)
	}
	if got := a.snip("rm ghost"); !strings.Contains(got.lines[0], "no such snippet") {
		t.Errorf("rm missing = %+v", got)
	}
}
