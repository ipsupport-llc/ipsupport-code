package tool

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// silent is an Approver whose prompt ran out: it says no, and that nobody said it.
type silent struct{}

func (silent) Approve(ctx context.Context, _, _ string) bool { NoAnswer(ctx); return false }

// A prompt that ran out used to read "denied by user", so the model took the
// user to be against the call. Every tool that asks now says what happened:
// nobody answered, and the model should go on without it.
func TestAnUnansweredPromptIsNotReportedAsTheUsersNo(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	repo := initRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		tl     Tool
		action string
		params map[string]any
	}{
		{"run", runToolFor(t, dir, "ask", silent{}, nil), "shell", map[string]any{"command": "echo hi"}},
		{"write", fileToolFor(t, dir, "ask", silent{}), "write", map[string]any{"path": "x.txt", "content": "x"}},
		{"edit", fileToolFor(t, dir, "ask", silent{}), "edit", map[string]any{"path": "f.txt", "find": "old", "replace": "new"}},
		{"mkdir", fileToolFor(t, dir, "ask", silent{}), "mkdir", map[string]any{"path": "sub"}},
		{"git", gitToolFor(t, repo, silent{}), "add", map[string]any{"paths": "a.txt"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.tl.Call(ctx, tc.action, tc.params)
			if !r.IsError || !strings.Contains(r.Content, "nobody answered") || strings.Contains(r.Content, "denied by user") {
				t.Errorf("result = %+v, want it to say nobody answered — not that the user denied it", r)
			}
			if strings.Contains(r.Content, "git git") {
				t.Errorf("result = %q names the call twice", r.Content)
			}
		})
	}
}

// NoAnswer marks only a no, and only under an Ask.
func TestNoAnswerMarksOnlyANo(t *testing.T) {
	saidYes := approverFunc(func(_, _ string) bool { return true })
	ok, gone := AskWith(context.Background(), "run", "ls", func(ctx context.Context, kind, detail string) bool {
		NoAnswer(ctx)
		return saidYes.Approve(ctx, kind, detail)
	})
	if !ok || gone != "" {
		t.Errorf("Ask = (%v, %q), want a yes to stay a yes", ok, gone)
	}
	if ok, gone := Ask(context.Background(), no(), "run", "ls"); ok || gone != "" {
		t.Errorf("Ask = (%v, %q), want a plain refusal with nothing to add", ok, gone)
	}
	NoAnswer(context.Background()) // outside an Ask: nothing to mark, and no panic
}

// git passes on the refusal of a path opening with an env reference — with
// its hint at where the path meant — instead of calling it outside the workspace.
func TestGitPassesOnTheUnexpandedPathHint(t *testing.T) {
	tl := gitToolFor(t, initRepo(t), yes())
	r := tl.Call(context.Background(), "clone", map[string]any{"url": "$HOME/repo"})
	if !r.IsError || !strings.Contains(r.Content, "not shell-expanded") {
		t.Errorf("clone $HOME/repo = %+v, want the not-shell-expanded refusal", r)
	}
}
