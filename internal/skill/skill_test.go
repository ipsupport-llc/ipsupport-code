package skill

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestSeedNewBuiltinOnUpgrade(t *testing.T) {
	dir := t.TempDir()
	// simulate an old install: the legacy single "1" marker + one built-in file
	// already present, but not the newer ones.
	if err := os.WriteFile(filepath.Join(dir, ".seeded"), []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "review.md"), []byte("---\nname: review\n---\nbody"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir, nil) // upgrade: should seed built-ins added since
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("subagents"); !ok {
		t.Error("a new built-in must be seeded on upgrade from the legacy marker")
	}
}

func TestRefreshUnmodifiedBuiltinOnUpgrade(t *testing.T) {
	dir := t.TempDir()
	old := []byte("---\nname: subagents\n---\nOLD built-in content")
	if err := os.WriteFile(filepath.Join(dir, "subagents.md"), old, 0o644); err != nil {
		t.Fatal(err)
	}
	// .seeded says we wrote exactly `old` → the user hasn't edited it
	seeded := `{"subagents":"` + hashBytes(old) + `"}`
	if err := os.WriteFile(filepath.Join(dir, ".seeded"), []byte(seeded), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sk, _ := s.Get("subagents"); strings.Contains(sk.Body, "OLD built-in content") {
		t.Error("an unmodified built-in should be refreshed to the embedded content on upgrade")
	}
}

func TestUserEditedBuiltinKept(t *testing.T) {
	dir := t.TempDir()
	edited := []byte("---\nname: subagents\n---\nMY OWN EDITS")
	if err := os.WriteFile(filepath.Join(dir, "subagents.md"), edited, 0o644); err != nil {
		t.Fatal(err)
	}
	// .seeded records a different hash → on-disk differs → treated as user-edited
	if err := os.WriteFile(filepath.Join(dir, ".seeded"), []byte(`{"subagents":"deadbeef"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sk, _ := s.Get("subagents"); !strings.Contains(sk.Body, "MY OWN EDITS") {
		t.Error("a user-edited built-in must be kept, not overwritten on upgrade")
	}
}

func TestRemovedBuiltinStaysRemoved(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, nil) // fresh install seeds every built-in
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("subagents"); !ok {
		t.Fatal("subagents should be seeded on a fresh install")
	}
	if err := s.Remove("subagents"); err != nil {
		t.Fatal(err)
	}
	s2, err := Open(dir, nil) // re-open must NOT resurrect it
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s2.Get("subagents"); ok {
		t.Error("a removed built-in must not be re-seeded")
	}
}

func TestBuiltinsSeededDisabled(t *testing.T) {
	s, err := Open(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	list := s.List()
	if len(list) == 0 {
		t.Fatal("expected built-in skills to be seeded")
	}
	for _, sk := range list {
		if sk.Enabled {
			t.Errorf("built-in %q seeded enabled; want disabled so the prompt stays lean", sk.Name)
		}
	}
	// Disabled skills must not leak into the prompt index.
	if s.Index() != "" || s.HasEnabled() {
		t.Errorf("disabled skills leaked: index=%q hasEnabled=%v", s.Index(), s.HasEnabled())
	}
}

func TestEnableExposesToPromptAndLoad(t *testing.T) {
	s, _ := Open(t.TempDir(), nil)
	name := s.List()[0].Name

	// Disabled: body load is refused.
	if _, err := s.Body(name); err == nil {
		t.Error("Body on a disabled skill should error")
	}
	if err := s.SetEnabled(name, true); err != nil {
		t.Fatal(err)
	}
	if !s.HasEnabled() || !strings.Contains(s.Index(), name) {
		t.Errorf("after enable: hasEnabled=%v index=%q", s.HasEnabled(), s.Index())
	}
	if body, err := s.Body(name); err != nil || strings.TrimSpace(body) == "" {
		t.Errorf("Body(%q) = %q, %v", name, body, err)
	}
}

func TestInstallFromURL(t *testing.T) {
	const md = "---\nname: my skill\ndescription: does a thing\n---\nFollow these steps carefully."
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(md))
	}))
	defer srv.Close()

	s, _ := Open(t.TempDir(), srv.Client())
	names, err := s.Install(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(names) != 1 || names[0] != "my-skill" { // name sanitized from frontmatter
		t.Fatalf("installed names = %v, want [my-skill]", names)
	}
	sk, ok := s.Get("my-skill")
	if !ok || !sk.Enabled { // installed skills are enabled (the user asked for them)
		t.Fatalf("installed skill = %+v, ok=%v", sk, ok)
	}
	if sk.Description != "does a thing" || !strings.Contains(sk.Body, "Follow these steps") {
		t.Errorf("parsed skill wrong: %+v", sk)
	}
	if !strings.Contains(s.Index(), "my-skill: does a thing") {
		t.Errorf("index missing installed skill: %q", s.Index())
	}

	if err := s.Remove("my-skill"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("my-skill"); ok {
		t.Error("skill still present after Remove")
	}
}

func TestStatePersistsAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir, nil)
	name := s.List()[0].Name
	if err := s.SetEnabled(name, true); err != nil {
		t.Fatal(err)
	}
	// A fresh store over the same dir must remember the toggle.
	s2, _ := Open(dir, nil)
	if sk, _ := s2.Get(name); !sk.Enabled {
		t.Errorf("enabled state not persisted for %q", name)
	}
}

// TestOpenNullStateFile covers a state.json whose content is the literal `null`
// — valid JSON, so json.Unmarshal returns no error, but it sets s.state itself
// to nil. seedBuiltins then panics writing into it unless Open guards against a
// nil result.
func TestOpenNullStateFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "state.json"), []byte("null"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	list := s.List()
	if len(list) == 0 {
		t.Fatal("expected built-in skills to be seeded, as on a fresh install")
	}
	for _, sk := range list {
		if sk.Enabled {
			t.Errorf("built-in %q seeded enabled; want disabled like a fresh install", sk.Name)
		}
	}
}

// TestStoreConcurrentAccess covers the real deployment shape: the same *Store
// is wired into both the foreground and background/sub-agent tool registries,
// so a mutator (SetEnabled) and readers (List, HasEnabled) run concurrently on
// the same underlying state. Run with -race.
func TestStoreConcurrentAccess(t *testing.T) {
	s, err := Open(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	list := s.List()
	if len(list) == 0 {
		t.Fatal("expected built-in skills to be seeded")
	}
	name := list[0].Name

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			if err := s.SetEnabled(name, i%2 == 0); err != nil {
				t.Error(err)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			s.List()
			s.HasEnabled()
		}
	}()
	wg.Wait()
}

// runGit runs a git command in dir, failing the test on error.
func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

// TestInstallGitRejectsSymlinkedSkillFile reproduces the symlink-escape bug:
// a cloned repo's guide.md is a symlink pointing at a file OUTSIDE the clone
// (e.g. an SSH key). installGit must not copy that external file's real
// content into the installed, enabled skill.
func TestInstallGitRejectsSymlinkedSkillFile(t *testing.T) {
	secretDir := t.TempDir()
	secretPath := filepath.Join(secretDir, "id_rsa")
	const marker = "MARKER-SECRET-CONTENT-DO-NOT-LEAK"
	if err := os.WriteFile(secretPath, []byte(marker), 0o600); err != nil {
		t.Fatal(err)
	}

	repo := t.TempDir()
	runGit(t, repo, "init")
	runGit(t, repo, "config", "user.email", "test@example.com")
	runGit(t, repo, "config", "user.name", "Test")
	if err := os.Symlink(secretPath, filepath.Join(repo, "guide.md")); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", "guide.md")
	runGit(t, repo, "commit", "-m", "add guide")

	s, err := Open(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	names, err := s.installGit(context.Background(), repo)
	if err != nil {
		return // failing the whole import is an acceptable, clean rejection
	}
	for _, name := range names {
		sk, _ := s.Get(name)
		if strings.Contains(sk.Body, marker) {
			t.Fatalf("installed skill %q leaked external file content via symlink: %q", name, sk.Body)
		}
	}
}

// TestInstallGitRejectsSymlinkedAncestorDir reproduces a gap in the leaf-symlink
// guard above: a cloned repo's "skills" directory is ITSELF a symlink pointing
// outside the clone, at a directory containing an ordinary (non-symlink) file.
// filepath.Glob follows the symlinked directory transparently, so the matched
// file's own Lstat reports a regular file — the leaf-symlink guard alone
// wouldn't catch this. installGit must not copy the external file's content
// into an installed, enabled skill.
func TestInstallGitRejectsSymlinkedAncestorDir(t *testing.T) {
	externalDir := t.TempDir()
	const marker = "TOP-SECRET-EXTERNAL-CONTENT"
	if err := os.WriteFile(filepath.Join(externalDir, "confidential.md"), []byte(marker), 0o644); err != nil {
		t.Fatal(err)
	}

	repo := t.TempDir()
	runGit(t, repo, "init")
	runGit(t, repo, "config", "user.email", "test@example.com")
	runGit(t, repo, "config", "user.name", "Test")
	if err := os.Symlink(externalDir, filepath.Join(repo, "skills")); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", "skills")
	runGit(t, repo, "commit", "-m", "add skills symlink")

	s, err := Open(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	names, err := s.installGit(context.Background(), repo)
	if err != nil {
		return // failing the whole import is an acceptable, clean rejection
	}
	for _, name := range names {
		sk, _ := s.Get(name)
		if strings.Contains(sk.Body, marker) {
			t.Fatalf("installed skill %q leaked external file content via symlinked ancestor dir: %q", name, sk.Body)
		}
	}
}

// TestInstallGitOrdinaryFiles confirms a normal git-sourced skill (a plain,
// non-symlinked .md file) still installs correctly after the symlink fix.
func TestInstallGitOrdinaryFiles(t *testing.T) {
	repo := t.TempDir()
	runGit(t, repo, "init")
	runGit(t, repo, "config", "user.email", "test@example.com")
	runGit(t, repo, "config", "user.name", "Test")
	const md = "---\nname: my skill\ndescription: does a thing\n---\nFollow these steps carefully."
	if err := os.WriteFile(filepath.Join(repo, "guide.md"), []byte(md), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", "guide.md")
	runGit(t, repo, "commit", "-m", "add guide")

	s, err := Open(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	names, err := s.installGit(context.Background(), repo)
	if err != nil {
		t.Fatalf("installGit: %v", err)
	}
	if len(names) != 1 || names[0] != "my-skill" {
		t.Fatalf("installed names = %v, want [my-skill]", names)
	}
	sk, ok := s.Get("my-skill")
	if !ok || !sk.Enabled || !strings.Contains(sk.Body, "Follow these steps") {
		t.Fatalf("installed skill = %+v, ok=%v", sk, ok)
	}
}

func TestIsGit(t *testing.T) {
	git := []string{"git@github.com:u/r.git", "https://github.com/u/r", "https://gitlab.com/u/r.git"}
	notGit := []string{"https://example.com/skill.md", "https://raw.githubusercontent.com/u/r/main/s.md"}
	for _, g := range git {
		if !isGit(g) {
			t.Errorf("isGit(%q) = false, want true", g)
		}
	}
	for _, f := range notGit {
		if isGit(f) {
			t.Errorf("isGit(%q) = true, want false", f)
		}
	}
}
