package skill

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
