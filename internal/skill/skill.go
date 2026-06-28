// Package skill manages downloadable, on-demand instruction packs ("skills").
// A skill is a markdown file with light YAML-ish frontmatter (name, description,
// optional when). Only ENABLED skills contribute a single index line to the
// system prompt; the model pulls a skill's full body on demand through the skill
// tool. So the base prompt stays lean no matter how many skills are installed —
// the same guides-on-demand idea, but user-extensible and downloadable.
package skill

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// builtinFS holds the curated skills that ship in the binary, so there is
// something to enable out of the box. They are seeded into the skills directory
// on first run, DISABLED by default — discoverable via /skills, opt-in so the
// base prompt stays lean.
//
//go:embed builtin/*.md
var builtinFS embed.FS

const maxSkillBytes = 100_000

// Skill is one instruction pack.
type Skill struct {
	Name        string
	Description string
	When        string
	Body        string
	Enabled     bool
	Source      string
}

// entry is the persisted per-skill state (the .md file holds the content).
type entry struct {
	Enabled bool   `json:"enabled"`
	Source  string `json:"source,omitempty"`
}

// Store is the on-disk skills directory plus the enabled/source state.
type Store struct {
	dir   string
	http  *http.Client
	state map[string]entry
}

// Open prepares the skills directory and loads the enabled/source state.
func Open(dir string, hc *http.Client) (*Store, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	s := &Store{dir: dir, http: hc, state: map[string]entry{}}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	if data, err := os.ReadFile(s.statePath()); err == nil {
		_ = json.Unmarshal(data, &s.state)
	}
	s.seedBuiltins()
	return s, nil
}

// seedBuiltins copies embedded skills into the directory, DISABLED by default, so
// there's something to enable out of the box without bloating the prompt. Each
// built-in is seeded at most once (tracked by name in .seeded), so a built-in the
// user later removed stays removed — but a NEW built-in added in a later release
// is still seeded on upgrade (the old design used a single marker and missed
// those entirely). It never clobbers a user file of the same name.
func (s *Store) seedBuiltins() {
	seeded := s.loadSeeded()
	entries, _ := builtinFS.ReadDir("builtin")
	changed := false
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".md")
		if seeded[name] {
			continue // already seeded once (even if the user later removed it)
		}
		seeded[name] = true
		changed = true
		if _, err := os.Stat(s.skillPath(name)); err == nil {
			continue // a user file already owns this name
		}
		data, err := builtinFS.ReadFile("builtin/" + e.Name())
		if err != nil {
			continue
		}
		if os.WriteFile(s.skillPath(name), data, 0o644) == nil {
			if _, ok := s.state[name]; !ok {
				s.state[name] = entry{Enabled: false, Source: "built-in"}
			}
		}
	}
	if changed {
		_ = s.saveState()
		_ = s.saveSeeded(seeded)
	}
}

func (s *Store) seededPath() string { return filepath.Join(s.dir, ".seeded") }

// loadSeeded reads the set of built-in names already seeded. A fresh install has
// no file (seed everything). The old format was a single "1" marker; migrate it
// by treating every built-in that currently has a file as already seeded, so the
// upgrade seeds only the genuinely new ones.
func (s *Store) loadSeeded() map[string]bool {
	set := map[string]bool{}
	data, err := os.ReadFile(s.seededPath())
	if err != nil {
		return set
	}
	var names []string
	if json.Unmarshal(data, &names) == nil {
		for _, n := range names {
			set[n] = true
		}
		return set
	}
	entries, _ := builtinFS.ReadDir("builtin") // old "1" marker → migrate
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".md")
		if _, err := os.Stat(s.skillPath(name)); err == nil {
			set[name] = true
		}
	}
	return set
}

func (s *Store) saveSeeded(set map[string]bool) error {
	names := make([]string, 0, len(set))
	for n := range set {
		names = append(names, n)
	}
	sort.Strings(names)
	data, err := json.Marshal(names)
	if err != nil {
		return err
	}
	return os.WriteFile(s.seededPath(), data, 0o644)
}

func (s *Store) statePath() string { return filepath.Join(s.dir, "state.json") }
func (s *Store) skillPath(name string) string {
	return filepath.Join(s.dir, name+".md")
}

func (s *Store) saveState() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.statePath(), data, 0o644)
}

// List returns every installed skill (enabled and disabled), sorted by name. A
// skill file with no state entry counts as enabled, so files dropped in by hand
// work without ceremony.
func (s *Store) List() []Skill {
	matches, _ := filepath.Glob(filepath.Join(s.dir, "*.md"))
	out := make([]Skill, 0, len(matches))
	for _, p := range matches {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		name := strings.TrimSuffix(filepath.Base(p), ".md")
		sk := parse(name, string(data))
		sk.Name = name // the file stem is the canonical id (frontmatter name is only used to derive it at install)
		st, ok := s.state[name]
		sk.Enabled = !ok || st.Enabled
		sk.Source = st.Source
		out = append(out, sk)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns one installed skill by name.
func (s *Store) Get(name string) (Skill, bool) {
	for _, sk := range s.List() {
		if sk.Name == name {
			return sk, true
		}
	}
	return Skill{}, false
}

// HasEnabled reports whether at least one skill is enabled (the skill tool and
// the prompt index are only wired in when this is true).
func (s *Store) HasEnabled() bool {
	for _, sk := range s.List() {
		if sk.Enabled {
			return true
		}
	}
	return false
}

// Index is the compact, prompt-facing catalog of ENABLED skills — one line each,
// "- name: description". Empty when none are enabled, so the base prompt is
// untouched until skills are actually in use.
func (s *Store) Index() string {
	var b strings.Builder
	for _, sk := range s.List() {
		if !sk.Enabled {
			continue
		}
		fmt.Fprintf(&b, "- %s: %s\n", sk.Name, oneLine(sk.Description))
	}
	return strings.TrimRight(b.String(), "\n")
}

// Body returns an enabled skill's full instructions for the model to follow.
func (s *Store) Body(name string) (string, error) {
	sk, ok := s.Get(name)
	if !ok {
		return "", fmt.Errorf("no skill named %q", name)
	}
	if !sk.Enabled {
		return "", fmt.Errorf("skill %q is disabled; enable it with /skills on %s", name, name)
	}
	return sk.Body, nil
}

// SetEnabled toggles a skill on or off and persists the choice.
func (s *Store) SetEnabled(name string, on bool) error {
	if _, ok := s.Get(name); !ok {
		return fmt.Errorf("no skill named %q", name)
	}
	e := s.state[name]
	e.Enabled = on
	s.state[name] = e
	return s.saveState()
}

// Remove deletes an installed skill.
func (s *Store) Remove(name string) error {
	if _, ok := s.Get(name); !ok {
		return fmt.Errorf("no skill named %q", name)
	}
	if err := os.Remove(s.skillPath(name)); err != nil && !os.IsNotExist(err) {
		return err
	}
	delete(s.state, name)
	return s.saveState()
}

// Install downloads skills from a source and enables them. The source is a URL
// to a single .md file, or a git repository (cloned; every *.md at its root or
// under skills/ is imported). Returns the installed skill names.
func (s *Store) Install(ctx context.Context, source string) ([]string, error) {
	source = strings.TrimSpace(source)
	if source == "" {
		return nil, fmt.Errorf("install needs a .md URL or a git repo")
	}
	if isGit(source) {
		return s.installGit(ctx, source)
	}
	return s.installFile(ctx, source)
}

// installFile fetches a single markdown skill over HTTP.
func (s *Store) installFile(ctx context.Context, rawURL string) ([]string, error) {
	body, err := s.fetch(ctx, rawURL)
	if err != nil {
		return nil, err
	}
	name := skillName(body, rawURL)
	if err := s.write(name, body, rawURL); err != nil {
		return nil, err
	}
	return []string{name}, nil
}

// installGit clones a repo and imports its skill markdown files.
func (s *Store) installGit(ctx context.Context, repo string) ([]string, error) {
	tmp, err := os.MkdirTemp("", "ips-skill-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)

	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", repo, tmp)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("git clone failed: %s", oneLine(string(out)))
	}

	var files []string
	for _, glob := range []string{filepath.Join(tmp, "*.md"), filepath.Join(tmp, "skills", "*.md")} {
		m, _ := filepath.Glob(glob)
		files = append(files, m...)
	}
	var names []string
	for _, p := range files {
		if strings.EqualFold(filepath.Base(p), "readme.md") {
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		name := skillName(string(data), p)
		if err := s.write(name, string(data), repo); err != nil {
			return names, err
		}
		names = append(names, name)
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("no skill markdown found in %s (looked at *.md and skills/*.md)", repo)
	}
	return names, nil
}

// write saves a skill file, records its source, and enables it.
func (s *Store) write(name, body, source string) error {
	clipped, _ := textutil.Clip(body, maxSkillBytes)
	if err := os.WriteFile(s.skillPath(name), []byte(clipped), 0o644); err != nil {
		return err
	}
	s.state[name] = entry{Enabled: true, Source: source}
	return s.saveState()
}

func (s *Store) fetch(ctx context.Context, rawURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return "", err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch %s: http %d", rawURL, resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxSkillBytes+1))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// isGit reports whether a source string is a git repository rather than a file.
func isGit(src string) bool {
	if strings.HasPrefix(src, "git@") || strings.HasSuffix(src, ".git") {
		return true
	}
	if strings.HasSuffix(strings.ToLower(src), ".md") {
		return false
	}
	if u, err := url.Parse(src); err == nil {
		switch u.Host {
		case "github.com", "gitlab.com", "bitbucket.org":
			return true
		}
	}
	return false
}

// skillName derives a safe skill name from the body's frontmatter, falling back
// to the source's base filename.
func skillName(body, source string) string {
	if sk := parse("", body); sk.Name != "" {
		return sanitize(sk.Name)
	}
	base := strings.TrimSuffix(filepath.Base(source), ".md")
	if base == "" || base == "." || base == "/" {
		base = "skill"
	}
	return sanitize(base)
}

// sanitize reduces a name to a safe, slug-like filename stem.
func sanitize(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == ' ' || r == '_' || r == '-' || r == '.':
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "skill"
	}
	return out
}

// parse splits a skill file into frontmatter fields and body. With no
// frontmatter, the whole text is the body and the fallback name is used.
func parse(fallbackName, text string) Skill {
	sk := Skill{Name: fallbackName, Body: strings.TrimSpace(text)}
	t := strings.ReplaceAll(text, "\r\n", "\n")
	if !strings.HasPrefix(t, "---\n") {
		return sk
	}
	end := strings.Index(t[4:], "\n---")
	if end < 0 {
		return sk
	}
	front := t[4 : 4+end]
	rest := t[4+end+4:]
	sk.Body = strings.TrimSpace(strings.TrimPrefix(rest, "\n"))
	for _, line := range strings.Split(front, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		val = strings.TrimSpace(val)
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "name":
			if val != "" {
				sk.Name = val
			}
		case "description":
			sk.Description = val
		case "when", "when_to_use":
			sk.When = val
		}
	}
	return sk
}

func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	out, _ := textutil.Clip(s, 200)
	return out
}
