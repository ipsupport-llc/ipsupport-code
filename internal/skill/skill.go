// Package skill manages downloadable, on-demand instruction packs ("skills").
// A skill is a markdown file with light YAML-ish frontmatter (name, description,
// optional when). Only ENABLED skills contribute a single index line to the
// system prompt; the model pulls a skill's full body on demand through the skill
// tool. So the base prompt stays lean no matter how many skills are installed —
// the same guides-on-demand idea, but user-extensible and downloadable.
package skill

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
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
		if err := json.Unmarshal(data, &s.state); err != nil {
			// Don't silently reset every skill's enabled/source to defaults on a
			// corrupt file — surface it so it can be looked at. saveState only
			// overwrites on the next explicit change.
			slog.Warn("skill state file is corrupt; using defaults until it's changed", "path", s.statePath(), "err", err)
		}
	}
	s.seedBuiltins()
	return s, nil
}

// seedBuiltins keeps the embedded skills in sync on disk, DISABLED by default.
// `.seeded` records the content hash we last wrote per built-in, which lets us:
//   - seed a NEW built-in added in a later release (it isn't in .seeded yet);
//   - REFRESH a built-in's content on upgrade when the user hasn't edited it (the
//     on-disk hash still matches what we wrote) — so fixes to built-in skills
//     actually reach existing installs;
//   - leave a built-in the user EDITED (hash differs) or REMOVED (no file) alone.
//
// It never clobbers a user file of the same name on first seed.
func (s *Store) seedBuiltins() {
	seeded := s.loadSeeded()
	entries, _ := builtinFS.ReadDir("builtin")
	changed := false
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".md")
		data, err := builtinFS.ReadFile("builtin/" + e.Name())
		if err != nil {
			continue
		}
		sum := hashBytes(data)
		prev, was := seeded[name]
		if !was { // never seeded → install unless a user file already owns the name
			if _, err := os.Stat(s.skillPath(name)); err != nil {
				if os.WriteFile(s.skillPath(name), data, 0o644) == nil {
					if _, ok := s.state[name]; !ok {
						s.state[name] = entry{Enabled: false, Source: "built-in"}
					}
				}
			}
			seeded[name], changed = sum, true
			continue
		}
		if prev == sum { // embedded content unchanged since we wrote it
			continue
		}
		// Embedded content changed: refresh ONLY if the on-disk file is still what
		// we last wrote (user hasn't edited it). prev == "" is a pre-hash upgrade —
		// treat the on-disk copy as a previous built-in version and refresh it.
		onDisk, rerr := os.ReadFile(s.skillPath(name))
		if rerr == nil && (prev == "" || hashBytes(onDisk) == prev) {
			if os.WriteFile(s.skillPath(name), data, 0o644) == nil {
				seeded[name], changed = sum, true
			}
		}
		// else: user edited it (keep their copy) or removed it (stays removed).
	}
	if changed {
		_ = s.saveState()
		_ = s.saveSeeded(seeded)
	}
}

func (s *Store) seededPath() string { return filepath.Join(s.dir, ".seeded") }

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// loadSeeded reads the per-built-in content hashes already written. Handles three
// formats: the current {name: hash} map; the previous ["name", …] list (unknown
// hashes → "" so they refresh once); and the oldest single "1" marker (migrate by
// treating every built-in that has a file on disk as seeded).
func (s *Store) loadSeeded() map[string]string {
	out := map[string]string{}
	data, err := os.ReadFile(s.seededPath())
	if err != nil {
		return out
	}
	if json.Unmarshal(data, &out) == nil && len(out) > 0 {
		return out
	}
	out = map[string]string{}
	var names []string
	if json.Unmarshal(data, &names) == nil {
		for _, n := range names {
			out[n] = ""
		}
		return out
	}
	entries, _ := builtinFS.ReadDir("builtin") // oldest "1" marker → migrate
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".md")
		if _, err := os.Stat(s.skillPath(name)); err == nil {
			out[name] = ""
		}
	}
	return out
}

func (s *Store) saveSeeded(m map[string]string) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return atomicfile.Write(s.seededPath(), data, 0o644)
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
	return atomicfile.Write(s.statePath(), data, 0o644)
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
	seen := map[string]bool{}
	for _, p := range files {
		if strings.EqualFold(filepath.Base(p), "readme.md") {
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		name := skillName(string(data), p)
		if seen[name] { // two files sanitize to the same slug — don't silently overwrite
			continue
		}
		seen[name] = true
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

func oneLine(s string) string { return textutil.OneLine(s, 200) }
