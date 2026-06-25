// Package policy is the workspace permission engine. It decides whether a shell
// command may run and whether a file may be written, using allow/deny globs from
// the workspace config, and it confines all file operations to an optional jail
// directory. The decision (Allow / Ask / Deny) is returned to the caller; the
// interactive "ask" prompt itself lives in the tool layer.
package policy

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// Decision is the outcome of a policy check.
type Decision int

const (
	Allow Decision = iota // run/write without prompting
	Ask                   // prompt the user (interactive y/n)
	Deny                  // refuse; never execute
)

func (d Decision) String() string {
	switch d {
	case Allow:
		return "allow"
	case Deny:
		return "deny"
	default:
		return "ask"
	}
}

// Engine resolves policy decisions for one workspace. Command globs are
// precompiled at construction. Allow globs match the whole command (anchored);
// deny globs match anywhere in the command, so a dangerous token is caught even
// when it's buried in `cd x && rm -rf /`.
type Engine struct {
	file       config.FilePolicy
	runDefault string
	allow      []*regexp.Regexp // anchored
	deny       []*regexp.Regexp // unanchored
	workspace  string           // absolute
	jailRoot   string           // absolute, symlink-resolved; "" disables the jail
}

// New builds an Engine from a Config, resolving the jail root (relative to the
// workspace, symlink-followed) and precompiling the command globs.
func New(c config.Config) (*Engine, error) {
	e := &Engine{
		file:       c.File,
		runDefault: c.Run.Default,
		allow:      compileGlobs(c.Run.Allow, true),
		deny:       compileGlobs(c.Run.Deny, false),
		workspace:  filepath.Clean(c.Workspace),
	}
	if c.File.Jail != "" {
		jail := c.File.Jail
		if !filepath.IsAbs(jail) {
			jail = filepath.Join(c.Workspace, jail)
		}
		jail = filepath.Clean(jail)
		if real, err := filepath.EvalSymlinks(jail); err == nil {
			jail = real
		}
		e.jailRoot = jail
	}
	return e, nil
}

// Run decides whether a shell command may execute. Deny (matched anywhere) wins
// over allow (whole-command); an unmatched command falls through to the default.
func (e *Engine) Run(command string) Decision {
	cmd := normWS(command)
	if anyMatch(e.deny, cmd) {
		return Deny
	}
	if anyMatch(e.allow, cmd) {
		return Allow
	}
	return parseDefault(e.runDefault)
}

// Write decides whether a file may be written: jail first (escape is an error),
// then deny/allow write globs, then the default.
func (e *Engine) Write(path string) (Decision, error) {
	abs, err := e.Resolve(path)
	if err != nil {
		return Deny, err
	}
	rel := e.rel(abs)
	if fileMatch(e.file.DenyWrite, rel) {
		return Deny, nil
	}
	if fileMatch(e.file.AllowWrite, rel) {
		return Allow, nil
	}
	return parseDefault(e.file.Default), nil
}

// Read enforces the jail for a read; reads are not glob-gated.
func (e *Engine) Read(path string) error {
	_, err := e.Resolve(path)
	return err
}

// Resolve returns the absolute, symlink-resolved path for a (possibly relative)
// input and errors if it escapes the jail. Relative paths resolve against the
// jail root (or the workspace when no jail is set).
func (e *Engine) Resolve(path string) (string, error) {
	abs := path
	if !filepath.IsAbs(abs) {
		base := e.jailRoot
		if base == "" {
			base = e.workspace
		}
		abs = filepath.Join(base, path)
	}
	abs = resolveSymlinks(filepath.Clean(abs))

	if e.jailRoot == "" {
		return abs, nil
	}
	if abs == e.jailRoot || strings.HasPrefix(abs, e.jailRoot+string(filepath.Separator)) {
		return abs, nil
	}
	return abs, fmt.Errorf("path %q escapes the workspace jail %q", path, e.jailRoot)
}

// resolveSymlinks follows symlinks in abs. For a not-yet-existing path it
// resolves the nearest existing ancestor and re-appends the missing tail, so a
// symlinked directory several levels up can't smuggle the path out of the jail.
func resolveSymlinks(abs string) string {
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		return real
	}
	var missing []string
	cur := abs
	for {
		parent := filepath.Dir(cur)
		if parent == cur {
			return abs // reached root without an existing ancestor
		}
		missing = append([]string{filepath.Base(cur)}, missing...)
		if real, err := filepath.EvalSymlinks(parent); err == nil {
			return filepath.Join(append([]string{real}, missing...)...)
		}
		cur = parent
	}
}

// rel returns the slash path of abs relative to the jail (or workspace) for glob
// matching; falls back to the absolute path if it sits outside that base.
func (e *Engine) rel(abs string) string {
	base := e.jailRoot
	if base == "" {
		base = e.workspace
	}
	if base != "" {
		if r, err := filepath.Rel(base, abs); err == nil && !strings.HasPrefix(r, "..") {
			return filepath.ToSlash(r)
		}
	}
	return filepath.ToSlash(abs)
}

func parseDefault(s string) Decision {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "allow":
		return Allow
	case "deny":
		return Deny
	default:
		return Ask
	}
}

// fileMatch uses path-aware globbing (supports ** and *.go) for file paths.
func fileMatch(patterns []string, rel string) bool {
	for _, p := range patterns {
		if ok, _ := doublestar.Match(p, rel); ok {
			return true
		}
	}
	return false
}

// compileGlobs turns command wildcard patterns into regexps. Path-aware globbing
// is wrong for commands ("rm -rf*" must catch "rm -rf /"), so * spans any
// characters. anchored=true wraps ^...$ (allow: whole command); anchored=false
// leaves it unanchored (deny: match anywhere).
func compileGlobs(patterns []string, anchored bool) []*regexp.Regexp {
	out := make([]*regexp.Regexp, 0, len(patterns))
	for _, p := range patterns {
		var b strings.Builder
		if anchored {
			b.WriteByte('^')
		}
		for _, r := range normWS(p) {
			switch r {
			case '*':
				b.WriteString(".*")
			case '?':
				b.WriteByte('.')
			default:
				b.WriteString(regexp.QuoteMeta(string(r)))
			}
		}
		if anchored {
			b.WriteByte('$')
		}
		if re, err := regexp.Compile(b.String()); err == nil {
			out = append(out, re)
		}
	}
	return out
}

func anyMatch(res []*regexp.Regexp, s string) bool {
	for _, re := range res {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

// normWS collapses runs of whitespace to single spaces so "rm  -rf" and
// "rm -rf" compare equal.
func normWS(s string) string { return strings.Join(strings.Fields(s), " ") }
