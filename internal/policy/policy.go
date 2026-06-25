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

// Engine resolves policy decisions for one workspace.
type Engine struct {
	run       config.RunPolicy
	file      config.FilePolicy
	workspace string // absolute
	jailRoot  string // absolute, symlink-resolved; "" disables the jail
}

// New builds an Engine from a Config, resolving the jail root (relative to the
// workspace) and following symlinks so jail-prefix checks are sound.
func New(c config.Config) (*Engine, error) {
	e := &Engine{run: c.Run, file: c.File, workspace: filepath.Clean(c.Workspace)}
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

// Run decides whether a shell command may execute. Deny globs win over allow
// globs; an unmatched command falls through to the configured default.
func (e *Engine) Run(command string) Decision {
	if cmdMatch(e.run.Deny, command) {
		return Deny
	}
	if cmdMatch(e.run.Allow, command) {
		return Allow
	}
	return parseDefault(e.run.Default)
}

// Write decides whether a file may be written. It first enforces the jail (an
// escape is an error, never a soft decision), then applies deny/allow write
// globs, then the default.
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

// Read enforces the jail for a read; reads are not glob-gated. Returns nil when
// the path is inside the jail (or the jail is disabled).
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
	abs = filepath.Clean(abs)

	// Resolve symlinks so a symlinked path can't smuggle us out of the jail.
	// For a not-yet-existing file, resolve the nearest existing ancestor.
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	} else if parent, err := filepath.EvalSymlinks(filepath.Dir(abs)); err == nil {
		abs = filepath.Join(parent, filepath.Base(abs))
	}

	if e.jailRoot == "" {
		return abs, nil
	}
	if abs == e.jailRoot || strings.HasPrefix(abs, e.jailRoot+string(filepath.Separator)) {
		return abs, nil
	}
	return abs, fmt.Errorf("path %q escapes the workspace jail %q", path, e.jailRoot)
}

// rel returns the slash-separated path of abs relative to the jail (or
// workspace) for glob matching; falls back to the absolute path if it sits
// outside that base.
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

// cmdMatch matches a shell command as a flat string where * spans any
// characters (including / and spaces) — path-aware globbing is wrong for
// commands, since "rm -rf*" must catch "rm -rf /".
func cmdMatch(patterns []string, command string) bool {
	for _, p := range patterns {
		if wildcardMatch(p, command) {
			return true
		}
	}
	return false
}

func wildcardMatch(pattern, s string) bool {
	var b strings.Builder
	b.WriteByte('^')
	for _, r := range pattern {
		switch r {
		case '*':
			b.WriteString(".*")
		case '?':
			b.WriteByte('.')
		default:
			b.WriteString(regexp.QuoteMeta(string(r)))
		}
	}
	b.WriteByte('$')
	re, err := regexp.Compile(b.String())
	if err != nil {
		return false
	}
	return re.MatchString(s)
}
