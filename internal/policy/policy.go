// Package policy is the workspace permission engine. It decides whether a shell
// command may run and whether a file may be written, using allow/deny globs from
// the workspace config, and it confines all file operations to an optional jail
// directory. The decision (Allow / Ask / Deny) is returned to the caller; the
// interactive "ask" prompt itself lives in the tool layer.
package policy

import (
	"fmt"
	"os"
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

// shellOps splits a command on the operators that chain one command into the
// next (&&, ||, ;, |, newline). A separator inside quotes only over-splits, which
// makes allow-matching MORE cautious (more segments to satisfy) — safe by design.
var shellOps = regexp.MustCompile(`&&|\|\||[;|\n]`)

// Run decides whether a shell command may execute:
//   - the hard floor (dangerous base exe / rm -r…, plus configured deny globs) → Deny;
//   - else EVERY chained segment must match an allow glob → Allow (so an allowed
//     prefix like "git *" can't smuggle a second command after && / |);
//   - else the default.
func (e *Engine) Run(command string) Decision {
	cmd := normWS(command)
	segs := shellOps.Split(cmd, -1)
	for _, s := range segs {
		if dangerousSegment(strings.TrimSpace(s)) {
			return Deny
		}
	}
	if anyMatch(e.deny, cmd) { // configured deny globs + the glob floor, matched anywhere
		return Deny
	}
	if e.allowsAll(cmd, segs) {
		return Allow
	}
	return parseDefault(e.runDefault)
}

// allowsAll reports whether every chained segment matches an allow glob. Command
// substitution ($()/backticks/${}) is never auto-allowed — it hides a subcommand
// an allow glob can't see.
func (e *Engine) allowsAll(cmd string, segs []string) bool {
	if len(e.allow) == 0 {
		return false
	}
	if strings.Contains(cmd, "$(") || strings.Contains(cmd, "`") || strings.Contains(cmd, "${") {
		return false
	}
	matchedAny := false
	for _, s := range segs {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		if !anyMatch(e.allow, s) {
			return false
		}
		matchedAny = true
	}
	return matchedAny
}

// dangerousSegment is the argv-aware hard floor: base executables unsafe whatever
// the flags, and rm with a recursive flag (reordering-proof, unlike a glob). It is
// denied even under an allow glob, and can't be turned off by config.
func dangerousSegment(seg string) bool {
	fields := strings.Fields(seg)
	if len(fields) == 0 {
		return false
	}
	switch filepath.Base(fields[0]) {
	case "sudo", "doas", "mkfs", "dd", "shutdown", "reboot", "halt", "poweroff", "init":
		return true
	case "rm":
		for _, a := range fields[1:] {
			if a == "--recursive" {
				return true
			}
			if strings.HasPrefix(a, "-") && !strings.HasPrefix(a, "--") && strings.ContainsAny(a, "rR") {
				return true // -r, -R, -rf, -fr, -Rf, ...
			}
		}
	}
	return false
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
// jail root (or the workspace when no jail is set). A leading ~ is expanded to
// the home dir FIRST — otherwise "~/x" is treated as a relative path and joined
// under the workspace, silently creating a literal "~" directory. The jail check
// still runs after expansion, so ~ can't be used to escape it.
func (e *Engine) Resolve(path string) (string, error) {
	abs := expandTilde(path)
	if !filepath.IsAbs(abs) {
		base := e.jailRoot
		if base == "" {
			base = e.workspace
		}
		abs = filepath.Join(base, abs)
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

// expandTilde turns a leading ~ or ~/ into the user's home directory, matching
// shell behavior so the model's "~/file" lands in $HOME instead of a literal "~"
// directory under the workspace.
func expandTilde(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
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
