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
	"runtime"
	"slices"
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
	workdir    string           // absolute base for relative paths (set by /cd); within the jail
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
	e.workdir = e.base() // default the relative-path base to the jail root / workspace
	return e, nil
}

// base is where relative paths resolve from: the session workdir (set by /cd) if
// any, otherwise the jail root, otherwise the workspace.
func (e *Engine) base() string {
	if e.workdir != "" {
		return e.workdir
	}
	if e.jailRoot != "" {
		return e.jailRoot
	}
	return e.workspace
}

// Workdir reports the current relative-path base.
func (e *Engine) Workdir() string { return e.base() }

// SetWorkdir points relative paths at dir (must resolve inside the jail). dir may
// be absolute or relative to the current base; ~ should be expanded by the caller.
func (e *Engine) SetWorkdir(dir string) (string, error) {
	abs, err := e.Resolve(dir) // resolves + enforces the jail
	if err != nil {
		return "", err
	}
	e.workdir = abs
	return abs, nil
}

// splitCommands splits a command line into the commands it chains — on &&, ||,
// ;, |, & and newline — the way the shell does: an operator inside quotes or
// after a backslash is text, not a separator. A single & matters as much as ;:
// "a & rm -fr x" backgrounds a and then runs the delete.
//
// Splitting where the shell doesn't is NOT the cautious direction. It used to
// be a regexp that ignored quoting, and `rm "a&b" -r` came apart into `rm "a`
// and `b" -r` — the executable in one segment, its -r in the other, and the
// floor saw neither as a recursive delete. The & of a redirection (2>&1, &>)
// still splits; that only parts a redirection target from its operator, never
// a command from its flags.
func splitCommands(cmd string) []string {
	var segs []string
	var b strings.Builder
	inSingle, inDouble := false, false
	rs := []rune(cmd)
	cut := func() { segs = append(segs, b.String()); b.Reset() }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case inSingle:
			if r == '\'' {
				inSingle = false
			}
		case r == '\\' && i+1 < len(rs):
			b.WriteRune(r)
			i++
			r = rs[i]
		case r == '"':
			inDouble = !inDouble
		case inDouble:
		case r == '\'':
			inSingle = true
		case r == '\n' || r == ';':
			cut()
			continue
		case r == '&' || r == '|':
			if i+1 < len(rs) && rs[i+1] == r { // && or ||
				i++
			}
			cut()
			continue
		}
		b.WriteRune(r)
	}
	return append(segs, b.String())
}

// quoteBlindOps is the other way to split: on the chaining operators without
// any notion of quoting (and not on a lone &, which also opens redirections).
var quoteBlindOps = regexp.MustCompile(`&&|\|\||[;|\n]`)

// commandSegments is every segment of BOTH splits. Neither alone is safe:
// the quote-aware split is fooled by a quote the shell ignores — one inside a
// comment (`ls # "⏎rm x⏎# "`) — or cuts at the & of a redirection
// (`rm 2>&1 -r x`); the quote-blind split is fooled by a quoted operator
// (`rm "a;b" -r`) and doesn't see a lone &. Their blind spots don't overlap,
// so the floor denies when EITHER finds a dangerous command, and an allow
// glob must match every segment of both. The cost is caution where the
// quote-blind split over-splits: `echo "a;b"` asks under an "echo *" glob.
func commandSegments(cmd string) []string {
	return append(splitCommands(cmd), quoteBlindOps.Split(cmd, -1)...)
}

// Run decides whether a shell command may execute:
//   - the hard floor (dangerous base exe / rm -r…, plus configured deny globs) → Deny;
//   - else EVERY chained segment must match an allow glob → Allow (so an allowed
//     prefix like "git *" can't smuggle a second command after && / |);
//   - else the default.
func (e *Engine) Run(command string) Decision {
	cmd := normWS(command)
	segs := commandSegments(cmd)
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
	// Windows runs commands in PowerShell (or cmd): ( ), $( ), @( ), { } and
	// [type]:: run nested code there, and cmd's %VAR% and ^ rewrite the line —
	// all invisible to a glob that sees the prefix. Never auto-allowed.
	if windowsShell && strings.ContainsAny(cmd, "()[]{}$@^%") {
		return false
	}
	if strings.Contains(cmd, "$(") || strings.Contains(cmd, "`") || strings.Contains(cmd, "${") {
		return false
	}
	// A lone & backgrounds a process. It is a separator now (see splitCommands), so
	// it no longer shows up inside a segment for the check below to refuse —
	// look for it on the whole command instead.
	if strings.Contains(strings.ReplaceAll(cmd, "&&", ""), "&") {
		return false
	}
	matchedAny := false
	for _, s := range segs {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		// Redirection/backgrounding (>, >>, <, &) can write outside the file jail or
		// detach a process — an allow glob can't see the target, so never auto-allow it.
		if strings.ContainsAny(s, "<>&") {
			return false
		}
		if !anyMatch(e.allow, s) {
			return false
		}
		matchedAny = true
	}
	return matchedAny
}

// cmdWrappers run another command given as their trailing argv (after their own
// flags / VAR=val assignments), so the hard floor must look THROUGH them — e.g.
// "xargs rm -rf" or "env FOO=1 rm -rf" would otherwise present base "xargs"/"env".
var cmdWrappers = map[string]bool{
	"xargs": true, "nohup": true, "nice": true, "ionice": true,
	"stdbuf": true, "env": true, "setsid": true, "time": true, "timeout": true,
	"command": true, "exec": true, // POSIX shell builtins that run their argument
}

// looksLikeArgValue reports whether a token is a bare number/duration (e.g. the "5"
// in "nice -n 5" or "timeout 5s"), so wrapper-flag values are skipped when looking
// through to the wrapped command. No real command is a bare number.
func looksLikeArgValue(s string) bool {
	s = strings.TrimRight(s, "smhdkgbSMHDKGB") // strip a trailing duration/size unit
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// dangerousSegment is the argv-aware hard floor: base executables unsafe whatever
// the flags, and rm with a recursive flag (reordering-proof, unlike a glob). It is
// denied even under an allow glob, and can't be turned off by config. Best-effort:
// it sees through common wrappers but not a full nested shell (sh -c "…"); the
// ask-default is the real backstop when allow-globs aren't set.
func dangerousSegment(seg string) bool { return dangerousArgv(strings.Fields(seg), 0) }

// windowsShell: commands run in PowerShell or cmd (tool.Shell), where names
// carry no case and maybe .exe, and the ways to wipe a tree or a disk have
// their own names. A variable so a test can play Windows.
var windowsShell = runtime.GOOS == "windows"

// windowsRemovers delete files: Remove-Item and its PowerShell aliases, which
// share cmd's del/erase/rd/rmdir names.
var windowsRemovers = map[string]bool{
	"remove-item": true, "ri": true, "rm": true, "rmdir": true, "rd": true, "del": true, "erase": true,
}

// windowsWipers end a machine or a disk outright, whatever their arguments.
var windowsWipers = map[string]bool{
	"format": true, "format-volume": true, "clear-disk": true, "initialize-disk": true,
	"remove-partition": true, "diskpart": true, "stop-computer": true, "restart-computer": true,
}

// windowsBase is how Windows names the program: no directory (either slash),
// no case, no executable extension.
func windowsBase(word string) string {
	if i := strings.LastIndexAny(word, `\/`); i >= 0 {
		word = word[i+1:]
	}
	word = strings.ToLower(word)
	for _, ext := range []string{".exe", ".com", ".bat", ".cmd"} {
		word = strings.TrimSuffix(word, ext)
	}
	return word
}

// windowsRecursive: -Recurse in any unambiguous PowerShell prefix (-r, -rec,
// -Recurse:$true), a short POSIX cluster for Git's rm.exe (-fr, -Rf), or cmd's
// /s, also inside a run of switches like /s/q. -Force is not one: a long
// option with an r in it is not recursion.
func windowsRecursive(args []string) bool {
	for _, a := range args {
		a = strings.ToLower(a)
		if strings.HasPrefix(a, "-r") || (len(a) <= 3 && strings.HasPrefix(a, "-") && !strings.HasPrefix(a, "--") && strings.Contains(a, "r")) {
			return true
		}
		if strings.HasPrefix(a, "/") && slices.Contains(strings.Split(a[1:], "/"), "s") {
			return true
		}
	}
	return false
}

func dangerousArgv(fields []string, depth int) bool {
	if len(fields) == 0 || depth > 4 {
		return false
	}
	// Judge the words the shell will actually run, not how they were typed:
	// `"rm" -rf`, `'rm' -r`, `r\m -fr` and `rm "-rf"` all reach the shell as
	// rm -rf, and comparing the raw text let every one of them past.
	words := make([]string, len(fields))
	for i, f := range fields {
		words[i] = shellUnquote(f)
	}
	raw := fields[0]
	fields = words
	base := filepath.Base(fields[0])
	if windowsShell {
		// From the word as typed, quotes aside: to PowerShell and cmd a
		// backslash is a path separator, not the escape shellUnquote removes.
		base = windowsBase(strings.Trim(raw, `"'`))
		if windowsWipers[base] {
			return true
		}
		// Before the POSIX rm rule below: there any r in a flag means
		// recursive, and PowerShell's rm -Force would be refused for good.
		if windowsRemovers[base] {
			return windowsRecursive(fields[1:])
		}
	}
	switch base {
	case "sudo", "doas", "mkfs", "dd", "shutdown", "reboot", "halt", "poweroff", "init":
		return true
	case "rm":
		for _, a := range fields[1:] {
			// GNU getopt takes any unambiguous prefix of a long option, and
			// --recursive is rm's only long option starting with r: --r,
			// --rec and --recursi all mean it (as does -\-r once unquoted).
			if name := strings.TrimPrefix(a, "--"); name != a && name != "" && strings.HasPrefix("recursive", name) {
				return true
			}
			if strings.HasPrefix(a, "-") && !strings.HasPrefix(a, "--") && strings.ContainsAny(a, "rR") {
				return true // -r, -R, -rf, -fr, -Rf, ...
			}
		}
	}
	if base == "command" {
		// `command -v x` / `command -V x` only look x up; they run nothing,
		// and treating them as running x denied a harmless `command -v dd`
		// for good. Any other form (plain, -p) runs its argument.
		for _, a := range fields[1:] {
			if !strings.HasPrefix(a, "-") || a == "--" {
				break
			}
			if strings.ContainsAny(a, "vV") {
				return false
			}
		}
	}
	if cmdWrappers[base] {
		rest := fields[1:] // skip the wrapper's flags, VAR=val assignments, and flag values
		for len(rest) > 0 && (strings.HasPrefix(rest[0], "-") || strings.Contains(rest[0], "=") || looksLikeArgValue(rest[0])) {
			rest = rest[1:]
		}
		return dangerousArgv(rest, depth+1)
	}
	return false
}

// shellUnquote removes the quoting a POSIX shell removes from one word: '…',
// "…", $'…' (bash), and a backslash before a character outside single quotes.
// Best-effort, like the rest of the floor: escapes inside $'…' (\x72) and
// expansions ($x, $(…)) are not evaluated — a static check can't see those,
// and the ask-default is the backstop for them.
func shellUnquote(s string) string {
	if !strings.ContainsAny(s, `'"\`) {
		return s
	}
	var b strings.Builder
	inSingle, inDouble := false, false
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case inSingle:
			if r == '\'' {
				inSingle = false
			} else {
				b.WriteRune(r)
			}
		case r == '\\' && i+1 < len(rs):
			i++
			b.WriteRune(rs[i])
		case r == '"':
			inDouble = !inDouble
		case r == '\'' && !inDouble:
			inSingle = true
		case r == '$' && !inDouble && i+1 < len(rs) && rs[i+1] == '\'':
			// $'…' opens an ANSI-C quoted word; the $ is not part of it
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Write decides whether a file may be written: jail first (escape is an error),
// then deny/allow write globs, then the default.
func (e *Engine) Write(path string) (Decision, error) {
	abs, err := e.Resolve(path)
	if err != nil {
		return Deny, err
	}
	if e.DeniedWrite(abs) {
		return Deny, nil
	}
	if fileMatch(e.file.AllowWrite, e.rel(abs)) {
		return Allow, nil
	}
	return parseDefault(e.file.Default), nil
}

// DeniedWrite reports whether an already-resolved absolute path is blocked by
// the DenyWrite floor (.env, *secret*, …). Write's own check above only
// covers the path as it resolved at that moment; a caller that waits on an
// (async, possibly slow) approval between Resolve calls must call this again
// against the freshly re-resolved path right before it actually opens the
// file, since the target could have been swapped to a symlink pointing at a
// denied file while approval was pending — Resolve alone only re-enforces the
// jail, not the deny-write glob.
func (e *Engine) DeniedWrite(abs string) bool {
	return fileMatchFold(e.file.DenyWrite, e.rel(abs))
}

// secretReadFloor blocks reading obvious credential files, so the agent can't
// slurp a .env / *secret* file and exfiltrate it (e.g. via the web tool). Reads
// are otherwise jail-checked only. Can't be turned off by config.
var secretReadFloor = []string{"**/*secret*", "**/.env", "**/.env.*"}

// Read enforces the jail for a read and refuses obvious secrets (credential files).
func (e *Engine) Read(path string) error {
	abs, err := e.Resolve(path)
	if err != nil {
		return err
	}
	if e.IsSecret(abs) {
		return fmt.Errorf("reading %s is blocked (it looks like a secrets/credentials file)", path)
	}
	return nil
}

// IsSecret reports whether an absolute path matches the secret-read floor, so the
// search/find walkers can skip it (not just the direct read path).
func (e *Engine) IsSecret(abs string) bool { return fileMatchFold(secretReadFloor, e.rel(abs)) }

// Resolve returns the absolute, symlink-resolved path for a (possibly relative)
// input and errors if it escapes the jail. Relative paths resolve against the
// jail root (or the workspace when no jail is set). A leading ~ is expanded to
// the home dir FIRST — otherwise "~/x" is treated as a relative path and joined
// under the workspace, silently creating a literal "~" directory. The jail check
// still runs after expansion, so ~ can't be used to escape it.
func (e *Engine) Resolve(path string) (string, error) {
	abs := expandTilde(path)
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(e.base(), abs)
	}
	abs, err := resolveSymlinks(filepath.Clean(abs))
	if err != nil {
		return "", err
	}

	if e.jailRoot == "" {
		return abs, nil
	}
	if abs == e.jailRoot || strings.HasPrefix(abs, e.jailRoot+string(filepath.Separator)) {
		return abs, nil
	}
	// Reported live: a model repeatedly tried different ways to reach a path
	// outside the jail (an absolute file path, then a run `cwd` set to "/",
	// then to another absolute dir) — because nothing had ever told it a jail
	// existed, each rejection just looked like a one-off failure to route
	// around, not a categorical boundary. Spelling that out here, in the one
	// error every jail-violating call already goes through (file paths, run's
	// cwd, git refs — anything via Resolve), means it doesn't depend on the
	// system prompt being remembered mid-task.
	// Deliberately doesn't mention /cd or any other escape mechanism: the
	// model can't invoke those itself (they're human-typed TUI commands), so
	// naming them here is noise, not guidance — all it needs is the plain
	// fact that retrying with a different path is pointless.
	// Resolve is shared by both a model's tool calls AND a human's own /cd —
	// reported live, a person running /cd themselves got told "Say so instead
	// of trying more variants", an instruction written for an agent that
	// makes no sense addressed to the human who just typed the command once.
	// Kept to a plain statement of fact instead: true and useful either way,
	// with no audience baked in.
	return abs, fmt.Errorf("path %q is outside the workspace jail %q — this is a hard boundary, not a one-off failure: no path or working directory outside it is reachable by ANY tool, and retrying with a different one won't help", path, e.jailRoot)
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

// maxSymlinkChase bounds resolveSymlinks' own-target chase (see below) against
// a symlink cycle (a -> b -> a) looping forever.
const maxSymlinkChase = 20

// resolveSymlinks follows symlinks in abs. For a not-yet-existing path it
// resolves the nearest existing ancestor and re-appends the missing tail, so a
// symlinked directory several levels up can't smuggle the path out of the
// jail. abs being ITSELF a symlink whose target doesn't exist (a "dangling"
// symlink) hits that same EvalSymlinks failure — but unlike an ordinary
// missing path, this one's own name isn't what the ancestor-walk should
// reconstruct: without following it first, the walk just reassembles abs's
// own (in-jail) location, the jail check approves it, and os.OpenFile then
// follows the symlink at the OS level and writes through it to wherever it
// points, jail or no jail. So a dangling symlink's own target is chased
// (possibly through a chain of them) before falling back to the ancestor walk.
func resolveSymlinks(abs string) (string, error) { return resolveSymlinksChase(abs, 0) }

func resolveSymlinksChase(abs string, depth int) (string, error) {
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		return real, nil
	}
	if depth < maxSymlinkChase {
		if target, err := os.Readlink(abs); err == nil {
			if !filepath.IsAbs(target) {
				target = filepath.Join(filepath.Dir(abs), target)
			}
			return resolveSymlinksChase(filepath.Clean(target), depth+1)
		}
	} else if _, err := os.Readlink(abs); err == nil {
		// The chase depth limit was hit, but abs is still a symlink with a
		// further hop we never followed (as opposed to a genuinely missing
		// path, where Readlink fails below). Falling through to the
		// missing-path walk would reconstruct abs's own in-jail location and
		// pass the jail check, while os.OpenFile/os.WriteFile later follow
		// the OS-level chain past this point to wherever it actually ends —
		// jail or not. Reject instead of silently trusting an unresolved tail.
		return "", fmt.Errorf("symlink chain too deep (> %d hops) resolving %q", maxSymlinkChase, abs)
	}
	var missing []string
	cur := abs
	for {
		parent := filepath.Dir(cur)
		if parent == cur {
			return abs, nil // reached root without an existing ancestor
		}
		missing = append([]string{filepath.Base(cur)}, missing...)
		if real, err := filepath.EvalSymlinks(parent); err == nil {
			return filepath.Join(append([]string{real}, missing...)...), nil
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

// fileMatchFold is fileMatch ignoring case, for DENY lists only. The default
// filesystems on macOS and Windows don't distinguish case: there .ENV is the
// .env file, SECRETS.yaml is as secret as secrets.yaml, and .GIT/config is the
// repository's own config. Allow lists stay case-sensitive — matching less is
// the cautious direction for them.
func fileMatchFold(patterns []string, rel string) bool {
	rel = strings.ToLower(rel)
	for _, p := range patterns {
		if ok, _ := doublestar.Match(strings.ToLower(p), rel); ok {
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

// normWS collapses runs of horizontal whitespace (space/tab) to a single
// space so "rm  -rf" and "rm -rf" compare equal. It deliberately does NOT
// touch newlines — strings.Fields would, since it treats \n as ordinary
// whitespace, which used to erase the newline BEFORE Run's splitCommands
// ever saw it: "echo safe\ncurl evil" collapsed to one "echo safe curl evil"
// segment, matching an "echo *" allow glob whole, while `sh -c` still ran
// both as separate commands (a real shell treats a bare newline exactly like
// `;`). Run must split (splitCommands) BEFORE normalizing, and normalizing must
// leave that split's newlines alone.
func normWS(s string) string {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == '\t' })
	return strings.Join(fields, " ")
}
