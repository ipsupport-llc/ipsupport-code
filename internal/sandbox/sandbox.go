// Package sandbox is an OS-level containment layer that runs shell commands
// inside the kernel sandbox of the host platform (Seatbelt on macOS; Landlock on
// Linux is planned). It sits UNDER the workspace policy engine — the policy
// decides allow/ask/deny and asks for approval; the sandbox then confines
// whatever runs so even an allowed command can't write outside the workspace.
//
// The profile generation is pure and cross-platform (so it's testable
// everywhere); only the exec wrapping is platform-specific (wrap_*.go).
package sandbox

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
)

// Modes for the `sandbox` config setting.
const (
	Off      = "off"
	Seatbelt = "seatbelt"
	Landlock = "landlock"
	Auto     = "auto" // pick the platform's mechanism (seatbelt on macOS, landlock on Linux)
)

// Spec describes the confinement for one workspace: the roots a command may
// write to (everything else is read-only), and whether the network is allowed.
type Spec struct {
	WritableRoots []string
	AllowNetwork  bool
}

// scratchDirs are writable under every spec, beside its roots: the temp dirs
// commands expect to write to.
var scratchDirs = map[string][]string{
	Seatbelt: {"/tmp", "/private/tmp", "/private/var/folders", "/private/var/tmp"},
	Landlock: {"/tmp", "/var/tmp", "/dev/shm"},
}

// Reaches reports whether a command confined by spec could write anywhere in
// path's tree: through a writable dir holding path, or one inside it. A mode
// this host can't confine reaches everything.
func Reaches(mode string, spec Spec, path string) bool {
	return !Available(mode) || reaches(resolveMode(mode), spec, path)
}

func reaches(mode string, spec Spec, path string) bool {
	p := resolved(path)
	for _, d := range append(append([]string(nil), scratchDirs[mode]...), spec.WritableRoots...) {
		if d := resolved(d); within(p, d) || within(d, p) {
			return true
		}
	}
	return false
}

// resolved is path made absolute, its symlinks followed (/tmp is /private/tmp
// on macOS) as far as it exists.
func resolved(path string) string {
	p, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return r
	}
	return p
}

// within reports whether p is dir or inside it.
func within(p, dir string) bool {
	rel, err := filepath.Rel(dir, p)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// resolveMode maps "auto" to the current platform's mechanism; other modes pass
// through unchanged.
func resolveMode(mode string) string {
	if mode != Auto {
		return mode
	}
	switch runtime.GOOS {
	case "darwin":
		return Seatbelt
	case "linux":
		return Landlock
	default:
		return Off
	}
}

// Wrap rewrites a command (name + args) into its sandboxed form for the given
// mode, or returns it unchanged (with wrapped=false) when the mode is off or the
// platform can't sandbox. The policy engine has already decided the command may
// run; this only adds kernel confinement.
func Wrap(mode string, spec Spec, name string, args []string) (string, []string, bool) {
	return wrap(mode, spec, name, args)
}

// Available reports whether the given mode can actually confine on this host.
func Available(mode string) bool { return available(mode) }

// Strict reports whether the given mode confines fully on this host: no
// content written or truncated outside the writable roots, and, with the
// network off, no TCP. Landlock does that from ABI 4 (truncation from 3, TCP
// from 4); below, BestEffort leaves them open.
func Strict(mode string) bool { return available(mode) && strict(resolveMode(mode)) }

// SeatbeltProfile builds an SBPL profile that allows everything EXCEPT writes
// outside the writable roots (plus the standard macOS temp dirs and dev nodes),
// and denies the network when allowNetwork is false. This "write-jail" posture
// is deliberately permissive on reads/exec — the point is that a shell command
// can't modify anything outside the workspace.
func SeatbeltProfile(writableRoots []string, allowNetwork bool) string {
	var b strings.Builder
	b.WriteString("(version 1)\n")
	b.WriteString("(allow default)\n")
	b.WriteString("(deny file-write*)\n")
	b.WriteString("(allow file-write*\n")
	// Standard macOS temp/scratch locations, then the caller's workspace roots.
	roots := append(append([]string(nil), scratchDirs[Seatbelt]...), writableRoots...)
	for _, r := range roots {
		fmt.Fprintf(&b, "    (subpath %s)\n", sbplString(r))
	}
	b.WriteString(")\n")
	b.WriteString(`(allow file-write-data (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (literal "/dev/dtracehelper"))` + "\n")
	if !allowNetwork {
		b.WriteString("(deny network*)\n")
	}
	return b.String()
}

// sbplString renders s as an SBPL double-quoted string literal.
func sbplString(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"`
}
