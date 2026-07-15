//go:build linux

package sandbox

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"

	"github.com/landlock-lsm/go-landlock/landlock"
	"golang.org/x/sys/unix"
)

// reexecFlag is the hidden argv[1] marker for the Landlock re-exec: Landlock
// restricts the CALLING process (inherited by children), so the wrapper re-runs
// our own binary, which self-restricts and then execs the real command.
const reexecFlag = "__landlock-exec"

// wrap rewrites the command to re-exec ourselves with the confinement spec when
// the mode resolves to landlock and the kernel supports it; else passthrough.
func wrap(mode string, spec Spec, name string, args []string) (string, []string, bool) {
	if resolveMode(mode) != Landlock || !kernelHasLandlock() {
		return name, args, false
	}
	exe, err := os.Executable()
	if err != nil {
		return name, args, false
	}
	data, err := json.Marshal(spec)
	if err != nil {
		return name, args, false
	}
	return exe, append([]string{reexecFlag, string(data), name}, args...), true
}

func available(mode string) bool {
	return resolveMode(mode) == Landlock && kernelHasLandlock()
}

// kernelHasLandlock probes the Landlock ABI version (>=1 means the LSM is
// enabled and usable). A raw syscall probe, so no ruleset is ever created.
func kernelHasLandlock() bool {
	v, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, 0, 0, 1 /* LANDLOCK_CREATE_RULESET_VERSION */)
	return errno == 0 && int(v) >= 1
}

// MaybeExecConfined handles the re-exec leg: if argv says we're the sandbox
// shim, apply the Landlock confinement and exec the real command — this call
// never returns on that path (it either execs or exits). A normal launch
// returns immediately.
func MaybeExecConfined() {
	if len(os.Args) < 4 || os.Args[1] != reexecFlag {
		return
	}
	var spec Spec
	if err := json.Unmarshal([]byte(os.Args[2]), &spec); err != nil {
		fatalShim("bad sandbox spec: " + err.Error())
	}
	if !kernelHasLandlock() { // gated at wrap time; a miss here means fail CLOSED
		fatalShim("landlock unavailable in this kernel")
	}

	// Write-jail posture, mirroring the Seatbelt profile: read/execute everywhere,
	// write only inside the writable roots (plus temp scratch and dev sinks).
	rw := []landlock.Rule{}
	var rwDirs []string
	for _, d := range append([]string{"/tmp", "/var/tmp", "/dev/shm"}, spec.WritableRoots...) {
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			rwDirs = append(rwDirs, d)
		}
	}
	rw = append(rw, landlock.RODirs("/"), landlock.RWDirs(rwDirs...))
	var devFiles []string
	for _, f := range []string{"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty"} {
		if _, err := os.Stat(f); err == nil {
			devFiles = append(devFiles, f)
		}
	}
	if len(devFiles) > 0 {
		rw = append(rw, landlock.RWFiles(devFiles...))
	}
	if err := landlock.V5.BestEffort().RestrictPaths(rw...); err != nil {
		fatalShim("landlock restrict failed: " + err.Error())
	}
	if !spec.AllowNetwork {
		// No rules = deny all TCP bind+connect (kernels with Landlock ABI >= 4;
		// BestEffort degrades silently below that — the app-level offline guard
		// remains the primary gate).
		if err := landlock.V5.BestEffort().RestrictNet(); err != nil {
			fatalShim("landlock net restrict failed: " + err.Error())
		}
	}

	name := os.Args[3]
	path, err := exec.LookPath(name)
	if err != nil {
		fatalShim("command not found: " + name)
	}
	if err := unix.Exec(path, os.Args[3:], os.Environ()); err != nil {
		fatalShim("exec failed: " + err.Error())
	}
}

func fatalShim(msg string) {
	fmt.Fprintln(os.Stderr, "sandbox: "+msg)
	os.Exit(126)
}
