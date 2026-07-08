//go:build darwin

package sandbox

import "os"

const sandboxExec = "/usr/bin/sandbox-exec"

// wrap runs the command under sandbox-exec with an inline SBPL profile when the
// mode resolves to seatbelt and sandbox-exec is present; otherwise passthrough.
func wrap(mode string, spec Spec, name string, args []string) (string, []string, bool) {
	if resolveMode(mode) != Seatbelt || !hasSandboxExec() {
		return name, args, false
	}
	profile := SeatbeltProfile(spec.WritableRoots, spec.AllowNetwork)
	out := append([]string{"-p", profile, name}, args...)
	return sandboxExec, out, true
}

func available(mode string) bool {
	return resolveMode(mode) == Seatbelt && hasSandboxExec()
}

func hasSandboxExec() bool {
	fi, err := os.Stat(sandboxExec)
	return err == nil && !fi.IsDir()
}
