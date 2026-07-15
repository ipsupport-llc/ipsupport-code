//go:build !darwin && !linux

package sandbox

// wrap is a passthrough on platforms without a supported kernel sandbox
// (Seatbelt on macOS, Landlock on Linux). The command runs unconfined.
func wrap(_ string, _ Spec, name string, args []string) (string, []string, bool) {
	return name, args, false
}

func available(string) bool { return false }
