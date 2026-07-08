//go:build !darwin

package sandbox

// wrap is a passthrough on platforms without a supported kernel sandbox yet
// (Landlock on Linux is planned). The command runs unconfined.
func wrap(_ string, _ Spec, name string, args []string) (string, []string, bool) {
	return name, args, false
}

func available(string) bool { return false }
