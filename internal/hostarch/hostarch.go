// Package hostarch reports the machine's own CPU architecture, which is not
// always the one this binary was built for: an amd64 build runs under
// emulation on Windows on ARM and under Rosetta on Apple silicon, and there
// runtime.GOARCH says amd64.
package hostarch

import "runtime"

// Native is the machine's architecture in GOARCH terms ("amd64", "arm64").
func Native() string {
	if a := native(); a != "" {
		return a
	}
	return runtime.GOARCH
}
