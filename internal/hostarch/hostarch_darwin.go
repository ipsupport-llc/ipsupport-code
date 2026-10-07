//go:build darwin

package hostarch

import "golang.org/x/sys/unix"

// hw.optional.arm64 is 1 on Apple silicon, Rosetta or not.
func native() string {
	if v, err := unix.SysctlUint32("hw.optional.arm64"); err == nil && v == 1 {
		return "arm64"
	}
	return ""
}
