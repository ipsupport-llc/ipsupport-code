//go:build !linux

package sandbox

// MaybeExecConfined is the Landlock re-exec hook; only Linux has one. On other
// platforms a normal launch just proceeds.
func MaybeExecConfined() {}
