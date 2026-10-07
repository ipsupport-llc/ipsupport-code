package hostarch

import (
	"runtime"
	"testing"
)

// A native build on a native machine reports its own architecture; nothing
// here runs under emulation, so the answer is the build's.
func TestNativeIsTheBuildsWhenNotEmulated(t *testing.T) {
	if got := Native(); got != runtime.GOARCH {
		t.Fatalf("Native() = %q, want %q", got, runtime.GOARCH)
	}
}
