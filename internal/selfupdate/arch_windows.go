//go:build windows

package selfupdate

import "golang.org/x/sys/windows"

const imageFileMachineARM64 = 0xAA64

// An x64 binary under emulation on Windows on ARM sees GOARCH amd64; the
// machine's own architecture is only visible through IsWow64Process2 (absent
// before Windows 10 1709, where it simply stays runtime.GOARCH).
func init() {
	var process, native uint16
	if windows.IsWow64Process2(windows.CurrentProcess(), &process, &native) == nil && native == imageFileMachineARM64 {
		nativeArch = "arm64"
	}
}
