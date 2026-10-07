//go:build windows

package hostarch

import "golang.org/x/sys/windows"

const imageFileMachineARM64 = 0xAA64

// IsWow64Process2 reports the native machine even to an emulated process
// (absent before Windows 10 1709, where the build's own is the answer).
func native() string {
	var process, machine uint16
	if windows.IsWow64Process2(windows.CurrentProcess(), &process, &machine) == nil && machine == imageFileMachineARM64 {
		return "arm64"
	}
	return ""
}
