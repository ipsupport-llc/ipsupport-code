package telemetry

import "golang.org/x/sys/unix"

func platformInfo() (osVersion, chip string, memoryGB int) {
	if v, err := unix.Sysctl("kern.osproductversion"); err == nil {
		osVersion = versionPrefix(v)
	}
	if b, err := unix.Sysctl("machdep.cpu.brand_string"); err == nil {
		chip = appleChip(b)
	}
	if m, err := unix.SysctlUint64("hw.memsize"); err == nil {
		memoryGB = gb(m)
	}
	return
}
