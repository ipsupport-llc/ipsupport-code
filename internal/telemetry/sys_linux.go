package telemetry

import "golang.org/x/sys/unix"

func platformInfo() (osVersion, chip string, memoryGB int) {
	var u unix.Utsname
	if err := unix.Uname(&u); err == nil {
		osVersion = versionPrefix(unix.ByteSliceToString(u.Release[:]))
	}
	var si unix.Sysinfo_t
	if err := unix.Sysinfo(&si); err == nil {
		memoryGB = gb(uint64(si.Totalram) * uint64(si.Unit))
	}
	return
}
