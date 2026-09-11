//go:build windows

package filelock

import (
	"os"

	"golang.org/x/sys/windows"
)

// lockFile opens (creating if absent) and locks path via LockFileEx, blocking
// until the exclusive lock is acquired.
func lockFile(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	ol := new(windows.Overlapped)
	allBytes := ^uint32(0)
	if err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK, 0, allBytes, allBytes, ol); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}
