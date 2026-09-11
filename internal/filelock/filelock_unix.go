//go:build !windows

package filelock

import (
	"os"

	"golang.org/x/sys/unix"
)

// lockFile opens (creating if absent) and flocks path, blocking until the
// exclusive lock is acquired.
func lockFile(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	var flockErr error
	for {
		flockErr = unix.Flock(int(f.Fd()), unix.LOCK_EX)
		if flockErr != unix.EINTR {
			break
		}
	}
	if flockErr != nil {
		f.Close()
		return nil, flockErr
	}
	return f, nil
}
