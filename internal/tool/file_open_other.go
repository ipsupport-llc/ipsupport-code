//go:build !windows

package tool

import (
	"errors"
	"os"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// linkCount is how many names the file has (hard links).
func linkCount(_ string, fi os.FileInfo) int {
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		return int(st.Nlink)
	}
	return 1
}

// hasXattrs reports whether the file carries extended attributes — POSIX ACLs
// among them — which a replacement file would not. An error reading them (an
// unsupporting filesystem) counts as none.
func hasXattrs(abs string) bool {
	n, err := unix.Listxattr(abs, nil)
	return err == nil && n > 0
}

// ownedByUs reports whether we own the file: a replacement is ours, and only
// root could hand it back to another owner.
func ownedByUs(fi os.FileInfo) bool {
	st, ok := fi.Sys().(*syscall.Stat_t)
	return !ok || int(st.Uid) == os.Getuid()
}

// keepOwner gives path the owner and group fi had, where we may: a replaced
// file otherwise belongs to whoever ran the agent. Errors are ignored — a
// non-root user can only keep a group it is in, and that is still worth doing.
func keepOwner(path string, fi os.FileInfo) {
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		if os.Chown(path, int(st.Uid), int(st.Gid)) != nil {
			os.Chown(path, -1, int(st.Gid))
		}
	}
}

// fifoReaderWait is how long append waits for a pipe's reader to open it.
const fifoReaderWait = time.Second

// openAppend opens abs for appending. A named pipe is opened without blocking:
// with no reader, opening it for writing would wait forever with the file lock
// held; it fails at once instead (ENXIO). Once open, writes block as usual.
func openAppend(abs string) (*os.File, error) {
	if !isFIFO(abs) {
		return os.OpenFile(abs, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	}
	// A reader that is about to open the pipe gets a moment to: a blocking
	// open would have waited for it; this one waits a second, not forever.
	var fh *os.File
	var err error
	for deadline := time.Now().Add(fifoReaderWait); ; time.Sleep(20 * time.Millisecond) {
		fh, err = os.OpenFile(abs, os.O_WRONLY|os.O_APPEND|syscall.O_NONBLOCK, 0)
		if !errors.Is(err, syscall.ENXIO) || time.Now().After(deadline) {
			break
		}
	}
	if errors.Is(err, syscall.ENXIO) {
		return nil, errors.New("nothing is reading this pipe, so nothing can be written to it")
	}
	if err != nil {
		return nil, err
	}
	if err := syscall.SetNonblock(int(fh.Fd()), false); err != nil {
		fh.Close()
		return nil, err
	}
	return fh, nil
}
