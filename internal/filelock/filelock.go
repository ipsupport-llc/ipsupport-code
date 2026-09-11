// Package filelock provides a cross-process, blocking, exclusive file lock.
// It exists so a store shared by more than one ipsupport-code process (the
// usage ledger, the knowledge base) can serialize its read-modify-write cycle
// around a Save: an in-process sync.Mutex only stops two goroutines in the
// SAME process from racing, so without this, one process's read could land
// before another process's write and then clobber it on its own later write.
package filelock

import (
	"os"
	"path/filepath"
)

// Lock acquires an exclusive OS-level lock on path+".lock" (created, along
// with its parent directory, if it doesn't already exist), blocking until the
// lock is available. The caller must call the returned unlock — even on an
// error return from whatever it guards — to release the lock; unlock closes
// the underlying file, which releases the OS-level lock too.
func Lock(path string) (unlock func() error, err error) {
	lockPath := path + ".lock"
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return nil, err
	}
	f, err := lockFile(lockPath)
	if err != nil {
		return nil, err
	}
	return f.Close, nil
}
