// Package atomicfile writes a file durably: to a temp sibling, fsync it, then
// rename over the target — so a crash mid-write can't leave a truncated or
// half-written file. It's the one home for the temp+rename pattern that config,
// the knowledge base, the usage ledger, skills, and self-update all need.
package atomicfile

import (
	"os"
	"path/filepath"
)

// Write writes data to path atomically (temp file → fsync → rename), creating the
// parent directory. perm is applied to the final file.
func Write(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename succeeds
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil { // flush to disk before the rename
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
