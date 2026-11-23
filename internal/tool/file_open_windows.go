//go:build windows

package tool

import "os"

// openAppend opens abs for appending.
func openAppend(abs string) (*os.File, error) {
	return os.OpenFile(abs, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

// linkCount is how many names the file has; not read on Windows.
func linkCount(os.FileInfo) int { return 1 }

// keepOwner: ownership on Windows is an ACL, left as the new file gets it.
func keepOwner(string, os.FileInfo) {}
