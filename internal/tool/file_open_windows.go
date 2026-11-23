//go:build windows

package tool

import "os"

// openAppend opens abs for appending.
func openAppend(abs string) (*os.File, error) {
	return os.OpenFile(abs, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}
