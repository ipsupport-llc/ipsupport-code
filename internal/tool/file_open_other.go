//go:build !windows

package tool

import (
	"errors"
	"os"
	"syscall"
)

// openAppend opens abs for appending. A named pipe is opened without blocking:
// with no reader, opening it for writing would wait forever with the file lock
// held; it fails at once instead (ENXIO). Once open, writes block as usual.
func openAppend(abs string) (*os.File, error) {
	if !isFIFO(abs) {
		return os.OpenFile(abs, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	}
	fh, err := os.OpenFile(abs, os.O_WRONLY|os.O_APPEND|syscall.O_NONBLOCK, 0)
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
