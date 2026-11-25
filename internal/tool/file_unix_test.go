//go:build !windows

package tool

import (
	"syscall"

	"golang.org/x/sys/unix"
)

func mkfifo(path string) error { return syscall.Mkfifo(path, 0o600) }

func setUmask(m int) int { return syscall.Umask(m) }

func setXattr(path, name, value string) error { return unix.Setxattr(path, name, []byte(value), 0) }

func getXattr(path, name string) (string, error) {
	buf := make([]byte, 64)
	n, err := unix.Getxattr(path, name, buf)
	if err != nil {
		return "", err
	}
	return string(buf[:n]), nil
}
