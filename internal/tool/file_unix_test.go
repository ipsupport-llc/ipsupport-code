//go:build !windows

package tool

import "syscall"

func mkfifo(path string) error { return syscall.Mkfifo(path, 0o600) }

func setUmask(m int) int { return syscall.Umask(m) }
