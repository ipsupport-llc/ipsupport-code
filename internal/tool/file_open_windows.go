//go:build windows

package tool

import (
	"os"
	"syscall"
)

// openAppend opens abs for appending.
func openAppend(abs string) (*os.File, error) {
	return os.OpenFile(abs, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

// linkCount is how many names the file has: NTFS hard links share one file,
// and a replacement renamed over one name would leave the others behind.
func linkCount(abs string, _ os.FileInfo) int {
	p, err := syscall.UTF16PtrFromString(abs)
	if err != nil {
		return 1
	}
	h, err := syscall.CreateFile(p, 0, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return 1
	}
	defer syscall.CloseHandle(h)
	var info syscall.ByHandleFileInformation
	if syscall.GetFileInformationByHandle(h, &info) != nil {
		return 1
	}
	return int(info.NumberOfLinks)
}

// hasXattrs: Windows keeps ACLs on the file object; a replacement inherits the
// directory's. Not detected here — the replacement keeps the write all-or-nothing.
func hasXattrs(string) bool { return false }

// ownedByUs: ownership on Windows is part of the ACL, not checked here.
func ownedByUs(os.FileInfo) bool { return true }

// keepOwner: ownership on Windows is an ACL, left as the new file gets it.
func keepOwner(string, os.FileInfo) {}
