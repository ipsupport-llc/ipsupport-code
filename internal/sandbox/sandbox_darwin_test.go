//go:build darwin

package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// These run only on macOS CI, where they exercise the real Seatbelt kernel
// sandbox via sandbox-exec — proving the profile actually confines writes.

func TestWrapUsesSandboxExec(t *testing.T) {
	if !available(Seatbelt) {
		t.Skip("sandbox-exec unavailable")
	}
	n, args, wrapped := Wrap(Seatbelt, Spec{WritableRoots: []string{"/w"}, AllowNetwork: true},
		"sh", []string{"-c", "echo"})
	if !wrapped || n != sandboxExec || len(args) < 4 || args[0] != "-p" {
		t.Fatalf("expected sandbox-exec wrap, got %s %v (wrapped=%v)", n, args, wrapped)
	}
}

func TestSeatbeltAllowsWriteInJail(t *testing.T) {
	if !available(Seatbelt) {
		t.Skip("sandbox-exec unavailable")
	}
	jail := t.TempDir()
	target := filepath.Join(jail, "f")
	n, args, _ := Wrap(Seatbelt, Spec{WritableRoots: []string{jail}, AllowNetwork: true},
		"sh", []string{"-c", "echo ok > " + target})
	if out, err := exec.Command(n, args...).CombinedOutput(); err != nil {
		t.Fatalf("write inside the jail was blocked: %v\n%s", err, out)
	}
	if b, _ := os.ReadFile(target); strings.TrimSpace(string(b)) != "ok" {
		t.Fatal("file not written inside the jail")
	}
}

func TestSeatbeltDeniesWriteOutsideJail(t *testing.T) {
	if !available(Seatbelt) {
		t.Skip("sandbox-exec unavailable")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	outside := filepath.Join(home, ".ipsupport-sandbox-write-test")
	os.Remove(outside)
	defer os.Remove(outside)

	jail := t.TempDir()
	n, args, _ := Wrap(Seatbelt, Spec{WritableRoots: []string{jail}, AllowNetwork: true},
		"sh", []string{"-c", "echo x > " + outside})
	_ = exec.Command(n, args...).Run() // expected to fail under the sandbox
	if _, err := os.Stat(outside); err == nil {
		t.Fatal("write OUTSIDE the jail was not blocked by the sandbox")
	}
}
