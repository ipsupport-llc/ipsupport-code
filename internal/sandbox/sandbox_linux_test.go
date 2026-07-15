//go:build linux

package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestMain lets THIS test binary serve as the re-exec shim: when the wrapped
// command re-runs os.Executable() (the test binary) with the sandbox argv, the
// shim leg applies Landlock and execs — exactly like the real binary's main().
func TestMain(m *testing.M) {
	MaybeExecConfined()
	os.Exit(m.Run())
}

// outsideDir returns a scratch dir NOT under any always-writable root
// (/tmp etc.), so a write there proves the jail leaks. $HOME qualifies.
func outsideDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	d, err := os.MkdirTemp(home, ".ipsupport-sandbox-test-")
	if err != nil {
		t.Skipf("cannot create scratch in home: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(d) })
	return d
}

func TestLandlockWrapShape(t *testing.T) {
	if !kernelHasLandlock() {
		t.Skip("kernel without landlock")
	}
	name, args, wrapped := Wrap(Landlock, Spec{WritableRoots: []string{"/w"}, AllowNetwork: true},
		"sh", []string{"-c", "echo"})
	if !wrapped || len(args) < 4 || args[0] != reexecFlag {
		t.Fatalf("expected a re-exec wrap, got %s %v (wrapped=%v)", name, args, wrapped)
	}
	if !strings.Contains(args[1], `"/w"`) {
		t.Errorf("spec JSON missing the writable root: %s", args[1])
	}
	if args[2] != "sh" || args[3] != "-c" {
		t.Errorf("original command not preserved: %v", args[2:])
	}
}

func TestLandlockAllowsWriteInJail(t *testing.T) {
	if !kernelHasLandlock() {
		t.Skip("kernel without landlock")
	}
	jail := outsideDir(t) // in $HOME, so only the jail rule (not /tmp) allows it
	target := filepath.Join(jail, "f")
	name, args, _ := Wrap(Landlock, Spec{WritableRoots: []string{jail}, AllowNetwork: true},
		"sh", []string{"-c", "echo ok > " + target})
	if out, err := exec.Command(name, args...).CombinedOutput(); err != nil {
		t.Fatalf("write inside the jail was blocked: %v\n%s", err, out)
	}
	if b, _ := os.ReadFile(target); strings.TrimSpace(string(b)) != "ok" {
		t.Fatal("file not written inside the jail")
	}
}

func TestLandlockDeniesWriteOutsideJail(t *testing.T) {
	if !kernelHasLandlock() {
		t.Skip("kernel without landlock")
	}
	outside := filepath.Join(outsideDir(t), "leak")
	jail := t.TempDir() // the confined command's own jail (under /tmp is fine)
	name, args, _ := Wrap(Landlock, Spec{WritableRoots: []string{jail}, AllowNetwork: true},
		"sh", []string{"-c", "echo x > " + outside})
	_ = exec.Command(name, args...).Run() // expected to fail under the sandbox
	if _, err := os.Stat(outside); err == nil {
		t.Fatal("write OUTSIDE the jail was not blocked by landlock")
	}
}

func TestLandlockReadStillWorks(t *testing.T) {
	if !kernelHasLandlock() {
		t.Skip("kernel without landlock")
	}
	jail := t.TempDir()
	// Reading a system file outside the jail must keep working (write-jail
	// posture: reads/exec stay open).
	name, args, _ := Wrap(Landlock, Spec{WritableRoots: []string{jail}, AllowNetwork: true},
		"sh", []string{"-c", "head -c 10 /etc/hostname && echo READOK"})
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "READOK") {
		t.Fatalf("read outside the jail broke: %v\n%s", err, out)
	}
}
