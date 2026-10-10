package sandbox

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestSeatbeltProfile(t *testing.T) {
	p := SeatbeltProfile([]string{"/work/proj"}, true)
	for _, want := range []string{
		"(version 1)", "(allow default)", "(deny file-write*)",
		`(subpath "/work/proj")`, `(subpath "/private/var/folders")`, `(literal "/dev/null")`,
	} {
		if !strings.Contains(p, want) {
			t.Errorf("profile missing %q:\n%s", want, p)
		}
	}
	if strings.Contains(p, "(deny network*)") {
		t.Error("network should be allowed when allowNetwork=true")
	}
}

func TestSeatbeltProfileDeniesNetwork(t *testing.T) {
	if !strings.Contains(SeatbeltProfile(nil, false), "(deny network*)") {
		t.Error("expected (deny network*) when allowNetwork=false")
	}
}

func TestSeatbeltProfileEscapesPaths(t *testing.T) {
	p := SeatbeltProfile([]string{`/a "b"\c`}, true)
	if !strings.Contains(p, `(subpath "/a \"b\"\\c")`) {
		t.Errorf("path not escaped in SBPL:\n%s", p)
	}
}

func TestResolveMode(t *testing.T) {
	if got := resolveMode(Off); got != Off {
		t.Errorf("resolveMode(off) = %q", got)
	}
	want := Off
	switch runtime.GOOS {
	case "darwin":
		want = Seatbelt
	case "linux":
		want = Landlock
	}
	if got := resolveMode(Auto); got != want {
		t.Errorf("resolveMode(auto) on %s = %q, want %q", runtime.GOOS, got, want)
	}
}

func TestWrapOffPassthrough(t *testing.T) {
	n, args, wrapped := Wrap(Off, Spec{}, "sh", []string{"-c", "echo"})
	if wrapped || n != "sh" || len(args) != 2 {
		t.Errorf("off must pass through unchanged, got %s %v (wrapped=%v)", n, args, wrapped)
	}
	if Available(Off) {
		t.Error("Available(off) must be false")
	}
}

func TestWrapSeatbeltPassthroughOffDarwin(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("darwin wrapping is covered by the darwin-only tests")
	}
	n, args, wrapped := Wrap(Seatbelt, Spec{WritableRoots: []string{"/w"}, AllowNetwork: true}, "sh", []string{"-c", "echo"})
	if wrapped || n != "sh" || len(args) != 2 {
		t.Fatalf("seatbelt must pass through off darwin, got %s %v (wrapped=%v)", n, args, wrapped)
	}
}

func TestReaches(t *testing.T) {
	ws := t.TempDir() // under a temp dir on most hosts: build the cases from a root of its own
	home := filepath.Join(ws, "home")
	proj := filepath.Join(home, "proj")
	cache := filepath.Join(home, ".cache")
	for _, d := range []string{proj, cache} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	link := filepath.Join(ws, "link")
	if err := os.Symlink(proj, link); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name  string
		roots []string
		path  string
		want  bool
	}{
		{"no roots", nil, proj, false},
		{"a root elsewhere", []string{cache}, proj, false},
		{"the path itself", []string{proj}, proj, true},
		{"a root holding it", []string{home}, proj, true},
		{"a root inside it", []string{cache}, home, true},
		{"through a symlink", []string{proj}, link, true},
		{"a sibling sharing a prefix", []string{proj + "x"}, proj, false},
	} {
		if got := reaches("", Spec{WritableRoots: c.roots}, c.path); got != c.want {
			t.Errorf("%s: reaches = %v, want %v", c.name, got, c.want)
		}
	}
	// The temp dirs every spec allows: a workspace in one is writable whatever the roots.
	if !reaches(Landlock, Spec{}, "/tmp/proj") || !reaches(Seatbelt, Spec{}, "/private/var/folders/x/T/proj") {
		t.Error("a path in a scratch dir must be reachable")
	}
	if reaches(Landlock, Spec{}, "/home/u/proj") {
		t.Error("a path outside every writable dir must not be reachable")
	}
	if !Reaches(Off, Spec{}, proj) {
		t.Error("an unconfined mode reaches everything")
	}
}
