//go:build windows

package policy

import (
	"os"
	"testing"
)

// os.UserHomeDir reads USERPROFILE on Windows, not the HOME the tests set, so
// point it at a throwaway dir to keep tests out of the real user config.
func TestMain(m *testing.M) {
	home, err := os.MkdirTemp("", "home")
	if err != nil {
		panic(err)
	}
	os.Setenv("USERPROFILE", home)
	code := m.Run()
	os.RemoveAll(home)
	os.Exit(code)
}
