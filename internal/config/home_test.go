package config

import "testing"

// setHome points the user's home at dir for this test. On Windows
// os.UserHomeDir reads USERPROFILE, not HOME: setting only HOME left every
// test in the package sharing TestMain's one home.
func setHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	t.Setenv("USERPROFILE", dir)
}
