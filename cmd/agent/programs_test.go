package main

import "github.com/ipsupport-llc/ipsupport-code/internal/config"

// The tests here see no installed programs a built-in provider's key command
// needs (the GitHub CLI): what they list and pick must not depend on the
// machine. A test about such a provider turns it back on itself.
func init() {
	config.HasProgram = func(string) bool { return false }
}
