// Package legal carries the licenses and notices of everything linked into
// the binary (NOTICES.txt, written by ./gen), so they travel with it as MIT
// and BSD require.
package legal

import (
	_ "embed"
	"strings"
)

//go:generate go run ./gen

//go:embed NOTICES.txt
var notices string

// marker separates the summary from the full texts; gen writes it.
const marker = "==== Full license texts ===="

// Full is every license and notice, texts included (ipsupport-code --license).
func Full() string { return notices }

// Summary is the component-and-license table without the texts (/license).
func Summary() string {
	if i := strings.Index(notices, marker); i >= 0 {
		return strings.TrimRight(notices[:i], "\n")
	}
	return notices
}
