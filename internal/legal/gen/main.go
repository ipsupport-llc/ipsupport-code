// Command gen writes internal/legal/NOTICES.txt: this program's license, Go's,
// and the license of every module linked into ipsupport-code, with its full
// text — MIT and BSD both require the notice to travel with the binary — plus
// the data the risk model was trained from.
//
// Run after changing dependencies, from anywhere in the repository:
//
//	go run ./internal/legal/gen   (or: go generate ./internal/legal)
//
// legal_test.go fails until it is run, so a new dependency cannot ship without
// its notice.
package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// spdxOverride names licenses the text match below cannot settle: dual
// licences, and texts that only say "see the BSD license".
var spdxOverride = map[string]string{
	"github.com/aymanbagabas/go-udiff":              "BSD-3-Clause AND MIT",
	"kernel.org/pub/linux/libs/security/libcap/psx": "BSD-3-Clause (dual BSD-3-Clause / GPL-2.0; used under BSD-3-Clause)",
	"github.com/andybalholm/cascadia":               "BSD-2-Clause",
	"github.com/alecthomas/chroma/v2":               "MIT",
}

// noLicenseFile covers a module that states its license only in its README:
// the standard text of that license, with the holder the README names.
var noLicenseFile = map[string]string{
	"github.com/mattn/go-localereader": "MIT License (stated in the module's README; the module ships no LICENSE file)\n\n" +
		"Copyright (c) Yasuhiro Matsumoto (mattn)\n\n" + mitText,
}

const mitText = `Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`

var licenseFile = regexp.MustCompile(`(?i)^(licen[cs]e|copying|notice)([-._].*)?$`)

type module struct {
	path, version, dir, spdx string
	texts                    []string
}

func main() {
	// go generate runs this in internal/legal; everything below is relative to
	// the repository root, so find it.
	for dir, _ := os.Getwd(); ; dir = filepath.Dir(dir) {
		if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil && strings.Contains(string(b), "module github.com/ipsupport-llc/ipsupport-code\n") {
			if err := os.Chdir(dir); err != nil {
				fail("chdir %s: %v", dir, err)
			}
			break
		}
		if filepath.Dir(dir) == dir {
			fail("not inside the ipsupport-code repository")
		}
	}
	// Every platform a release ships for: landlock is Linux's alone, coninput
	// Windows', and a notice missing for one of them is missing.
	var lines []string
	for _, goos := range []string{"linux", "darwin", "windows"} {
		cmd := exec.Command("go", "list", "-deps", "-f",
			`{{if not .Standard}}{{with .Module}}{{if not .Main}}{{.Path}}	{{.Version}}	{{.Dir}}{{end}}{{end}}{{end}}`,
			"./cmd/agent")
		cmd.Env = append(os.Environ(), "GOOS="+goos)
		out, err := cmd.Output()
		if err != nil {
			fail("go list (%s): %v", goos, err)
		}
		lines = append(lines, strings.Split(strings.TrimSpace(string(out)), "\n")...)
	}
	seen := map[string]*module{}
	for _, line := range lines {
		f := strings.Split(line, "\t")
		if len(f) != 3 || seen[f[0]] != nil {
			continue
		}
		m := &module{path: f[0], version: f[1], dir: f[2]}
		entries, _ := os.ReadDir(m.dir)
		for _, e := range entries {
			if !e.IsDir() && licenseFile.MatchString(e.Name()) {
				b, err := os.ReadFile(filepath.Join(m.dir, e.Name()))
				if err == nil {
					m.texts = append(m.texts, strings.TrimSpace(string(b)))
				}
			}
		}
		if len(m.texts) == 0 && noLicenseFile[m.path] != "" {
			m.texts = []string{noLicenseFile[m.path]}
		}
		if len(m.texts) == 0 {
			fail("%s %s: no license file in %s", m.path, m.version, m.dir)
		}
		m.spdx = spdxOverride[m.path]
		if m.spdx == "" {
			m.spdx = classify(strings.Join(m.texts, "\n"))
		}
		if m.spdx == "" {
			fail("%s: unrecognised license; add it to spdxOverride", m.path)
		}
		seen[m.path] = m
	}
	var mods []*module
	for _, m := range seen {
		mods = append(mods, m)
	}
	sort.Slice(mods, func(i, j int) bool { return strings.ToLower(mods[i].path) < strings.ToLower(mods[j].path) })

	self, err := os.ReadFile("LICENSE")
	if err != nil {
		fail("read LICENSE (run from the repository root): %v", err)
	}
	goroot, _ := exec.Command("go", "env", "GOROOT").Output()
	golicense, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(goroot)), "LICENSE"))
	if err != nil {
		fail("read Go's LICENSE: %v", err)
	}

	var b bytes.Buffer
	b.WriteString("ipsupport-code — licenses and notices\n\n")
	b.WriteString("ipsupport-code is MIT-licensed. It is built with Go and links the modules\nbelow; their licenses require these notices to accompany the binary.\n\n")
	b.WriteString("Component                                              License\n")
	fmt.Fprintf(&b, "%-54s %s\n", "ipsupport-code", "MIT")
	fmt.Fprintf(&b, "%-54s %s\n", "Go standard library and runtime", "BSD-3-Clause")
	for _, m := range mods {
		fmt.Fprintf(&b, "%-54s %s\n", m.path+" "+m.version, m.spdx)
	}
	b.WriteString("\nData the risk model (internal/risk/model.bin) was trained from:\n")
	b.WriteString("  gitleaks rule vocabulary            MIT, Copyright (c) 2019 Zachary Rice\n")
	b.WriteString("  github/gitignore templates          CC0-1.0\n")
	b.WriteString("\n" + Marker + "\n\n")
	section(&b, "ipsupport-code", string(self))
	section(&b, "Go", string(golicense))
	for _, m := range mods {
		section(&b, m.path+" "+m.version+" ("+m.spdx+")", strings.Join(m.texts, "\n\n"))
	}
	if err := os.WriteFile(filepath.Join("internal", "legal", "NOTICES.txt"), b.Bytes(), 0o644); err != nil {
		fail("write: %v", err)
	}
	fmt.Printf("wrote internal/legal/NOTICES.txt: %d modules\n", len(mods))
}

// Marker separates the summary /license shows from the full texts --license
// adds. Kept in sync with internal/legal by legal_test.go.
const Marker = "==== Full license texts ===="

func section(b *bytes.Buffer, title, text string) {
	fmt.Fprintf(b, "---- %s ----\n\n%s\n\n", title, strings.TrimSpace(text))
}

func classify(t string) string {
	t = strings.Join(strings.Fields(strings.ToLower(t)), " ")
	switch {
	case strings.Contains(t, "apache license") && strings.Contains(t, "version 2.0"):
		return "Apache-2.0"
	case strings.Contains(t, "permission is hereby granted, free of charge"):
		return "MIT"
	case strings.Contains(t, "neither the name"):
		return "BSD-3-Clause"
	case strings.Contains(t, "redistribution and use in source and binary forms"):
		return "BSD-2-Clause"
	case strings.Contains(t, "mozilla public license"):
		return "MPL-2.0"
	case strings.Contains(t, "permission to use, copy, modify, and/or distribute"):
		return "ISC"
	}
	return ""
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "gen: "+format+"\n", a...)
	os.Exit(1)
}
