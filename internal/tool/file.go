package tool

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	udiff "github.com/aymanbagabas/go-udiff"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

const maxReadBytes = 200_000

type fileTool struct {
	pol *policy.Engine
	ap  Approver
}

// NewFile returns the file tool, gated by the policy engine and (for "ask"
// decisions) the approver.
func NewFile(p *policy.Engine, ap Approver) Tool {
	f := &fileTool{pol: p, ap: ap}
	return NewDomain(DomainSpec{
		Name:    "file",
		Summary: "Read, write, append, edit, list files/dirs in the workspace (paths relative, confined to it).",
		NotHere: "NOT here — shell → run; web/fetch → web; math → calc.",
		Actions: []Action{
			{Name: "read", Params: []Param{Req("path", "str")}, Run: f.read},
			{Name: "write", Mutates: true, Params: []Param{Req("path", "str"), Opt("content", "str", "")}, Note: "(overwrites; omit content for an empty file)", Run: f.write},
			{Name: "append", Mutates: true, Params: []Param{Req("path", "str"), Req("content", "str")}, Run: f.appendFile},
			{Name: "edit", Mutates: true, Params: []Param{Req("path", "str"), Req("find", "str"), Req("replace", "str")}, Note: "(replaces first match; prefer this for partial changes)", Run: f.edit},
			{Name: "list", Params: []Param{Opt("path", "str", ".")}, Run: f.list},
			{Name: "search", Params: []Param{Req("query", "str"), Opt("path", "str", ".")}, Note: "(regex; find matching lines across files — file:line: match)", Run: f.search},
			{Name: "mkdir", Mutates: true, Params: []Param{Req("path", "str")}, Run: f.mkdir},
		},
	})
}

func (f *fileTool) read(_ context.Context, a Args) Result {
	path := a.Str("path")
	if err := f.pol.Read(path); err != nil {
		return Err(err.Error())
	}
	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return Err("cannot read " + path + ": " + err.Error())
	}
	out, truncated := textutil.Clip(string(data), maxReadBytes)
	if truncated {
		out += fmt.Sprintf("\n…[truncated; %d bytes total]", len(data))
	}
	return Ok(out)
}

// search greps the workspace for a regex (literal if it doesn't compile), under
// the jail, skipping VCS/dep/build dirs and binary or oversized files. Read-only.
func (f *fileTool) search(_ context.Context, a Args) Result {
	query := a.Str("query")
	root := a.Str("path")
	if root == "" {
		root = "."
	}
	if err := f.pol.Read(root); err != nil {
		return Err(err.Error())
	}
	abs, err := f.pol.Resolve(root)
	if err != nil {
		return Err(err.Error())
	}
	re, err := regexp.Compile(query)
	if err != nil {
		re = regexp.MustCompile(regexp.QuoteMeta(query)) // not valid regex → literal
	}

	const maxMatches = 200
	skipDir := map[string]bool{".git": true, "node_modules": true, "vendor": true, "dist": true, ".agent": true}
	var b strings.Builder
	n := 0
	_ = filepath.WalkDir(abs, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if p != abs && (skipDir[d.Name()] || strings.HasPrefix(d.Name(), ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type()&fs.ModeSymlink != 0 { // don't follow symlinks — they can point out of the jail
			return nil
		}
		if info, e := d.Info(); e != nil || info.Size() > 1<<20 { // skip files > 1 MiB
			return nil
		}
		data, e := os.ReadFile(p)
		if e != nil {
			return nil
		}
		if bytes.IndexByte(data[:min(len(data), 512)], 0) >= 0 { // binary
			return nil
		}
		rel, _ := filepath.Rel(abs, p)
		for i, line := range strings.Split(string(data), "\n") {
			if !re.MatchString(line) {
				continue
			}
			clipped, _ := textutil.Clip(strings.TrimSpace(line), 200)
			fmt.Fprintf(&b, "%s:%d: %s\n", rel, i+1, clipped)
			if n++; n >= maxMatches {
				fmt.Fprintf(&b, "… stopped at %d matches\n", maxMatches)
				return filepath.SkipAll
			}
		}
		return nil
	})
	if n == 0 {
		return Ok("(no matches for " + query + ")")
	}
	return Ok(strings.TrimRight(b.String(), "\n"))
}

func (f *fileTool) write(_ context.Context, a Args) Result { return f.writeFile("write", a, false) }
func (f *fileTool) appendFile(_ context.Context, a Args) Result {
	return f.writeFile("append", a, true)
}

func (f *fileTool) writeFile(action string, a Args, appendMode bool) Result {
	path := a.Str("path")
	content := a.Str("content")

	d, err := f.pol.Write(path)
	if err != nil {
		return Err(err.Error()) // jail escape
	}
	switch d {
	case policy.Deny:
		return Err(action + " " + path + " denied by workspace policy")
	case policy.Ask:
		if !f.ap.Approve(action, path) {
			return Err(action + " " + path + " denied by user")
		}
	}

	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	var old string
	if !appendMode {
		if data, e := os.ReadFile(abs); e == nil {
			old = string(data)
		}
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return Fail("file", action, "cannot create parent directory for "+path, err)
	}
	flag := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	fh, err := os.OpenFile(abs, flag, 0o644)
	if err != nil {
		return Fail("file", action, "cannot open "+path, err)
	}
	defer fh.Close()
	if _, err := fh.WriteString(content); err != nil {
		return Fail("file", action, "write to "+path+" failed", err)
	}

	verb := "wrote"
	if appendMode {
		verb = "appended to"
	}
	res := Ok(fmt.Sprintf("%s %s (%d bytes)", verb, path, len(content)))
	if !appendMode && old != content { // show the change — for a new file, all additions
		res.Diff = udiff.Unified(path, path, old, content)
	}
	return res
}

func (f *fileTool) edit(_ context.Context, a Args) Result {
	path := a.Str("path")
	find := a.Str("find")
	replace := a.Str("replace")

	d, err := f.pol.Write(path)
	if err != nil {
		return Err(err.Error())
	}
	switch d {
	case policy.Deny:
		return Err("edit " + path + " denied by workspace policy")
	case policy.Ask:
		if !f.ap.Approve("edit", path) {
			return Err("edit " + path + " denied by user")
		}
	}
	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return Err("cannot read " + path + ": " + err.Error())
	}
	old := string(data)
	if !strings.Contains(old, find) {
		return Err("edit: the 'find' text is not present in " + path)
	}
	updated := strings.Replace(old, find, replace, 1)
	if err := os.WriteFile(abs, []byte(updated), 0o644); err != nil {
		return Fail("file", "edit", "write to "+path+" failed", err)
	}
	diff := udiff.Unified(path, path, old, updated)
	add, del := diffStat(diff)
	return Result{Content: fmt.Sprintf("edited %s (+%d -%d)", path, add, del), Diff: diff}
}

// diffStat counts added and removed lines in a unified diff.
func diffStat(diff string) (added, removed int) {
	for _, ln := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(ln, "+++"), strings.HasPrefix(ln, "---"):
		case strings.HasPrefix(ln, "+"):
			added++
		case strings.HasPrefix(ln, "-"):
			removed++
		}
	}
	return added, removed
}

func (f *fileTool) mkdir(_ context.Context, a Args) Result {
	path := a.Str("path")
	d, err := f.pol.Write(path)
	if err != nil {
		return Err(err.Error())
	}
	switch d {
	case policy.Deny:
		return Err("mkdir " + path + " denied by workspace policy")
	case policy.Ask:
		if !f.ap.Approve("mkdir", path) {
			return Err("mkdir " + path + " denied by user")
		}
	}
	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return Fail("file", "mkdir", "mkdir "+path+" failed", err)
	}
	return Ok("created directory " + path)
}

func (f *fileTool) list(_ context.Context, a Args) Result {
	path := a.Str("path")
	if path == "" {
		path = "."
	}
	if err := f.pol.Read(path); err != nil {
		return Err(err.Error())
	}
	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return Err("cannot list " + path + ": " + err.Error())
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() {
			n += "/"
		}
		names = append(names, n)
	}
	sort.Strings(names)
	if len(names) == 0 {
		return Ok("(empty)")
	}
	return Ok(strings.Join(names, "\n"))
}
