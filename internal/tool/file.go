package tool

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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
func NewFile(p *policy.Engine, a Approver) Tool { return &fileTool{pol: p, ap: a} }

func (*fileTool) Name() string { return "file" }
func (*fileTool) Actions() []string {
	return []string{"read", "write", "append", "edit", "list", "mkdir"}
}

func (*fileTool) Description() string {
	return strings.TrimSpace(`Read, write, append, edit, list files/dirs in the workspace (paths relative, confined to it).
Actions:
  - read:   {"path": str}
  - write:  {"path": str, "content": str}   (overwrites the whole file)
  - append: {"path": str, "content": str}
  - edit:   {"path": str, "find": str, "replace": str}   (replaces first match; prefer this for partial changes)
  - list:   {"path"?: str="."}
  - mkdir:  {"path": str}
NOT here — shell → run; web/fetch → web; math → calc.`)
}

func (f *fileTool) Call(_ context.Context, action string, params map[string]any) Result {
	switch action {
	case "read":
		return f.read(params)
	case "write":
		return f.writeFile(action, params, false)
	case "append":
		return f.writeFile(action, params, true)
	case "edit":
		return f.edit(params)
	case "list":
		return f.list(params)
	case "mkdir":
		return f.mkdir(params)
	}
	return Err("file: unknown action " + action)
}

func (f *fileTool) read(params map[string]any) Result {
	if err := Require(params, "path"); err != nil {
		return Err(err.Error())
	}
	path := Str(params, "path")
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

func (f *fileTool) writeFile(action string, params map[string]any, appendMode bool) Result {
	if err := Require(params, "path", "content"); err != nil {
		return Err(err.Error())
	}
	path := Str(params, "path")
	content := Str(params, "content")

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

func (f *fileTool) edit(params map[string]any) Result {
	if err := Require(params, "path", "find", "replace"); err != nil {
		return Err(err.Error())
	}
	path := Str(params, "path")
	find := Str(params, "find")
	replace := Str(params, "replace")

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

func (f *fileTool) mkdir(params map[string]any) Result {
	if err := Require(params, "path"); err != nil {
		return Err(err.Error())
	}
	path := Str(params, "path")
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

func (f *fileTool) list(params map[string]any) Result {
	path := Str(params, "path")
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
