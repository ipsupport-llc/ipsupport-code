package tool

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

const maxReadBytes = 200_000

type fileTool struct {
	pol *policy.Engine
	ap  Approver
}

// NewFile returns the file tool, gated by the policy engine and (for "ask"
// decisions) the approver.
func NewFile(p *policy.Engine, a Approver) Tool { return &fileTool{pol: p, ap: a} }

func (*fileTool) Name() string     { return "file" }
func (*fileTool) Actions() []string { return []string{"read", "write", "append", "list", "mkdir"} }

func (*fileTool) Description() string {
	return strings.TrimSpace(`Read, write, append, list, and create files/dirs in the workspace.
Actions:
  - read:   {"path": str}
  - write:  {"path": str, "content": str}   (overwrites)
  - append: {"path": str, "content": str}
  - list:   {"path"?: str}                  (defaults to ".")
  - mkdir:  {"path": str}
Paths are relative to the workspace and confined to it.
NOT here — shell commands → run; web/search/fetch → web; arithmetic → calc.`)
}

func (f *fileTool) Call(_ context.Context, action string, params map[string]any) Result {
	switch action {
	case "read":
		return f.read(params)
	case "write":
		return f.writeFile(action, params, false)
	case "append":
		return f.writeFile(action, params, true)
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
	out := string(data)
	if len(data) > maxReadBytes {
		out = out[:maxReadBytes] + fmt.Sprintf("\n…[truncated; %d bytes total]", len(data))
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
	return Ok(fmt.Sprintf("%s %s (%d bytes)", verb, path, len(content)))
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
