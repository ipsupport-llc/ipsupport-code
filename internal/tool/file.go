package tool

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	udiff "github.com/aymanbagabas/go-udiff"
	"github.com/bmatcuk/doublestar/v4"

	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

const maxReadBytes = 200_000

// fileMu serializes the actual read-modify-write filesystem work in write/
// append/edit/mkdir — NOT the approval prompt before it (that can wait
// arbitrarily long on the user, and would otherwise block an unrelated
// parallel sub-agent's approval prompt from even showing). Parallel
// sub-agents share a workspace by default (a fan-out of `agent` calls with no
// explicit dir all get the SAME registry), so two of them editing the same
// file is a real, unlocked read-then-write race: each reads the original,
// computes its own edit, and the later write silently discards the earlier
// one — both report success. One process-wide lock is enough; file I/O isn't
// the bottleneck next to the LLM calls around it, so per-path locking would
// be unneeded complexity.
var fileMu sync.Mutex

// Snapshotter is called with the absolute path of a file about to be modified,
// BEFORE the change, so a checkpoint can capture its prior content (for /rewind).
type Snapshotter func(absPath string)

type fileTool struct {
	pol  *policy.Engine
	ap   Approver
	snap Snapshotter
}

// NewFile returns the file tool, gated by the policy engine and (for "ask"
// decisions) the approver. snap (may be nil) is notified before each file change.
func NewFile(p *policy.Engine, ap Approver, snap Snapshotter) Tool {
	f := &fileTool{pol: p, ap: ap, snap: snap}
	return NewDomain(DomainSpec{
		Name:    "file",
		Summary: "Read/write/append/edit/list/find/search files in the workspace (relative paths, jailed).",
		NotHere: "NOT here — shell → run; web/fetch → web; math → calc.",
		Actions: []Action{
			{Name: "read", Params: []Param{Req("path", "str"), Opt("offset", "int", "0"), Opt("limit", "int", "0")}, Note: "(offset/limit: line window, big files)", Run: f.read},
			{Name: "write", Mutates: true, Params: []Param{Req("path", "str"), Opt("content", "str", "")}, Note: "(overwrites; omit content = empty file)", Run: f.write},
			{Name: "append", Mutates: true, Params: []Param{Req("path", "str"), Req("content", "str")}, Run: f.appendFile},
			{Name: "edit", Mutates: true, Params: []Param{Req("path", "str"), Opt("find", "str", ""), Opt("replace", "str", ""), Opt("replace_all", "bool", ""), Opt("edits", "str", "")}, Note: "(1st match; replace_all=all; or edits=JSON [{find,replace},…])", Run: f.edit},
			{Name: "list", Params: []Param{Opt("path", "str", ".")}, Run: f.list},
			{Name: "find", Params: []Param{Req("pattern", "str"), Opt("path", "str", ".")}, Note: "(glob names, e.g. **/*.go)", Run: f.find},
			{Name: "search", Params: []Param{Req("query", "str"), Opt("path", "str", ".")}, Note: "(regex; file:line: match)", Run: f.search},
			{Name: "mkdir", Mutates: true, Params: []Param{Req("path", "str")}, Run: f.mkdir},
		},
		// When a model omits "action", recover a READ-ONLY intent from the params
		// (never a mutation — if content/edits/find are present it's ambiguous with
		// a write, so make the model say so). This unblocks read-heavy work (e.g.
		// "review the repo") on models that skip the action field.
		Infer: func(p map[string]any) string {
			switch {
			case !isEmpty(p["content"]) || !isEmpty(p["edits"]) || !isEmpty(p["find"]):
				return "" // looks like a write/edit — don't guess a mutation
			case !isEmpty(p["query"]):
				return "search"
			case !isEmpty(p["pattern"]):
				return "find"
			case !isEmpty(p["path"]):
				return "read"
			}
			return ""
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
	// Optional line window: stream a slice of a big file instead of loading the
	// whole thing into memory, to keep the context lean (offset is 1-based;
	// limit 0 = to the end).
	if offset, limit := a.Int("offset", 0), a.Int("limit", 0); offset > 0 || limit > 0 {
		return f.readWindow(abs, path, offset, limit)
	}
	return f.readPlain(abs, path)
}

// readPlain returns the whole file, capped at maxReadBytes — via a bounded
// read instead of os.ReadFile, so a multi-gigabyte file in the workspace is
// never loaded fully into memory just to be immediately clipped back down. It
// reads at most maxReadBytes+1 bytes: enough to tell whether the file is
// larger than the cap (for the "truncated" footer) without ever holding more
// of it than that.
func (f *fileTool) readPlain(abs, path string) Result {
	file, err := os.Open(abs)
	if err != nil {
		return Err("cannot read " + path + ": " + err.Error())
	}
	defer file.Close()

	buf := make([]byte, maxReadBytes+1)
	n, err := io.ReadFull(file, buf)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return Err("cannot read " + path + ": " + err.Error())
	}
	buf = buf[:n]
	if n <= maxReadBytes {
		return Ok(string(buf))
	}

	total := int64(n)
	if info, statErr := file.Stat(); statErr == nil {
		total = info.Size()
	}
	out, _ := textutil.Clip(string(buf), maxReadBytes)
	return Ok(out + fmt.Sprintf("\n…[truncated; %d bytes total]", total))
}

// readWindow returns lines [offset, offset+limit) of the file at abs, streamed
// line by line so a huge file never has to sit fully in memory just to serve a
// small window near the end of it — only the requested lines (plus a running
// line count) are held. Splitting semantics match strings.Split(content, "\n")
// exactly, including its trailing empty element for content ending in "\n".
func (f *fileTool) readWindow(abs, path string, offset, limit int) Result {
	file, err := os.Open(abs)
	if err != nil {
		return Err("cannot read " + path + ": " + err.Error())
	}
	defer file.Close()

	start := offset - 1
	if start < 0 {
		start = 0
	}
	end := -1 // unknown until total is known; -1 means "to the end"
	if limit > 0 {
		end = start + limit
	}

	br := bufio.NewReader(file)
	var window []string
	total := 0
	for {
		line, rerr := br.ReadString('\n')
		if rerr != nil && rerr != io.EOF {
			return Err("cannot read " + path + ": " + rerr.Error())
		}
		line = strings.TrimSuffix(line, "\n")
		if total >= start && (end < 0 || total < end) {
			window = append(window, line)
		}
		total++
		if rerr == io.EOF {
			break
		}
	}

	if start > total {
		start = total
	}
	winEnd := end
	if winEnd < 0 || winEnd > total {
		winEnd = total
	}
	hdr := fmt.Sprintf("(lines %d–%d of %d)\n", start+1, winEnd, total)
	out, truncated := textutil.Clip(strings.Join(window, "\n"), maxReadBytes)
	if truncated {
		out += "\n…[truncated]"
	}
	return Ok(hdr + out)
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
		if f.pol.IsSecret(p) { // don't surface .env / *secret* contents
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

func (f *fileTool) write(ctx context.Context, a Args) Result {
	return f.writeFile(ctx, "write", a, false)
}
func (f *fileTool) appendFile(ctx context.Context, a Args) Result {
	return f.writeFile(ctx, "append", a, true)
}

func (f *fileTool) writeFile(ctx context.Context, action string, a Args, appendMode bool) Result {
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
		if !f.ap.Approve(ctx, action, path) {
			return Err(action + " " + path + " denied by user")
		}
	}

	fileMu.Lock()
	defer fileMu.Unlock()

	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	f.snapshot(abs) // checkpoint the prior content before we change it
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

type editPair struct {
	Find    string `json:"find"`
	Replace string `json:"replace"`
}

// editPairs pulls the edit list from an edit call, accepting every shape a weak
// model reaches for: a native JSON array (the model did the right thing), a
// stringified JSON array, a single {find,replace} object, or top-level
// find/replace. Returns a clear error only when nothing usable was given.
func editPairs(a Args) ([]editPair, error) {
	const usage = "edit needs {find, replace} (or edits=[{find,replace},…]), plus {path}"
	switch v := a.m["edits"].(type) {
	case nil: // no edits key → single top-level find/replace
		if strings.TrimSpace(a.Str("find")) == "" && strings.TrimSpace(a.Str("replace")) == "" {
			return nil, errors.New(usage)
		}
		return []editPair{{Find: a.Str("find"), Replace: a.Str("replace")}}, nil
	case string: // stringified JSON array (or empty → fall back to find/replace)
		if strings.TrimSpace(v) == "" {
			return editPairs(Args{m: withoutKey(a.m, "edits")})
		}
		var pairs []editPair
		if err := json.Unmarshal([]byte(v), &pairs); err != nil {
			return nil, errors.New("edit: 'edits' must be a JSON array of {find,replace}: " + err.Error())
		}
		if len(pairs) == 0 {
			return nil, errors.New(usage)
		}
		return pairs, nil
	default: // a native array, or a single object — round-trip through JSON
		blob, err := json.Marshal(v)
		if err != nil {
			return nil, errors.New(usage)
		}
		var pairs []editPair
		if json.Unmarshal(blob, &pairs) == nil && len(pairs) > 0 {
			return pairs, nil
		}
		var one editPair
		if json.Unmarshal(blob, &one) == nil && (one.Find != "" || one.Replace != "") {
			return []editPair{one}, nil
		}
		return nil, errors.New("edit: 'edits' must be a JSON array of {find,replace} — " + usage)
	}
}

// withoutKey returns a shallow copy of m with key removed (so an empty "edits"
// string falls back to the top-level find/replace path).
func withoutKey(m map[string]any, key string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if k != key {
			out[k] = v
		}
	}
	return out
}

func (f *fileTool) edit(ctx context.Context, a Args) Result {
	path := a.Str("path")

	pairs, err := editPairs(a)
	if err != nil {
		return Err(err.Error())
	}
	// Validate the edits up front — before policy prompt, snapshot, and read — so a
	// weak model gets one clean, early error instead of a late one after the churn.
	for i, p := range pairs {
		if p.Find == "" {
			return Err(fmt.Sprintf("edit: empty 'find' in edit #%d — give the exact text to replace (with {path})", i+1))
		}
	}

	d, err := f.pol.Write(path)
	if err != nil {
		return Err(err.Error())
	}
	switch d {
	case policy.Deny:
		return Err("edit " + path + " denied by workspace policy")
	case policy.Ask:
		if !f.ap.Approve(ctx, "edit", path) {
			return Err("edit " + path + " denied by user")
		}
	}

	fileMu.Lock()
	defer fileMu.Unlock()

	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	f.snapshot(abs) // checkpoint the prior content before we change it
	data, err := os.ReadFile(abs)
	if err != nil {
		return Err("cannot read " + path + ": " + err.Error())
	}
	old := string(data)

	n := 1
	if a.Bool("replace_all") {
		n = -1 // strings.Replace: n<0 replaces every occurrence
	}
	updated := old
	for i, p := range pairs {
		if !strings.Contains(updated, p.Find) {
			clip, _ := textutil.Clip(p.Find, 60)
			return Err(fmt.Sprintf("edit: 'find' text not present in %s (edit #%d): %s", path, i+1, clip))
		}
		updated = strings.Replace(updated, p.Find, p.Replace, n)
	}
	if updated == old {
		return Err("edit: no change made to " + path)
	}
	if err := os.WriteFile(abs, []byte(updated), 0o644); err != nil {
		return Fail("file", "edit", "write to "+path+" failed", err)
	}
	diff := udiff.Unified(path, path, old, updated)
	add, del := diffStat(diff)
	return Result{Content: fmt.Sprintf("edited %s (+%d -%d)", path, add, del), Diff: diff}
}

// find globs filenames under the jail (paths only), skipping VCS/dep/build dirs
// and symlinks — locate a file without listing whole trees or reading contents.
func (f *fileTool) find(_ context.Context, a Args) Result {
	pattern := a.Str("pattern")
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
	const maxFind = 300
	skipDir := map[string]bool{".git": true, "node_modules": true, "vendor": true, "dist": true, ".agent": true}
	var out []string
	stopped := false
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
		if d.Type()&fs.ModeSymlink != 0 {
			return nil
		}
		rel, _ := filepath.Rel(abs, p)
		ok, _ := doublestar.Match(pattern, rel)
		if !ok { // also match the bare name, so "*.go" works without a **/ prefix
			ok, _ = doublestar.Match(pattern, d.Name())
		}
		if ok {
			out = append(out, rel)
			if len(out) >= maxFind {
				stopped = true
				return filepath.SkipAll
			}
		}
		return nil
	})
	if len(out) == 0 {
		return Ok("(no files match " + pattern + ")")
	}
	sort.Strings(out)
	if stopped {
		out = append(out, fmt.Sprintf("… stopped at %d", maxFind))
	}
	return Ok(strings.Join(out, "\n"))
}

func (f *fileTool) snapshot(abs string) {
	if f.snap != nil {
		f.snap(abs)
	}
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

func (f *fileTool) mkdir(ctx context.Context, a Args) Result {
	path := a.Str("path")
	d, err := f.pol.Write(path)
	if err != nil {
		return Err(err.Error())
	}
	switch d {
	case policy.Deny:
		return Err("mkdir " + path + " denied by workspace policy")
	case policy.Ask:
		if !f.ap.Approve(ctx, "mkdir", path) {
			return Err("mkdir " + path + " denied by user")
		}
	}

	fileMu.Lock()
	defer fileMu.Unlock()

	abs, err := f.pol.Resolve(path)
	if err != nil {
		return Err(err.Error())
	}
	// os.MkdirAll is a silent no-op when the directory already exists — check
	// first so the model is TOLD it already existed instead of always seeing
	// "created", which reads as a fresh start and hides the one fact (this
	// isn't new) that would let it skip a step it doesn't need to repeat.
	existed := false
	if info, statErr := os.Stat(abs); statErr == nil && info.IsDir() {
		existed = true
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return Fail("file", "mkdir", "mkdir "+path+" failed", err)
	}
	if existed {
		return Ok(path + " already exists — nothing to do")
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
