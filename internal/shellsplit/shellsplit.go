// Package shellsplit cuts a command line into the commands a shell would run,
// by that shell's own rules: what separates two commands, what quotes and
// escapes hide a separator, which text is data rather than a command (a
// heredoc body, a here-string, a comment), and which code is nested inside a
// command (command substitution, a PowerShell subexpression or script block,
// a heredoc body that is expanded or fed to an interpreter).
//
// It is for scoring, not for executing: a nested command is returned as a
// command of its own as well as inside the one that holds it, and a line the
// rules can't finish (an unclosed quote) still yields what was read. Where the
// rules are ambiguous it errs toward showing a command, never toward hiding one.
//
// scripts/shellsplit.py is a line-by-line port; testdata/golden.jsonl holds the
// two to the same output.
package shellsplit

import (
	"encoding/base64"
	"path"
	"slices"
	"strings"
	"unicode/utf16"
)

// Dialect is the shell a command line is written for.
type Dialect int

const (
	Sh         Dialect = iota // POSIX sh and its kin (dash, bash, zsh): what `run` uses off Windows
	PowerShell                // Windows PowerShell 5.1 and PowerShell 7
	Cmd                       // cmd.exe
)

// Parsed is a line cut into what is scored.
type Parsed struct {
	// Commands, in order, each trimmed; nested ones follow the one holding them.
	Commands []string
	// Groups are commands joined by | || && (|& in sh) into one unit with two
	// or more members — a pipeline's danger can be in the joining (`curl … | sh`).
	Groups []string
	// Code is the line without its data — heredoc bodies, PowerShell
	// here-strings, comments — with its commands and the operators between
	// them. (A sh <<< here-string is one word of its command and stays.)
	Code string
	// Incomplete: code nested deeper than maxDepth, or past maxParses, was
	// kept as text, not cut.
	Incomplete bool

	untracked bool // see result.untracked: eval hands it on
	varProg   bool // see result.varProg: every nesting hands it on
	dropped   bool // see result.dropped: every nesting hands it on
}

// maxDepth bounds nesting — $( $( $( … ) ) ), script blocks, heredocs fed to
// a shell. Deeper code is kept whole as one command, and the parse says so.
const maxDepth = 32

// maxParses bounds the work across all of a line's nesting: depth alone does
// not — eval $( eval $( … ) ) reads each substitution twice, as the line's and
// as eval's code, doubling the work per level. Past it, code is kept whole.
const maxParses = 1024

// Split returns the commands in line (see Parsed.Commands).
func Split(d Dialect, line string) []string { return Parse(d, line).Commands }

// Code returns line as a program, without its data (see Parsed.Code).
func Code(d Dialect, line string) string { return Parse(d, line).Code }

// Parse cuts line once.
func Parse(d Dialect, line string) Parsed {
	budget := maxParses
	return parse(d, line, 0, &budget)
}

func parse(d Dialect, line string, depth int, budget *int) Parsed {
	r := &result{d: d, depth: depth, budget: budget}
	if *budget--; depth >= maxDepth || *budget < 0 {
		if s := Trim(line); s != "" {
			r.cmds = append(r.cmds, s)
		}
		return Parsed{Commands: r.cmds, Code: Trim(line), Incomplete: true}
	}
	switch d {
	case PowerShell:
		splitPowerShell(line, r)
	case Cmd:
		splitCmd(line, r)
	default:
		splitSh(line, r)
	}
	r.endGroup()
	// read x <<EOF … EOF; $e "$x": a body dropped as data can still be run by
	// a variable, or a name, set up after it was read.
	if r.dropped && (r.untracked || r.varProg) {
		r.incomplete = true
	}
	code := strings.TrimSuffix(r.code.String(), " ;")
	return Parsed{Commands: append(r.cmds, r.inner...), Groups: append(r.groups, r.innerG...), Code: Trim(code), Incomplete: r.incomplete,
		untracked: r.untracked, varProg: r.varProg, dropped: r.dropped}
}

// Trim is strings.TrimSpace — named so the Python port can match it exactly
// (Python's str.strip also drops \x1c-\x1f, which Go keeps).
func Trim(s string) string { return strings.TrimSpace(s) }

type result struct {
	d          Dialect
	depth      int
	budget     *int // parses left, shared by the whole line
	incomplete bool
	cmds       []string
	piped      []bool   // cmds[i]'s output goes into cmds[i+1]: | or |&
	pipeTo     [][]int  // what cmds[i] sends on is read by each cmds[pipeTo[i][k]]
	toBlock    [][]int  // … and by every command of each blocks[toBlock[i][k]]
	opens      []open   // the blocks still open, innermost last
	blocks     [][2]int // the ( … ) and { … } closed, [start, end)
	closed     []int    // the block just closed, [start, end): sent on by the next cut, if it sends
	feeding    [][2]int // what the last cut sent on, for a block opening next: | { …; }
	links      int      // reader links made: bounded by maxLinks
	input      []bool   // cmds[i] is in a <( … ): what it writes is read as a file
	funcName   bool     // the last cut was a single word before (: name () …
	untracked  bool     // a function, coproc or alias was set up: a heredoc's output can reach a reader later, unseen
	varProg    bool     // a variable's value runs as code: $e "$x", sh -c "$x"
	dropped    bool     // a heredoc was read as data, its body scored as nothing
	groups     []string
	inner      []string // what a command runs inside it: sh -c '…', eval …
	innerG     []string
	code       strings.Builder
	group      []string // commands of the group being read
	gtext      strings.Builder
}

// joins are the operators that keep a group going.
var joins = map[string]bool{"|": true, "||": true, "&&": true, "|&": true}

// cut ends the command being read at an operator.
func (r *result) cut(b *buf, op string) {
	s := Trim(b.String())
	pipe := op == "|" || op == "|&"
	if s != "" {
		r.cmds = append(r.cmds, s)
		r.piped = append(r.piped, pipe)
		r.pipeTo = append(r.pipeTo, nil)
		r.toBlock = append(r.toBlock, nil)
		r.input = append(r.input, false)
		r.innerCode(s)
		if r.code.Len() > 0 {
			r.code.WriteString(" ")
		}
		r.code.WriteString(s)
		if len(r.group) > 0 {
			r.gtext.WriteString(" ")
		}
		r.gtext.WriteString(s)
		r.group = append(r.group, s)
	}
	if op != "" && r.code.Len() > 0 {
		r.code.WriteString(" " + op)
	}
	if joins[op] {
		if len(r.group) > 0 {
			r.gtext.WriteString(" " + op)
		}
	} else {
		r.endGroup()
	}
	r.block(s, op, pipe)
	b.reset()
}

// open is a block not yet closed: where it started (an index into cmds), and
// what closes it — ) or }, esac (inside a case a ) ends a pattern, not a
// block), fi, or done.
type open struct {
	start   int
	end     string
	feeders [][2]int // what is sent into a ( or {: read by every command in it
	input   bool     // <( … ): what it writes is read as a file — by sh, source, python3 as well
}

// closers are the compound commands a keyword opens, by the word that ends
// each.
var closers = map[string]string{"case": "esac", "if": "fi", "for": "done", "while": "done", "until": "done", "select": "done"}

// block keeps track of ( … ), { … }, case … esac, if … fi and loops' … done,
// and of what each cut sends on — | and |& pipe it, > >(…) and >(…) write it
// into a process substitution. What is sent on is read by the next command,
// and by every command of a block opening next: cat <<EOF | { :; sh; },
// (cat <<EOF) > >(sh). What a block writes is sent on by the cut after it,
// whatever redirection comes between ((…) 2>/dev/null | sh); blocks nest, and
// each one sending on adds readers: ((cat <<EOF) | sh) | grep x is read by sh
// and by grep.
func (r *result) block(s, op string, pipe bool) {
	var ended [][2]int // blocks whose output this cut sends on
	if r.closed != nil {
		ended = append(ended, [2]int{r.closed[0], r.closed[1]})
	}
	r.closed = nil
	if s != "" && r.d == Sh {
		k, n := keyword(s), len(r.opens)
		switch {
		case closers[k] != "":
			r.opens = append(r.opens, open{start: len(r.cmds) - 1, end: closers[k]})
		case n > 0 && k == r.opens[n-1].end: // esac, fi, done
			ended = append(ended, [2]int{r.opens[n-1].start, len(r.cmds) - 1})
			r.opens = r.opens[:n-1]
		}
	}
	var feeders [][2]int
	if pipe || op == "(" && strings.HasSuffix(s, ">") {
		feeders = ended
		if s != "" {
			feeders = append(feeders, [2]int{len(r.cmds) - 1, len(r.cmds)})
		}
		for _, f := range feeders {
			r.readBy(f[0], f[1], len(r.cmds))
		}
		if op == "(" && r.d == Sh && lead(s) == "exec" {
			r.incomplete = true // exec > >(sh): all written after it goes there
		}
	}
	if r.d == Sh {
		// f () { cat <<EOF; } … f | sh, coproc sh; cat <<EOF >&${COPROC[1]},
		// alias cat=sh, $e "$x": a heredoc reaches its reader through a name, a
		// descriptor or a variable set up earlier, not through anything on its
		// line. (A $( ) program, as eval $(…) runs, is the substitution's
		// output: that substitution is scored.)
		l := lead(s)
		if r.funcName && op == ")" && s == "" || keyword(s) == "function" || keyword(s) == "coproc" || l == "alias" || expands(l) {
			r.untracked = true
		}
		if expands(l) {
			r.varProg = true
		}
		r.funcName = op == "(" && len(strings.Fields(s)) == 1 && !strings.ContainsAny(s, "<>=") // not arr=( )
	}
	pending := r.feeding
	r.feeding = nil
	n := len(r.opens)
	switch {
	case op == "(" || op == "{":
		o := open{start: len(r.cmds), end: ")", feeders: feeders, input: op == "(" && strings.HasSuffix(s, "<")}
		if feeders == nil && s == "" {
			o.feeders = pending
		}
		r.opens = append(r.opens, o)
	case (op == ")" || op == "}") && n > 0 && r.opens[n-1].end == ")":
		o := r.opens[n-1]
		r.opens = r.opens[:n-1]
		r.closed = []int{o.start, len(r.cmds)}
		r.blocks = append(r.blocks, [2]int{o.start, len(r.cmds)})
		if o.input && r.link(len(r.cmds)-o.start) { // sh <(cat <<EOF): read as a script, maybe
			for j := o.start; j < len(r.cmds); j++ {
				r.input[j] = true
			}
		}
		for _, f := range o.feeders {
			if !r.link(f[1] - f[0] + len(r.cmds) - o.start) { // readers reads the block through
				break
			}
			for j := f[0]; j < f[1]; j++ {
				r.toBlock[j] = append(r.toBlock[j], len(r.blocks)-1)
			}
		}
	default:
		r.feeding = feeders
	}
}

// keyword is a command's first word past the ! and time (-p, --) that may
// come before a compound command: ! case …, time -p for ….
func keyword(s string) string {
	for _, w := range strings.Fields(s) {
		if w != "!" && w != "time" && w != "-p" && w != "--" {
			return w
		}
	}
	return ""
}

// maxLinks bounds the reader links a line makes: ( ( … ) | sh ) | sh … nested
// n deep makes n²/2 of them. Past it, the parse is incomplete.
const maxLinks = 1 << 16

// link counts n reader links; whether there is room for them.
func (r *result) link(n int) bool {
	if r.links += n; r.links > maxLinks {
		r.incomplete = true
		return false
	}
	return true
}

// expands: s holds a $ that isn't a $( … ) substitution — a variable, ${ },
// $(( )) — whose value eval would run.
func expands(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '$' && !(i+1 < len(s) && s[i+1] == '(' && !(i+2 < len(s) && s[i+2] == '(')) {
			return true
		}
	}
	return false
}

// lead is the builtin a simple command runs: its first word past !, time,
// assignments, redirections, and command or builtin with their flags —
// quotes removed, and a redirection it carries (exec>log) left off. exec,
// alias and a program named by a variable ($e "$x") are found however they
// are written.
func lead(s string) string {
	ws := words(s)
	for i := 0; i < len(ws); i++ {
		switch w := ws[i]; {
		case operator(w):
			i++
		case w == "!" || w == "time" || w == "command" || w == "builtin" || w == "-p" || w == "-v" || w == "-V" || w == "--" || assignment(w):
		default:
			if p, ok := program(w); ok {
				return p
			}
		}
	}
	return ""
}

// readBy adds cmds[t] as a reader of cmds[from:to].
func (r *result) readBy(from, to, t int) {
	if !r.link(to - from) {
		return
	}
	for j := from; j < to; j++ {
		r.pipeTo[j] = append(r.pipeTo[j], t)
	}
}

func (r *result) endGroup() {
	if len(r.group) > 1 {
		r.groups = append(r.groups, Trim(strings.TrimRight(Trim(r.gtext.String()), "|&")))
	}
	r.group = nil
	r.gtext.Reset()
}

// nest adds what a nested piece of code holds, after the line's own.
func (r *result) nest(n *[]string, ng *[]string, d Dialect, code string) Parsed {
	p := parse(d, code, r.depth+1, r.budget)
	*n = append(*n, p.Commands...)
	*ng = append(*ng, p.Groups...)
	r.incomplete = r.incomplete || p.Incomplete
	r.varProg = r.varProg || p.varProg
	r.dropped = r.dropped || p.dropped
	return p
}

// innerCode adds the code a command hands to another interpreter as an
// argument: sh -c '…', pwsh -Command "…", powershell -EncodedCommand …,
// cmd /c …, python -c '…', eval …. The interpreter is looked for among all the
// command's words, not just its program (see runner), in one pass: a flag
// belongs to the latest interpreter named before it.
func (r *result) innerCode(cmd string) {
	ws := words(cmd)
	if r.d == Cmd { // cmd's own words aren't sh's; /c is read as the rest of the line
		ws = strings.Fields(cmd)
	}
	prog := ""
	for i := 0; i < len(ws); i++ {
		switch w := ws[i]; {
		case operator(w):
			i++ // its operand: << sh is a heredoc's word, > sh a file
		case wordRunner(w) == "eval":
			arg := strings.Join(ws[i+1:], " ")
			// eval runs in this shell: a coproc, function or alias it sets up
			// stays — and eval "$x" runs a value that isn't on the line.
			if p := r.nest(&r.inner, &r.innerG, r.d, arg); p.untracked {
				r.untracked = true
			}
			if r.d == Sh && expands(arg) {
				r.incomplete = true
			}
			return
		case wordRunner(w) != "":
			prog = wordRunner(w)
		case prog != "" && i+1 < len(ws) && r.handOff(prog, w, ws[i+1:]):
			return
		}
	}
}

// handOff adds the code prog is handed by flag w, rest being what follows the
// flag; whether w was such a flag.
func (r *result) handOff(prog, w string, rest []string) bool {
	ps := prog == "pwsh" || prog == "powershell"
	switch {
	case shells[prog] && strings.HasPrefix(w, "-") && !strings.HasPrefix(w, "--") && strings.Contains(w, "c"):
		r.nest(&r.inner, &r.innerG, Sh, rest[0])
	case ps && (strings.EqualFold(w, "-c") || strings.EqualFold(w, "-command")):
		r.nest(&r.inner, &r.innerG, PowerShell, strings.Join(rest, " "))
	case ps && encodedFlag(w):
		if code, ok := decodeUTF16(rest[0]); ok {
			r.nest(&r.inner, &r.innerG, PowerShell, code)
		} else {
			r.incomplete = true // a script in a form that can't be read
		}
	case prog == "cmd" && (strings.EqualFold(w, "/c") || strings.EqualFold(w, "/k")):
		r.nest(&r.inner, &r.innerG, Cmd, strings.Join(rest, " "))
	case interpreters[prog] && (w == "-c" || w == "-e" || w == "--eval"):
		if t := Trim(rest[0]); t != "" {
			r.inner = append(r.inner, t) // another language: scored as itself
		}
	default:
		return false
	}
	if r.d == Sh && expands(strings.Join(rest, " ")) {
		r.varProg = true // sh -c "$x", python3 -c "$x": a value not on the line runs
	}
	return true
}

// encodedFlag is PowerShell's -EncodedCommand as it may be written: any prefix
// of it (-e, -enc, -encoded), -ec, with - or /.
func encodedFlag(w string) bool {
	l := strings.ToLower(w)
	if len(l) < 2 || (l[0] != '-' && l[0] != '/') {
		return false
	}
	return l[1:] == "ec" || strings.HasPrefix("encodedcommand", l[1:])
}

// decodeUTF16 reads -EncodedCommand's argument: base64 of UTF-16LE text.
func decodeUTF16(s string) (string, bool) {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil || len(b)%2 != 0 {
		return "", false
	}
	u := make([]uint16, len(b)/2)
	for i := range u {
		u[i] = uint16(b[2*i]) | uint16(b[2*i+1])<<8
	}
	return string(utf16.Decode(u)), true
}

// words splits a simple command into its words with sh quote removal:
// '…' literal, "…" with \ escapes, \ outside quotes.
func words(cmd string) []string {
	var out []string
	var w strings.Builder
	in := false
	rs := []rune(cmd)
	for i := 0; i < len(rs); i++ {
		c := rs[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n':
			if in {
				out = append(out, w.String())
				w.Reset()
				in = false
			}
		case c == '\'':
			in = true
			j := indexFrom(rs, i+1, '\'')
			w.WriteString(string(rs[i+1 : min(j, len(rs))]))
			i = j
		case c == '"':
			in = true
			i++
			for i < len(rs) && rs[i] != '"' {
				if rs[i] == '\\' && i+1 < len(rs) && strings.ContainsRune("\"\\$`", rs[i+1]) {
					i++
				}
				w.WriteRune(rs[i])
				i++
			}
		case c == '\\' && i+1 < len(rs):
			in = true
			i++
			w.WriteRune(rs[i])
		default:
			in = true
			w.WriteRune(c)
		}
	}
	if in {
		out = append(out, w.String())
	}
	return out
}

// buf is the command being read. It remembers its last rune and whether an
// escape produced it, so "is this the start of a word" and "is this & part of
// a redirection" are answered without re-reading the buffer — and an escaped
// space or > is not mistaken for a real one.
type buf struct {
	b       strings.Builder
	last    rune
	lastEsc bool
}

func (b *buf) write(r rune, esc bool) { b.b.WriteRune(r); b.last, b.lastEsc = r, esc }
func (b *buf) writeStr(s string) {
	if s == "" {
		return
	}
	b.b.WriteString(s)
	rs := []rune(s)
	b.last, b.lastEsc = rs[len(rs)-1], false
}
func (b *buf) reset()         { b.b.Reset(); b.last, b.lastEsc = 0, false }
func (b *buf) String() string { return b.b.String() }
func (b *buf) empty() bool    { return b.b.Len() == 0 }

// after reports whether the last rune is one of set, written as itself.
func (b *buf) after(set string) bool {
	return !b.empty() && !b.lastEsc && strings.ContainsRune(set, b.last)
}

// --- sh -----------------------------------------------------------------

// shells read a heredoc as a script; interpreters read it as their own code.
var (
	// ssh and su run a heredoc as a shell script too — on another host, as
	// another user.
	shells       = map[string]bool{"sh": true, "bash": true, "zsh": true, "dash": true, "ksh": true, "ash": true, "mksh": true, "ssh": true, "su": true}
	interpreters = map[string]bool{"python": true, "python3": true, "python2": true, "perl": true, "ruby": true, "node": true, "nodejs": true,
		"php": true, "lua": true, "deno": true, "bun": true, "osascript": true, "tclsh": true, "Rscript": true}
)

// runner is the interpreter a word names, by its base name — a shell,
// PowerShell, cmd, another language, or eval — or "".
//
// What runs code is looked for among all of a command's words, not only its
// program: telling the program from its wrappers' arguments (sudo -u deploy sh,
// timeout -s KILL 5 sh, env -u FOO sh, &>log sh) takes knowing every option of
// every wrapper, and getting one wrong hides the code. A word taken for an
// interpreter that isn't one costs only a little more scoring. A version in
// the name doesn't hide it: python3.11, ksh93, perl5.38 are python, ksh, perl.
func runner(w string) string {
	b := name(w)
	switch l := strings.ToLower(b); l {
	case "pwsh", "powershell", "cmd":
		return l
	}
	if v := strings.TrimRight(b, "0123456789."); v != "" {
		b = v
	}
	if b == "eval" || shells[b] || interpreters[b] {
		return b
	}
	return ""
}

// name is a program word's base name: quotes (cmd keeps them), directories
// and .exe left off.
func name(w string) string {
	b := path.Base(strings.ReplaceAll(strings.Trim(w, `"`), `\`, "/"))
	if len(b) > 4 && strings.EqualFold(b[len(b)-4:], ".exe") {
		b = b[:len(b)-4]
	}
	return b
}

// operator: a redirection operator standing alone and waiting for its operand,
// the next word (<<, <<-, >, 2>, &>, 2>&) — not one complete in itself (2>&1,
// <&0, 2>&-), which the next word is not part of.
func operator(w string) bool {
	if !strings.ContainsAny(w, "<>") || strings.Trim(w, "0123456789&<>-|") != "" {
		return false
	}
	return strings.HasSuffix(w, "<<-") || strings.ContainsRune("<>&|", rune(w[len(w)-1]))
}

// wordRunner is runner for a word that may carry a redirection with it:
// sh<<EOF, python3<<'EOF', sh>log — the word before the operator.
func wordRunner(w string) string {
	if i := strings.IndexAny(w, "<>"); i >= 0 {
		w = strings.TrimRight(w[:i], "0123456789&")
	}
	return runner(w)
}

// feed is what reads a heredoc: a shell (its commands run), another language
// (scored as itself), both — an interpreter's arguments can name a shell — a
// program not known to read data (what it does with it can't be told), or
// none of these (data).
type feed uint8

const (
	script feed = 1 << iota
	code
	unknown
	psScript  // read by PowerShell: pwsh -Command -, pwsh -File -
	cmdScript // read by cmd
)

// scored is what has a heredoc's body scored.
const scored = script | code | psScript | cmdScript

// dataReaders are programs that read a heredoc as data: they store, show,
// search or transform it, and run none of it. Not sed (its e command runs
// lines), awk (system), patch (writes the files the body names), less
// (LESSOPEN); git and gh only for the subcommands in dataSubcommands. A heredoc is data only for one
// of them; any other reader — sudo, xargs, psql, docker, $SH — is unknown, and
// the parse incomplete: the list of what runs its input is never complete,
// the list of what doesn't can be.
var dataReaders = map[string]bool{":": true, "cat": true, "tee": true, "git": true, "gh": true, "grep": true,
	"egrep": true, "fgrep": true, "rg": true, "sort": true, "uniq": true, "wc": true, "head": true, "tail": true,
	"cut": true, "tr": true, "base64": true, "jq": true, "yq": true, "diff": true, "cmp": true,
	"column": true, "fold": true, "nl": true, "od": true, "xxd": true, "hexdump": true, "more": true,
	"read": true, "echo": true, "printf": true, "tac": true, "rev": true, "paste": true, "envsubst": true,
	"md5sum": true, "sha256sum": true, "shasum": true}

// readerRuns are the flags that make a data reader run a program on what it
// reads (sort --compress-program, which GNU sort takes abbreviated) or on the
// files it is given (rg --pre).
var readerRuns = map[string][]string{"sort": {"--com"}, "rg": {"--pre"}}

// feeds is what cmd runs what it reads as: every interpreter among its words
// counts (see runner) — a match adds what is scored, never takes it away —
// and a program that is neither an interpreter nor a data reader makes it
// unknown. (What it writes into a process substitution, > >(sh), is read by
// that: see block.)
func feeds(cmd string) feed {
	var f feed
	prog, found, at, env := "", false, 0, false
	ws := words(cmd)
	for i := 0; i < len(ws); i++ {
		w := ws[i]
		if operator(w) {
			i++
			continue
		}
		switch r := wordRunner(w); {
		case shells[r]:
			f |= script
		case interpreters[r]:
			f |= code
		case r == "pwsh" || r == "powershell":
			f |= psScript
		case r == "cmd":
			f |= cmdScript
		}
		if !found && !assignment(w) {
			prog, found = program(w)
			at = i
		}
		env = env || !found && assignment(w) && (strings.HasPrefix(w, "GIT_") || strings.HasPrefix(w, "GH_"))
	}
	switch {
	case !found:
		f |= unknown
	case (prog == "git" || prog == "gh") && (env || !dataSubcommand(prog, ws[at+1:])):
		f |= unknown // git apply, an alias (git -c alias.x='!sh' x), a gh extension, GIT_EDITOR=sh
	case slices.ContainsFunc(ws[at+1:], func(w string) bool {
		return slices.ContainsFunc(readerRuns[prog], func(f string) bool { return strings.HasPrefix(w, f) })
	}):
		f |= unknown // sort --compress-program=sh pipes what it reads through sh
	case dataReaders[prog], wordRunner(prog) != "":
	default:
		f |= unknown
	}
	return f
}

// program is the program a word names, if it names one: not a redirection
// with its operand (2>/dev/null), and before one it carries (cat>f).
func program(w string) (string, bool) {
	if i := strings.IndexAny(w, "<>"); i >= 0 {
		if strings.Trim(w[:i], "0123456789&") == "" {
			return "", false
		}
		w = w[:i]
	}
	return name(w), true
}

// assignment is a NAME=value word before a command.
func assignment(w string) bool {
	i := strings.IndexByte(w, '=')
	if i <= 0 {
		return false
	}
	for j, c := range w[:i] {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || j > 0 && c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

// dataSubcommands are what git and gh do with a heredoc as data: a message,
// an object, a request body, paths to check. Not git apply, am, fast-import,
// update-index or update-ref — the body says what they change — nor an alias
// or a gh extension, which can be anything.
var dataSubcommands = map[string]map[string]bool{
	"git": {"commit": true, "tag": true, "notes": true, "hash-object": true, "cat-file": true, "check-ignore": true,
		"check-attr": true, "check-mailmap": true, "rev-list": true, "log": true, "diff": true, "show": true,
		"status": true, "stripspace": true, "interpret-trailers": true, "mktree": true, "mktag": true,
		"commit-tree": true, "rev-parse": true, "ls-files": true, "grep": true, "shortlog": true, "blame": true},
	"gh": {"api": true, "issue": true, "pr": true, "release": true, "gist": true, "secret": true, "variable": true,
		"repo": true, "label": true, "workflow": true, "run": true, "search": true},
}

// dataSubcommand: ws, the words after git or gh, name one of its
// dataSubcommands — past the options before it, but no -c or --config-env,
// which can define an alias or a command to run — and don't open an editor
// (-e, --edit): git commit -e -F - hands the body to whatever core.editor is,
// and sh as an editor runs it.
func dataSubcommand(prog string, ws []string) bool {
	for i := 0; i < len(ws); i++ {
		w := ws[i]
		switch {
		case operator(w):
			i++
		case strings.HasPrefix(w, "-c") || strings.HasPrefix(w, "--config-env") || strings.HasPrefix(w, "--exec-path"):
			return false
		case w == "-C" || w == "--git-dir" || w == "--work-tree" || w == "--namespace" || w == "-R" || w == "--repo":
			i++
		case strings.HasPrefix(w, "-"):
		default:
			sub, ok := program(w)
			return ok && dataSubcommands[prog][sub] && !slices.ContainsFunc(ws[i+1:], func(w string) bool {
				return w == "-e" || w == "--edit" || w == "--editor"
			})
		}
	}
	return false
}

// splitSh follows POSIX sh: ; & && || | |& and newlines separate commands;
// '…' hides everything, "…" and \ hide separators; # at the start of a word
// comments out the rest of the line; a heredoc's body (<<WORD … WORD) is data
// for a data reader, code for a shell or an interpreter, and for any other
// reader not known: the parse is incomplete (see feeds) — and an unquoted
// word's $( ) and ` ` run whoever reads it; $( … ) and `…` hold commands of
// their own; ( ) and { } group commands.
func splitSh(line string, out *result) {
	rs := []rune(line)
	var b buf
	var heredocs []heredoc // pending on this line: their bodies start at the next newline
	var nested, ngroups []string
	cut := func(op string) { out.cut(&b, op) }
	// Operators cut the command, so a word starts after a blank or at the
	// start — a ) written into the buffer closed a $( ) and starts nothing.
	wordStart := func() bool { return b.empty() || b.after(" \t\n") }
	blankNext := func(i int) bool { return i+1 >= len(rs) || strings.ContainsRune(" \t\n;&|)", rs[i+1]) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '\\' && i+1 < len(rs):
			if rs[i+1] == '\n' { // a line continuation: the command goes on
				i++
				continue
			}
			b.write(r, true)
			i++
			b.write(rs[i], true)
		case r == '\'':
			j := min(indexFrom(rs, i+1, '\'')+1, len(rs)) // past the closing quote
			b.writeStr(string(rs[i:j]))
			i = j - 1
		case r == '"':
			j := i + 1
			for ; j < len(rs) && rs[j] != '"'; j++ {
				switch {
				case rs[j] == '\\':
					j++
				case rs[j] == '$' && j+1 < len(rs) && rs[j+1] == '(':
					end := closeParen(rs, j+2, Sh)
					out.nest(&nested, &ngroups, Sh, string(rs[j+2:min(end, len(rs))]))
					j = end
				case rs[j] == '`':
					end := indexFrom(rs, j+1, '`')
					out.nest(&nested, &ngroups, Sh, string(rs[j+1:min(end, len(rs))]))
					j = end
				}
			}
			j = min(j+1, len(rs))
			b.writeStr(string(rs[i:j]))
			i = j - 1
		case r == '$' && i+1 < len(rs) && rs[i+1] == '(' && !(i+2 < len(rs) && rs[i+2] == '('):
			end := closeParen(rs, i+2, Sh)
			out.nest(&nested, &ngroups, Sh, string(rs[i+2:min(end, len(rs))]))
			b.writeStr(string(rs[i:min(end+1, len(rs))]))
			i = end
		case r == '`':
			end := indexFrom(rs, i+1, '`')
			out.nest(&nested, &ngroups, Sh, string(rs[i+1:min(end, len(rs))]))
			b.writeStr(string(rs[i:min(end+1, len(rs))]))
			i = end
		case r == '#' && wordStart():
			for i < len(rs) && rs[i] != '\n' {
				i++
			}
			i-- // the newline itself still ends the command
		case r == '<' && i+2 < len(rs) && rs[i+1] == '<' && rs[i+2] == '<': // <<< a here-string: one word, read on
			b.writeStr("<<<")
			i += 2
		case r == '<' && i+1 < len(rs) && rs[i+1] == '<':
			h, next := readHeredoc(rs, i+2)
			h.cmd = len(out.cmds) // the command it belongs to; what it feeds is read when that ends
			heredocs = append(heredocs, h)
			b.writeStr(string(rs[i:next]))
			i = next - 1
		case r == '\n':
			cut(";")
			var fed []feed
			if len(heredocs) > 0 {
				fed = out.readers(heredocs[0].cmd)
			}
			for _, h := range heredocs {
				var f feed
				if h.cmd < len(out.cmds) {
					f = fed[h.cmd-heredocs[0].cmd]
				}
				start := i + 1
				end, next := heredocBody(rs, start, h)
				body := string(rs[start:end])
				if !h.quoted {
					substitutions(body, out, &nested, &ngroups) // expanded first: its $( ) and ` ` run, whatever reads it
				}
				switch {
				case f&unknown != 0 && f&scored == 0:
					out.incomplete = true // read by a program that may run it, and scored as nothing
				case f&scored == 0:
					out.dropped = true
				}
				if f&script != 0 {
					out.nest(&nested, &ngroups, Sh, body) // its commands run
				}
				if f&code != 0 {
					if s := Trim(body); s != "" {
						nested = append(nested, s) // code in another language: scored as itself
					}
				}
				if f&psScript != 0 {
					out.nest(&nested, &ngroups, PowerShell, body)
				}
				if f&cmdScript != 0 {
					out.nest(&nested, &ngroups, Cmd, body)
				}
				i = next - 1
			}
			heredocs = nil
		case r == ';' || r == '(' || r == ')':
			cut(string(r))
		case r == '{' && wordStart() && blankNext(i), r == '}' && wordStart() && blankNext(i):
			cut(string(r))
		case r == '|':
			op := "|"
			if i+1 < len(rs) && (rs[i+1] == '|' || rs[i+1] == '&') {
				i++
				op += string(rs[i])
			}
			cut(op)
		case r == '&':
			switch {
			case i+1 < len(rs) && rs[i+1] == '&':
				i++
				cut("&&")
			case i+1 < len(rs) && rs[i+1] == '>', // &> file
				b.after("<>"): // 2>&1, <&3 — a real >, not an escaped one
				b.write(r, false)
			default: // a lone & runs what precedes it in the background
				cut("&")
			}
		default:
			b.write(r, false)
		}
	}
	cut("")
	out.cmds = append(out.cmds, nested...)
	out.groups = append(out.groups, ngroups...)
}

// readers is, for each command from the first on, what reads its heredoc:
// the command, every command its output is piped into (cat <<EOF | sh), and
// what the output of a block it is in is piped into ((cat <<EOF) | sh). A
// pipe that goes on past the line, or a block still open at its end, leaves
// the reader unknown.
func (r *result) readers(first int) []feed {
	if first >= len(r.cmds) {
		return nil
	}
	unclosed := len(r.cmds) // from here on, commands are in a block not yet closed
	if len(r.opens) > 0 {
		unclosed = r.opens[0].start
	}
	fed := make([]feed, len(r.cmds)-first)
	into := func(t int) feed {
		if t < len(r.cmds) {
			return fed[t-first]
		}
		return unknown
	}
	// A block reads as all its commands do; worked out once per block.
	blockFed := map[int]feed{}
	intoBlock := func(id int) feed {
		f, ok := blockFed[id]
		if !ok {
			for t := r.blocks[id][0]; t < r.blocks[id][1]; t++ {
				f |= into(t)
			}
			blockFed[id] = f
		}
		return f
	}
	for j := len(r.cmds) - 1; j >= first; j-- {
		f := feeds(r.cmds[j])
		if r.piped[j] {
			f |= into(j + 1)
		}
		for _, t := range r.pipeTo[j] {
			f |= into(t)
		}
		for _, id := range r.toBlock[j] {
			f |= intoBlock(id)
		}
		// Unknown: in a block still open (what is sent into one reaches its first
		// command, so that too), in a <( … ) — read as a file by whatever is
		// given it — or after a function or coproc is set up.
		if j >= unclosed || r.input[j] || r.untracked {
			f |= unknown
		}
		fed[j-first] = f
	}
	return fed
}

// substitutions adds the commands in the $( ) and ` ` of expanded text.
func substitutions(body string, out *result, nested, ngroups *[]string) {
	rs := []rune(body)
	for i := 0; i < len(rs); i++ {
		switch {
		case rs[i] == '\\':
			i++
		case rs[i] == '$' && i+1 < len(rs) && rs[i+1] == '(':
			end := closeParen(rs, i+2, Sh)
			out.nest(nested, ngroups, Sh, string(rs[i+2:min(end, len(rs))]))
			i = end
		case rs[i] == '`':
			end := indexFrom(rs, i+1, '`')
			out.nest(nested, ngroups, Sh, string(rs[i+1:min(end, len(rs))]))
			i = end
		}
	}
}

type heredoc struct {
	word   string
	tabs   bool // <<- : the closing word may be indented with tabs
	quoted bool // any quoting in the word: the body is not expanded
	cmd    int  // the index of the command it feeds
}

// readHeredoc reads the word after << (or <<-), with the shell's quote
// removal; next is where the command line goes on.
func readHeredoc(rs []rune, i int) (heredoc, int) {
	h := heredoc{}
	if i < len(rs) && rs[i] == '-' {
		h.tabs = true
		i++
	}
	for i < len(rs) && (rs[i] == ' ' || rs[i] == '\t') {
		i++
	}
	var w strings.Builder
	for i < len(rs) && !strings.ContainsRune(" \t\n;&|<>()", rs[i]) {
		switch rs[i] {
		case '\'':
			h.quoted = true
			j := indexFrom(rs, i+1, '\'')
			w.WriteString(string(rs[i+1 : min(j, len(rs))]))
			i = min(j+1, len(rs))
			continue
		case '"':
			h.quoted = true
			i++
			for i < len(rs) && rs[i] != '"' {
				if rs[i] == '\\' && i+1 < len(rs) && rs[i+1] == '\n' { // a line continuation
					i += 2
					continue
				}
				if rs[i] == '\\' && i+1 < len(rs) && strings.ContainsRune("\"\\$`", rs[i+1]) {
					i++
				}
				w.WriteRune(rs[i])
				i++
			}
			i = min(i+1, len(rs))
			continue
		case '\\':
			if i+1 < len(rs) && rs[i+1] == '\n' { // a line continuation, not quoting
				i += 2
				continue
			}
			h.quoted = true
			i++
			if i >= len(rs) {
				continue
			}
		}
		w.WriteRune(rs[i])
		i++
	}
	h.word = w.String()
	return h, i
}

// heredocBody returns where the body ends (the start of its closing line) and
// where the line after the closing word starts; both the end of the text when
// it never closes.
func heredocBody(rs []rune, i int, h heredoc) (end, next int) {
	for i < len(rs) {
		e := indexFrom(rs, i, '\n')
		l := string(rs[i:e])
		// An unquoted body takes line continuations: EO\⏎F closes it as EOF —
		// but data\\⏎ ends in an escaped backslash, and the line ends there.
		for !h.quoted && continued(l) && e < len(rs) {
			n := indexFrom(rs, e+1, '\n')
			l = l[:len(l)-1] + string(rs[e+1:n])
			e = n
		}
		if h.tabs {
			l = strings.TrimLeft(l, "\t")
		}
		if l == h.word {
			return i, min(e+1, len(rs))
		}
		i = e + 1
	}
	return len(rs), len(rs)
}

// continued: the line ends in an odd run of backslashes — its newline escaped.
func continued(l string) bool {
	return (len(l)-len(strings.TrimRight(l, "\\")))%2 == 1
}

// --- PowerShell -----------------------------------------------------------

// splitPowerShell follows PowerShell: ; | && || and newlines separate
// commands (& is the call operator, not a separator); '…' (” escapes) hides
// everything, "…" and the backtick escape hide separators; # at a word's start
// and <# … #> are comments; here-strings (@'…'@, @"…"@) are data when what
// reads them is a psDataCommand, and scored as code when not (see psReads);
// $( ), @( ),
// ( ) and { } hold code of their own, as does $( ) inside "…" and @"…"@ —
// quotes of its own included.
func splitPowerShell(line string, out *result) {
	rs := []rune(line)
	var b buf
	var nested, ngroups []string
	var hs []hereString
	cut := func(op string) { out.cut(&b, op) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '`' && i+1 < len(rs):
			b.write(r, true)
			i++
			b.write(rs[i], true)
		case r == '@' && i+1 < len(rs) && (rs[i+1] == '\'' || rs[i+1] == '"') && hereStringStart(rs, i+2):
			q := rs[i+1]
			end := hereStringEnd(rs, i+2, q)
			if q == '"' {
				subexpressions(rs[i+2:min(end, len(rs))], out, &nested, &ngroups)
			}
			stop := len(rs) // the body ends before the \n'@ closing it
			if end < len(rs) {
				stop = end - 1
			}
			hs = append(hs, hereString{body: string(rs[i+2 : stop]), cmd: len(out.cmds)})
			b.writeStr("@" + string(q) + "…" + string(q) + "@") // the body is left out of the command
			i = end                                             // the line after it goes on
		case r == '\'':
			j := i + 1
			for j < len(rs) {
				if rs[j] == '\'' {
					if j+1 < len(rs) && rs[j+1] == '\'' {
						j += 2
						continue
					}
					break
				}
				j++
			}
			j = min(j+1, len(rs))
			b.writeStr(string(rs[i:j]))
			i = j - 1
		case r == '"':
			j := i + 1
			for j < len(rs) && rs[j] != '"' {
				switch {
				case rs[j] == '`':
					j += 2
				case rs[j] == '$' && j+1 < len(rs) && rs[j+1] == '(':
					end := closeParen(rs, j+2, PowerShell) // its own quotes don't end the string
					out.nest(&nested, &ngroups, PowerShell, string(rs[j+2:min(end, len(rs))]))
					j = end + 1
				default:
					j++
				}
			}
			j = min(j+1, len(rs))
			b.writeStr(string(rs[i:j]))
			i = j - 1
		case r == '<' && i+1 < len(rs) && rs[i+1] == '#': // <# block comment #>
			j := i + 2
			for j+1 < len(rs) && !(rs[j] == '#' && rs[j+1] == '>') {
				j++
			}
			i = j + 1
		case r == '#' && (b.empty() || b.after(" \t")):
			for i < len(rs) && rs[i] != '\n' {
				i++
			}
			i--
		case r == '(' || r == '{':
			end := closeParen(rs, i+1, PowerShell)
			if r == '{' {
				end = closeBrace(rs, i+1)
			}
			out.nest(&nested, &ngroups, PowerShell, string(rs[i+1:min(end, len(rs))]))
			b.writeStr(string(rs[i:min(end+1, len(rs))]))
			i = end
		case r == '\n' || r == ';':
			cut(";")
		case r == '|':
			op := "|"
			if i+1 < len(rs) && rs[i+1] == '|' {
				i++
				op = "||"
			}
			cut(op)
		case r == '&' && i+1 < len(rs) && rs[i+1] == '&':
			i++
			cut("&&")
		default:
			b.write(r, false)
		}
	}
	cut("")
	for _, h := range hs {
		if psReads(out, nested, h.cmd) {
			continue
		}
		if t := Trim(h.body); t != "" {
			nested = append(nested, t) // as itself: it may be another language
			out.nest(&nested, &ngroups, PowerShell, h.body)
		}
	}
	out.cmds = append(out.cmds, nested...)
	out.groups = append(out.groups, ngroups...)
}

type hereString struct {
	body string
	cmd  int // the index of the command it is in
}

// psReads: what reads the here-string in out.cmds[k] reads it as data — that
// command, and every one it is piped into (@'…'@ | Out-File f), each a
// psDataCommand. Assigned to a variable ($c = @'…'@), it is read wherever a
// variable is: every command of the line that names one must be one too
// (iex $c, & ([scriptblock]::Create($c))).
func psReads(out *result, nested []string, k int) bool {
	if k >= len(out.cmds) {
		return true // nothing reads it
	}
	for j := k; ; j++ {
		if !psDataCommand(out.cmds[j]) {
			return false
		}
		if !out.piped[j] || j+1 >= len(out.cmds) {
			break
		}
	}
	if _, ok := psAssignment(out.cmds[k]); !ok {
		return true
	}
	return !slices.ContainsFunc(slices.Concat(out.cmds, nested), func(c string) bool {
		return strings.Contains(c, "$") && !psDataCommand(c)
	})
}

// psAssignment: cmd assigns to a variable ($c = …, $c=…, $c += …); rest is
// the words of what it assigns.
func psAssignment(cmd string) (rest []string, ok bool) {
	ws := strings.Fields(cmd)
	if len(ws) == 0 {
		return nil, false
	}
	if v, after, eq := strings.Cut(ws[0], "="); eq {
		return append(strings.Fields(after), ws[1:]...), psVariable(strings.TrimRight(v, "+-*/"))
	}
	if len(ws) < 2 || !psVariable(ws[0]) {
		return nil, false
	}
	if op := strings.TrimLeft(ws[1], "+-*/"); strings.HasPrefix(op, "=") {
		return append(strings.Fields(op[1:]), ws[2:]...), true
	}
	return nil, false
}

// psDataCommands are PowerShell commands that take a here-string as data:
// they write, show, parse or search it, and run none of it.
var psDataCommands = map[string]bool{"set-content": true, "add-content": true, "out-file": true, "new-item": true,
	"write-output": true, "write-host": true, "write": true, "echo": true, "write-verbose": true,
	"write-information": true, "out-string": true, "out-null": true, "set-clipboard": true, "sc": true, "ac": true,
	"convertfrom-json": true, "convertfrom-csv": true, "convertfrom-stringdata": true, "select-string": true,
	"measure-object": true, "sort-object": true, "git": true, "gh": true}

// psDataCommand: cmd, past an assignment to a variable ($c = …), is nothing,
// a string or a bare variable, or a psDataCommands one (git and gh only for
// their dataSubcommands).
func psDataCommand(cmd string) bool {
	ws := strings.Fields(cmd)
	if rest, ok := psAssignment(cmd); ok {
		ws = rest
	}
	if len(ws) == 0 || psVariable(ws[0]) || strings.ContainsRune("'\"@", rune(ws[0][0])) {
		return true
	}
	prog := strings.ToLower(name(ws[0]))
	if prog == "git" || prog == "gh" {
		return dataSubcommand(prog, ws[1:])
	}
	return psDataCommands[prog]
}

// psVariable: w is a bare variable, $name or $scope:name — not a member or a
// call on one ($x.Invoke(), $ExecutionContext.InvokeCommand…).
func psVariable(w string) bool {
	if len(w) < 2 || w[0] != '$' {
		return false
	}
	for _, c := range w[1:] {
		if !(c == '_' || c == ':' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

// hereStringStart: a here-string's opening @' or @" ends its line.
func hereStringStart(rs []rune, i int) bool {
	for ; i < len(rs) && rs[i] != '\n'; i++ {
		if rs[i] != ' ' && rs[i] != '\t' && rs[i] != '\r' {
			return false
		}
	}
	return i < len(rs)
}

// hereStringEnd returns the index of the @ closing a here-string: '@ or "@
// at the start of a line.
func hereStringEnd(rs []rune, i int, q rune) int {
	for ; i+1 < len(rs); i++ {
		if rs[i] == '\n' && i+2 < len(rs) && rs[i+1] == q && rs[i+2] == '@' {
			return i + 2
		}
	}
	return len(rs)
}

// subexpressions adds the commands in the $( … ) of expandable text.
func subexpressions(rs []rune, out *result, nested, ngroups *[]string) {
	for i := 0; i+1 < len(rs); i++ {
		if rs[i] == '`' {
			i++
			continue
		}
		if rs[i] == '$' && rs[i+1] == '(' {
			end := closeParen(rs, i+2, PowerShell)
			out.nest(nested, ngroups, PowerShell, string(rs[i+2:min(end, len(rs))]))
			i = end
		}
	}
}

// --- cmd.exe --------------------------------------------------------------

// splitCmd follows cmd.exe: & && || | and newlines separate commands; ^
// escapes the next character; "…" hides separators (there is no escape inside
// it); ( ) group commands.
func splitCmd(line string, out *result) {
	rs := []rune(line)
	var b buf
	cut := func(op string) { out.cut(&b, op) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '^' && i+1 < len(rs):
			b.write(r, true)
			i++
			b.write(rs[i], true)
		case r == '"':
			j := min(indexFrom(rs, i+1, '"')+1, len(rs))
			b.writeStr(string(rs[i:j]))
			i = j - 1
		case r == '\n':
			cut("&")
		case r == '(' || r == ')':
			cut(string(r))
		case r == '&' || r == '|':
			if r == '&' && b.after("<>") { // 2>&1 — a real >, not ^>
				b.write(r, false)
				continue
			}
			op := string(r)
			if i+1 < len(rs) && rs[i+1] == r {
				i++
				op += op
			}
			cut(op)
		default:
			b.write(r, false)
		}
	}
	cut("")
}

// --- shared -----------------------------------------------------------------

// indexFrom is the index of r at or after i, or len(rs).
func indexFrom(rs []rune, i int, r rune) int {
	for ; i < len(rs); i++ {
		if rs[i] == r {
			return i
		}
	}
	return len(rs)
}

// closeParen returns the index of the ) matching an opening one just before
// i, past nested parens and quotes; len(rs) when it never closes.
func closeParen(rs []rune, i int, d Dialect) int {
	depth := 1
	for ; i < len(rs); i++ {
		switch r := rs[i]; {
		case d == Sh && r == '\\', d == PowerShell && r == '`':
			i++
		case r == '\'' || r == '"':
			i = indexFrom(rs, i+1, r)
		case r == '(':
			depth++
		case r == ')':
			if depth--; depth == 0 {
				return i
			}
		}
	}
	return len(rs)
}

// closeBrace is closeParen for a PowerShell script block's { }.
func closeBrace(rs []rune, i int) int {
	depth := 1
	for ; i < len(rs); i++ {
		switch r := rs[i]; {
		case r == '`':
			i++
		case r == '\'' || r == '"':
			i = indexFrom(rs, i+1, r)
		case r == '{':
			depth++
		case r == '}':
			if depth--; depth == 0 {
				return i
			}
		}
	}
	return len(rs)
}

// String is the dialect's name as data rows spell it: sh, powershell, cmd.
func (d Dialect) String() string {
	switch d {
	case PowerShell:
		return "powershell"
	case Cmd:
		return "cmd"
	}
	return "sh"
}
