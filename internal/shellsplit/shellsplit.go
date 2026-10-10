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
	"path"
	"strings"
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
	// Code is the line without its data — heredoc bodies, here-strings,
	// comments — with its commands and the operators between them.
	Code string
}

// Split returns the commands in line (see Parsed.Commands).
func Split(d Dialect, line string) []string { return Parse(d, line).Commands }

// Code returns line as a program, without its data (see Parsed.Code).
func Code(d Dialect, line string) string { return Parse(d, line).Code }

// Parse cuts line once.
func Parse(d Dialect, line string) Parsed {
	r := &result{}
	switch d {
	case PowerShell:
		splitPowerShell(line, r)
	case Cmd:
		splitCmd(line, r)
	default:
		splitSh(line, r)
	}
	r.endGroup()
	code := strings.TrimSuffix(r.code.String(), " ;")
	return Parsed{Commands: r.cmds, Groups: r.groups, Code: Trim(code)}
}

// Trim is strings.TrimSpace — named so the Python port can match it exactly
// (Python's str.strip also drops \x1c-\x1f, which Go keeps).
func Trim(s string) string { return strings.TrimSpace(s) }

type result struct {
	cmds   []string
	groups []string
	code   strings.Builder
	group  []string // commands of the group being read
	gtext  strings.Builder
}

// joins are the operators that keep a group going.
var joins = map[string]bool{"|": true, "||": true, "&&": true, "|&": true}

// cut ends the command being read at an operator.
func (r *result) cut(b *buf, op string) {
	if s := Trim(b.String()); s != "" {
		r.cmds = append(r.cmds, s)
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
	b.reset()
}

func (r *result) endGroup() {
	if len(r.group) > 1 {
		r.groups = append(r.groups, Trim(strings.TrimRight(Trim(r.gtext.String()), "|&")))
	}
	r.group = nil
	r.gtext.Reset()
}

// nest adds what a nested piece of code holds, after the line's own.
func (r *result) nest(n *[]string, ng *[]string, d Dialect, code string) {
	p := Parse(d, code)
	*n = append(*n, p.Commands...)
	*ng = append(*ng, p.Groups...)
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
	shells       = map[string]bool{"sh": true, "bash": true, "zsh": true, "dash": true, "ksh": true, "ash": true, "mksh": true}
	interpreters = map[string]bool{"python": true, "python3": true, "python2": true, "perl": true, "ruby": true, "node": true,
		"php": true, "lua": true, "deno": true, "bun": true, "osascript": true, "tclsh": true, "Rscript": true}
	wrappers = map[string]bool{"sudo": true, "env": true, "exec": true, "nohup": true, "time": true, "command": true, "doas": true}
)

// program is the command's program name, past sudo/env and the like.
func program(cmd string) string {
	for _, f := range strings.Fields(cmd) {
		base := path.Base(f)
		if wrappers[base] || strings.HasPrefix(f, "-") || strings.Contains(f, "=") {
			continue
		}
		return base
	}
	return ""
}

// splitSh follows POSIX sh: ; & && || | |& and newlines separate commands;
// '…' hides everything, "…" and \ hide separators; # at the start of a word
// comments out the rest of the line; a heredoc's body (<<WORD … WORD) is data —
// unless its word is unquoted (its $( ) and ` ` run) or it feeds a shell or an
// interpreter (then it is code); $( … ) and `…` hold commands of their own;
// ( ) and { } group commands.
func splitSh(line string, out *result) {
	rs := []rune(line)
	var b buf
	var heredocs []heredoc // pending on this line: their bodies start at the next newline
	var nested, ngroups []string
	cut := func(op string) { out.cut(&b, op) }
	wordStart := func() bool { return b.empty() || b.after(" \t\n;&|()") }
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
			h.prog = program(b.String())
			heredocs = append(heredocs, h)
			b.writeStr(string(rs[i:next]))
			i = next - 1
		case r == '\n':
			cut(";")
			for _, h := range heredocs {
				start := i + 1
				end, next := heredocBody(rs, start, h)
				body := string(rs[start:end])
				switch {
				case shells[h.prog]:
					out.nest(&nested, &ngroups, Sh, body) // a script: its commands run
				case interpreters[h.prog]:
					if s := Trim(body); s != "" {
						nested = append(nested, s) // code in another language: scored as itself
					}
				case !h.quoted:
					substitutions(body, out, &nested, &ngroups) // expanded: its $( ) and ` ` run
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
	tabs   bool   // <<- : the closing word may be indented with tabs
	quoted bool   // any quoting in the word: the body is not expanded
	prog   string // the program reading it
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
				if rs[i] == '\\' && i+1 < len(rs) && strings.ContainsRune("\"\\$`", rs[i+1]) {
					i++
				}
				w.WriteRune(rs[i])
				i++
			}
			i = min(i+1, len(rs))
			continue
		case '\\':
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

// --- PowerShell -----------------------------------------------------------

// splitPowerShell follows PowerShell: ; | && || and newlines separate
// commands (& is the call operator, not a separator); '…' (” escapes) hides
// everything, "…" and the backtick escape hide separators; # at a word's start
// and <# … #> are comments; here-strings (@'…'@, @"…"@) are data; $( ), @( ),
// ( ) and { } hold code of their own, as does $( ) inside "…" and @"…"@ —
// quotes of its own included.
func splitPowerShell(line string, out *result) {
	rs := []rune(line)
	var b buf
	var nested, ngroups []string
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
			b.writeStr("@" + string(q) + "…" + string(q) + "@") // the body is data
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
		case r == '#' && (b.empty() || b.after(" \t;|&(){}")):
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
	out.cmds = append(out.cmds, nested...)
	out.groups = append(out.groups, ngroups...)
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
