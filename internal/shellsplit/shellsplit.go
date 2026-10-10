// Package shellsplit cuts a command line into the commands a shell would run,
// by that shell's own rules: what separates two commands, what quotes and
// escapes hide a separator, which text is data rather than a command (a
// heredoc body, a here-string, a comment), and which code is nested inside a
// command (command substitution, a PowerShell subexpression or script block).
//
// It is for scoring, not for executing: a nested command is returned as a
// command of its own as well as inside the one that holds it, and a line the
// rules can't finish (an unclosed quote) still yields what was read.
package shellsplit

import "strings"

// Dialect is the shell a command line is written for.
type Dialect int

const (
	Sh         Dialect = iota // POSIX sh and its kin (dash, bash, zsh): what `run` uses off Windows
	PowerShell                // Windows PowerShell 5.1 and PowerShell 7
	Cmd                       // cmd.exe
)

// Split returns the commands in line, in order, each trimmed; empty ones are
// dropped. Commands nested in another (`echo $(rm -r x)`) follow the one that
// holds them.
func Split(d Dialect, line string) []string { return parse(d, line).cmds }

// Code is line without its data — heredoc bodies, here-strings, comments —
// with its commands and the operators between them: the line as a program,
// for what only the combination says (`curl … | sh`).
func Code(d Dialect, line string) string {
	return strings.TrimSpace(strings.TrimSuffix(parse(d, line).code.String(), " ;"))
}

type result struct {
	cmds []string
	code strings.Builder
}

func parse(d Dialect, line string) *result {
	r := &result{}
	switch d {
	case PowerShell:
		splitPowerShell(line, r)
	case Cmd:
		splitCmd(line, r)
	default:
		splitSh(line, r)
	}
	return r
}

// cut ends the command being read at an operator.
func (r *result) cut(b *strings.Builder, op string) {
	if s := strings.TrimSpace(b.String()); s != "" {
		r.cmds = append(r.cmds, s)
		if r.code.Len() > 0 {
			r.code.WriteString(" ")
		}
		r.code.WriteString(s)
	}
	if op != "" && r.code.Len() > 0 {
		r.code.WriteString(" " + op)
	}
	b.Reset()
}

// --- sh -----------------------------------------------------------------

// splitSh follows POSIX sh: ; & && || | |& and newlines separate commands;
// '…' hides everything, "…" and \ hide separators; # at the start of a word
// comments out the rest of the line; a heredoc's body (<<WORD … WORD) is data;
// $( … ) and `…` hold commands of their own; ( ) and { } group commands.
func splitSh(line string, out *result) {
	rs := []rune(line)
	var b strings.Builder
	var heredocs []heredoc // pending on this line: their bodies start at the next newline
	var nested []string    // commands inside $( ) and ` `, after the ones that hold them
	cut := func(op string) { out.cut(&b, op) }
	wordStart := func() bool { return b.Len() == 0 || strings.ContainsRune(" \t\n;&|()", lastRune(b.String())) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '\\' && i+1 < len(rs):
			if rs[i+1] == '\n' { // a line continuation: the command goes on
				i++
				continue
			}
			b.WriteRune(r)
			i++
			b.WriteRune(rs[i])
		case r == '\'':
			j := min(indexFrom(rs, i+1, '\'')+1, len(rs)) // past the closing quote
			b.WriteString(string(rs[i:j]))
			i = j - 1
		case r == '"':
			j := i + 1
			for ; j < len(rs) && rs[j] != '"'; j++ {
				switch {
				case rs[j] == '\\':
					j++
				case rs[j] == '$' && j+1 < len(rs) && rs[j+1] == '(':
					end := closeParen(rs, j+2, Sh)
					nested = append(nested, Split(Sh, string(rs[j+2:min(end, len(rs))]))...)
					j = end
				case rs[j] == '`':
					end := indexFrom(rs, j+1, '`')
					nested = append(nested, Split(Sh, string(rs[j+1:min(end, len(rs))]))...)
					j = end
				}
			}
			j = min(j+1, len(rs))
			b.WriteString(string(rs[i:j]))
			i = j - 1
		case r == '$' && i+1 < len(rs) && rs[i+1] == '(' && !(i+2 < len(rs) && rs[i+2] == '('):
			end := closeParen(rs, i+2, Sh)
			nested = append(nested, Split(Sh, string(rs[i+2:min(end, len(rs))]))...)
			b.WriteString(string(rs[i:min(end+1, len(rs))]))
			i = end
		case r == '`':
			end := indexFrom(rs, i+1, '`')
			nested = append(nested, Split(Sh, string(rs[i+1:min(end, len(rs))]))...)
			b.WriteString(string(rs[i:min(end+1, len(rs))]))
			i = end
		case r == '#' && wordStart():
			for i < len(rs) && rs[i] != '\n' {
				i++
			}
			i-- // the newline itself still ends the command
		case r == '<' && i+1 < len(rs) && rs[i+1] == '<' && !(i+2 < len(rs) && rs[i+2] == '<'):
			h, next := readHeredoc(rs, i+2)
			heredocs = append(heredocs, h)
			b.WriteString(string(rs[i:next]))
			i = next - 1
		case r == '\n':
			cut(";")
			for _, h := range heredocs {
				i = skipHeredocBody(rs, i+1, h) - 1
			}
			heredocs = nil
		case r == ';' || r == '(' || r == ')' || r == '{' && wordStart() || r == '}' && wordStart():
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
				b.Len() > 0 && strings.ContainsRune("<>", lastRune(b.String())): // 2>&1, <&3
				b.WriteRune(r)
			default: // a lone & runs what precedes it in the background
				cut("&")
			}
		default:
			b.WriteRune(r)
		}
	}
	cut("")
	out.cmds = append(out.cmds, nested...)
}

type heredoc struct {
	word string
	tabs bool // <<- : the closing word may be indented with tabs
}

// readHeredoc reads the word after << (or <<-), quoted or not; next is where
// the command line goes on.
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
		case '\'', '"':
			q := rs[i]
			j := indexFrom(rs, i+1, q)
			w.WriteString(string(rs[i+1 : min(j, len(rs))]))
			i = min(j+1, len(rs))
			continue
		case '\\':
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

// skipHeredocBody returns the index just past the body's closing line, or the
// end of the line when it never closes.
func skipHeredocBody(rs []rune, i int, h heredoc) int {
	for i < len(rs) {
		end := indexFrom(rs, i, '\n')
		l := string(rs[i:end])
		if h.tabs {
			l = strings.TrimLeft(l, "\t")
		}
		if l == h.word {
			return min(end+1, len(rs))
		}
		i = end + 1
	}
	return len(rs)
}

// --- PowerShell -----------------------------------------------------------

// splitPowerShell follows PowerShell: ; | && || and newlines separate
// commands (& is the call operator, not a separator); '…' (” escapes) hides
// everything, "…" and the backtick escape hide separators; # and <# … #> are
// comments; here-strings (@'…'@, @"…"@) are data; $( ), @( ), ( ) and { }
// hold code of their own, as does $( ) inside "…" and @"…"@.
func splitPowerShell(line string, out *result) {
	rs := []rune(line)
	var b strings.Builder
	var nested []string
	cut := func(op string) { out.cut(&b, op) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '`' && i+1 < len(rs):
			b.WriteRune(r)
			i++
			b.WriteRune(rs[i])
		case r == '@' && i+1 < len(rs) && (rs[i+1] == '\'' || rs[i+1] == '"') && hereStringStart(rs, i+2):
			q := rs[i+1]
			end := hereStringEnd(rs, i+2, q)
			if q == '"' {
				nested = append(nested, subexpressions(rs[i+2:min(end, len(rs))])...)
			}
			b.WriteString("@" + string(q) + "…" + string(q) + "@") // the body is data
			i = end                                                // the line after it goes on
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
			b.WriteString(string(rs[i:j]))
			i = j - 1
		case r == '"':
			j := i + 1
			for j < len(rs) && rs[j] != '"' {
				if rs[j] == '`' {
					j++
				}
				j++
			}
			body := rs[i+1 : min(j, len(rs))]
			nested = append(nested, subexpressions(body)...)
			j = min(j+1, len(rs))
			b.WriteString(string(rs[i:j]))
			i = j - 1
		case r == '<' && i+1 < len(rs) && rs[i+1] == '#': // <# block comment #>
			j := i + 2
			for j+1 < len(rs) && !(rs[j] == '#' && rs[j+1] == '>') {
				j++
			}
			i = j + 1
		case r == '#' && (b.Len() == 0 || strings.ContainsRune(" \t;|&(){}", lastRune(b.String()))):
			for i < len(rs) && rs[i] != '\n' {
				i++
			}
			i--
		case r == '(' || r == '{':
			end := closeParen(rs, i+1, PowerShell)
			if r == '{' {
				end = closeBrace(rs, i+1)
			}
			inner := string(rs[i+1 : min(end, len(rs))])
			nested = append(nested, Split(PowerShell, inner)...)
			b.WriteString(string(rs[i:min(end+1, len(rs))]))
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
			b.WriteRune(r)
		}
	}
	cut("")
	out.cmds = append(out.cmds, nested...)
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

// subexpressions are the commands in the $( … ) of expandable text.
func subexpressions(rs []rune) []string {
	var out []string
	for i := 0; i+1 < len(rs); i++ {
		if rs[i] == '`' {
			i++
			continue
		}
		if rs[i] == '$' && rs[i+1] == '(' {
			end := closeParen(rs, i+2, PowerShell)
			out = append(out, Split(PowerShell, string(rs[i+2:min(end, len(rs))]))...)
			i = end
		}
	}
	return out
}

// --- cmd.exe --------------------------------------------------------------

// splitCmd follows cmd.exe: & && || | and newlines separate commands; ^
// escapes the next character; "…" hides separators (there is no escape inside
// it); ( ) group commands.
func splitCmd(line string, out *result) {
	rs := []rune(line)
	var b strings.Builder
	cut := func(op string) { out.cut(&b, op) }
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '^' && i+1 < len(rs):
			b.WriteRune(r)
			i++
			b.WriteRune(rs[i])
		case r == '"':
			j := indexFrom(rs, i+1, '"')
			j = min(j+1, len(rs))
			b.WriteString(string(rs[i:j]))
			i = j - 1
		case r == '\n':
			cut("&")
		case r == '(' || r == ')':
			cut(string(r))
		case r == '&' || r == '|':
			if r == '&' && b.Len() > 0 && strings.ContainsRune("<>", lastRune(b.String())) { // 2>&1
				b.WriteRune(r)
				continue
			}
			op := string(r)
			if i+1 < len(rs) && rs[i+1] == r {
				i++
				op += op
			}
			cut(op)
		default:
			b.WriteRune(r)
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

func lastRune(s string) rune {
	rs := []rune(s)
	if len(rs) == 0 {
		return 0
	}
	return rs[len(rs)-1]
}
