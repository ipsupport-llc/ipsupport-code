package main

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

// textField is a one-line editable value for the /config panel: the inline
// edits (address, key, model, context size) and the add-provider form. It has
// a cursor and the keys a shell line has — ←/→, home/end (ctrl+a/ctrl+e),
// backspace/delete, ctrl+u (clear to the start), ctrl+k (to the end), typing
// and paste. A masked field shows ● per character, for keys.
type textField struct {
	r      []rune
	cur    int
	masked bool
}

func newTextField(value string, masked bool) textField {
	f := textField{masked: masked}
	f.set(value)
	return f
}

func (f *textField) set(s string) {
	f.r = []rune(s)
	f.cur = len(f.r)
}

func (f textField) value() string { return string(f.r) }

// key edits the field and reports whether k was an editing key; anything else
// (enter, esc, tab) is the caller's.
func (f *textField) key(k tea.KeyMsg) bool {
	switch k.String() {
	case "left", "ctrl+b":
		if f.cur > 0 {
			f.cur--
		}
	case "right", "ctrl+f":
		if f.cur < len(f.r) {
			f.cur++
		}
	case "home", "ctrl+a":
		f.cur = 0
	case "end", "ctrl+e":
		f.cur = len(f.r)
	case "backspace", "ctrl+h":
		if f.cur > 0 {
			f.r = append(f.r[:f.cur-1], f.r[f.cur:]...)
			f.cur--
		}
	case "delete":
		if f.cur < len(f.r) {
			f.r = append(f.r[:f.cur], f.r[f.cur+1:]...)
		}
	case "ctrl+u":
		f.r = append([]rune{}, f.r[f.cur:]...)
		f.cur = 0
	case "ctrl+k":
		f.r = f.r[:f.cur]
	default:
		s, ok := insertedText(k)
		if !ok {
			return false
		}
		// One line: a pasted value's line breaks are not part of it.
		s = strings.NewReplacer("\r\n", "", "\n", "", "\r", "").Replace(s)
		ins := []rune(s)
		f.r = append(f.r[:f.cur], append(ins, f.r[f.cur:]...)...)
		f.cur += len(ins)
	}
	return true
}

// view renders the value with the cursor as a reversed cell, ● per character
// when masked.
func (f textField) view() string {
	shown := f.r
	if f.masked {
		shown = []rune(strings.Repeat("●", len(f.r)))
	}
	var b strings.Builder
	b.WriteString(string(shown[:f.cur]))
	at := " "
	if f.cur < len(shown) {
		at = string(shown[f.cur])
	}
	b.WriteString("\x1b[7m" + at + "\x1b[27m")
	if f.cur < len(shown) {
		b.WriteString(string(shown[f.cur+1:]))
	}
	return b.String()
}
