package risk

import (
	"strings"
	"unicode/utf8"
)

// Localize rewrites the workspace's own absolute path in text as ".", so a
// project file reads as one: `wc -l /app/data.txt` with the workspace at /app
// is `wc -l ./data.txt`. The model can't know where a workspace is; the scorer
// does, and an absolute path into it otherwise read as outside — the largest
// source of false alarms on real agents' commands.
//
// Only a path that really is the workspace or inside it: the workspace must
// start the path (nothing that could continue a name before it) and either go
// on into it (a separator) or end there — at the end, at a closing quote, or,
// outside quotes, at a blank or an operator. So /application, /x/app/y,
// "/app backup/x" and /app,old are left alone. On Windows paths, case and
// / versus \ don't matter, as they don't to Windows. scripts/train_risk.py
// mirrors this (LOCALIZE_VECTORS).
func Localize(text, workspace string) string {
	ws := strings.TrimRight(workspace, `/\`)
	if ws == "" || ws == "." || ws == "~" || utf8.RuneCountInString(ws) < 2 {
		return text
	}
	hay, needle := text, ws
	if windowsPath(ws) {
		hay, needle = foldWindows(text), foldWindows(ws)
	}
	var b strings.Builder
	var quote byte
	for i := 0; i < len(text); {
		if strings.HasPrefix(hay[i:], needle) {
			end := i + len(needle)
			startOK := i == 0 || !strings.ContainsRune(pathRunes, rune(text[i-1]))
			endOK := end == len(text) || text[end] == '/' || text[end] == '\\' ||
				(quote == 0 && strings.ContainsRune(" \t\n;&|)<>", rune(text[end]))) ||
				(quote != 0 && text[end] == quote)
			if startOK && endOK {
				b.WriteString(".")
				i = end
				continue
			}
		}
		c := text[i]
		switch {
		case quote == 0 && (c == '"' || c == '\''):
			quote = c
		case c == quote:
			quote = 0
		}
		b.WriteByte(c)
		i++
	}
	return b.String()
}

const pathRunes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/\\~$"

// windowsPath: a drive letter or a backslash.
func windowsPath(p string) bool {
	return strings.Contains(p, `\`) || (len(p) >= 2 && p[1] == ':')
}

// foldWindows lowercases ASCII and turns / into \ — byte for byte, so offsets
// in the folded text are offsets in the original.
func foldWindows(s string) string {
	b := []byte(s)
	for i, c := range b {
		switch {
		case c >= 'A' && c <= 'Z':
			b[i] = c + 32
		case c == '/':
			b[i] = '\\'
		}
	}
	return string(b)
}

// localizeParams is a copy of params with every string value localized.
func localizeParams(params map[string]any, workspace string) map[string]any {
	out := make(map[string]any, len(params))
	for k, v := range params {
		if s, ok := v.(string); ok {
			v = Localize(s, workspace)
		}
		out[k] = v
	}
	return out
}
