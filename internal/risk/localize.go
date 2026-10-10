package risk

import "strings"

// Localize rewrites the workspace's own absolute path in text as ".", so a
// project file reads as one: `wc -l /app/data.txt` with the workspace at /app
// is `wc -l ./data.txt`. The model can't know where a workspace is; the scorer
// does, and an absolute path into it otherwise read as outside — the largest
// source of false alarms on real agents' commands.
//
// Only at a path's edges: the workspace must start the path (nothing that
// could continue a name before it) and end there (a separator, a quote, an
// operator or the end), so /application and /x/app/y are left alone.
// scripts/train_risk.py mirrors this (LOCALIZE_VECTORS).
func Localize(text, workspace string) string {
	ws := strings.TrimRight(workspace, `/\`)
	if ws == "" || ws == "." || ws == "~" || len(ws) < 2 {
		return text
	}
	var b strings.Builder
	for i := 0; i < len(text); {
		j := strings.Index(text[i:], ws)
		if j < 0 {
			b.WriteString(text[i:])
			break
		}
		j += i
		end := j + len(ws)
		startOK := j == 0 || !strings.ContainsRune(pathRunes, rune(text[j-1]))
		endOK := end == len(text) || strings.ContainsRune(pathEnds, rune(text[end]))
		b.WriteString(text[i:j])
		if startOK && endOK {
			b.WriteString(".")
		} else {
			b.WriteString(ws)
		}
		i = end
	}
	return b.String()
}

const (
	pathRunes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/\\~$"
	pathEnds  = "/\\ \t\n\"';&|)<>,:="
)

// localizeParams is params with every string value localized; the same map
// when there is nothing to do.
func localizeParams(params map[string]any, workspace string) map[string]any {
	if strings.TrimRight(workspace, `/\`) == "" {
		return params
	}
	out := make(map[string]any, len(params))
	for k, v := range params {
		if s, ok := v.(string); ok {
			v = Localize(s, workspace)
		}
		out[k] = v
	}
	return out
}
