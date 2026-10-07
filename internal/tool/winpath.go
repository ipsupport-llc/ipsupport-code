package tool

import "strings"

// mergePath is cur followed by every directory in fresh that cur lacks — the
// Windows ';' list, compared as Windows does (case, a trailing backslash).
// cur goes first so whatever the launching terminal put ahead (a venv, a
// pinned toolchain) still wins; what was installed since only adds.
func mergePath(cur string, fresh ...string) string {
	seen := map[string]bool{}
	var out []string
	for _, list := range append([]string{cur}, fresh...) {
		for _, dir := range strings.Split(list, ";") {
			dir = strings.TrimSpace(dir)
			key := strings.ToLower(strings.TrimRight(dir, `\`))
			if dir == "" || seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, dir)
		}
	}
	return strings.Join(out, ";")
}

// withEnv returns env with key set to value, replacing any entry for it —
// matched without case, as Windows names environment variables.
func withEnv(env []string, key, value string) []string {
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if k, _, ok := strings.Cut(kv, "="); ok && strings.EqualFold(k, key) {
			continue
		}
		out = append(out, kv)
	}
	return append(out, key+"="+value)
}
