package llm

import (
	"regexp"
	"strings"
)

// snapshotSuffix is the dated-snapshot tail hosted providers append to an
// alias when they report which model answered: gpt-4o → gpt-4o-2024-08-06,
// grok-4 → grok-4-0709.
var snapshotSuffix = regexp.MustCompile(`^-[0-9][0-9-]*$`)

// SameModel reports whether served — the model a server says answered — is the
// model that was asked for. It tolerates differences that don't change the
// model: letter case, a publisher prefix (google/x vs x), a quantization tag
// one side leaves off (x@q8_0 vs x), Ollama's :latest, and a dated snapshot of
// an alias. An empty served (the server didn't say) counts as the same.
func SameModel(asked, served string) bool {
	a, s := normModel(asked), normModel(served)
	if s == "" || a == s {
		return true
	}
	if baseModel(a) == baseModel(s) && (!strings.Contains(a, "@") || !strings.Contains(s, "@")) {
		return true // one side names the quantization, the other doesn't
	}
	return strings.HasPrefix(s, a) && snapshotSuffix.MatchString(s[len(a):])
}

func normModel(id string) string {
	id = strings.ToLower(strings.TrimSpace(id))
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	return strings.TrimSuffix(id, ":latest")
}

// baseModel drops an LM Studio quantization tag (x@q8_0 → x).
func baseModel(id string) string {
	if i := strings.Index(id, "@"); i >= 0 {
		return id[:i]
	}
	return id
}
