package llm

import (
	"regexp"
	"strings"
)

// aliasSuffix is the tail that tells an alias from the snapshot it points at:
// a date (gpt-4o-2024-08-06, claude-3-5-sonnet-20241022, grok-4-0709) or
// "-latest". Hosted providers often answer an alias with the snapshot's id.
var aliasSuffix = regexp.MustCompile(`-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}|latest)$`)

// SameModel reports whether served — the model a server says answered — is the
// model that was asked for. It tolerates differences that don't change the
// model: letter case, a publisher prefix only one side names (google/x vs x),
// a quantization tag only one side names (x@q8_0 vs x), Ollama's :latest, and
// an alias against its dated snapshot. An empty served (the server didn't say)
// counts as the same.
func SameModel(asked, served string) bool {
	a, s := strings.ToLower(strings.TrimSpace(asked)), strings.ToLower(strings.TrimSpace(served))
	if s == "" || a == s {
		return true
	}
	if !strings.Contains(a, "/") || !strings.Contains(s, "/") {
		a, s = dropPublisher(a), dropPublisher(s) // two named publishers must agree
	}
	a, s = strings.TrimSuffix(a, ":latest"), strings.TrimSuffix(s, ":latest")
	if a == s {
		return true
	}
	if ba, bs := baseModel(a), baseModel(s); ba == bs && (ba == a || bs == s) {
		return true // one side names the quantization, the other doesn't
	}
	return aliasSuffix.ReplaceAllString(a, "") == aliasSuffix.ReplaceAllString(s, "")
}

func dropPublisher(id string) string {
	if i := strings.LastIndex(id, "/"); i >= 0 {
		return id[i+1:]
	}
	return id
}

// baseModel drops an LM Studio quantization tag (x@q8_0 → x).
func baseModel(id string) string {
	if i := strings.Index(id, "@"); i >= 0 {
		return id[:i]
	}
	return id
}
