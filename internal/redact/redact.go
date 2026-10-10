// Package redact masks credentials out of text before it is written down —
// the trace log keeps every command and every tool output, and a token that
// passed through the agent once would otherwise stay in it for good, and go
// wherever the log goes next (a fine-tuning set, a bug report).
//
// It masks the shapes credentials actually come in; a secret with no marker
// at all (a bare random password) is not recognisable and passes.
package redact

import (
	"regexp"
	"strings"
)

// Mask replaces what was a credential.
const Mask = "<REDACTED>"

var (
	privateKey = regexp.MustCompile(`(?s)-----BEGIN ([A-Z ]*)PRIVATE KEY-----.*?-----END ([A-Z ]*)PRIVATE KEY-----`)
	authHeader = regexp.MustCompile(`(?i)(authorization:\s*(?:bearer|basic|token)\s+)[^\s'"]+`)
	urlPass    = regexp.MustCompile(`\b([a-z][a-z0-9+.-]*://[^\s/:@]+:)[^\s/@]+@`)
	prefixed   = regexp.MustCompile(`\b(?:gh[pousr]_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AKIA[A-Z0-9]{12,}|eyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+)`)
	// NAME=value / name: value where the name says what it holds. The value
	// must be six characters or more and not a call — `token := getToken()`
	// and `tokens: 4096` are code and counts, not credentials.
	keyed = regexp.MustCompile(`(?i)\b([\w-]*(?:token|secret|passwo?r?d|passwd|api[_-]?key|credential|access[_-]?key)[\w-]*\s*[=:]\s*)(["']?)([^\s"'(),;{}]{6,})`)
)

// Secrets returns s with every recognisable credential replaced by Mask.
func Secrets(s string) string {
	if s == "" {
		return s
	}
	s = privateKey.ReplaceAllString(s, "-----BEGIN ${1}PRIVATE KEY-----"+Mask+"-----END ${2}PRIVATE KEY-----")
	s = authHeader.ReplaceAllString(s, "${1}"+Mask)
	s = urlPass.ReplaceAllString(s, "${1}"+Mask+"@")
	s = prefixed.ReplaceAllString(s, Mask)
	var b strings.Builder
	last := 0
	for _, m := range keyed.FindAllStringSubmatchIndex(s, -1) {
		valStart, valEnd := m[6], m[7]
		if valEnd < len(s) && s[valEnd] == '(' {
			continue // a call, not a value
		}
		if s[valStart:valEnd] == Mask {
			continue // already masked above
		}
		b.WriteString(s[last:valStart])
		b.WriteString(Mask)
		last = valEnd
	}
	if last == 0 {
		return s
	}
	b.WriteString(s[last:])
	return b.String()
}

// Value masks every string inside v — a trace record's fields, nested maps
// and lists included — and leaves every other value as it is.
func Value(v any) any {
	switch x := v.(type) {
	case string:
		return Secrets(x)
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = Value(e)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = Value(e)
		}
		return out
	case []string:
		out := make([]string, len(x))
		for i, e := range x {
			out[i] = Secrets(e)
		}
		return out
	}
	return v
}
