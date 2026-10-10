package redact

import (
	"strings"
	"testing"
)

func TestSecretsAreMasked(t *testing.T) {
	for _, c := range []struct{ in, gone, kept string }{
		{"export GITHUB_TOKEN=ghp_abcdefghijklmnop1234", "ghp_abcdefghijklmnop1234", "GITHUB_TOKEN="},
		{"PASSWORD=hunter22 ./run.sh", "hunter22", "PASSWORD=<REDACTED> ./run.sh"},
		{`curl -H "Authorization: Bearer eyJhbGciOi.eyJzdWIiOiIx.c2lnbmF0dXJl" https://api.example.com`, "eyJhbGciOi", "https://api.example.com"},
		{"psql postgres://app:s3cr3tpass@db.internal:5432/app", "s3cr3tpass", "postgres://app:<REDACTED>@db.internal"},
		{"aws configure set aws_access_key_id AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"},
		{"OPENAI_API_KEY=sk-proj-abcdefgh12345678", "sk-proj-abcdefgh12345678", "OPENAI_API_KEY="},
		{"slack: xoxb-1234-5678-abcdefgh", "xoxb-1234-5678-abcdefgh", "slack:"},
		{`api_key: "q9w8e7r6t5y4"`, "q9w8e7r6t5y4", "api_key:"},
		{"-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjEAAAAA", "PRIVATE KEY"},
		{"curl -H 'X-Api-Key: 7f3c9e1a2b4d' https://x", "7f3c9e1a2b4d", "X-Api-Key:"},
	} {
		got := Secrets(c.in)
		if strings.Contains(got, c.gone) || !strings.Contains(got, c.kept) || !strings.Contains(got, Mask) {
			t.Errorf("Secrets(%q) = %q; want %q gone, %q kept", c.in, got, c.gone, c.kept)
		}
	}
}

// Ordinary text and code stay as they are: the log is still for reading and
// for training, and a mask over `token := getToken()` would only corrupt it.
func TestOrdinaryTextIsKept(t *testing.T) {
	for _, s := range []string{
		"go test ./... && git push origin main",
		"token := getToken(ctx)",
		"if password == \"\" { return errEmpty }",
		"curl -s https://api.github.com/repos/o/r",
		"the token budget is 8192",
		"tokens: 4096",
	} {
		if got := Secrets(s); got != s {
			t.Errorf("Secrets(%q) = %q; want it unchanged", s, got)
		}
	}
}

func TestValueWalksNestedRecords(t *testing.T) {
	v := Value(map[string]any{
		"params": map[string]any{"command": "export TOKEN=abcdef123456"},
		"list":   []any{"PASSWORD=hunter22", 7},
	}).(map[string]any)
	if s := v["params"].(map[string]any)["command"].(string); strings.Contains(s, "abcdef123456") {
		t.Errorf("nested map not masked: %q", s)
	}
	if s := v["list"].([]any)[0].(string); strings.Contains(s, "hunter22") {
		t.Errorf("nested list not masked: %q", s)
	}
	if v["list"].([]any)[1] != 7 {
		t.Error("a non-string value changed")
	}
}
