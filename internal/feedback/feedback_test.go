package feedback

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// The request is what the server's Submission requires: JSON, an
// Idempotency-Key UUID, no unknown fields, an empty author left out.
func TestSubmitSendsTheServersShape(t *testing.T) {
	var got map[string]any
	var key string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key = r.Header.Get("Idempotency-Key")
		json.NewDecoder(r.Body).Decode(&got)
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte(`{"id":"x","status":"pending"}`))
	}))
	defer ts.Close()
	err := Submit(context.Background(), ts.URL, "ipsupport-code/0.61.0",
		Review{Product: "ipsupport-code", Rating: 5, Text: "works with my local model", Version: "0.61.0"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`).MatchString(key) {
		t.Errorf("Idempotency-Key %q is not a UUID", key)
	}
	if _, ok := got["author"]; ok {
		t.Errorf("an anonymous review sent an author: %v", got)
	}
	if got["product"] != "ipsupport-code" || got["rating"] != float64(5) {
		t.Errorf("body = %v", got)
	}
}

func TestSubmitExplainsRefusals(t *testing.T) {
	for code, want := range map[int]string{
		http.StatusTooManyRequests: "too many reviews",
		http.StatusBadRequest:      "invalid_text",
	} {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(code)
			w.Write([]byte(`{"error":"invalid_text"}`))
		}))
		err := Submit(context.Background(), ts.URL, "", Review{Product: "p", Rating: 3, Text: "ok"}, nil)
		ts.Close()
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%d: err = %v, want it to mention %q", code, err, want)
		}
	}
}

func TestValidateBeforeSending(t *testing.T) {
	for _, r := range []Review{
		{Rating: 0, Text: "x"}, {Rating: 6, Text: "x"}, {Rating: 3, Text: "  "},
		{Rating: 3, Text: strings.Repeat("я", 2001)}, {Rating: 3, Text: "x", Author: strings.Repeat("a", 65)},
	} {
		if r.Validate() == nil {
			t.Errorf("accepted %+v", r)
		}
	}
	if (Review{Rating: 5, Text: strings.Repeat("я", 2000)}).Validate() != nil {
		t.Error("2000 characters (not bytes) must pass")
	}
}

func TestPromptPolicy(t *testing.T) {
	now := time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC)
	day := 24 * time.Hour
	for _, tc := range []struct {
		name string
		s    PromptState
		want bool
	}{
		{"never launched", PromptState{}, false},
		{"first day", PromptState{FirstLaunch: now.Add(-time.Hour)}, false},
		{"after a day", PromptState{FirstLaunch: now.Add(-2 * day)}, true},
		{"snoozed", PromptState{FirstLaunch: now.Add(-20 * day), SnoozedUntil: now.Add(day)}, false},
		{"snooze over", PromptState{FirstLaunch: now.Add(-20 * day), SnoozedUntil: now.Add(-time.Minute)}, true},
		{"rated", PromptState{FirstLaunch: now.Add(-20 * day), Reviewed: true}, false},
		{"never", PromptState{FirstLaunch: now.Add(-20 * day), Never: true}, false},
	} {
		if got := tc.s.ShouldPrompt(now); got != tc.want {
			t.Errorf("%s: %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestUpdatePromptPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rate.json")
	now := time.Now()
	UpdatePrompt(path, func(s *PromptState) { s.FirstLaunch = now })
	s, _ := UpdatePrompt(path, func(s *PromptState) { Snooze(s, now) })
	if !s.FirstLaunch.Equal(now) || s.SnoozedUntil.Sub(now) != snooze {
		t.Errorf("state = %+v", s)
	}
}
