// Package feedback sends a rating of this program to ipsupport.us and decides
// when to suggest one (adr/0016). The server's contract is ipsupport-api's
// openapi.yaml (Submission) and its ADRs 4 and 6: a review is moderated
// before it is published, and a POST must carry an Idempotency-Key.
package feedback

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
)

// Review is one rating as submitted.
type Review struct {
	Product string `json:"product"`
	Rating  int    `json:"rating"`
	Text    string `json:"text"`
	Version string `json:"version,omitempty"`
	Author  string `json:"author,omitempty"` // empty: anonymous
}

// Validate applies the server's limits before anything is sent, so a typo
// gets a clear message instead of a round trip and an error code.
func (r Review) Validate() error {
	switch {
	case r.Rating < 1 || r.Rating > 5:
		return errors.New("the rating is 1 to 5 stars")
	case strings.TrimSpace(r.Text) == "":
		return errors.New("say a few words — the review text can't be empty")
	case utf8.RuneCountInString(r.Text) > 2000:
		return errors.New("the review is longer than 2000 characters")
	case utf8.RuneCountInString(r.Author) > 64:
		return errors.New("the name is longer than 64 characters")
	}
	return nil
}

// Submit sends a review. It is stored as pending and appears once it has
// been moderated.
func Submit(ctx context.Context, endpoint, userAgent string, r Review, client *http.Client) error {
	if err := r.Validate(); err != nil {
		return err
	}
	body, err := json.Marshal(r)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Idempotency-Key", newUUID())
	if userAgent != "" {
		req.Header.Set("User-Agent", userAgent)
	}
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("couldn't reach ipsupport.us: %w", err)
	}
	defer resp.Body.Close()
	var e struct {
		Error string `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&e)
	switch {
	case resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusOK:
		return nil
	case resp.StatusCode == http.StatusTooManyRequests:
		return errors.New("too many reviews from this address — try again later")
	case e.Error != "":
		return fmt.Errorf("the server refused it (%s)", e.Error)
	}
	return fmt.Errorf("the server answered %s", resp.Status)
}

// Prompt policy, as LLMTray's ReviewPromptPolicy: not on the first day, not
// before the program has done something, never again once rated or told
// never, and at most every two weeks.
const (
	minimumAge = 24 * time.Hour
	snooze     = 14 * 24 * time.Hour
)

// PromptState is what the policy remembers (rate.json in the config dir).
type PromptState struct {
	FirstLaunch  time.Time `json:"first_launch"`
	SnoozedUntil time.Time `json:"snoozed_until,omitempty"`
	Reviewed     bool      `json:"reviewed,omitempty"`
	Never        bool      `json:"never,omitempty"`
}

// ShouldPrompt reports whether to suggest a rating after a finished task.
func (s PromptState) ShouldPrompt(now time.Time) bool {
	if s.Reviewed || s.Never || s.FirstLaunch.IsZero() {
		return false
	}
	return now.Sub(s.FirstLaunch) >= minimumAge && !now.Before(s.SnoozedUntil)
}

// UpdatePrompt changes the stored policy state under a lock: several
// sessions share it.
func UpdatePrompt(path string, f func(*PromptState)) (PromptState, error) {
	unlock, err := filelock.Lock(path)
	if err != nil {
		return PromptState{}, err
	}
	defer unlock()
	var s PromptState
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &s) // a damaged file starts over; it holds nothing that matters
	} else if !errors.Is(err, fs.ErrNotExist) {
		return s, err
	}
	before := s
	f(&s)
	if s == before {
		return s, nil
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return s, err
	}
	return s, atomicfile.Write(path, data, 0o600)
}

// Snooze puts the suggestion off for two weeks.
func Snooze(s *PromptState, now time.Time) { s.SnoozedUntil = now.Add(snooze) }

func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
