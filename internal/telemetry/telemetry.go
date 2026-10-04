// Package telemetry keeps the anonymous daily usage counters and sends them
// to ipsupport.us (adr/0016). What a report may contain is fixed by the
// server's spec — ipsupport-api docs/telemetry.md, its ADRs 9, 10 and 12 —
// and the shape here mirrors it: counts of features and the coarse families
// of models used, never prompts, code, commands, paths or model names.
//
// The state is one small file shared by every session on the machine
// (telemetry.json in the config directory): the random install ID and the
// counters per local day. It is read and written under a file lock, the same
// way the usage ledger is.
package telemetry

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
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
)

// Product is this program's name in the reports.
const Product = "ipsupport-code"

// DefaultAPIBase is where reports and reviews go. IPS_API_BASE overrides it,
// which is how a development build is pointed at a local server instead.
const DefaultAPIBase = "https://ipsupport.us/api"

// Features this program counts — the server's allowlist for the product.
var Features = []string{
	"tasks", "tool_calls", "goals", "subagents", "external_agents",
	"mcp", "skills", "plan_mode", "local_model", "cloud_model",
}

const (
	// maxDays is how far back a day may still be sent; the server refuses
	// older ones, so they are dropped unsent.
	maxDays = 7
	// retryAfter is the wait after a timeout, a server error or no answer.
	retryAfter = time.Hour
)

// State is the file's content.
type State struct {
	InstallID string          `json:"install_id"`
	Days      map[string]*Day `json:"days,omitempty"`
	// NextAttempt holds sends back after a 429 or a failed attempt.
	NextAttempt time.Time `json:"next_attempt,omitempty"`
}

// Day is one local day's counters.
type Day struct {
	Features map[string]int `json:"features,omitempty"`
	Families []string       `json:"families,omitempty"`
}

// Counts is what one recording adds to a day.
type Counts struct {
	Features map[string]int
	Families []string
}

// Enable makes sure the state exists with an install ID: a new random one
// when there is none — which is also what turning telemetry back on means,
// since Disable removes the file.
func Enable(path string) error {
	return update(path, func(s *State) bool {
		if s.InstallID != "" {
			return false
		}
		s.InstallID = newUUID()
		return true
	})
}

// Disable deletes the install ID and every unsent counter.
func Disable(path string) error {
	unlock, err := filelock.Lock(path)
	if err != nil {
		return err
	}
	defer unlock()
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

// ResetID replaces the install ID, keeping the counters.
func ResetID(path string) error {
	return update(path, func(s *State) bool {
		if s.InstallID == "" {
			return false
		}
		s.InstallID = newUUID()
		return true
	})
}

// Load reads the state; a missing file is an empty state.
func Load(path string) (State, error) {
	var s State
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return s, err
	}
	if err := json.Unmarshal(data, &s); err != nil {
		return State{}, fmt.Errorf("%s: %w", path, err)
	}
	return s, nil
}

// Record adds c to day (a local date, yyyy-mm-dd). The day is created even
// with nothing to add: a day the program ran on is reported. Nothing is
// recorded while telemetry is off (no install ID).
func Record(path, day string, c Counts) error {
	return update(path, func(s *State) bool {
		if s.InstallID == "" {
			return false
		}
		if s.Days == nil {
			s.Days = map[string]*Day{}
		}
		d := s.Days[day]
		if d == nil {
			d = &Day{}
			s.Days[day] = d
		}
		for k, n := range c.Features {
			if n <= 0 || !slices.Contains(Features, k) {
				continue
			}
			if d.Features == nil {
				d.Features = map[string]int{}
			}
			d.Features[k] += n
		}
		for _, f := range c.Families {
			if f != "" && !slices.Contains(d.Families, f) {
				d.Families = append(d.Families, f)
			}
		}
		sort.Strings(d.Families)
		return true
	})
}

// update runs f on the state under the file lock and writes it back when f
// reports a change.
func update(path string, f func(*State) bool) error {
	unlock, err := filelock.Lock(path)
	if err != nil {
		return err
	}
	defer unlock()
	s, err := Load(path)
	if err != nil {
		return err
	}
	if !f(&s) {
		return nil
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, data, 0o600)
}

// Info describes the machine, as far as a report says anything about it.
type Info struct {
	OS        string // runtime.GOOS
	Arch      string // runtime.GOARCH
	OSVersion string // macOS 15.1.1, a Linux kernel's 6.8.0, Windows 10.0.26100
	Chip      string // "Apple M3 Pro" on Apple silicon, else ""
	MemoryGB  int
	Locale    string // e.g. "de_DE.UTF-8"; the server keeps the language only
}

// Report is one day as sent — the server's TelemetryReport.
type Report struct {
	Product       string         `json:"product"`
	InstallID     string         `json:"install_id"`
	Day           string         `json:"day"`
	AppVersion    string         `json:"app_version"`
	OSVersion     string         `json:"os_version,omitempty"`
	Chip          string         `json:"chip"`
	MemoryGB      int            `json:"memory_gb,omitempty"`
	Locale        string         `json:"locale,omitempty"`
	Features      map[string]int `json:"features"`
	ModelFamilies []string       `json:"model_families"`
	OS            string         `json:"os"`
	Arch          string         `json:"arch"`
}

// BuildReport is the report for one day of s.
func BuildReport(s State, day string, version string, info Info) Report {
	r := Report{
		Product: Product, InstallID: s.InstallID, Day: day, AppVersion: AppVersion(version),
		OSVersion: info.OSVersion, Chip: info.Chip, MemoryGB: info.MemoryGB, Locale: info.Locale,
		Features: map[string]int{}, ModelFamilies: []string{}, OS: info.OS, Arch: info.Arch,
	}
	if r.Chip == "" {
		r.Chip = "other"
	}
	if d := s.Days[day]; d != nil {
		for k, n := range d.Features {
			r.Features[k] = n
		}
		r.ModelFamilies = append(r.ModelFamilies, d.Families...)
	}
	return r
}

var releaseVersion = regexp.MustCompile(`^(v\d+\.\d+\.\d+|nightly-\d{8}-[0-9a-f]+)$`)

// IsRelease reports whether version is a published build. Anything else — a
// local `make build`, `dev` — sends nothing to ipsupport.us: a developer's
// runs are not usage.
func IsRelease(version string) bool { return releaseVersion.MatchString(version) }

// AppVersion is the version as reported: "0.61.0" rather than "v0.61.0".
func AppVersion(version string) string { return strings.TrimPrefix(version, "v") }

// SendOptions configure Send.
type SendOptions struct {
	Path     string
	Endpoint string // the telemetry URL
	Version  string
	Info     Info
	Client   *http.Client
	Now      func() time.Time
	// UserAgent identifies the program, not the system.
	UserAgent string
	// Keep, when set, is asked before each day is posted; false stops the
	// round (the session went offline or turned reporting off mid-batch).
	Keep func() bool
}

// SendResult says what one Send did.
type SendResult struct {
	Sent, Dropped int
	Deferred      bool // stopped early: rate limited, a server error or no answer
}

// Send reports every finished day, oldest first. Today waits until it ends,
// so a user who turns telemetry off on the first day has sent nothing.
//
// The answers mirror LLMTray's (its ADR 15): 204 sent; 429 waits for
// Retry-After; 408, 5xx and no answer try again after an hour; any other
// 4xx drops that day. A resend of a day replaces it on the server, so a
// retry needs no idempotency key. The lock is not held across the network:
// another session must not wait on it.
func Send(ctx context.Context, o SendOptions) (SendResult, error) {
	var res SendResult
	now := o.Now
	if now == nil {
		now = time.Now
	}
	t := now()
	today := t.Format(time.DateOnly)
	// Seven days back counted from the UTC day, as the server does: a machine
	// behind UTC would otherwise send a day it refuses.
	oldest := t.UTC().AddDate(0, 0, -maxDays).Format(time.DateOnly)

	var s State
	var due []string
	err := update(o.Path, func(st *State) bool {
		s = *st
		if st.InstallID == "" || t.Before(st.NextAttempt) {
			return false
		}
		changed := false
		for day := range st.Days {
			switch {
			case day < oldest || day > today:
				delete(st.Days, day) // too old for the server, or a clock set back
				res.Dropped++
				changed = true
			case day < today:
				due = append(due, day)
			}
		}
		sort.Strings(due)
		s = *st
		return changed
	})
	if err != nil || len(due) == 0 {
		return res, err
	}

	done := map[string]bool{}
	next := time.Time{}
	for _, day := range due {
		// Consent is checked again before every day, not once per round: the
		// session may have gone offline, or the user turned reporting off —
		// here or in another session, which deletes the state.
		if o.Keep != nil && !o.Keep() {
			return res, nil
		}
		if cur, err := Load(o.Path); err != nil || cur.InstallID != s.InstallID {
			return res, nil
		}
		code, wait, err := post(ctx, o, BuildReport(s, day, o.Version, o.Info))
		switch {
		case err != nil || code == http.StatusRequestTimeout || code >= 500:
			next = t.Add(retryAfter)
		case code == http.StatusTooManyRequests:
			next = t.Add(wait)
		case code == http.StatusNoContent || (code >= 200 && code < 300):
			done[day] = true
			res.Sent++
			continue
		default: // any other 4xx: this day will never be accepted
			done[day] = true
			res.Dropped++
			continue
		}
		res.Deferred = true
		break
	}
	err = update(o.Path, func(st *State) bool {
		if st.InstallID != s.InstallID { // turned off and on, or reset, meanwhile
			return false
		}
		for day := range done {
			delete(st.Days, day)
		}
		st.NextAttempt = next
		return true
	})
	return res, err
}

func post(ctx context.Context, o SendOptions, r Report) (code int, wait time.Duration, err error) {
	body, err := json.Marshal(r)
	if err != nil {
		return 0, 0, err
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.Endpoint, bytes.NewReader(body))
	if err != nil {
		return 0, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if o.UserAgent != "" {
		req.Header.Set("User-Agent", o.UserAgent)
	}
	client := o.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	wait = retryAfter
	if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
		wait = time.Duration(s) * time.Second
	}
	return resp.StatusCode, wait, nil
}

// familyMarks map a model name onto the server's families. Order matters:
// "gpt-oss" before "gpt", "mixtral" and friends are Mistral's.
var familyMarks = []struct{ mark, family string }{
	{"nemotron", "nemotron"}, {"gpt-oss", "gpt-oss"},
	{"qwen", "qwen"}, {"llama", "llama"}, {"gemma", "gemma"},
	{"mistral", "mistral"}, {"mixtral", "mistral"}, {"devstral", "mistral"},
	{"codestral", "mistral"}, {"magistral", "mistral"},
	{"deepseek", "deepseek"}, {"glm", "glm"}, {"phi-", "phi"},
	{"claude", "claude"}, {"grok", "grok"}, {"gemini", "gemini"}, {"gpt", "gpt"},
}

// Family is the coarse family of a model, or "other". Only the last path
// component is looked at ("mlx-community/Qwen3-30B" -> "qwen30b…"), and only
// the family leaves the machine — never the name.
func Family(model string) string {
	m := strings.ToLower(strings.TrimSpace(model))
	if i := strings.LastIndex(m, "/"); i >= 0 {
		m = m[i+1:]
	}
	if m == "" {
		return ""
	}
	for _, f := range familyMarks {
		if strings.Contains(m, f.mark) {
			return f.family
		}
	}
	if strings.HasPrefix(m, "phi") {
		return "phi"
	}
	return "other"
}

// newUUID is a random (version 4) UUID.
func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
