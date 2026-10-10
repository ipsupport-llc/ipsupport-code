package telemetry

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// Off means off: no install ID, so nothing is counted — not even the day.
func TestNothingIsRecordedWhileOff(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	if err := Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}}); err != nil {
		t.Fatal(err)
	}
	if s, _ := Load(path); s.InstallID != "" || len(s.Days) != 0 {
		t.Fatalf("recorded while off: %+v", s)
	}
}

func TestRecordCountsOnlyKnownFeatures(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	if err := Enable(path); err != nil {
		t.Fatal(err)
	}
	Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 2, "chat": 5}, Families: []string{"qwen"}})
	Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}, Families: []string{"qwen", "claude"}})
	Record(path, "2026-10-04", Counts{}) // a day the program ran on, with nothing done
	s, _ := Load(path)
	d := s.Days["2026-10-03"]
	if d.Features["tasks"] != 3 || d.Features["chat"] != 0 {
		t.Errorf("features = %v — chat is another product's key", d.Features)
	}
	if !slices.Equal(d.Families, []string{"claude", "qwen"}) {
		t.Errorf("families = %v", d.Families)
	}
	if _, ok := s.Days["2026-10-04"]; !ok {
		t.Error("an empty day was not kept")
	}
}

// Turning it off forgets the install and everything unsent; turning it back
// on starts a new install.
func TestDisableForgetsAndEnableStartsOver(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	first, _ := Load(path)
	Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}})
	if err := Disable(path); err != nil {
		t.Fatal(err)
	}
	if s, _ := Load(path); s.InstallID != "" || len(s.Days) != 0 {
		t.Fatalf("disable kept state: %+v", s)
	}
	Enable(path)
	if second, _ := Load(path); second.InstallID == "" || second.InstallID == first.InstallID {
		t.Errorf("re-enabling must make a new install ID: %q then %q", first.InstallID, second.InstallID)
	}
}

type server struct {
	mu      sync.Mutex
	reports []Report
	status  func(day string) (int, string)
}

func (s *server) handler(w http.ResponseWriter, r *http.Request) {
	var rep Report
	body, _ := io.ReadAll(r.Body)
	json.Unmarshal(body, &rep)
	s.mu.Lock()
	s.reports = append(s.reports, rep)
	s.mu.Unlock()
	code, retry := http.StatusNoContent, ""
	if s.status != nil {
		code, retry = s.status(rep.Day)
	}
	if retry != "" {
		w.Header().Set("Retry-After", retry)
	}
	if code == http.StatusBadRequest { // as ipsupport-api refuses: the field, by code
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		io.WriteString(w, `{"error":"invalid_day"}`)
		return
	}
	w.WriteHeader(code)
}

func sendAt(t *testing.T, path, url string, now time.Time) SendResult {
	t.Helper()
	res, err := Send(context.Background(), SendOptions{
		Path: path, Endpoint: url, Version: "v0.61.0",
		Info: Info{OS: "linux", Arch: "amd64", OSVersion: "6.8.0", MemoryGB: 32, Locale: "de_DE.UTF-8"},
		Now:  func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// Finished days go, oldest first; today waits until it ends; a day too old
// for the server, or one after today (a clock set back), is dropped unsent.
func TestSendFinishedDaysOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	for _, d := range []string{"2026-09-20", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-09"} {
		Record(path, d, Counts{Features: map[string]int{"tasks": 1}, Families: []string{"nemotron"}})
	}
	srv := &server{}
	ts := httptest.NewServer(http.HandlerFunc(srv.handler))
	defer ts.Close()

	res := sendAt(t, path, ts.URL, time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local))
	var days []string
	for _, r := range srv.reports {
		days = append(days, r.Day)
	}
	if !slices.Equal(days, []string{"2026-10-02", "2026-10-03"}) {
		t.Fatalf("sent %v, want the two finished days oldest first", days)
	}
	if res.Sent != 2 || res.Dropped != 2 {
		t.Errorf("result %+v", res)
	}
	s, _ := Load(path)
	if len(s.Days) != 1 || s.Days["2026-10-04"] == nil {
		t.Errorf("left %v, want only today", s.Days)
	}
	r := srv.reports[0]
	if r.Product != Product || r.AppVersion != "0.61.0" || r.OS != "linux" || r.Arch != "amd64" ||
		r.Chip != "other" || r.Features["tasks"] != 1 || !slices.Equal(r.ModelFamilies, []string{"nemotron"}) {
		t.Errorf("report = %+v", r)
	}
}

// A 429 stops the round and waits for Retry-After; a 400 drops that day for
// good; a 5xx keeps the day and waits an hour.
func TestSendAnswers(t *testing.T) {
	now := time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local)
	for _, tc := range []struct {
		name      string
		code      int
		retry     string
		wantLeft  int
		wantAfter time.Duration
	}{
		{"rate limited", http.StatusTooManyRequests, "120", 2, 2 * time.Minute},
		{"server error", http.StatusBadGateway, "", 2, time.Hour},
		{"refused", http.StatusBadRequest, "", 0, 0},
		// Not this server's answer: a proxy wanting a login. The days are
		// not at fault — kept, and tried again later.
		{"proxy", http.StatusProxyAuthRequired, "", 2, time.Hour},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "telemetry.json")
			Enable(path)
			Record(path, "2026-10-02", Counts{})
			Record(path, "2026-10-03", Counts{})
			srv := &server{status: func(string) (int, string) { return tc.code, tc.retry }}
			ts := httptest.NewServer(http.HandlerFunc(srv.handler))
			defer ts.Close()

			sendAt(t, path, ts.URL, now)
			s, _ := Load(path)
			if len(s.Days) != tc.wantLeft {
				t.Errorf("%d day(s) left, want %d", len(s.Days), tc.wantLeft)
			}
			if got := s.NextAttempt.Sub(now); got != tc.wantAfter && !(tc.wantAfter == 0 && s.NextAttempt.IsZero()) {
				t.Errorf("next attempt in %v, want %v", got, tc.wantAfter)
			}
			if tc.wantAfter > 0 {
				before := len(srv.reports)
				sendAt(t, path, ts.URL, now.Add(tc.wantAfter/2))
				if len(srv.reports) != before {
					t.Error("sent again before the wait was over")
				}
			}
		})
	}
}

func TestFamily(t *testing.T) {
	for in, want := range map[string]string{
		"roman220220/Nemotron-3.5-Lightning-30B-A3B": "nemotron",
		"mlx-community/Qwen3-Coder-30B":              "qwen",
		"openai/gpt-oss-20b":                         "gpt-oss",
		"gpt-4o":                                     "gpt",
		"anthropic/claude-sonnet-4":                  "claude",
		"x-ai/grok-4":                                "grok",
		"devstral-small":                             "mistral",
		"phi-4":                                      "phi",
		"kimi-k2":                                    "other",
		"":                                           "",
	} {
		if got := Family(in); got != want {
			t.Errorf("Family(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestIsRelease(t *testing.T) {
	for v, want := range map[string]bool{
		"v0.61.0": true, "nightly-20261002-95af39a": true,
		"dev": false, "v0.61.0-3-gabcdef1-dirty": false, "95af39a": false,
	} {
		if IsRelease(v) != want {
			t.Errorf("IsRelease(%q) = %v", v, !want)
		}
	}
}

// What this machine says about itself must pass the server's own checks.
func TestCollectInfoFitsTheServer(t *testing.T) {
	info := CollectInfo()
	if info.OS != runtime.GOOS || info.Arch != runtime.GOARCH {
		t.Errorf("os/arch = %q/%q", info.OS, info.Arch)
	}
	if info.OSVersion != "" && !regexp.MustCompile(`^\d{1,5}(\.\d{1,5}){0,3}$`).MatchString(info.OSVersion) {
		t.Errorf("os_version %q would be refused", info.OSVersion)
	}
	if runtime.GOOS == "linux" && (info.MemoryGB <= 0 || info.OSVersion == "") {
		t.Errorf("linux info incomplete: %+v", info)
	}
	t.Logf("%+v", info)
}

// The OS reports a little less memory than is installed; the report names
// the size the machine was sold with, or the server counts it as "other".
func TestGBSnapsToShippedSizes(t *testing.T) {
	const gib = 1 << 30
	for in, want := range map[float64]int{
		31.2: 32, 15.5: 16, 7.6: 8, 16: 16, 36: 36, 62.7: 64, 20: 20, 1000: 1000,
	} {
		if got := gb(uint64(in * gib)); got != want {
			t.Errorf("gb(%.1f GiB) = %d, want %d", in, got, want)
		}
	}
}

// Turning reporting off while a batch is going out stops the rest of it —
// whether the session closed its gate or another session deleted the state.
func TestAnOptOutMidBatchStopsTheRest(t *testing.T) {
	for _, how := range []string{"gate", "state deleted"} {
		t.Run(how, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "telemetry.json")
			Enable(path)
			for _, d := range []string{"2026-10-01", "2026-10-02", "2026-10-03"} {
				Record(path, d, Counts{})
			}
			open := true
			srv := &server{status: func(string) (int, string) {
				if how == "gate" {
					open = false
				} else {
					Disable(path)
				}
				return http.StatusNoContent, ""
			}}
			ts := httptest.NewServer(http.HandlerFunc(srv.handler))
			defer ts.Close()
			Send(context.Background(), SendOptions{Path: path, Endpoint: ts.URL, Version: "v0.61.0",
				Now:  func() time.Time { return time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local) },
				Keep: func() bool { return open }})
			if len(srv.reports) != 1 {
				t.Errorf("sent %d report(s) after the opt-out, want just the one in flight", len(srv.reports))
			}
		})
	}
}

// A count recorded into a day while that day's report is in flight must not be
// deleted with it: the day stays, and its next send carries the whole day.
func TestARecordDuringTheSendIsKept(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}})
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()
	sendAt(t, path, ts.URL, time.Date(2026, 10, 4, 0, 0, 1, 0, time.Local))
	s, _ := Load(path)
	if d := s.Days["2026-10-03"]; d == nil || d.Features["tasks"] != 2 {
		t.Fatalf("day after the send = %+v, want it kept with both tasks", d)
	}
}

// The state remembers how the last send went, so /telemetry can say whether
// reports actually leave — accepted, refused (and why), or failed.
func TestSendRemembersTheLastAttempt(t *testing.T) {
	now := time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local)
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	Record(path, "2026-10-02", Counts{})
	Record(path, "2026-10-03", Counts{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var rep Report
		json.NewDecoder(r.Body).Decode(&rep)
		if rep.Day == "2026-10-02" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":"invalid_day"}`)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	sendAt(t, path, ts.URL, now)
	ts.Close()
	s, _ := Load(path)
	if s.Last == nil || !s.Last.Time.Equal(now) || s.Last.Sent != 1 || len(s.Last.Refused) != 1 || s.Last.Refused[0] != "2026-10-02: invalid_day" {
		t.Fatalf("last = %+v, want 1 sent and 2026-10-02 refused as invalid_day at %v", s.Last, now)
	}

	// A send that cannot reach the server says so, and keeps the day.
	Record(path, "2026-10-03", Counts{Features: map[string]int{"tasks": 1}})
	sendAt(t, path, ts.URL, now.Add(2*time.Hour)) // ts is closed: connection refused
	s, _ = Load(path)
	if s.Last == nil || s.Last.Error == "" || s.Last.Sent != 0 || s.Days["2026-10-03"] == nil {
		t.Fatalf("last = %+v, days %v; want the failure recorded and the day kept", s.Last, s.Days)
	}
}

// Ctrl-C during the launch send is not a failure: no "failed" status, and
// nothing holds back the next launch.
func TestACancelledSendHoldsNothingBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	Record(path, "2026-10-03", Counts{})
	ctx, cancel := context.WithCancel(context.Background())
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cancel() // Ctrl-C while the report is in flight
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusNoContent) // too late: the client has given up
	}))
	defer ts.Close()
	_, err := Send(ctx, SendOptions{Path: path, Endpoint: ts.URL, Version: "v0.61.0",
		Now: func() time.Time { return time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local) }})
	if err != nil {
		t.Fatal(err)
	}
	s, _ := Load(path)
	if !s.NextAttempt.IsZero() || s.Last != nil || s.Days["2026-10-03"] == nil {
		t.Fatalf("next %v, last %+v, days %v; want nothing held back, no status, the day kept", s.NextAttempt, s.Last, s.Days)
	}
}

// Days dropped for age leave a trace: they are the data a long gap loses.
func TestExpiredDaysAreRecorded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "telemetry.json")
	Enable(path)
	Record(path, "2026-09-20", Counts{})
	sendAt(t, path, "http://127.0.0.1:1", time.Date(2026, 10, 4, 12, 0, 0, 0, time.Local))
	s, _ := Load(path)
	if s.Last == nil || len(s.Last.Expired) != 1 || s.Last.Expired[0] != "2026-09-20" {
		t.Fatalf("last = %+v, want 2026-09-20 recorded as expired", s.Last)
	}
}

// A server's error text reaches the terminal only as a plain code.
func TestErrorCodeKeepsOnlyACode(t *testing.T) {
	for in, want := range map[string]string{
		"invalid_day": "invalid_day", "\x1b]0;pwned\x07\nfake": "", "Not Found": "",
		strings.Repeat("a", 41): "",
	} {
		if got := errorCode(in); got != want {
			t.Errorf("errorCode(%q) = %q, want %q", in, got, want)
		}
	}
}
