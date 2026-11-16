package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// TestKeyHelperProcess is the api_key_cmd the tests run: the test binary
// itself, so it works on every OS. It counts its runs in KEY_HELPER_COUNT and
// prints key-<n>, or fails/stays silent/sleeps as KEY_HELPER_MODE says.
func TestKeyHelperProcess(t *testing.T) {
	if os.Getenv("KEY_HELPER") != "1" {
		return
	}
	f := os.Getenv("KEY_HELPER_COUNT")
	b, _ := os.ReadFile(f)
	n, _ := strconv.Atoi(strings.TrimSpace(string(b)))
	n++
	os.WriteFile(f, []byte(strconv.Itoa(n)), 0o600)
	switch os.Getenv("KEY_HELPER_MODE") {
	case "fail":
		fmt.Fprintln(os.Stderr, "sign-in required")
		os.Exit(3)
	case "empty":
		os.Exit(0)
	case "slow":
		time.Sleep(300 * time.Millisecond)
	case "hang":
		time.Sleep(3 * time.Second)
	case "leak":
		fmt.Printf("secret-%d\n", n)
		fmt.Fprintf(os.Stderr, "rejected secret-%d\n", n)
		os.Exit(1)
	case "slowfirst": // the first run ends after the second
		if n == 1 {
			time.Sleep(600 * time.Millisecond)
		} else {
			time.Sleep(100 * time.Millisecond)
		}
	case "talk":
		fmt.Fprintln(os.Stderr, "open https://sso.example/device and enter ABCD")
		time.Sleep(300 * time.Millisecond)
	case "orphan":
		// Leave a child holding stdout, as a helper that starts a background
		// refresher would.
		c := exec.Command(os.Args[0], os.Args[1:]...)
		c.Env = append(os.Environ(), "KEY_HELPER_MODE=hang", "KEY_HELPER_COUNT="+f+".child")
		c.Stdout = os.Stdout
		c.Start()
	}
	fmt.Printf("  key-%d\n", n)
	os.Exit(0)
}

// keyHelper sets up the helper and returns its argv and a run counter.
func keyHelper(t *testing.T, mode string) ([]string, func() int) {
	t.Helper()
	count := filepath.Join(t.TempDir(), "count")
	t.Setenv("KEY_HELPER", "1")
	t.Setenv("KEY_HELPER_COUNT", count)
	t.Setenv("KEY_HELPER_MODE", mode)
	// A unique arg keeps each test's command apart in the shared keyCmds cache.
	argv := []string{os.Args[0], "-test.run=^TestKeyHelperProcess$", "-test.count=1", "--", t.Name()}
	return argv, func() int {
		b, _ := os.ReadFile(count)
		n, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		return n
	}
}

func TestKeyCmdRunsOnceAndKeepsTheKey(t *testing.T) {
	argv, runs := keyHelper(t, "")
	l := config.LLM{APIKey: "stale-from-env", APIKeyCmd: argv}
	for i := 0; i < 3; i++ {
		k, err := APIKey(context.Background(), l)
		if err != nil || k != "key-1" {
			t.Fatalf("APIKey = %q, %v; want key-1 (trimmed, and over api_key)", k, err)
		}
	}
	if runs() != 1 {
		t.Errorf("command ran %d times, want 1", runs())
	}
	if k, _ := APIKey(context.Background(), config.LLM{APIKey: "plain"}); k != "plain" {
		t.Errorf("no command: APIKey = %q, want the configured key", k)
	}
}

func TestKeyCmdRunsAgainAfterTheTTL(t *testing.T) {
	argv, runs := keyHelper(t, "")
	l := config.LLM{APIKeyCmd: argv, APIKeyTTL: 1}
	k := keyCommand(l)
	if key, _ := APIKey(context.Background(), l); key != "key-1" {
		t.Fatalf("first key %q", key)
	}
	k.mu.Lock()
	k.at = k.at.Add(-2 * time.Second) // the key is now older than its ttl
	k.mu.Unlock()
	if key, _ := APIKey(context.Background(), l); key != "key-2" || runs() != 2 {
		t.Errorf("after the ttl: key %q, runs %d; want key-2, 2", key, runs())
	}
}

func TestKeyCmdSharesOneRunBetweenCallers(t *testing.T) {
	argv, runs := keyHelper(t, "slow")
	l := config.LLM{APIKeyCmd: argv}
	var wg sync.WaitGroup
	keys := make([]string, 5)
	for i := range keys {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			keys[i], _ = APIKey(context.Background(), l)
		}(i)
	}
	wg.Wait()
	for _, k := range keys {
		if k != "key-1" {
			t.Errorf("keys %v, want all key-1", keys)
			break
		}
	}
	if runs() != 1 {
		t.Errorf("command ran %d times for concurrent callers, want 1", runs())
	}
}

// A caller that gives up stops waiting; the run goes on and its key is kept.
func TestKeyCmdOutlivesACallerThatGivesUp(t *testing.T) {
	argv, runs := keyHelper(t, "slow")
	l := config.LLM{APIKeyCmd: argv}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := APIKey(ctx, l); err == nil {
		t.Fatal("a caller past its deadline got a key")
	}
	if k, err := APIKey(context.Background(), l); k != "key-1" || err != nil || runs() != 1 {
		t.Errorf("then: %q, %v, runs %d; want key-1 from the same run", k, err, runs())
	}
}

func TestKeyCmdFailuresSayWhy(t *testing.T) {
	argv, _ := keyHelper(t, "fail")
	_, err := APIKey(context.Background(), config.LLM{APIKeyCmd: argv})
	if err == nil || !strings.Contains(err.Error(), "sign-in required") {
		t.Errorf("err = %v, want the command's stderr", err)
	}
	argv, _ = keyHelper(t, "empty")
	argv = append(argv, "empty") // a different command, not the cached failure above
	_, err = APIKey(context.Background(), config.LLM{APIKeyCmd: argv})
	if err == nil || !strings.Contains(err.Error(), "printed no key") {
		t.Errorf("err = %v, want 'printed no key'", err)
	}
}

// The chat client sends the command's key; a 401 gets a fresh key and one
// retry, and a second 401 is reported.
func TestChatRefreshesARejectedKeyOnce(t *testing.T) {
	argv, runs := keyHelper(t, "")
	var mu sync.Mutex
	var seen []string
	accept := "key-2"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		got := r.Header.Get("Authorization")
		seen = append(seen, got)
		ok := got == "Bearer "+accept
		mu.Unlock()
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":{"message":"Expired Key"}}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKey: "ignored", APIKeyCmd: argv})
	msg, err := cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err != nil || msg.Content != "ok" {
		t.Fatalf("Chat = %q, %v; want ok after one refresh", msg.Content, err)
	}
	if strings.Join(seen, ",") != "Bearer key-1,Bearer key-2" || runs() != 2 {
		t.Errorf("sent %v with %d runs; want key-1 then key-2", seen, runs())
	}

	mu.Lock()
	accept, seen = "never", nil
	mu.Unlock()
	_, err = cl.Chat(context.Background(), []Message{User("hi")}, nil)
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("err = %v, want the 401 reported", err)
	}
	if len(seen) != 2 {
		t.Errorf("sent %d requests on a key that stays refused, want 2 (one retry)", len(seen))
	}

	// Without a command a 401 is reported straight away, as before.
	mu.Lock()
	seen = nil
	mu.Unlock()
	plain := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKey: "k"})
	_, err = plain.Chat(context.Background(), []Message{User("hi")}, nil)
	mu.Lock()
	defer mu.Unlock()
	if err == nil || len(seen) != 1 {
		t.Errorf("plain key: err %v after %d requests, want a 401 after 1", err, len(seen))
	}
}

// A slow command is not the server going quiet: the idle watchdog starts
// after the key is in hand.
func TestASlowKeyCmdIsNotAStall(t *testing.T) {
	argv, _ := keyHelper(t, "slow")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()
	cl := NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKeyCmd: argv, RetryAttempts: 1})
	cl.idle = 100 * time.Millisecond
	if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
		t.Errorf("Chat: %v — the key's wait counted against the idle timeout", err)
	}
}

// Two connections naming the same command share it, whatever their ttl.
func TestKeyCmdIsSharedAcrossTTLs(t *testing.T) {
	argv, runs := keyHelper(t, "")
	a, _ := APIKey(context.Background(), config.LLM{APIKeyCmd: argv})
	b, _ := APIKey(context.Background(), config.LLM{APIKeyCmd: argv, APIKeyTTL: 300})
	if a != "key-1" || b != "key-1" || runs() != 1 {
		t.Errorf("keys %q %q after %d runs; want one shared key-1", a, b, runs())
	}
}

// A failed run stands for a moment, so a declined sign-in isn't reopened by
// every probe and request behind it; then it is tried again.
func TestAFailedKeyCmdIsNotRerunAtOnce(t *testing.T) {
	argv, runs := keyHelper(t, "fail")
	l := config.LLM{APIKeyCmd: argv}
	for i := 0; i < 3; i++ {
		if _, err := APIKey(context.Background(), l); err == nil {
			t.Fatal("a failing command gave a key")
		}
	}
	if runs() != 1 {
		t.Errorf("ran %d times in a row, want 1", runs())
	}
	k := keyCommand(l)
	k.mu.Lock()
	k.failAt = k.failAt.Add(-keyCmdRetryAfter)
	k.mu.Unlock()
	APIKey(context.Background(), l)
	if runs() != 2 {
		t.Errorf("after the pause: %d runs, want 2", runs())
	}
}

// The key never rides out in an error: not in a failing helper's stderr, and
// not in a server's reply quoting it.
func TestTheKeyStaysOutOfErrors(t *testing.T) {
	argv, _ := keyHelper(t, "leak")
	_, err := APIKey(context.Background(), config.LLM{APIKeyCmd: argv})
	if err == nil || strings.Contains(err.Error(), "secret-1") || !strings.Contains(err.Error(), "rejected [api key]") {
		t.Errorf("helper error = %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintf(w, `{"error":"key %s not allowed"}`, strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	}))
	defer srv.Close()
	_, err = NewOpenAIClient(config.LLM{BaseURL: srv.URL, Model: "m", APIKey: "sk-opaque-123"}).Chat(context.Background(), []Message{User("hi")}, nil)
	if err == nil || strings.Contains(err.Error(), "sk-opaque-123") {
		t.Errorf("chat error = %v", err)
	}
	_, err = ListModels(context.Background(), srv.URL, "sk-opaque-123", srv.Client())
	if err == nil || strings.Contains(err.Error(), "sk-opaque-123") {
		t.Errorf("list error = %v", err)
	}
}

// Listing models refreshes a refused key once, like a chat request.
func TestListModelsForRefreshesARejectedKey(t *testing.T) {
	argv, runs := keyHelper(t, "")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer key-2" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
	}))
	defer srv.Close()
	ids, err := ListModelsFor(context.Background(), config.LLM{BaseURL: srv.URL, APIKeyCmd: argv}, srv.Client())
	if err != nil || len(ids) != 1 || runs() != 2 {
		t.Errorf("ids %v, err %v, runs %d; want [m] after one refresh", ids, err, runs())
	}
}

// A helper that leaves a child holding its stdout still hands over its key,
// once the wait for the pipes runs out.
func TestAKeyCmdThatLeavesAChildBehindStillWorks(t *testing.T) {
	defer func(d time.Duration) { keyCmdWaitDelay = d }(keyCmdWaitDelay)
	keyCmdWaitDelay = 200 * time.Millisecond
	argv, _ := keyHelper(t, "orphan")
	start := time.Now()
	k, err := APIKey(context.Background(), config.LLM{APIKeyCmd: argv})
	if err != nil || k != "key-1" {
		t.Errorf("key %q, err %v", k, err)
	}
	if d := time.Since(start); d > 20*time.Second {
		t.Errorf("took %s — the child's open pipe held the run", d)
	}
}

// A key refused while a refresh was already running isn't answered by that
// run, which may return the very key refused: the next caller gets a run of
// its own.
func TestAKeyRefusedDuringARefreshGetsAFreshRun(t *testing.T) {
	argv, runs := keyHelper(t, "slowfirst")
	l := config.LLM{APIKeyCmd: argv, APIKeyTTL: 1}
	k := keyCommand(l)
	k.mu.Lock()
	k.key, k.at = "K1", time.Now().Add(-2*time.Second) // due for a refresh
	k.mu.Unlock()
	go APIKey(context.Background(), l) // the refresh, under way
	time.Sleep(50 * time.Millisecond)
	k.mu.Lock()
	k.key, k.at = "K1", time.Now() // still the cached key while the refresh runs
	k.mu.Unlock()
	k.expire("K1") // a request with K1 just got a 401
	key, err := APIKey(context.Background(), l)
	if err != nil || runs() != 2 {
		t.Errorf("after the 401: key %q, err %v, runs %d; want a second run of its own", key, err, runs())
	}
	time.Sleep(700 * time.Millisecond) // the first run ends now, after the second
	if got, _ := APIKey(context.Background(), l); got != key {
		t.Errorf("the run begun before the 401 overwrote its answer: %q, want %q", got, key)
	}
}

// A background probe never starts the command; it uses a key already fetched.
func TestKnownKeyNeverRunsTheCommand(t *testing.T) {
	argv, runs := keyHelper(t, "")
	l := config.LLM{APIKeyCmd: argv}
	if _, ok := KnownKey(l); ok || runs() != 0 {
		t.Fatalf("before any request: ok %v, runs %d; want no key and no run", ok, runs())
	}
	APIKey(context.Background(), l)
	if k, ok := KnownKey(l); !ok || k != "key-1" || runs() != 1 {
		t.Errorf("after one: %q %v, runs %d", k, ok, runs())
	}
	if k, ok := KnownKey(config.LLM{APIKey: "plain"}); !ok || k != "plain" {
		t.Errorf("no command: %q %v", k, ok)
	}
}

// What a helper says on stderr reaches the user while it is still running.
func TestKeyCmdStderrIsShownAsItComes(t *testing.T) {
	argv, _ := keyHelper(t, "talk")
	var mu sync.Mutex
	var lines []string
	var heard time.Time
	SetKeyCmdOutput(func(l string) {
		mu.Lock()
		lines, heard = append(lines, l), time.Now()
		mu.Unlock()
	})
	defer SetKeyCmdOutput(nil)
	if _, err := APIKey(context.Background(), config.LLM{APIKeyCmd: argv}); err != nil {
		t.Fatal(err)
	}
	done := time.Now()
	mu.Lock()
	defer mu.Unlock()
	if len(lines) != 1 || !strings.Contains(lines[0], "enter ABCD") {
		t.Fatalf("lines %q", lines)
	}
	if done.Sub(heard) < 200*time.Millisecond {
		t.Errorf("the line came %s before the end — not while the helper waited", done.Sub(heard))
	}
}
