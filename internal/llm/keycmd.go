package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// keyCmdTimeout bounds one run of an api_key_cmd. It is long because the
// command may open a browser for a sign-in and wait for the user.
var keyCmdTimeout = 5 * time.Minute

// keyCmdRetryAfter is how long a failed run's error stands before the command
// is tried again, so a declined sign-in isn't reopened by every request that
// follows it at once.
var keyCmdRetryAfter = 15 * time.Second

// keyCmdWaitDelay is how long a finished (or killed) command may leave its
// output pipes open — a child it started holding them — before Run stops
// waiting on them.
var keyCmdWaitDelay = 5 * time.Second

// keyCmdOutput is SetKeyCmdOutput's function.
var keyCmdOutput atomic.Pointer[func(line string)]

// SetKeyCmdOutput sets what gets each line an api_key_cmd writes to stderr, as
// it writes it, so a helper that says "open this URL and enter the code" is
// heard while it waits, not after it gives up.
func SetKeyCmdOutput(f func(line string)) { keyCmdOutput.Store(&f) }

// keyCmd is an api_key_cmd: a command whose trimmed stdout is the API key. The
// key is kept in memory only, never written to config. It is fetched on first
// use, again once a caller's ttl has passed, and again after the server
// rejects it (expire).
type keyCmd struct {
	argv []string

	mu     sync.Mutex
	key    string
	at     time.Time
	failed error      // the last run's error, standing until failAt+keyCmdRetryAfter
	failAt time.Time  //
	flight *keyFlight // the run in progress, shared by everyone waiting on it
	gen    int        // bumped by expire: a run begun before it can't answer after it
}

type keyFlight struct {
	done chan struct{}
	gen  int
	key  string
	err  error
}

// keyCmds holds one keyCmd per distinct command, so every client built for a
// connection (wire() rebuilds them often), and every connection naming the
// same command, share one key and one run.
var keyCmds sync.Map

func keyCommand(l config.LLM) *keyCmd {
	if len(l.APIKeyCmd) == 0 {
		return nil
	}
	id := strings.Join(l.APIKeyCmd, "\x00")
	k, _ := keyCmds.LoadOrStore(id, &keyCmd{argv: append([]string(nil), l.APIKeyCmd...)})
	return k.(*keyCmd)
}

// Key returns the current key, running the command when there is none or it
// is older than ttl (0: no age limit). Callers arriving during a run wait for
// that run. The run itself isn't tied to ctx: a caller giving up (a short
// probe, an esc) stops waiting, but doesn't kill a sign-in in progress.
func (k *keyCmd) Key(ctx context.Context, ttl time.Duration) (string, error) {
	k.mu.Lock()
	if key, ok := k.cachedLocked(ttl); ok {
		k.mu.Unlock()
		return key, nil
	}
	if k.failed != nil && time.Since(k.failAt) < keyCmdRetryAfter {
		err := k.failed
		k.mu.Unlock()
		return "", err
	}
	f := k.flight
	if f == nil || f.gen != k.gen {
		f = &keyFlight{done: make(chan struct{}), gen: k.gen}
		k.flight = f
		go k.run(f)
	}
	k.mu.Unlock()
	select {
	case <-f.done:
		return f.key, f.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (k *keyCmd) run(f *keyFlight) {
	f.key, f.err = runKeyCmd(k.argv)
	k.mu.Lock()
	if f.gen == k.gen { // else a key was refused since this run began, and this may be it
		if f.err == nil {
			k.key, k.at, k.failed = f.key, time.Now(), nil
		} else {
			k.failed, k.failAt = f.err, time.Now()
		}
	}
	if k.flight == f {
		k.flight = nil
	}
	k.mu.Unlock()
	close(f.done)
}

// expire drops the key the server just rejected, so the next Key runs the
// command again. A key someone else has already replaced is left alone. A run
// already under way may be about to return the rejected key, so the next Key
// starts a run of its own instead of waiting on that one.
func (k *keyCmd) expire(rejected string) {
	k.mu.Lock()
	if k.key == rejected {
		k.key = ""
		k.gen++
	}
	k.mu.Unlock()
}

func (k *keyCmd) cachedLocked(ttl time.Duration) (string, bool) {
	if k.key != "" && (ttl <= 0 || time.Since(k.at) < ttl) {
		return k.key, true
	}
	return "", false
}

func runKeyCmd(argv []string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), keyCmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	var stdout bytes.Buffer
	stderr := &lineWriter{}
	cmd.Stdout, cmd.Stderr = &stdout, stderr
	// The timeout kills the whole tree, and Run stops waiting on pipes a child
	// left open (see procgroup.Set).
	procgroup.Set(cmd)
	cmd.WaitDelay = keyCmdWaitDelay
	err := cmd.Run()
	if errors.Is(err, exec.ErrWaitDelay) && ctx.Err() == nil {
		err = nil // it exited fine; only a child it left behind kept the pipes open
	}
	if err != nil {
		if ctx.Err() != nil {
			err = fmt.Errorf("no key after %s", keyCmdTimeout)
		}
		// A key it printed before failing stays out of the message.
		msg := strings.TrimSpace(redactKey(stderr.all.String(), strings.TrimSpace(stdout.String())))
		if msg != "" {
			return "", fmt.Errorf("api_key_cmd %s: %v: %s", argv[0], err, textutil.OneLine(msg, 300))
		}
		return "", fmt.Errorf("api_key_cmd %s: %v", argv[0], err)
	}
	key := strings.TrimSpace(stdout.String())
	if key == "" {
		return "", errors.New("api_key_cmd " + argv[0] + " printed no key")
	}
	return key, nil
}

// lineWriter keeps what a command writes to stderr (up to a limit, for the
// error message) and hands each finished line to SetKeyCmdOutput's function.
type lineWriter struct {
	all     bytes.Buffer
	partial []byte
}

func (w *lineWriter) Write(p []byte) (int, error) {
	if w.all.Len() < 16<<10 {
		w.all.Write(p)
	}
	w.partial = append(w.partial, p...)
	for {
		i := bytes.IndexByte(w.partial, '\n')
		if i < 0 {
			break
		}
		if f := keyCmdOutput.Load(); f != nil && *f != nil {
			if line := strings.TrimSpace(string(w.partial[:i])); line != "" {
				(*f)(line)
			}
		}
		w.partial = w.partial[i+1:]
	}
	if len(w.partial) > 4<<10 {
		w.partial = w.partial[:0] // a line this long is not a message for a person
	}
	return len(p), nil
}

// redactKey hides key in s, so an error quoting what a server or helper said
// can't carry the key into the log or the screen.
func redactKey(s, key string) string {
	if key == "" {
		return s
	}
	return strings.ReplaceAll(s, key, "[api key]")
}

// APIKey is the key to send for a connection: from its api_key_cmd when it has
// one, else its api_key (which may have come from an environment variable).
func APIKey(ctx context.Context, l config.LLM) (string, error) {
	if k := keyCommand(l); k != nil {
		return k.Key(ctx, keyTTL(l))
	}
	return l.APIKey, nil
}

// KnownKey is APIKey for a background probe: it never runs api_key_cmd, so a
// probe can't open a sign-in nobody asked for. ok is false when the command
// has no usable key yet; the probe should skip it, and a later one, once a
// request has fetched the key, gets it.
func KnownKey(l config.LLM) (key string, ok bool) {
	k := keyCommand(l)
	if k == nil {
		return l.APIKey, true
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.cachedLocked(keyTTL(l))
}

func keyTTL(l config.LLM) time.Duration { return time.Duration(l.APIKeyTTL) * time.Second }

// ListModelsFor is ListModels on a connection's own key. A key from
// api_key_cmd that the server refuses is replaced once, as Chat does.
func ListModelsFor(ctx context.Context, l config.LLM, hc *http.Client) ([]string, error) {
	key, err := APIKey(ctx, l)
	if err != nil {
		return nil, err
	}
	ids, err := ListModels(ctx, l.BaseURL, key, hc)
	var ae *authError
	if k := keyCommand(l); k != nil && errors.As(err, &ae) {
		k.expire(key)
		if key, err = k.Key(ctx, keyTTL(l)); err != nil {
			return nil, err
		}
		ids, err = ListModels(ctx, l.BaseURL, key, hc)
	}
	return ids, err
}
