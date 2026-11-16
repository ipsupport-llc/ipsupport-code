package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// keyCmdTimeout bounds one run of an api_key_cmd. It is long because the
// command may open a browser for a sign-in and wait for the user.
var keyCmdTimeout = 5 * time.Minute

// keyCmd is a connection's api_key_cmd: a command whose trimmed stdout is the
// API key. The key is kept in memory only, never written to config. It is
// fetched on first use, again once ttl has passed (ttl 0: never on time), and
// again after the server rejects it (expire).
type keyCmd struct {
	argv []string
	ttl  time.Duration

	mu     sync.Mutex
	key    string
	at     time.Time
	flight *keyFlight // the run in progress, shared by everyone waiting on it
}

type keyFlight struct {
	done chan struct{}
	key  string
	err  error
}

// keyCmds holds one keyCmd per distinct command, so every client built for a
// connection (wire() rebuilds them often) shares one key and one run.
var keyCmds sync.Map

func keyCommand(l config.LLM) *keyCmd {
	if len(l.APIKeyCmd) == 0 {
		return nil
	}
	ttl := time.Duration(l.APIKeyTTL) * time.Second
	id := fmt.Sprintf("%d\x00%s", ttl, strings.Join(l.APIKeyCmd, "\x00"))
	k, _ := keyCmds.LoadOrStore(id, &keyCmd{argv: append([]string(nil), l.APIKeyCmd...), ttl: ttl})
	return k.(*keyCmd)
}

// Key returns the current key, running the command when there is none or it
// has outlived its ttl. Callers arriving during a run wait for that run. The
// run itself isn't tied to ctx: a caller giving up (a short probe, an esc)
// stops waiting, but doesn't kill a sign-in someone is in the middle of.
func (k *keyCmd) Key(ctx context.Context) (string, error) {
	k.mu.Lock()
	if k.key != "" && (k.ttl <= 0 || time.Since(k.at) < k.ttl) {
		key := k.key
		k.mu.Unlock()
		return key, nil
	}
	f := k.flight
	if f == nil {
		f = &keyFlight{done: make(chan struct{})}
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
	if f.err == nil {
		k.key, k.at = f.key, time.Now()
	}
	k.flight = nil
	k.mu.Unlock()
	close(f.done)
}

// expire drops the key the server just rejected, so the next Key runs the
// command again. A key someone else has already replaced is left alone.
func (k *keyCmd) expire(rejected string) {
	k.mu.Lock()
	if k.key == rejected {
		k.key = ""
	}
	k.mu.Unlock()
}

func runKeyCmd(argv []string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), keyCmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			err = fmt.Errorf("no key after %s", keyCmdTimeout)
		}
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
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

// APIKey is the key to send for a connection: from its api_key_cmd when it has
// one, else its api_key (which may have come from an environment variable).
func APIKey(ctx context.Context, l config.LLM) (string, error) {
	if k := keyCommand(l); k != nil {
		return k.Key(ctx)
	}
	return l.APIKey, nil
}
