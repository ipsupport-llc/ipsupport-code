package mcp

import (
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"
)

// A cancelled/timed-out roundTrip must not leave a reader blocked on the shared
// stream: a late response for the dead id is dropped, and the NEXT call still
// works. Previously each call spawned its own reader goroutine, so a timeout
// leaked one and the next call raced it, corrupting the transport.
func TestStdioTimeoutDoesNotCorruptNextCall(t *testing.T) {
	reqR, reqW := io.Pipe()
	respR, respW := io.Pipe()
	tr := newStdio(reqW, respR, func() { reqW.Close(); respW.Close() })
	go io.Copy(io.Discard, reqR) // drain the requests it writes so writes don't block

	// First call: never answered before its deadline → a timeout.
	id1 := 1
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := tr.roundTrip(ctx, rpcMsg{ID: &id1, Method: "slow"}); err == nil {
		t.Fatal("expected a timeout error for the unanswered call")
	}

	// A late response for the now-dead id must be harmlessly dropped.
	fmt.Fprintln(respW, `{"jsonrpc":"2.0","id":1,"result":{"late":true}}`)

	// The next call must succeed — the transport is intact.
	id2 := 2
	go func() {
		time.Sleep(5 * time.Millisecond)
		fmt.Fprintln(respW, `{"jsonrpc":"2.0","id":2,"result":{"ok":true}}`)
	}()
	raw, err := tr.roundTrip(context.Background(), rpcMsg{ID: &id2, Method: "m"})
	if err != nil {
		t.Fatalf("second call errored: %v", err)
	}
	if !strings.Contains(string(raw), "ok") {
		t.Errorf("second call result = %s, want it to contain ok", raw)
	}
}

// A blocked stdin write must not defeat the call's context. io.Pipe has no
// internal buffering — unlike a real OS pipe — so a Write blocks until
// something Reads it; leaving reqR undrained here deterministically
// reproduces "the child stopped reading stdin, the pipe write hangs forever"
// without needing to actually fill a real kernel pipe buffer.
func TestStdioRoundTripCancelsBlockedWrite(t *testing.T) {
	reqR, reqW := io.Pipe()
	respR, respW := io.Pipe()
	tr := newStdio(reqW, respR, func() { reqW.Close(); respW.Close() })
	_ = reqR // deliberately never drained/closed — the write must hang on it

	id := 1
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := tr.roundTrip(ctx, rpcMsg{ID: &id, Method: "m"})
	if err == nil {
		t.Fatal("expected a context-deadline error")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("roundTrip took %s to return — the blocked write defeated ctx", elapsed)
	}
}
