package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"

	"github.com/ipsupport-llc/ipsupport-code/internal/procgroup"
)

// stdioTransport speaks newline-delimited JSON-RPC to a subprocess over its
// stdin/stdout. A single long-lived reader loop dispatches each response to the
// waiter registered for its id, so a cancelled/timed-out call just unregisters its
// waiter — it never leaves a goroutine blocked on the shared reader (which would
// then race the next call and corrupt the stream).
type stdioTransport struct {
	w       io.Writer
	br      *bufio.Reader
	closeFn func()

	mu      sync.Mutex
	waiters map[int]chan rpcResp
	done    chan struct{} // closed when the reader loop exits (EOF/error)
	readErr error
}

func dialStdio(name string, s Server) (transport, error) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, s.Command, s.Args...)
	cmd.Env = os.Environ()
	for k, v := range s.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stderr = io.Discard
	// Kill the WHOLE process group on close, and bound Wait so a grandchild
	// that outlives the server (e.g. a shell's backgrounded child) holding the
	// stdout pipe open can't hang teardown forever — same wedge class
	// internal/tool/run.go and git.go already guard against.
	procgroup.Set(cmd)
	in, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start %s: %w", s.Command, err)
	}
	return newStdio(in, out, func() {
		_ = in.Close()
		cancel()
		_ = cmd.Wait()
	}), nil
}

func newStdio(w io.Writer, r io.Reader, closeFn func()) *stdioTransport {
	t := &stdioTransport{
		w: w, br: bufio.NewReaderSize(r, 1<<20), closeFn: closeFn,
		waiters: map[int]chan rpcResp{},
		done:    make(chan struct{}),
	}
	go t.readLoop()
	return t
}

// readLoop reads responses for the transport's whole lifetime and hands each to
// the waiter for its id. Runs until the stream errors/EOFs (e.g. close() kills the
// process), then wakes any remaining waiters via done.
func (t *stdioTransport) readLoop() {
	defer close(t.done)
	for {
		line, err := t.br.ReadBytes('\n')
		if err != nil {
			t.mu.Lock()
			t.readErr = err
			t.mu.Unlock()
			return
		}
		if line = bytes.TrimSpace(line); len(line) == 0 {
			continue
		}
		var m rpcResp
		if json.Unmarshal(line, &m) != nil || m.ID == nil {
			continue // notification, log line, or unparseable
		}
		t.mu.Lock()
		ch, ok := t.waiters[*m.ID]
		if ok {
			delete(t.waiters, *m.ID)
		}
		t.mu.Unlock()
		if ok {
			ch <- m // ch is buffered(1) → never blocks the reader
		}
	}
}

func (t *stdioTransport) close() {
	if t.closeFn != nil {
		t.closeFn()
	}
}

func (t *stdioTransport) notify(ctx context.Context, msg rpcMsg) error {
	// Same reasoning as roundTrip: t.write can't be cancelled once called
	// directly, so run it on its own goroutine and let ctx/t.done still win
	// if the pipe is stuck (a stopped-reading child, a full OS buffer).
	writeErr := make(chan error, 1)
	go func() { writeErr <- t.write(msg) }()
	select {
	case err := <-writeErr:
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-t.done:
		return io.EOF
	}
}

func (t *stdioTransport) roundTrip(ctx context.Context, msg rpcMsg) (json.RawMessage, error) {
	if msg.ID == nil {
		return nil, fmt.Errorf("roundTrip requires a request id")
	}
	id := *msg.ID
	ch := make(chan rpcResp, 1)
	t.mu.Lock()
	t.waiters[id] = ch
	t.mu.Unlock()
	unregister := func() {
		t.mu.Lock()
		delete(t.waiters, id)
		t.mu.Unlock()
	}

	// t.write is a blocking pipe write with no cancellation support (Go's
	// os.File/pipe writes can't be interrupted by a context) — if the child
	// stops reading stdin and the OS pipe buffer fills, a synchronous write
	// here would hang forever regardless of ctx, defeating the caller's whole
	// timeout. Do it on its own goroutine instead: writeErr is buffered so
	// that goroutine can always deliver and exit even after this call has
	// already returned via ctx.Done()/t.done below (it then leaks only until
	// the pipe itself unblocks or closes — no worse than an abandoned
	// synchronous write would have been).
	writeErr := make(chan error, 1)
	go func() { writeErr <- t.write(msg) }()

	// pending starts as writeErr; once that case fires with a nil error, it's
	// set to nil so the loop's next iteration blocks only on ctx/done/ch (a
	// nil channel is never selectable) — i.e. "wait for the write to
	// succeed, THEN wait for the response", either phase interruptible.
	pending := writeErr
	for {
		select {
		case err := <-pending:
			if err != nil {
				unregister()
				return nil, err
			}
			pending = nil
		case <-ctx.Done():
			unregister()
			return nil, ctx.Err()
		case <-t.done:
			unregister()
			t.mu.Lock()
			err := t.readErr
			t.mu.Unlock()
			if err == nil {
				err = io.EOF
			}
			return nil, fmt.Errorf("%s: transport closed: %w", msg.Method, err)
		case m := <-ch:
			if m.Error != nil {
				return nil, fmt.Errorf("%s: %s", msg.Method, m.Error.Message)
			}
			return m.Result, nil
		}
	}
}

func (t *stdioTransport) write(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = t.w.Write(append(b, '\n'))
	return err
}

// newClient builds a Client over a raw stdio pipe (used directly by tests).
func newClient(name string, w io.Writer, r io.Reader, closeFn func()) *Client {
	return &Client{name: name, tr: newStdio(w, r, closeFn)}
}
