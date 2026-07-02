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
	cmd := exec.Command(s.Command, s.Args...)
	cmd.Env = os.Environ()
	for k, v := range s.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stderr = io.Discard
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", s.Command, err)
	}
	return newStdio(in, out, func() {
		_ = in.Close()
		_ = cmd.Process.Kill()
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

func (t *stdioTransport) notify(_ context.Context, msg rpcMsg) error {
	return t.write(msg)
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

	if err := t.write(msg); err != nil {
		unregister()
		return nil, err
	}
	select {
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
