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
)

// stdioTransport speaks newline-delimited JSON-RPC to a subprocess over its
// stdin/stdout. One request/response at a time (serialized by the caller's id
// loop); interleaved notifications from the server are skipped.
type stdioTransport struct {
	w       io.Writer
	br      *bufio.Reader
	closeFn func()
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
	return &stdioTransport{w: w, br: bufio.NewReaderSize(r, 1<<20), closeFn: closeFn}
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
	if err := t.write(msg); err != nil {
		return nil, err
	}
	type result struct {
		raw json.RawMessage
		err error
	}
	done := make(chan result, 1)
	go func() {
		for {
			line, err := t.br.ReadBytes('\n')
			if err != nil {
				done <- result{nil, err}
				return
			}
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var m rpcResp
			if json.Unmarshal(line, &m) != nil || m.ID == nil || msg.ID == nil || *m.ID != *msg.ID {
				continue // notification, log line, or a different id
			}
			if m.Error != nil {
				done <- result{nil, fmt.Errorf("%s: %s", msg.Method, m.Error.Message)}
				return
			}
			done <- result{m.Result, nil}
			return
		}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		return r.raw, r.err
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
