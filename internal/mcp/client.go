// Package mcp is a minimal Model Context Protocol client: it launches a
// user-configured MCP server (stdio transport, newline-delimited JSON-RPC 2.0),
// lists the tools it offers, and calls them. Deliberately lean — the agent
// exposes every MCP tool through ONE proxy tool so the prompt catalog (and a
// small model's context) stays small instead of ballooning with server schemas.
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
	"strings"
	"sync"
)

// Server is a configured MCP server launched over stdio.
type Server struct {
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// Tool is one tool advertised by a server.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// Client is a live connection to one MCP server.
type Client struct {
	name    string
	w       io.Writer
	br      *bufio.Reader
	closeFn func()
	tools   []Tool

	mu sync.Mutex // serializes request/response on the single stdio channel
	id int
}

type rpcMsg struct {
	JSONRPC string `json:"jsonrpc"`
	ID      *int   `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResp struct {
	ID     *int            `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// Connect launches the server over stdio and completes the handshake + tool list.
func Connect(ctx context.Context, name string, s Server) (*Client, error) {
	if strings.TrimSpace(s.Command) == "" {
		return nil, fmt.Errorf("mcp %q: empty command", name)
	}
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
		return nil, fmt.Errorf("mcp %q: start %s: %w", name, s.Command, err)
	}
	c := newClient(name, in, out, func() {
		_ = in.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	if err := c.handshake(ctx); err != nil {
		c.Close()
		return nil, fmt.Errorf("mcp %q: %w", name, err)
	}
	tools, err := c.listTools(ctx)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("mcp %q: %w", name, err)
	}
	c.tools = tools
	return c, nil
}

// newClient wires a client to a raw transport (used directly by tests).
func newClient(name string, w io.Writer, r io.Reader, closeFn func()) *Client {
	return &Client{name: name, w: w, br: bufio.NewReaderSize(r, 1<<20), closeFn: closeFn}
}

// Tools returns the cached tool list from the handshake.
func (c *Client) Tools() []Tool { return c.tools }

// Close shuts the server down.
func (c *Client) Close() {
	if c.closeFn != nil {
		c.closeFn()
	}
}

func (c *Client) handshake(ctx context.Context) error {
	if _, err := c.rpc(ctx, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "ipsupport-code", "version": "1"},
	}); err != nil {
		return err
	}
	return c.write(rpcMsg{JSONRPC: "2.0", Method: "notifications/initialized"}) // no id → notification
}

func (c *Client) listTools(ctx context.Context) ([]Tool, error) {
	raw, err := c.rpc(ctx, "tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}
	var out struct {
		Tools []Tool `json:"tools"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out.Tools, nil
}

// Call invokes a tool and returns its text content. A tool that reports an error
// (isError) is surfaced as a Go error so the model sees it as a failed call.
func (c *Client) Call(ctx context.Context, tool string, args map[string]any) (string, error) {
	raw, err := c.rpc(ctx, "tools/call", map[string]any{"name": tool, "arguments": args})
	if err != nil {
		return "", err
	}
	var out struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, item := range out.Content {
		if item.Text != "" {
			sb.WriteString(item.Text)
			sb.WriteByte('\n')
		}
	}
	text := strings.TrimSpace(sb.String())
	if out.IsError {
		if text == "" {
			text = "(tool reported an error)"
		}
		return "", fmt.Errorf("%s", text)
	}
	return text, nil
}

// rpc sends one request and returns its result, skipping any interleaved
// notifications. Serialized by the mutex; honors ctx cancellation/timeout.
func (c *Client) rpc(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.id++
	id := c.id
	if err := c.write(rpcMsg{JSONRPC: "2.0", ID: &id, Method: method, Params: params}); err != nil {
		return nil, err
	}
	type result struct {
		raw json.RawMessage
		err error
	}
	done := make(chan result, 1)
	go func() {
		for {
			line, err := c.br.ReadBytes('\n')
			if err != nil {
				done <- result{nil, err}
				return
			}
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var m rpcResp
			if json.Unmarshal(line, &m) != nil || m.ID == nil || *m.ID != id {
				continue // notification, log line, or a different id
			}
			if m.Error != nil {
				done <- result{nil, fmt.Errorf("%s: %s", method, m.Error.Message)}
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

func (c *Client) write(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = c.w.Write(append(b, '\n'))
	return err
}
