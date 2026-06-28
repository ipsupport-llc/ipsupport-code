// Package mcp is a minimal Model Context Protocol client. It connects to a
// user-configured server over stdio (a local subprocess) or HTTP (a remote URL,
// with auth headers), lists the tools it offers, and calls them. Deliberately
// lean — the agent exposes every MCP tool through ONE proxy tool so the prompt
// catalog (and a small model's context) stays small instead of ballooning with
// per-server schemas.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// Server is a configured MCP server: stdio (Command) or HTTP (URL). Headers carry
// auth (e.g. {"Authorization": "Bearer …"}) and apply to the HTTP transport.
type Server struct {
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
}

// Tool is one tool advertised by a server.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// transport is one server connection: a request/response RPC, fire-and-forget
// notifications, and teardown. stdio and HTTP each implement it.
type transport interface {
	roundTrip(ctx context.Context, msg rpcMsg) (json.RawMessage, error)
	notify(ctx context.Context, msg rpcMsg) error
	close()
}

// Client is a live connection to one MCP server.
type Client struct {
	name  string
	tr    transport
	tools []Tool
	idMu  sync.Mutex
	id    int
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

// Connect opens the server (stdio if Command is set, else HTTP for URL) and
// completes the handshake + tool list.
func Connect(ctx context.Context, name string, s Server) (*Client, error) {
	var tr transport
	var err error
	switch {
	case strings.TrimSpace(s.Command) != "":
		tr, err = dialStdio(name, s)
	case strings.TrimSpace(s.URL) != "":
		tr, err = dialHTTP(name, s)
	default:
		return nil, fmt.Errorf("mcp %q: set either \"command\" (stdio) or \"url\" (http)", name)
	}
	if err != nil {
		return nil, fmt.Errorf("mcp %q: %w", name, err)
	}
	c := &Client{name: name, tr: tr}
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

// Tools returns the cached tool list from the handshake.
func (c *Client) Tools() []Tool { return c.tools }

// Close shuts the connection down.
func (c *Client) Close() {
	if c.tr != nil {
		c.tr.close()
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
	return c.notify(ctx, "notifications/initialized")
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

func (c *Client) rpc(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.idMu.Lock()
	c.id++
	id := c.id
	c.idMu.Unlock()
	return c.tr.roundTrip(ctx, rpcMsg{JSONRPC: "2.0", ID: &id, Method: method, Params: params})
}

func (c *Client) notify(ctx context.Context, method string) error {
	return c.tr.notify(ctx, rpcMsg{JSONRPC: "2.0", Method: method})
}
