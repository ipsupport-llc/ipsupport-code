package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"testing"
)

// fakeMCP is a minimal MCP server speaking newline-delimited JSON-RPC over conn.
func fakeMCP(conn net.Conn) {
	defer conn.Close()
	br := bufio.NewReader(conn)
	for {
		line, err := br.ReadBytes('\n')
		if err != nil {
			return
		}
		var req struct {
			ID     *int   `json:"id"`
			Method string `json:"method"`
		}
		if json.Unmarshal(bytes.TrimSpace(line), &req) != nil || req.ID == nil {
			continue // bad line or a notification (no response)
		}
		result := `{}`
		switch req.Method {
		case "tools/list":
			result = `{"tools":[{"name":"echo","description":"echo back","inputSchema":{"type":"object"}}]}`
		case "tools/call":
			result = `{"content":[{"type":"text","text":"echoed!"}],"isError":false}`
		}
		fmt.Fprintf(conn, `{"jsonrpc":"2.0","id":%d,"result":%s}`+"\n", *req.ID, result)
	}
}

func TestClientHandshakeListCall(t *testing.T) {
	cConn, sConn := net.Pipe()
	go fakeMCP(sConn)
	c := newClient("fake", cConn, cConn, func() { cConn.Close() })
	defer c.Close()
	ctx := context.Background()

	if err := c.handshake(ctx); err != nil {
		t.Fatal(err)
	}
	tools, err := c.listTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "echo" {
		t.Fatalf("tools = %+v, want [echo]", tools)
	}
	out, err := c.Call(ctx, "echo", map[string]any{"msg": "hi"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(out, "echoed!") {
		t.Errorf("call result = %q, want echoed!", out)
	}
}

func TestClientCallSurfacesToolError(t *testing.T) {
	cConn, sConn := net.Pipe()
	go func() { // a server whose tools/call reports isError
		defer sConn.Close()
		br := bufio.NewReader(sConn)
		for {
			line, err := br.ReadBytes('\n')
			if err != nil {
				return
			}
			var req struct {
				ID     *int   `json:"id"`
				Method string `json:"method"`
			}
			if json.Unmarshal(bytes.TrimSpace(line), &req) != nil || req.ID == nil {
				continue
			}
			result := `{}`
			if req.Method == "tools/call" {
				result = `{"content":[{"type":"text","text":"boom"}],"isError":true}`
			}
			fmt.Fprintf(sConn, `{"jsonrpc":"2.0","id":%d,"result":%s}`+"\n", *req.ID, result)
		}
	}()
	c := newClient("fake", cConn, cConn, func() { cConn.Close() })
	defer c.Close()
	if _, err := c.Call(context.Background(), "x", nil); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("a tool isError must surface as an error, got %v", err)
	}
}
