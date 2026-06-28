package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPTransport(t *testing.T) {
	var sawAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID     *int   `json:"id"`
			Method string `json:"method"`
		}
		_ = json.Unmarshal(body, &req)
		if req.ID == nil { // a notification (initialized) → just accept it
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Mcp-Session-Id", "sess-1")
		result := `{}`
		switch req.Method {
		case "tools/list":
			result = `{"tools":[{"name":"ping","description":"ping"}]}`
		case "tools/call":
			result = `{"content":[{"type":"text","text":"pong"}]}`
		}
		io.WriteString(w, `{"jsonrpc":"2.0","id":`+itoa(*req.ID)+`,"result":`+result+`}`)
	}))
	defer srv.Close()

	c, err := Connect(context.Background(), "remote", Server{
		URL:     srv.URL,
		Headers: map[string]string{"Authorization": "Bearer secret"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	if sawAuth != "Bearer secret" {
		t.Errorf("auth header not sent: %q", sawAuth)
	}
	if len(c.Tools()) != 1 || c.Tools()[0].Name != "ping" {
		t.Fatalf("tools = %+v, want [ping]", c.Tools())
	}
	out, err := c.Call(context.Background(), "ping", nil)
	if err != nil || !strings.Contains(out, "pong") {
		t.Errorf("call = %q, %v; want pong", out, err)
	}
	// the session id from initialize must be echoed back on later requests
	if c.tr.(*httpTransport).session != "sess-1" {
		t.Errorf("session id not captured: %q", c.tr.(*httpTransport).session)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
