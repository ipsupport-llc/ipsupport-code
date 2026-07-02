package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// httpTransport speaks the MCP "streamable HTTP" transport: each JSON-RPC message
// is POSTed to one URL; the response comes back as application/json or as an SSE
// stream. Config Headers carry auth (e.g. Authorization: Bearer …). A
// Mcp-Session-Id handed back at initialize is echoed on later requests.
type httpTransport struct {
	url     string
	headers map[string]string
	hc      *http.Client

	mu      sync.Mutex
	session string
}

func dialHTTP(_ string, s Server) (transport, error) {
	if !strings.HasPrefix(s.URL, "http://") && !strings.HasPrefix(s.URL, "https://") {
		return nil, fmt.Errorf("url must start with http:// or https://")
	}
	return &httpTransport{url: s.URL, headers: s.Headers, hc: &http.Client{}}, nil
}

func (t *httpTransport) close() {}

func (t *httpTransport) notify(ctx context.Context, msg rpcMsg) error {
	_, err := t.do(ctx, msg) // a notification has no id; the server just 202s
	return err
}

func (t *httpTransport) roundTrip(ctx context.Context, msg rpcMsg) (json.RawMessage, error) {
	m, err := t.do(ctx, msg)
	if err != nil {
		return nil, err
	}
	if m == nil {
		return nil, fmt.Errorf("%s: empty response", msg.Method)
	}
	if m.Error != nil {
		return nil, fmt.Errorf("%s: %s", msg.Method, m.Error.Message)
	}
	return m.Result, nil
}

func (t *httpTransport) do(ctx context.Context, msg rpcMsg) (*rpcResp, error) {
	b, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers { // auth and any custom headers
		req.Header.Set(k, v)
	}
	t.mu.Lock()
	sess := t.session
	t.mu.Unlock()
	if sess != "" {
		req.Header.Set("Mcp-Session-Id", sess)
	}

	resp, err := t.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		t.mu.Lock()
		t.session = sid
		t.mu.Unlock()
	}
	switch {
	case resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusNoContent:
		return nil, nil // accepted notification, no body
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return nil, fmt.Errorf("http %d (check the server's auth headers)", resp.StatusCode)
	case resp.StatusCode >= 400:
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("http %d: %s", resp.StatusCode, oneLine(string(data)))
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		return readSSE(resp.Body, msg.ID)
	}
	var m rpcResp
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

// readSSE scans an event stream for the JSON-RPC response matching id.
func readSSE(r io.Reader, id *int) (*rpcResp, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var m rpcResp
		if json.Unmarshal([]byte(payload), &m) != nil || m.ID == nil || id == nil || *m.ID != *id {
			continue
		}
		return &m, nil
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no matching response in event stream")
}

func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if clipped, truncated := textutil.Clip(s, 150); truncated { // rune-safe, unlike a raw byte slice
		s = clipped + "…"
	}
	return s
}
