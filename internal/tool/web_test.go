package tool

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebSearch(t *testing.T) {
	page := `<div class="result__body">
		<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fgo&rut=x">Go example</a>
		<a class="result__snippet">A page about Go</a>
	</div>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, page)
	}))
	defer srv.Close()
	old := ddgSearchURL
	ddgSearchURL = srv.URL
	defer func() { ddgSearchURL = old }()

	r := NewWeb(srv.Client(), false).Call(context.Background(), "search", map[string]any{"query": "golang"})
	if r.IsError {
		t.Fatalf("search error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "Go example") || !strings.Contains(r.Content, "https://example.com/go") {
		t.Errorf("search content = %q", r.Content)
	}
}

func TestWebFetch(t *testing.T) {
	webAllowPrivate = true // the test server is on loopback
	t.Cleanup(func() { webAllowPrivate = false })
	page := `<html><head><title>Hello</title></head><body><article>
		<h1>Big Heading</h1>
		<p>Some paragraph text that is reasonably long so the reader keeps it around.</p>
	</article></body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		io.WriteString(w, page)
	}))
	defer srv.Close()

	r := NewWeb(srv.Client(), false).Call(context.Background(), "fetch", map[string]any{"url": srv.URL})
	if r.IsError {
		t.Fatalf("fetch error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "Big Heading") {
		t.Errorf("fetch markdown = %q, want heading text", r.Content)
	}
	if strings.Contains(r.Content, "<h1>") {
		t.Errorf("html not converted: %q", r.Content)
	}
}

func TestWebFetchBlocksPrivate(t *testing.T) {
	// SSRF guard ON (default): fetching loopback/metadata must be refused.
	for _, u := range []string{"http://127.0.0.1:80/", "http://localhost:8080/x", "http://169.254.169.254/latest/meta-data/"} {
		r := NewWeb(http.DefaultClient, false).Call(context.Background(), "fetch", map[string]any{"url": u})
		if !r.IsError || !strings.Contains(r.Content, "SSRF") {
			t.Errorf("fetch(%q) = %+v, want an SSRF refusal", u, r)
		}
	}
}

func TestWebStackExchange(t *testing.T) {
	js := `{"items":[{"title":"How to defer in Go?","link":"https://stackoverflow.com/q/1","score":42,"answer_count":3,"is_answered":true}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, js)
	}))
	defer srv.Close()
	old := stackExURL
	stackExURL = srv.URL
	defer func() { stackExURL = old }()

	r := NewWeb(srv.Client(), false).Call(context.Background(), "stackexchange", map[string]any{"query": "defer"})
	if r.IsError {
		t.Fatalf("stackexchange error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "How to defer in Go?") || !strings.Contains(r.Content, "stackoverflow.com/q/1") {
		t.Errorf("stackexchange content = %q", r.Content)
	}
}

func TestWebOfflineRefuses(t *testing.T) {
	w := NewWeb(http.DefaultClient, true) // offline
	for _, act := range []string{"search", "fetch", "stackexchange"} {
		params := map[string]any{"query": "x"}
		if act == "fetch" {
			params = map[string]any{"url": "http://example.com"}
		}
		r := w.Call(context.Background(), act, params)
		if !r.IsError || !strings.Contains(r.Content, "offline") {
			t.Errorf("%s while offline = %+v, want an offline refusal", act, r)
		}
	}
}
