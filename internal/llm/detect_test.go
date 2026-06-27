package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLMStudioModelsURL(t *testing.T) {
	cases := map[string]string{
		"http://localhost:1234/v1":  "http://localhost:1234/api/v0/models",
		"http://localhost:1234/v1/": "http://localhost:1234/api/v0/models",
		"http://host:9/v1":          "http://host:9/api/v0/models",
	}
	for in, want := range cases {
		if got := lmStudioModelsURL(in); got != want {
			t.Errorf("lmStudioModelsURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDetectContextWindow(t *testing.T) {
	const body = `{"data":[
		{"id":"other","state":"not-loaded","max_context_length":32768},
		{"id":"m","state":"loaded","loaded_context_length":16384,"max_context_length":131072}
	]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v0/models" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		io.WriteString(w, body)
	}))
	defer srv.Close()

	// Prefer the loaded window of our model.
	if got := DetectContextWindow(context.Background(), srv.URL+"/v1", "m", srv.Client()); got != 16384 {
		t.Errorf("got %d, want 16384 (loaded window of model m)", got)
	}
	// Unknown model → first loaded entry.
	if got := DetectContextWindow(context.Background(), srv.URL+"/v1", "nope", srv.Client()); got != 16384 {
		t.Errorf("got %d, want 16384 (first loaded)", got)
	}
}

func TestDetectContextWindowFallsToMax(t *testing.T) {
	const body = `{"data":[{"id":"m","state":"loaded","max_context_length":8192}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, body)
	}))
	defer srv.Close()
	if got := DetectContextWindow(context.Background(), srv.URL+"/v1", "m", srv.Client()); got != 8192 {
		t.Errorf("got %d, want 8192 (max when no loaded length)", got)
	}
}

func TestDetectContextWindowUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound) // not LM Studio's native API
	}))
	defer srv.Close()
	if got := DetectContextWindow(context.Background(), srv.URL+"/v1", "m", srv.Client()); got != 0 {
		t.Errorf("got %d, want 0 (so the caller keeps its default)", got)
	}
}
