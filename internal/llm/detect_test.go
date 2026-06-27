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

// An unloaded model reports only max_context_length — we must NOT adopt it
// (that's the 260k bug); return 0 so the caller keeps its default and re-detects
// once the model is loaded.
func TestListLMStudioModels(t *testing.T) {
	const body = `{"data":[
		{"id":"a","state":"loaded","loaded_context_length":8192,"max_context_length":131072,"quantization":"Q4_K_M"},
		{"id":"b","state":"not-loaded","max_context_length":4096}
	]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v0/models" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		io.WriteString(w, body)
	}))
	defer srv.Close()

	ms, err := ListLMStudioModels(context.Background(), srv.URL+"/v1", srv.Client())
	if err != nil || len(ms) != 2 {
		t.Fatalf("models=%+v err=%v", ms, err)
	}
	if ms[0].ID != "a" || ms[0].State != "loaded" || ms[0].LoadedContextLength != 8192 || ms[0].Quantization != "Q4_K_M" {
		t.Errorf("parsed wrong: %+v", ms[0])
	}
}

func TestDetectContextWindowUnloadedReturnsZero(t *testing.T) {
	const body = `{"data":[{"id":"m","state":"not-loaded","max_context_length":262144}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, body)
	}))
	defer srv.Close()
	if got := DetectContextWindow(context.Background(), srv.URL+"/v1", "m", srv.Client()); got != 0 {
		t.Errorf("got %d, want 0 (don't trust max for an unloaded model)", got)
	}
}

func TestDetectModelContext(t *testing.T) {
	// OpenRouter-style /v1/models with context_length (top-level and nested).
	const body = `{"data":[
		{"id":"x-ai/grok-4.3","context_length":131072},
		{"id":"nested/model","top_provider":{"context_length":200000}}
	]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		io.WriteString(w, body)
	}))
	defer srv.Close()

	if got := DetectModelContext(context.Background(), srv.URL+"/v1", "k", "x-ai/grok-4.3", srv.Client()); got != 131072 {
		t.Errorf("top-level context_length = %d, want 131072", got)
	}
	if got := DetectModelContext(context.Background(), srv.URL+"/v1", "k", "nested/model", srv.Client()); got != 200000 {
		t.Errorf("nested top_provider.context_length = %d, want 200000", got)
	}
	if got := DetectModelContext(context.Background(), srv.URL+"/v1", "k", "unknown", srv.Client()); got != 0 {
		t.Errorf("unknown model = %d, want 0", got)
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
