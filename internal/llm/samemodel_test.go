package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

func TestSameModelToleratesOnlyHarmlessDifferences(t *testing.T) {
	same := [][2]string{
		{"gemma-4-e2b", "google/gemma-4-e2b"},        // publisher prefix
		{"google/gemma-4-e2b", "gemma-4-e2b"},        // either side
		{"Gemma-4-E2B", "gemma-4-e2b"},               // case
		{"gemma-4-e2b@q8_0", "gemma-4-e2b"},          // server leaves the quant off
		{"gemma-4-e2b", "gemma-4-e2b@q4_k_m"},        // or adds it
		{"llama3", "llama3:latest"},                  // Ollama's default tag
		{"gpt-4o", "gpt-4o-2024-08-06"},              // dated snapshot of an alias
		{"grok-4", "grok-4-0709"},                    // ditto
		{"anything", ""},                             // the server didn't say
		{"openai/gpt-4o-mini", "openai/gpt-4o-mini"}, // identical
	}
	for _, c := range same {
		if !SameModel(c[0], c[1]) {
			t.Errorf("SameModel(%q, %q) = false, want true", c[0], c[1])
		}
	}
	differ := [][2]string{
		{"gemma-4-26b-a4b-it-jang-imatrix@q8_0", "google/gemma-4-e2b"}, // the reported fallback
		{"gemma-4", "gemma-4-e2b"},                                     // a longer name is another model, not a snapshot
		{"gemma-4-e2b@q8_0", "gemma-4-e2b@q4_k_m"},                     // two different quantizations
		{"qwen3:8b", "qwen3:14b"},                                      // Ollama tags other than :latest matter
		{"gpt-4o", "gpt-4o-mini"},
	}
	for _, c := range differ {
		if SameModel(c[0], c[1]) {
			t.Errorf("SameModel(%q, %q) = true, want false", c[0], c[1])
		}
	}
}

// A server that answers with another model than the one asked for — LM Studio
// does this, with a 200, for an id it can't resolve — is reported, on both the
// streaming and the plain JSON path. A harmless difference is not.
func TestAReplyFromAnotherModelIsReported(t *testing.T) {
	stream := func(model string) string {
		return sseServer(t,
			`{"model":"`+model+`","choices":[{"delta":{"content":"hi"}}]}`,
			`{"model":"`+model+`","choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`,
		)
	}
	plain := func(model string) string {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `{"model":"`+model+`","choices":[{"message":{"role":"assistant","content":"hi"}}]}`)
		}))
		t.Cleanup(srv.Close)
		return srv.URL
	}
	for name, serve := range map[string]func(string) string{"stream": stream, "json": plain} {
		for served, want := range map[string]bool{"google/gemma-4-e2b": true, "publisher/wanted": false} {
			cl := NewOpenAIClient(config.LLM{BaseURL: serve(served), Model: "wanted"})
			var got [][2]string
			cl.OnModelMismatch = func(asked, s string) { got = append(got, [2]string{asked, s}) }
			if _, err := cl.Chat(context.Background(), []Message{User("hi")}, nil); err != nil {
				t.Fatal(err)
			}
			if want && (len(got) != 1 || got[0] != [2]string{"wanted", served}) {
				t.Errorf("%s, served %q: mismatch reports = %v, want one (wanted, %s)", name, served, got, served)
			}
			if !want && len(got) != 0 {
				t.Errorf("%s, served %q: reported %v for a harmless difference", name, served, got)
			}
		}
	}
}

// LM Studio's list says whether the configured id is there at all, and when
// it isn't loaded, which loaded model the window was borrowed from.
func TestDetectContextWindowSaysWhereTheWindowCameFrom(t *testing.T) {
	const body = `{"data":[
		{"id":"wanted@q8_0","state":"not-loaded","max_context_length":131072},
		{"id":"google/gemma-4-e2b","state":"loaded","loaded_context_length":8192,"max_context_length":131072}
	]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, body) }))
	defer srv.Close()
	get := func(model string) LMWindow {
		return DetectContextWindow(context.Background(), srv.URL+"/v1", model, srv.Client())
	}

	if w := get("gemma-4-e2b"); !w.Known || !w.Listed || w.Tokens != 8192 || w.Borrowed("gemma-4-e2b") {
		t.Errorf("loaded model (prefix left off): %+v, want listed, own window, not borrowed", w)
	}
	if w := get("wanted"); !w.Listed || w.Tokens != 8192 || !w.Borrowed("wanted") || w.From != "google/gemma-4-e2b" {
		t.Errorf("listed but unloaded: %+v, want listed, window borrowed from google/gemma-4-e2b", w)
	}
	if w := get("typo"); !w.Known || w.Listed || !w.Borrowed("typo") {
		t.Errorf("unknown id: %+v, want known list, not listed, borrowed", w)
	}

	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, `{"object":"error"}`) }))
	defer other.Close()
	if w := DetectContextWindow(context.Background(), other.URL+"/v1", "m", other.Client()); w.Known {
		t.Errorf("a 200 without a model list: %+v — not a list that lacks our model", w)
	}

	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNotFound) }))
	defer down.Close()
	if w := DetectContextWindow(context.Background(), down.URL+"/v1", "m", down.Client()); w.Known || w.Listed {
		t.Errorf("no list from the server: %+v — Listed must not read as 'missing'", w)
	}
}
