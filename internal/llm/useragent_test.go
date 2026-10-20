package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// Every request to the model server names this program: a proxy in front of
// it saw only "Go-http-client/1.1", the same as any other Go tool.
func TestEveryModelRequestCarriesTheUserAgent(t *testing.T) {
	defer func(ua string) { UserAgent = ua }(UserAgent)
	UserAgent = "ipsupport-code/9.9.9"

	var mu sync.Mutex
	var seen []string // "path user-agent", one per request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.URL.Path+" "+r.Header.Get("User-Agent"))
		mu.Unlock()
		switch r.URL.Path {
		case "/v1/chat/completions":
			io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
		default:
			io.WriteString(w, `{"data":[{"id":"m","state":"loaded","loaded_context_length":8192}]}`)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	base := srv.URL + "/v1"
	calls := map[string]func(){
		"ListModels":          func() { _, _ = ListModels(ctx, base, "", srv.Client()) },
		"DetectModelContext":  func() { _ = DetectModelContext(ctx, base, "", "m", srv.Client()) },
		"ListLMStudioModels":  func() { _, _ = ListLMStudioModels(ctx, base, srv.Client()) },
		"DetectContextWindow": func() { _ = DetectContextWindow(ctx, base, "m", srv.Client()) },
		"Chat": func() {
			_, _ = NewOpenAIClient(config.LLM{BaseURL: base, Model: "m"}).Chat(ctx, []Message{User("hi")}, nil)
		},
	}
	for name, call := range calls {
		mu.Lock()
		seen = nil
		mu.Unlock()
		call()
		mu.Lock()
		if len(seen) == 0 {
			t.Errorf("%s made no request", name)
		}
		for _, s := range seen {
			if !strings.HasSuffix(s, " "+UserAgent) {
				t.Errorf("%s: request %q, want User-Agent %q", name, s, UserAgent)
			}
		}
		mu.Unlock()
	}
}
