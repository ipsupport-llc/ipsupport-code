package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// oneLineList collapses a (possibly JSON/HTML) error body to a short single line.
func oneLineList(s string) string { return textutil.OneLine(s, 200) }

// ListModels returns the model IDs an OpenAI-compatible server advertises at
// /v1/models (sorted). Works for LM Studio, OpenAI, xAI, Groq, OpenRouter, etc.
func ListModels(ctx context.Context, baseURL, apiKey string, hc *http.Client) ([]string, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	setUserAgent(req)
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		err := fmt.Errorf("list models: http %d", resp.StatusCode)
		if msg := strings.TrimSpace(redactKey(string(body), apiKey)); msg != "" {
			err = fmt.Errorf("list models: http %d: %s", resp.StatusCode, oneLineList(msg))
		}
		if resp.StatusCode == http.StatusUnauthorized {
			return nil, &authError{err: err, key: apiKey}
		}
		return nil, err
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(out.Data))
	for _, m := range out.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	sort.Strings(ids)
	return ids, nil
}

// DetectModelContext returns the advertised context length of `model` from an
// OpenAI-style /v1/models list. OpenRouter lists context_length (top-level and
// under top_provider), GitHub Copilot capabilities.limits; most others omit it
// → 0. Best-effort: 0 on any failure, so the caller keeps its current window.
func DetectModelContext(ctx context.Context, baseURL, apiKey, model string, hc *http.Client) int {
	return lookupModel(ctx, baseURL, apiKey, model, hc).window
}

// modelInfo is what a /models list says about one model.
type modelInfo struct {
	window int
	// endpoints are the paths the server says serve the model (GitHub
	// Copilot's supported_endpoints, e.g. "/chat/completions", "/responses");
	// nil when the list doesn't say.
	endpoints []string
}

// lookupModel reads model's entry from the server's /models list. Best-effort:
// the zero value on any failure.
func lookupModel(ctx context.Context, baseURL, apiKey, model string, hc *http.Client) modelInfo {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/models", nil)
	if err != nil {
		return modelInfo{}
	}
	setUserAgent(req)
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := hc.Do(req)
	if err != nil {
		return modelInfo{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return modelInfo{}
	}
	var out struct {
		Data []struct {
			ID            string `json:"id"`
			ContextLength int    `json:"context_length"`
			TopProvider   struct {
				ContextLength int `json:"context_length"`
			} `json:"top_provider"`
			Capabilities struct {
				Limits struct {
					// The prompt cap is what a request must stay under; the
					// whole window also holds the reply.
					MaxPromptTokens        int `json:"max_prompt_tokens"`
					MaxContextWindowTokens int `json:"max_context_window_tokens"`
				} `json:"limits"`
			} `json:"capabilities"`
			SupportedEndpoints []string `json:"supported_endpoints"`
		} `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil {
		return modelInfo{}
	}
	for _, m := range out.Data {
		if m.ID != model {
			continue
		}
		info := modelInfo{endpoints: m.SupportedEndpoints}
		lim := m.Capabilities.Limits
		for _, n := range []int{m.ContextLength, m.TopProvider.ContextLength, lim.MaxPromptTokens, lim.MaxContextWindowTokens} {
			if n > 0 {
				info.window = n
				break
			}
		}
		return info
	}
	return modelInfo{}
}

// LMModel is one entry from LM Studio's native /api/v0/models.
type LMModel struct {
	ID                  string `json:"id"`
	State               string `json:"state"` // "loaded" | "not-loaded"
	Quantization        string `json:"quantization"`
	LoadedContextLength int    `json:"loaded_context_length"`
	MaxContextLength    int    `json:"max_context_length"`
}

// ListLMStudioModels returns the rich model list from LM Studio's native API
// (state, context length, quant) — more than the bare OpenAI /v1/models ids.
func ListLMStudioModels(ctx context.Context, baseURL string, hc *http.Client) ([]LMModel, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, lmStudioModelsURL(baseURL), nil)
	if err != nil {
		return nil, err
	}
	setUserAgent(req)
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list models: http %d", resp.StatusCode)
	}
	var out struct {
		Data []LMModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].ID < out.Data[j].ID })
	return out.Data, nil
}

// LMWindow is what LM Studio's native model list says about the configured
// model: the context window to size against, which loaded model it came from,
// and whether the configured id is in the list at all.
type LMWindow struct {
	Tokens int    // loaded context length; 0 when nothing usable is loaded
	From   string // id of the loaded model Tokens was read from
	Listed bool   // the configured id is in the server's model list
	Known  bool   // the server returned a model list at all (Listed means something)
}

// Borrowed reports whether the window was read from a loaded model other than
// the configured one, because the configured one isn't loaded.
func (w LMWindow) Borrowed(model string) bool {
	return w.Tokens > 0 && !SameModel(model, w.From)
}

// DetectContextWindow asks an LM Studio server for the loaded model's context
// length via its native /api/v0/models endpoint (the OpenAI /v1 surface doesn't
// report it). Tokens is 0 when unavailable — not LM Studio, an older version,
// the model isn't loaded yet, or the call fails — so the caller keeps its
// configured default. Best-effort: never errors.
func DetectContextWindow(ctx context.Context, baseURL, model string, hc *http.Client) LMWindow {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, lmStudioModelsURL(baseURL), nil)
	if err != nil {
		return LMWindow{}
	}
	setUserAgent(req)
	resp, err := hc.Do(req)
	if err != nil {
		return LMWindow{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return LMWindow{}
	}

	var out struct {
		Data []struct {
			ID                  string `json:"id"`
			State               string `json:"state"`
			LoadedContextLength int    `json:"loaded_context_length"`
			MaxContextLength    int    `json:"max_context_length"`
		} `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil {
		return LMWindow{}
	}

	// Only trust loaded_context_length — the window the model is actually running
	// with. max_context_length is the model's theoretical max (often huge), which
	// is all an UNLOADED model reports; using it would size auto-compact against,
	// say, 260k instead of the real 8k. When nothing is loaded Tokens is 0 so the
	// caller keeps its default and re-detects once the model is up.
	w := LMWindow{Known: out.Data != nil} // a reply without a list says nothing about ours
	// The exact id first: SameModel would also accept a sibling (x for x@q8_0)
	// whose window may differ.
	for _, exact := range []bool{true, false} {
		for _, m := range out.Data {
			if exact && m.ID != model || !exact && !SameModel(model, m.ID) {
				continue
			}
			w.Listed = true
			if m.LoadedContextLength > 0 {
				w.Tokens, w.From = m.LoadedContextLength, m.ID
				return w
			}
		}
	}
	// Our model isn't loaded: fall back to whatever is, because that is often
	// what will answer (LM Studio serves an unresolved id with the loaded model).
	// The caller sees From differ and says so.
	for _, m := range out.Data {
		if m.State == "loaded" && m.LoadedContextLength > 0 {
			w.Tokens, w.From = m.LoadedContextLength, m.ID
			return w
		}
	}
	return w
}

// lmStudioModelsURL turns an OpenAI base URL (…/v1) into LM Studio's native
// models endpoint (…/api/v0/models).
func lmStudioModelsURL(baseURL string) string {
	b := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	return b + "/api/v0/models"
}
