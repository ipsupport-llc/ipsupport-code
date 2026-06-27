package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

// DetectContextWindow asks an LM Studio server for the loaded model's context
// length via its native /api/v0/models endpoint (the OpenAI /v1 surface doesn't
// report it). It returns 0 when unavailable — not LM Studio, an older version,
// the model isn't loaded yet, or the call fails — so the caller keeps its
// configured default. Best-effort: never errors.
func DetectContextWindow(ctx context.Context, baseURL, model string, hc *http.Client) int {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, lmStudioModelsURL(baseURL), nil)
	if err != nil {
		return 0
	}
	resp, err := hc.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
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
		return 0
	}

	// Prefer the entry for our model; otherwise the first loaded one.
	for _, m := range out.Data {
		if m.ID == model {
			return ctxLen(m.LoadedContextLength, m.MaxContextLength)
		}
	}
	for _, m := range out.Data {
		if m.State == "loaded" {
			return ctxLen(m.LoadedContextLength, m.MaxContextLength)
		}
	}
	return 0
}

// ctxLen prefers the loaded window (what's actually in use) over the model max.
func ctxLen(loaded, max int) int {
	if loaded > 0 {
		return loaded
	}
	return max
}

// lmStudioModelsURL turns an OpenAI base URL (…/v1) into LM Studio's native
// models endpoint (…/api/v0/models).
func lmStudioModelsURL(baseURL string) string {
	b := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	return b + "/api/v0/models"
}
