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
		if msg := strings.TrimSpace(string(body)); msg != "" {
			return nil, fmt.Errorf("list models: http %d: %s", resp.StatusCode, oneLineList(msg))
		}
		return nil, fmt.Errorf("list models: http %d", resp.StatusCode)
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
// OpenAI-style /v1/models list that carries a context_length field. OpenRouter
// does (top-level and under top_provider); most others omit it → 0. Best-effort:
// 0 on any failure, so the caller keeps its current window.
func DetectModelContext(ctx context.Context, baseURL, apiKey, model string, hc *http.Client) int {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/models", nil)
	if err != nil {
		return 0
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
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
			ID            string `json:"id"`
			ContextLength int    `json:"context_length"`
			TopProvider   struct {
				ContextLength int `json:"context_length"`
			} `json:"top_provider"`
		} `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil {
		return 0
	}
	for _, m := range out.Data {
		if m.ID == model {
			if m.ContextLength > 0 {
				return m.ContextLength
			}
			return m.TopProvider.ContextLength
		}
	}
	return 0
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

	// Only trust loaded_context_length — the window the model is actually running
	// with. max_context_length is the model's theoretical max (often huge), which
	// is all an UNLOADED model reports; using it would size auto-compact against,
	// say, 260k instead of the real 8k. When nothing is loaded we return 0 so the
	// caller keeps its default and re-detects once the model is up.
	for _, m := range out.Data {
		if m.ID == model && m.LoadedContextLength > 0 {
			return m.LoadedContextLength
		}
	}
	for _, m := range out.Data {
		if m.State == "loaded" && m.LoadedContextLength > 0 {
			return m.LoadedContextLength
		}
	}
	return 0
}

// lmStudioModelsURL turns an OpenAI base URL (…/v1) into LM Studio's native
// models endpoint (…/api/v0/models).
func lmStudioModelsURL(baseURL string) string {
	b := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	return b + "/api/v0/models"
}
