// Package llm is the model boundary. Everything that reasons or reflects depends
// only on the Chatter interface, so the concrete backend — LM Studio, OpenAI, a
// LiteLLM proxy (all OpenAI-compatible, swapped by base_url/api_key), or an
// Anthropic adapter — is interchangeable.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// ToolCall is a function call the model wants to make. Arguments is the raw JSON
// argument string exactly as the model emitted it.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// Message is one chat message in either direction.
type Message struct {
	Role       string // system | user | assistant | tool
	Content    string
	ToolCalls  []ToolCall // assistant → tools
	ToolCallID string     // tool → which call this answers
	Name       string     // tool name (on role=tool)
}

// Chatter is the one-method abstraction over any chat model with tool calling.
type Chatter interface {
	Chat(ctx context.Context, msgs []Message, tools []map[string]any) (Message, error)
}

// Convenience constructors.
func System(content string) Message { return Message{Role: "system", Content: content} }
func User(content string) Message   { return Message{Role: "user", Content: content} }
func ToolResult(callID, name, content string) Message {
	return Message{Role: "tool", ToolCallID: callID, Name: name, Content: content}
}

// OpenAIClient talks to an OpenAI-compatible /chat/completions endpoint.
type OpenAIClient struct {
	baseURL string
	model   string
	apiKey  string
	temp    float64
	hc      *http.Client
}

// NewOpenAIClient builds a client from LLM config (LM Studio by default).
func NewOpenAIClient(c config.LLM) *OpenAIClient {
	return &OpenAIClient{
		baseURL: strings.TrimRight(c.BaseURL, "/"),
		model:   c.Model,
		apiKey:  c.APIKey,
		temp:    c.Temperature,
		hc:      &http.Client{Timeout: 120 * time.Second},
	}
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

// Chat sends the conversation (and tool catalog) and returns the model's reply.
func (c *OpenAIClient) Chat(ctx context.Context, msgs []Message, tools []map[string]any) (Message, error) {
	wm := make([]wireMessage, len(msgs))
	for i, m := range msgs {
		wm[i] = toWire(m)
	}

	body := map[string]any{
		"model":       c.model,
		"temperature": c.temp,
		"messages":    wm,
	}
	if len(tools) > 0 {
		body["tools"] = tools
		body["tool_choice"] = "auto"
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return Message{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return Message{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, err
	}
	if resp.StatusCode >= 400 {
		return Message{}, fmt.Errorf("llm http %d: %s", resp.StatusCode, truncate(string(data), 500))
	}

	var out struct {
		Choices []struct {
			Message wireMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return Message{}, fmt.Errorf("decode llm response: %w", err)
	}
	if len(out.Choices) == 0 {
		return Message{}, fmt.Errorf("llm returned no choices")
	}
	return fromWire(out.Choices[0].Message), nil
}

func toWire(m Message) wireMessage {
	w := wireMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID, Name: m.Name}
	for _, tc := range m.ToolCalls {
		var wtc wireToolCall
		wtc.ID = tc.ID
		wtc.Type = "function"
		wtc.Function.Name = tc.Name
		wtc.Function.Arguments = tc.Arguments
		w.ToolCalls = append(w.ToolCalls, wtc)
	}
	return w
}

func fromWire(w wireMessage) Message {
	m := Message{Role: w.Role, Content: w.Content, ToolCallID: w.ToolCallID, Name: w.Name}
	for _, tc := range w.ToolCalls {
		m.ToolCalls = append(m.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	return m
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
