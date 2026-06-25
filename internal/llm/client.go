// Package llm is the model boundary. Everything that reasons or reflects depends
// only on the Chatter interface, so the concrete backend — LM Studio, OpenAI, a
// LiteLLM proxy (all OpenAI-compatible, swapped by base_url/api_key), or an
// Anthropic adapter — is interchangeable.
package llm

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
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
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

	mu       sync.Mutex
	promptTk int
	complTk  int
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
		"model":          c.model,
		"temperature":    c.temp,
		"messages":       wm,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
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
	req.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("llm http %d: %s", resp.StatusCode, truncate(string(data), 500))
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		return c.parseStream(resp.Body)
	}
	return c.parseJSON(resp.Body) // server ignored stream (e.g. a test fake)
}

// parseStream reads an SSE stream, accumulating content and tool calls, and ticks
// the live completion-token counter as deltas arrive so the UI updates in real
// time (LM Studio sends roughly one token per chunk). The final usage chunk
// reconciles the estimate with the real count.
func (c *OpenAIClient) parseStream(r io.Reader) (Message, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var content strings.Builder
	calls := map[int]*ToolCall{}
	var order []int
	reqCompl := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "[DONE]" {
			break
		}
		var ch streamChunk
		if json.Unmarshal([]byte(payload), &ch) != nil {
			continue
		}
		if len(ch.Choices) > 0 {
			d := ch.Choices[0].Delta
			if d.Content != "" {
				content.WriteString(d.Content)
				reqCompl++
				c.bumpToken()
			}
			for _, tc := range d.ToolCalls {
				cur, ok := calls[tc.Index]
				if !ok {
					cur = &ToolCall{}
					calls[tc.Index] = cur
					order = append(order, tc.Index)
				}
				if tc.ID != "" {
					cur.ID = tc.ID
				}
				if tc.Function.Name != "" {
					cur.Name = tc.Function.Name
				}
				if tc.Function.Arguments != "" {
					cur.Arguments += tc.Function.Arguments
					reqCompl++
					c.bumpToken()
				}
			}
		}
		if ch.Usage != nil {
			c.mu.Lock()
			c.promptTk += ch.Usage.PromptTokens
			c.complTk += ch.Usage.CompletionTokens - reqCompl
			c.mu.Unlock()
		}
	}
	if err := sc.Err(); err != nil {
		return Message{}, err
	}
	msg := Message{Role: "assistant", Content: content.String()}
	for _, idx := range order {
		msg.ToolCalls = append(msg.ToolCalls, *calls[idx])
	}
	return msg, nil
}

func (c *OpenAIClient) parseJSON(r io.Reader) (Message, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Message{}, err
	}
	var out struct {
		Choices []struct {
			Message wireMessage `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return Message{}, fmt.Errorf("decode llm response: %w", err)
	}
	if len(out.Choices) == 0 {
		return Message{}, fmt.Errorf("llm returned no choices")
	}
	c.mu.Lock()
	c.promptTk += out.Usage.PromptTokens
	c.complTk += out.Usage.CompletionTokens
	c.mu.Unlock()
	return fromWire(out.Choices[0].Message), nil
}

func (c *OpenAIClient) bumpToken() {
	c.mu.Lock()
	c.complTk++
	c.mu.Unlock()
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// Usage returns cumulative prompt/completion tokens reported by the server
// across this client's lifetime (zero if the server omits usage).
func (c *OpenAIClient) Usage() (prompt, completion int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.promptTk, c.complTk
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
	out, _ := textutil.Clip(s, n)
	return out
}
