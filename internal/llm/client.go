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

	// OnRetry, if set, is called before each backoff so the UI can show that
	// we're retrying/backing off (e.g. while LM Studio reloads an unloaded
	// model) rather than just "thinking".
	OnRetry func(attempt int, wait time.Duration, reason string)

	// idle bounds how long we wait with NO response or stream data before giving
	// up on a request and retrying. It's an idle (not total) deadline, so a slow-
	// but-progressing generation isn't killed, while a server that went silent
	// (LM Studio idle, connection still open) fails fast instead of hanging
	// forever. Set once at construction; tests shorten it before use.
	idle time.Duration

	// maxRespTk caps the tokens one turn may generate before we abort it. A turn
	// producing far more than the context window has stopped making progress —
	// usually a reasoning model looping in its own monologue — and would otherwise
	// stream for many minutes. Derived from the context window at construction.
	maxRespTk int

	mu           sync.Mutex
	promptTk     int
	complTk      int
	lastPromptTk int // prompt size of the most recent request (context fullness)
}

// NewOpenAIClient builds a client from LLM config (LM Studio by default). The
// http client has no total timeout on purpose — a long generation can run for
// minutes; the idle watchdog in send() is what guards against a true hang.
func NewOpenAIClient(c config.LLM) *OpenAIClient {
	return &OpenAIClient{
		baseURL:   strings.TrimRight(c.BaseURL, "/"),
		model:     c.Model,
		apiKey:    c.APIKey,
		temp:      c.Temperature,
		hc:        &http.Client{},
		idle:      90 * time.Second,
		maxRespTk: maxResponseTokens(c.ContextWindow),
	}
}

// maxResponseTokens is the per-turn generation cap: a generous 32k floor, or 4×
// the context window when that's larger (big-context models may legitimately
// write more). A turn beyond this is looping, not working.
func maxResponseTokens(ctxWindow int) int {
	cap := 32768
	if n := 4 * ctxWindow; n > cap {
		cap = n
	}
	return cap
}

// startIdleWatchdog cancels the request if neither the response nor any stream
// chunk arrives within c.idle. It returns a tick func the reader calls on each
// chunk to push the deadline back; the goroutine exits when ctx is done.
func (c *OpenAIClient) startIdleWatchdog(ctx context.Context, cancel context.CancelFunc) func() {
	d := c.idle
	reset := make(chan struct{}, 1)
	go func() {
		t := time.NewTimer(d)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-reset:
				if !t.Stop() {
					select {
					case <-t.C:
					default:
					}
				}
				t.Reset(d)
			case <-t.C:
				cancel() // gone silent — abort so the caller can retry
				return
			}
		}
	}()
	return func() {
		select {
		case reset <- struct{}{}:
		default:
		}
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
		"messages":       wm,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	// Only send temperature when explicitly set (> 0). Some newer hosted models
	// (e.g. OpenAI gpt-5.x / chat-latest) reject any non-default temperature with
	// a 400; omitting the field lets the server use its default and keeps them
	// working, while local models still honor a configured value.
	if c.temp > 0 {
		body["temperature"] = c.temp
	}
	if len(tools) > 0 {
		body["tools"] = tools
		body["tool_choice"] = "auto"
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return Message{}, err
	}

	// Local servers (LM Studio) hiccup with transient 5xx and need time to
	// reload a model unloaded by the idle timeout. Retry those (and network
	// errors) with exponential backoff instead of failing the whole task.
	const maxAttempts = 5
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		msg, err, retriable := c.send(ctx, buf)
		if err == nil {
			return msg, nil
		}
		lastErr = err
		if !retriable || ctx.Err() != nil || attempt == maxAttempts {
			break
		}
		wait := backoff(attempt)
		if c.OnRetry != nil {
			c.OnRetry(attempt, wait, err.Error())
		}
		select {
		case <-ctx.Done():
			return Message{}, ctx.Err()
		case <-time.After(wait):
		}
	}
	return Message{}, lastErr
}

// backoff grows exponentially (500ms, 1s, 2s, 4s…) capped at 8s.
func backoff(attempt int) time.Duration {
	d := 500 * time.Millisecond << (attempt - 1)
	if d > 8*time.Second {
		d = 8 * time.Second
	}
	return d
}

// send makes one attempt; the bool reports whether the failure is worth a retry.
func (c *OpenAIClient) send(ctx context.Context, buf []byte) (Message, error, bool) {
	reqCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	tick := c.startIdleWatchdog(reqCtx, cancel)

	// stalled reports a watchdog-induced cancel (vs. the user cancelling ctx), so
	// the caller knows to retry rather than abort.
	stalled := func() bool { return reqCtx.Err() != nil && ctx.Err() == nil }

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return Message{}, err, false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if strings.Contains(c.baseURL, "openrouter.ai") {
		// OpenRouter uses these to attribute traffic to the app (rankings, some
		// free-tier access). Harmless elsewhere, so only set for OpenRouter.
		req.Header.Set("HTTP-Referer", "https://github.com/ipsupport-llc/ipsupport-code")
		req.Header.Set("X-Title", "ipsupport-code")
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		if stalled() {
			return Message{}, fmt.Errorf("llm timed out (no response for %s)", c.idle), true
		}
		return Message{}, err, true // network hiccup — retry
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode >= 500:
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return Message{}, fmt.Errorf("llm server error (http %d)", resp.StatusCode), true
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusRequestTimeout:
		// Rate-limit / request-timeout is transient — back off and retry rather
		// than aborting the task (the common failure on hosted providers).
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return Message{}, fmt.Errorf("llm rate-limited (http %d)", resp.StatusCode), true
	case resp.StatusCode >= 400:
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Message{}, fmt.Errorf("llm http %d: %s", resp.StatusCode, oneLine(string(data))), false
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		m, err := c.parseStream(resp.Body, tick, c.maxRespTk)
		if err != nil && stalled() {
			return Message{}, fmt.Errorf("llm stream stalled (no data for %s)", c.idle), true
		}
		return m, err, false // runaway / parse errors are not retriable
	}
	m, err := c.parseJSON(resp.Body) // server ignored stream (e.g. a test fake)
	return m, err, false
}

// oneLine collapses a (possibly HTML) error body to a short single line.
func oneLine(s string) string {
	return truncate(strings.Join(strings.Fields(s), " "), 150)
}

// parseStream reads an SSE stream, accumulating content and tool calls, and ticks
// the live completion-token counter as deltas arrive so the UI updates in real
// time (LM Studio sends roughly one token per chunk). The final usage chunk
// reconciles the estimate with the real count.
func (c *OpenAIClient) parseStream(r io.Reader, tick func(), maxTk int) (Message, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var content strings.Builder
	calls := map[int]*ToolCall{}
	var order []int
	reqCompl := 0
	for sc.Scan() {
		tick() // data arrived — push the idle deadline back
		if maxTk > 0 && reqCompl > maxTk {
			return Message{}, fmt.Errorf("the model generated over %d tokens in one turn without finishing — it's looping in its own reasoning, not making progress. Try a stronger model, give it more context, or rephrase the task", maxTk)
		}
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
			// Count reasoning deltas toward live progress (reconciled to the
			// server's real total by the usage chunk) — but don't show them.
			if d.ReasoningContent != "" || d.Reasoning != "" {
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
			if ch.Usage.PromptTokens > 0 {
				c.lastPromptTk = ch.Usage.PromptTokens
			}
			c.mu.Unlock()
		}
	}
	if err := sc.Err(); err != nil {
		return Message{}, err
	}
	msg := Message{Role: "assistant", Content: content.String()}
	for _, idx := range order {
		c := *calls[idx]
		c.Arguments = validArgs(c.Arguments)
		msg.ToolCalls = append(msg.ToolCalls, c)
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
	if out.Usage.PromptTokens > 0 {
		c.lastPromptTk = out.Usage.PromptTokens
	}
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
			Content string `json:"content"`
			// Reasoning models stream their hidden thinking here before any
			// content/tool calls (xAI: reasoning_content; OpenRouter: reasoning).
			// We don't show it, but we count it so the UI shows live progress
			// instead of looking frozen for minutes.
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
			ToolCalls        []struct {
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

// SeedUsage carries the running token totals from a previous client, so
// rebuilding the stack (a /skills or /permissions toggle, /login) doesn't zero
// the session's cumulative count.
func (c *OpenAIClient) SeedUsage(prompt, completion int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.promptTk, c.complTk = prompt, completion
}

// Context returns the prompt size of the most recent request — a proxy for how
// full the model's context window is right now.
func (c *OpenAIClient) Context() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastPromptTk
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
		m.ToolCalls = append(m.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: validArgs(tc.Function.Arguments)})
	}
	return m
}

// validArgs normalizes a tool call's arguments to valid JSON. Small models
// sometimes emit empty or malformed arguments; echoing those back in the
// conversation can break a server's chat template (LM Studio 500s), so coerce
// them to "{}" — the dispatcher then reports the missing action/params normally.
func validArgs(s string) string {
	if json.Valid([]byte(s)) {
		return s
	}
	return "{}"
}

func truncate(s string, n int) string {
	out, _ := textutil.Clip(s, n)
	return out
}
