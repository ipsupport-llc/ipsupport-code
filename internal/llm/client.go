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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"regexp"
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
	extra   map[string]any // extra top-level request params (per-model reasoning, etc.)
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

	// disableLoopDetection turns off the degenerate-repetition detectors
	// (config.LLM.DisableLoopDetection) — set once at construction. The
	// token-count runaway cap (maxRespTk) is unaffected; this only covers the
	// character- and phrase-repetition checks.
	disableLoopDetection bool

	mu           sync.Mutex
	promptTk     int
	complTk      int
	lastPromptTk int             // prompt size of the most recent request (context fullness)
	live         strings.Builder // the current (or most recent) call's live output — reasoning text if the model streams any, else the answer text as it's generated

	// liveReasoning/liveContent mirror live but keep each channel's text
	// separate, so checkPhraseRepeat can compare a channel only against its
	// own prior text — never across the reasoning→content boundary, where a
	// model's reasoning legitimately drafts its final sentence and the
	// content phase then restates that same sentence verbatim (a common CoT
	// pattern), which is not a repetition loop.
	liveReasoning strings.Builder
	liveContent   strings.Builder
}

// deadlineConn arms a fresh read deadline before every Read, so a single read
// can't block longer than idle. A read that returns re-arms the next one, so a
// live stream is unaffected; a wedged read is force-woken with a deadline error
// (which send() treats as retriable). The netpoller enforces this regardless of
// socket/kernel state — unlike ctx-cancel, nothing else has to fire. (The timer
// is monotonic, so on a laptop suspend it counts from RESUME, not during sleep.)
type deadlineConn struct {
	net.Conn
	idle time.Duration
}

func (c *deadlineConn) Read(b []byte) (int, error) {
	_ = c.Conn.SetReadDeadline(time.Now().Add(c.idle))
	return c.Conn.Read(b)
}

// NewOpenAIClient builds a client from LLM config (LM Studio by default). The
// http client has no TOTAL timeout on purpose — a long generation can run for
// minutes; the idle watchdog in send() guards against a silent stream. But the
// transport DOES bound the connect/handshake/response-header phases so a dead
// connection (e.g. after the laptop sleeps) can't wedge a request forever —
// short of that, a stuck socket read ignores context cancellation.
func NewOpenAIClient(c config.LLM) *OpenAIClient {
	idle := 90 * time.Second
	if c.IdleTimeoutSeconds > 0 {
		idle = time.Duration(c.IdleTimeoutSeconds) * time.Second
	}
	return &OpenAIClient{
		baseURL: strings.TrimRight(c.BaseURL, "/"),
		model:   c.Model,
		apiKey:  c.APIKey,
		temp:    c.Temperature,
		extra:   c.Extra,
		hc: &http.Client{
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
				// Wrap the conn with a ROLLING read deadline: every read is bounded by
				// the idle window, re-armed as data flows. This is the self-contained
				// floor under the idle watchdog — a wedged read (e.g. a dead socket
				// after the laptop sleeps) wakes on the netpoller's own timer without
				// depending on ctx propagation or the transport closing the right conn.
				DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
					cn, err := (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext(ctx, network, addr)
					if err != nil {
						return nil, err
					}
					return &deadlineConn{Conn: cn, idle: idle}, nil
				},
				ForceAttemptHTTP2:     true,
				MaxIdleConns:          100,
				IdleConnTimeout:       60 * time.Second, // drop pooled conns so a post-sleep zombie isn't reused
				TLSHandshakeTimeout:   30 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
				ResponseHeaderTimeout: idle, // no response headers within the idle window → abort
			},
		},
		idle:                 idle,
		maxRespTk:            maxResponseTokens(c.ContextWindow),
		disableLoopDetection: c.DisableLoopDetection,
	}
}

// maxResponseTokens is the per-turn generation cap: a generous 32k floor, or 4×
// the context window when that's larger (big-context models may legitimately
// write more). A turn beyond this is looping, not working.
func maxResponseTokens(ctxWindow int) int {
	limit := 32768
	if n := 4 * ctxWindow; n > limit {
		limit = n
	}
	return limit
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

// argString decodes a tool call's "arguments" field. Per the OpenAI spec it's a
// JSON-encoded STRING — but some gateways (e.g. an Anthropic→OpenAI converter
// passing tool_use.input through) emit it as a raw OBJECT. Rejecting that made
// the whole chunk unparseable, silently dropping the arguments: the model then
// looked like it called the tool with no action at all. Accept both shapes;
// either way the value becomes the raw JSON text the tool-args parser expects.
// Marshals back as a plain string, so outgoing requests are unchanged.
type argString string

func (s *argString) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' { // spec shape: a JSON string
		var str string
		if err := json.Unmarshal(data, &str); err != nil {
			return err
		}
		*s = argString(str)
		return nil
	}
	if string(data) == "null" {
		*s = ""
		return nil
	}
	*s = argString(data) // object/array: keep its raw JSON text
	return nil
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string    `json:"name"`
		Arguments argString `json:"arguments"`
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
	// Per-model reasoning controls (and any other extra params) — the user supplies
	// the provider's own shape; we just merge it in. Doesn't clobber core fields.
	for k, v := range c.extra {
		if _, core := body[k]; !core {
			body[k] = v
		}
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
	const maxAttempts = 8 // ride out a longer network glitch before giving up (it's the internet)
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// reqCompl counts THIS attempt's own live per-delta bumps (c.bumpToken),
		// so a retriable failure can roll back exactly this attempt's own
		// contribution before the retry re-counts it (otherwise the failed
		// attempt's tokens are double-counted) — a delta, like the success-path
		// reconciliation below, not a snapshot-restore of the whole shared
		// counter (which would also wipe out a DIFFERENT concurrent Chat()
		// call's legitimate progress made while this attempt was in flight).
		var reqCompl int
		msg, err, retriable := c.send(ctx, buf, &reqCompl)
		if err == nil {
			return msg, nil
		}
		lastErr = err
		if !retriable || ctx.Err() != nil || attempt == maxAttempts {
			break
		}
		c.rollbackCompletionCount(reqCompl) // undo only this attempt's own bumped estimate
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

// runawayError marks a token-cap runaway — the model looping in its own
// reasoning. Unlike a network error, retrying it won't help, so it's not
// retriable.
type runawayError struct{ max int }

func (e *runawayError) Error() string {
	return fmt.Sprintf("the model generated over %d tokens in one turn without finishing — it's looping in its own reasoning, not making progress. Try a stronger model, give it more context, or rephrase the task", e.max)
}

// degenerateRunThreshold is how many times the SAME rune must repeat back to
// back in the model's live output before it's treated as a stuck, degenerate
// loop rather than legitimate content — high enough that real output
// (a dashed separator, repeated braces/whitespace in code, ASCII art)
// essentially never crosses it, but far below runawayError's much larger
// token-count cap, catching it in seconds instead of the many minutes a
// model can spend looping on one character before that cap ever fires
// (observed live: 14m52s and 11.2k tokens in, still short of a 32k cap).
const degenerateRunThreshold = 300

// degenerateOutputError marks the model looping on a single repeated
// character — obviously worthless output from the very first run of
// repeats, not a legitimate long generation. Like runawayError, retrying
// won't help.
type degenerateOutputError struct {
	r rune
	n int
}

func (e *degenerateOutputError) Error() string {
	return fmt.Sprintf("the model got stuck repeating the same character (%q) %d times in a row instead of making progress — it's looping, not thinking. Try a stronger model, or rephrase the task", e.r, e.n)
}

// phraseRepeatMatchLen/phraseRepeatWindow catch a DIFFERENT collapse than
// degenerateRunThreshold: a model stuck re-emitting the same SENTENCE or
// phrase, not a single repeated character. Observed live: the same ~150-byte
// sentence ("We decided to use Q-Learning (Reinforcement Learning) instead
// of a simple Perceptron...") repeating back to back in the live output. An
// exact match this long, found this close to where it just occurred, is
// essentially never legitimate — a genuinely repeated code idiom or prose
// phrase is either much shorter than this or spaced much further apart in
// real content; requiring the match to be found within a short, IMMEDIATE
// trailing window (not just "somewhere earlier in the whole response") is
// what tells a true degenerate loop apart from normal reuse.
const (
	phraseRepeatMatchLen = 80
	phraseRepeatWindow   = 400
)

// phraseRepeatError marks the model looping on a repeated phrase/sentence.
// Like runawayError and degenerateOutputError, retrying won't help.
type phraseRepeatError struct{ phrase string }

func (e *phraseRepeatError) Error() string {
	shown, _ := textutil.Clip(e.phrase, 60)
	return fmt.Sprintf("the model got stuck repeating the same phrase over and over instead of making progress — it's looping, not thinking (%q…). Try a stronger model, or rephrase the task", shown)
}

// checkPhraseRepeat reports a degenerate loop when the trailing
// phraseRepeatMatchLen bytes of ONE CHANNEL's live output so far (reasoning
// text or content text — never both mixed together) also occur earlier
// within that same channel's preceding phraseRepeatWindow bytes.
func checkPhraseRepeat(all string) error {
	if len(all) < phraseRepeatMatchLen*2 {
		return nil
	}
	tail := all[len(all)-phraseRepeatMatchLen:]
	if isSingleRune(tail) {
		// A run of one repeated character (a dashed separator, a row of "=")
		// trivially "matches" any earlier same-length window of itself — that's
		// not a repeated PHRASE, and it's already degenerateRunThreshold's job
		// to catch it (at its own, much higher bar) if it's actually a loop.
		return nil
	}
	winStart := len(all) - phraseRepeatMatchLen - phraseRepeatWindow
	if winStart < 0 {
		winStart = 0
	}
	haystack := all[winStart : len(all)-phraseRepeatMatchLen]
	if strings.Contains(haystack, tail) {
		return &phraseRepeatError{tail}
	}
	return nil
}

// isSingleRune reports whether s consists of a single rune repeated
// throughout (or is empty).
func isSingleRune(s string) bool {
	var first rune
	set := false
	for _, r := range s {
		if !set {
			first, set = r, true
			continue
		}
		if r != first {
			return false
		}
	}
	return true
}

// send makes one attempt; the bool reports whether the failure is worth a retry.
func (c *OpenAIClient) send(ctx context.Context, buf []byte, reqCompl *int) (Message, error, bool) {
	c.mu.Lock()
	c.live.Reset() // this attempt's own live output, not a dropped attempt's leftovers
	c.liveReasoning.Reset()
	c.liveContent.Reset()
	c.mu.Unlock()

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
		m, err := c.parseStream(resp.Body, tick, c.maxRespTk, reqCompl)
		if err != nil {
			var re *runawayError
			var de *degenerateOutputError
			var pe *phraseRepeatError
			switch {
			case errors.As(err, &re), errors.As(err, &de), errors.As(err, &pe):
				return m, err, false // model looping — a retry won't help
			case stalled():
				return Message{}, fmt.Errorf("llm stream stalled (no data for %s)", c.idle), true
			default:
				// A mid-stream read error (connection reset, unexpected EOF) is a
				// transient network problem — retry, unless the USER cancelled.
				return Message{}, err, ctx.Err() == nil
			}
		}
		return m, nil, false
	}
	m, err := c.parseJSON(resp.Body) // server ignored stream (e.g. a test fake)
	return m, err, false
}

// oneLine collapses a (possibly HTML) error body to a short single line.
func oneLine(s string) string { return textutil.OneLine(s, 150) }

// parseStream reads an SSE stream, accumulating content and tool calls, and ticks
// the live completion-token counter as deltas arrive so the UI updates in real
// time (LM Studio sends roughly one token per chunk). The final usage chunk
// reconciles the estimate with the real count.
func (c *OpenAIClient) parseStream(r io.Reader, tick func(), maxTk int, reqCompl *int) (Message, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var content strings.Builder
	calls := map[int]*ToolCall{}
	var order []int
	// progress marks a REAL token delta: it counts it and pushes the idle deadline
	// back. Only real progress resets the watchdog — SSE heartbeats / keep-alive
	// comments (": ping", blank lines from proxies) must NOT, or a wedged-but-
	// heartbeating stream would "think" forever without ever producing a token.
	// *reqCompl is this attempt's own bump count, owned by the caller (Chat's
	// retry loop) so a retriable failure can roll back exactly this attempt's
	// contribution to the shared c.complTk — see Chat.
	progress := func() {
		(*reqCompl)++
		c.bumpToken()
		tick()
	}
	// degenerate-repetition detector, shared across content and reasoning text
	// (whichever the model is currently emitting) — see degenerateRunThreshold.
	var lastRune rune
	var runLen int
	checkDegenerate := func(s string) error {
		for _, r := range s {
			if r == lastRune {
				runLen++
			} else {
				lastRune, runLen = r, 1
			}
			if runLen >= degenerateRunThreshold {
				return &degenerateOutputError{lastRune, runLen}
			}
		}
		return nil
	}
	done := false // only set at a real "[DONE]" — see the check after the loop
	for sc.Scan() {
		// reqCompl counts stream deltas, not true tokens (there's no per-chunk token
		// count mid-stream). LM Studio sends ~1 token/chunk so the cap is accurate
		// there; a server that batches many tokens per chunk trips it late. It's a
		// runaway backstop, not a billing meter — the usage chunk reconciles the real
		// count — so an approximate trigger point is acceptable.
		if maxTk > 0 && *reqCompl > maxTk {
			return Message{}, &runawayError{maxTk}
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "[DONE]" {
			done = true
			break
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(payload), &ch); err != nil {
			// A chunk we can't parse is DROPPED — if that ever bites again (a
			// gateway inventing a new shape), IPS_LOG=debug shows the evidence.
			slog.Debug("stream chunk unparsed", "err", err, "payload", textutil.OneLine(payload, 240))
			continue
		}
		if len(ch.Choices) > 0 {
			d := ch.Choices[0].Delta
			if d.Content != "" {
				content.WriteString(d.Content)
				// Also buffer into the live-output view (see Live) — a plain
				// model with no separate reasoning phase (most local models,
				// e.g. gemma) never sends reasoning_content/reasoning at all, so
				// the live view has to fall back to the actual answer text as it
				// streams, or it would show nothing while tokens are visibly
				// still ticking up (reported live: "counter grows, screen empty
				// — Open WebUI shows what it's doing" for exactly this model).
				c.mu.Lock()
				c.live.WriteString(d.Content)
				c.liveContent.WriteString(d.Content)
				chanAll := c.liveContent.String()
				c.mu.Unlock()
				if !c.disableLoopDetection {
					if err := checkDegenerate(d.Content); err != nil {
						return Message{}, err
					}
					if err := checkPhraseRepeat(chanAll); err != nil {
						return Message{}, err
					}
				}
				progress()
			}
			// Count reasoning deltas toward live progress (reconciled to the
			// server's real total by the usage chunk), and buffer the text itself
			// so a UI tick can show it live (see Live) — never sent to the model
			// or included in the final Message, just for display.
			if rc := d.ReasoningContent; rc != "" {
				c.mu.Lock()
				c.live.WriteString(rc)
				c.liveReasoning.WriteString(rc)
				chanAll := c.liveReasoning.String()
				c.mu.Unlock()
				if !c.disableLoopDetection {
					if err := checkDegenerate(rc); err != nil {
						return Message{}, err
					}
					if err := checkPhraseRepeat(chanAll); err != nil {
						return Message{}, err
					}
				}
				progress()
			} else if rc := d.Reasoning; rc != "" {
				c.mu.Lock()
				c.live.WriteString(rc)
				c.liveReasoning.WriteString(rc)
				chanAll := c.liveReasoning.String()
				c.mu.Unlock()
				if !c.disableLoopDetection {
					if err := checkDegenerate(rc); err != nil {
						return Message{}, err
					}
					if err := checkPhraseRepeat(chanAll); err != nil {
						return Message{}, err
					}
				}
				progress()
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
					cur.Arguments += string(tc.Function.Arguments)
					progress()
				}
			}
		}
		if ch.Usage != nil {
			c.mu.Lock()
			c.promptTk += ch.Usage.PromptTokens
			c.complTk += ch.Usage.CompletionTokens - *reqCompl
			if ch.Usage.PromptTokens > 0 {
				c.lastPromptTk = ch.Usage.PromptTokens
			}
			c.mu.Unlock()
		}
	}
	if err := sc.Err(); err != nil {
		return Message{}, err
	}
	if !done {
		// The connection closed cleanly (bufio.Scanner's own EOF, not an error)
		// before a "[DONE]" ever arrived — a proxy or the server itself cut the
		// stream mid-generation. Without this check, whatever partial content/
		// tool-call arguments had accumulated so far was returned as a normal,
		// complete answer: a truncated final message, or a half-formed tool
		// call argument string, executed as if the model had actually finished.
		// The caller's retry classification (client.go's roundTrip) already
		// treats a plain error like this as a transient, retriable mid-stream
		// failure — same bucket as a connection reset.
		return Message{}, fmt.Errorf("llm stream ended without completing (no [DONE])")
	}
	msg := Message{Role: "assistant", Content: stripChannelTokens(content.String())}
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
	msg := fromWire(out.Choices[0].Message)
	msg.Content = stripChannelTokens(msg.Content)
	return msg, nil
}

func (c *OpenAIClient) bumpToken() {
	c.mu.Lock()
	c.complTk++
	c.mu.Unlock()
}

// rollbackCompletionCount undoes a failed attempt's own contribution to the
// running completion estimate: a delta of exactly what THAT attempt bumped via
// bumpToken, not a snapshot-restore of the whole shared counter — the same
// delta pattern parseStream's success-path reconciliation uses, applied to the
// failure path too (see Chat's retry loop).
func (c *OpenAIClient) rollbackCompletionCount(n int) {
	c.mu.Lock()
	c.complTk -= n
	c.mu.Unlock()
}

// channelTokenPattern matches OpenAI Harmony-style control tokens
// (<|start|>, <|end|>, <|message|>, <|channel|>, <|constrain|>, <|return|>,
// <|call|>, <|refusal|>) — including the leading-pipe-dropped form some
// quantized local models emit (e.g. "<channel|>" instead of "<|channel|>").
// A server/template that properly separates the model's reasoning/final
// channels never puts these in the plain content field; one that doesn't
// leaks them verbatim into what the user sees. The vocabulary is small and
// fixed and never occurs in ordinary prose, so stripping it can't eat real
// content.
var channelTokenPattern = regexp.MustCompile(`<\|?(start|end|message|channel|constrain|return|call|refusal)\|>`)

// stripChannelTokens removes leaked Harmony-style control tokens from model
// output. See channelTokenPattern.
func stripChannelTokens(s string) string {
	if !strings.Contains(s, "|>") {
		return s // fast path: this exact substring never appears in normal prose
	}
	return channelTokenPattern.ReplaceAllString(s, "")
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
					Name      string    `json:"name"`
					Arguments argString `json:"arguments"`
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

// Live returns the output streamed so far for the most recent call — live, so
// a caller can poll it on a UI tick to show what the model is currently doing
// while it's still generating. This is reasoning text for a model that
// streams a separate reasoning phase (reasoning_content/reasoning); for a
// plain model with no such phase (most local models), it's the answer text
// itself as it's generated — otherwise the live view would show nothing while
// tokens are visibly still ticking up. Reset at the start of every send
// attempt, so a retry doesn't mix a dropped attempt's partial text into the
// next one's.
func (c *OpenAIClient) Live() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.live.String()
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
		wtc.Function.Arguments = argString(tc.Arguments)
		w.ToolCalls = append(w.ToolCalls, wtc)
	}
	return w
}

func fromWire(w wireMessage) Message {
	m := Message{Role: w.Role, Content: w.Content, ToolCallID: w.ToolCallID, Name: w.Name}
	for _, tc := range w.ToolCalls {
		m.ToolCalls = append(m.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: validArgs(string(tc.Function.Arguments))})
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
