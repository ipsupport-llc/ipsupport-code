package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
)

// The /responses wire format (OpenAI's Responses API), for models a server
// serves only there. The conversation is sent whole every turn (store:false),
// exactly as /chat/completions is: nothing is kept on the server between turns.

// responsesBody builds a /responses request from the same conversation and
// tool catalog a /chat/completions request would carry.
func (c *OpenAIClient) responsesBody(msgs []Message, tools []map[string]any, t *tuning) ([]byte, error) {
	body := map[string]any{
		"model":  c.model,
		"input":  responsesInput(msgs),
		"stream": true,
		"store":  false,
	}
	if s := leadingSystem(msgs); s != "" {
		body["instructions"] = s
	}
	if t.temp > 0 {
		body["temperature"] = t.temp
	}
	if t.topP > 0 {
		body["top_p"] = t.topP
	}
	if t.maxOutputTok > 0 {
		body["max_output_tokens"] = t.maxOutputTok
	}
	for k, v := range t.extra {
		if k == "reasoning_effort" { // the chat-completions shape of the same setting
			if _, set := t.extra["reasoning"]; !set {
				body["reasoning"] = map[string]any{"effort": v}
			}
			continue
		}
		if _, core := body[k]; !core {
			body[k] = v
		}
	}
	if len(tools) > 0 {
		body["tools"] = responsesTools(tools)
		body["tool_choice"] = "auto"
	}
	return json.Marshal(body)
}

// leadingSystem joins the system messages that open the conversation: on
// /responses they are the request's instructions.
func leadingSystem(msgs []Message) string {
	var parts []string
	for _, m := range msgs {
		if m.Role != "system" {
			break
		}
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n\n")
}

// responsesInput turns the conversation into /responses input items. A tool
// call goes back as a function_call item without the server's own item id:
// with store:false that id refers to nothing the server kept, and a
// reasoning model's server rejects a call item whose id points at a reasoning
// item that isn't sent along.
func responsesInput(msgs []Message) []map[string]any {
	items := []map[string]any{}
	i := 0
	for i < len(msgs) && msgs[i].Role == "system" {
		i++ // sent as instructions
	}
	for _, m := range msgs[i:] {
		switch m.Role {
		case "tool":
			items = append(items, map[string]any{"type": "function_call_output", "call_id": m.ToolCallID, "output": m.Content})
		case "assistant":
			if m.Content != "" {
				items = append(items, map[string]any{"type": "message", "role": "assistant", "content": m.Content})
			}
			for _, tc := range m.ToolCalls {
				items = append(items, map[string]any{"type": "function_call", "call_id": tc.ID, "name": tc.Name, "arguments": validArgs(tc.Arguments)})
			}
		default: // user, and a system note further down the conversation
			items = append(items, map[string]any{"type": "message", "role": m.Role, "content": m.Content})
		}
	}
	return items
}

// responsesTools flattens chat-completions tool definitions
// ({"type":"function","function":{name,…}}) into the /responses shape
// ({"type":"function","name":…}). strict is off unless a definition asks for
// it: /responses defaults it on, and strict mode refuses any schema with an
// optional parameter.
func responsesTools(tools []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		fn, ok := t["function"].(map[string]any)
		if !ok {
			out = append(out, t)
			continue
		}
		ft := map[string]any{"type": "function", "strict": false}
		for k, v := range fn {
			ft[k] = v
		}
		out = append(out, ft)
	}
	return out
}

// responsesEvent is one /responses stream event — only the fields read here.
type responsesEvent struct {
	Type    string          `json:"type"`
	Delta   string          `json:"delta"`
	ItemID  string          `json:"item_id"`
	Index   *int            `json:"output_index"`
	Args    *argString      `json:"arguments"`
	Item    *responsesItem  `json:"item"`
	Resp    *responsesReply `json:"response"`
	Code    any             `json:"code"`
	Message string          `json:"message"`
}

type responsesItem struct {
	Type      string    `json:"type"`
	ID        string    `json:"id"`
	CallID    string    `json:"call_id"`
	Name      string    `json:"name"`
	Arguments argString `json:"arguments"`
	Role      string    `json:"role"`
	Content   []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

type responsesReply struct {
	Status            string          `json:"status"`
	Model             string          `json:"model"`
	Output            []responsesItem `json:"output"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Usage *responsesUsage `json:"usage"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// finishReason maps a finished response's status onto the chat-completions
// finish_reason the rest of the program reads.
func (r *responsesReply) finishReason(calls bool) string {
	if r.Status == "incomplete" {
		if r.IncompleteDetails != nil && r.IncompleteDetails.Reason == "max_output_tokens" {
			return "length"
		}
		if r.IncompleteDetails != nil && r.IncompleteDetails.Reason != "" {
			return r.IncompleteDetails.Reason
		}
		return "length"
	}
	if calls {
		return "tool_calls"
	}
	return "stop"
}

// responsesFailure turns an error the server reported into the same error the
// chat-completions stream gives for it, so retries treat both alike.
func responsesFailure(code any, msg string) error {
	if transientStreamError(nil, code, msg) {
		return fmt.Errorf("llm stream: the server reported a transient error mid-stream: %s", msg)
	}
	return &serverStreamError{msg: msg}
}

// parseResponsesStream is parseStream for a /responses event stream. The
// stream is complete only at response.completed or response.incomplete: one
// that ends before either was cut off, whatever it carried.
func (c *OpenAIClient) parseResponsesStream(r io.Reader, tick, alive func(), maxTk int, reqCompl *int, promptEstimate int) (Message, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	started := false
	progress := func() {
		started = true
		(*reqCompl)++
		c.bumpToken()
		tick()
	}
	text := &streamText{c: c, progress: progress}
	// A call is kept under its call_id: the one id servers hold steady (GitHub
	// Copilot reportedly gives an item a new id on every event). An event that
	// carries no call_id — argument deltas and their done — finds its call by
	// item id, else by output_index, else it belongs to the call streaming now.
	calls := map[string]*ToolCall{}
	var order []string
	byItem, byIndex, last := map[string]string{}, map[int]string{}, ""
	find := func(itemID string, idx *int) string {
		if k, ok := byItem[itemID]; ok && itemID != "" {
			return k
		}
		if idx != nil {
			return byIndex[*idx]
		}
		return ""
	}
	at := func(key, itemID string, idx *int) *ToolCall {
		cur, ok := calls[key]
		if !ok {
			cur = &ToolCall{}
			calls[key] = cur
			order = append(order, key)
		}
		if itemID != "" {
			byItem[itemID] = key
		}
		if idx != nil {
			byIndex[*idx] = key
		}
		last = key
		return cur
	}
	// call is the call an item describes (output_item events, the finished
	// response's output).
	call := func(it *responsesItem, idx *int) *ToolCall {
		key := it.CallID
		if _, known := calls[key]; key == "" || !known {
			if k := find(it.ID, idx); k != "" {
				key = k // the same call, seen before under its item id or index
			} else if key == "" {
				key = "item:" + it.ID
			}
		}
		cur := at(key, it.ID, idx)
		if it.CallID != "" {
			cur.ID = it.CallID // what the tool's output answers, not the item id
		}
		if it.Name != "" {
			cur.Name = it.Name
		}
		return cur
	}
	// argsOf is the call an arguments event is about.
	argsOf := func(itemID string, idx *int) *ToolCall {
		key := find(itemID, idx)
		if key == "" {
			key = last
		}
		if key == "" {
			key = "item:" + itemID
		}
		return at(key, itemID, idx)
	}
	events, dropped := 0, 0
	var final *responsesReply
	for sc.Scan() {
		if maxTk > 0 && *reqCompl > maxTk {
			return Message{}, &runawayError{maxTk}
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			// "event:" lines repeat the type the data carries. A comment before the
			// first token is a server reading the prompt — see parseStream.
			if !started && alive != nil && strings.HasPrefix(line, ":") {
				alive()
			}
			continue
		}
		payload := strings.TrimSpace(line[len("data:"):])
		if payload == "[DONE]" { // not part of /responses, but a gateway may add it
			continue
		}
		events++
		var ev responsesEvent
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			slog.Debug("responses event unparsed", "err", err, "payload", textutil.OneLine(payload, 240))
			dropped++
			continue
		}
		switch ev.Type {
		case "response.output_text.delta":
			if err := text.addContent(ev.Delta); err != nil {
				return Message{}, err
			}
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			if err := text.addReasoning(ev.Delta); err != nil {
				return Message{}, err
			}
		case "response.output_item.added", "response.output_item.done":
			if ev.Item == nil || ev.Item.Type != "function_call" {
				continue
			}
			if cur := call(ev.Item, ev.Index); ev.Item.Arguments != "" {
				cur.Arguments = string(ev.Item.Arguments) // the finished item is the whole call
			}
		case "response.function_call_arguments.delta":
			if ev.Delta != "" {
				argsOf(ev.ItemID, ev.Index).Arguments += ev.Delta
				progress()
			}
		case "response.function_call_arguments.done":
			if ev.Args != nil && *ev.Args != "" {
				argsOf(ev.ItemID, ev.Index).Arguments = string(*ev.Args)
			}
		case "response.completed", "response.incomplete":
			final = ev.Resp
			if final == nil {
				final = &responsesReply{Status: strings.TrimPrefix(ev.Type, "response.")}
			}
		case "response.failed":
			msg := "the response failed"
			var code any
			if ev.Resp != nil && ev.Resp.Error != nil {
				msg, code = ev.Resp.Error.Message, ev.Resp.Error.Code
			}
			return Message{}, responsesFailure(code, msg)
		case "error":
			return Message{}, responsesFailure(ev.Code, ev.Message)
		}
		if final != nil {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return Message{}, err
	}
	if final == nil {
		var argsLen int
		for _, id := range order {
			argsLen += len(calls[id].Arguments)
		}
		return Message{}, fmt.Errorf("llm stream ended without completing (no response.completed): %d events, %d content bytes, %d reasoning bytes, %d tool-call arg bytes accumulated before it cut off",
			events, text.content.Len(), text.reasoning.Len(), argsLen)
	}
	// The finished response lists its output again. With no calls streamed, its
	// calls are the turn's; with some, it only fills in what they lack — a call
	// it names that matches none of them is not run, since a server that gave
	// it a fresh id there would have the same call run twice.
	streamedCalls := len(order) > 0
	for i := range final.Output {
		it := final.Output[i]
		switch it.Type {
		case "function_call":
			if _, known := calls[it.CallID]; streamedCalls && !known && find(it.ID, nil) == "" {
				continue
			}
			if cur := call(&it, nil); cur.Arguments == "" {
				cur.Arguments = string(it.Arguments)
			}
		case "message":
			if text.content.Len() == 0 {
				for _, part := range it.Content {
					if part.Type == "output_text" {
						text.content.WriteString(part.Text)
					}
				}
			}
		}
	}
	if dropped > 0 && len(order) > 0 {
		return Message{}, fmt.Errorf("llm stream: %d unparseable event(s) in a turn with tool calls — not running arguments that may be incomplete", dropped)
	}
	c.commitUsage(final.Usage, *reqCompl, promptEstimate)
	msg := Message{Role: "assistant", Content: stripChannelTokens(text.content.String()), Reasoning: text.reasoning.String(), FinishReason: final.finishReason(len(order) > 0)}
	for _, id := range order {
		tc := *calls[id]
		tc.Arguments = validArgs(tc.Arguments)
		msg.ToolCalls = append(msg.ToolCalls, tc)
	}
	c.checkServed(final.Model)
	return msg, nil
}

// commitUsage records a finished /responses reply's token counts — the real
// ones when the server reported them, else the request-size estimate (see
// parseStream for why a missing report must not leave the context at 0).
// streamed is what this attempt already counted live, one per delta.
func (c *OpenAIClient) commitUsage(u *responsesUsage, streamed, promptEstimate int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if u != nil && u.InputTokens > 0 {
		c.promptTk += u.InputTokens
		c.complTk += u.OutputTokens - streamed
		c.lastPromptTk = u.InputTokens
		return
	}
	c.promptTk += promptEstimate
	c.lastPromptTk = promptEstimate
}

// parseResponsesJSON reads a /responses reply sent whole instead of streamed.
func (c *OpenAIClient) parseResponsesJSON(r io.Reader, promptEstimate int) (Message, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Message{}, err
	}
	var out responsesReply
	if err := json.Unmarshal(data, &out); err != nil {
		return Message{}, fmt.Errorf("decode llm response: %w", err)
	}
	if out.Status == "failed" {
		msg := "the response failed"
		var code any
		if out.Error != nil {
			msg, code = out.Error.Message, out.Error.Code
		}
		return Message{}, responsesFailure(code, msg)
	}
	msg := Message{Role: "assistant"}
	var content strings.Builder
	for _, it := range out.Output {
		switch it.Type {
		case "message":
			for _, part := range it.Content {
				if part.Type == "output_text" {
					content.WriteString(part.Text)
				}
			}
		case "function_call":
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{ID: it.CallID, Name: it.Name, Arguments: validArgs(string(it.Arguments))})
		}
	}
	msg.Content = stripChannelTokens(content.String())
	msg.FinishReason = out.finishReason(len(msg.ToolCalls) > 0)
	if out.Usage == nil || out.Usage.InputTokens == 0 {
		c.mu.Lock()
		c.complTk += len(data) / 4 // nothing was counted live on this path
		c.mu.Unlock()
	}
	c.commitUsage(out.Usage, 0, promptEstimate)
	c.checkServed(out.Model)
	return msg, nil
}
