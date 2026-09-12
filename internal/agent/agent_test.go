package agent

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// scriptLLM returns a fixed sequence of replies and records the last messages
// it was given.
type scriptLLM struct {
	replies  []llm.Message
	i        int
	lastMsgs []llm.Message
}

func (s *scriptLLM) Chat(_ context.Context, msgs []llm.Message, _ []map[string]any) (llm.Message, error) {
	s.lastMsgs = msgs
	if s.i >= len(s.replies) {
		return llm.Message{Role: "assistant", Content: "(no more replies)"}, nil
	}
	m := s.replies[s.i]
	s.i++
	return m, nil
}

func toolCallReply(id, name, args string) llm.Message {
	return llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: id, Name: name, Arguments: args}}}
}

func toolObservation(msgs []llm.Message) []llm.Message {
	var out []llm.Message
	for _, m := range msgs {
		if m.Role == "tool" {
			out = append(out, m)
		}
	}
	return out
}

func TestRunFiresToolThenFinal(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		{Role: "assistant", Content: "the answer is 4"},
	}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, err := a.Run(context.Background(), "what is 2+2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "the answer is 4" {
		t.Errorf("final = %q", tr.Final)
	}
	if tr.Steps != 2 {
		t.Errorf("steps = %d, want 2", tr.Steps)
	}
	obs := toolObservation(tr.Messages)
	if len(obs) != 1 || !strings.Contains(obs[0].Content, "4") {
		t.Errorf("tool observation = %+v, want result containing 4", obs)
	}
}

// The "model turn" debug log is the main tool for diagnosing a degenerate
// empty reply (empty content, no tool calls, no clue why) — it must actually
// show the model's own reasoning text, the server's stated finish_reason, and
// the current context fullness, not just the (possibly blank) final content.
func TestModelTurnDebugLogIncludesReasoningFinishReasonAndContext(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(prev)

	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "", Reasoning: "thinking it over", FinishReason: "stop"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	if _, err := a.Run(context.Background(), "do something"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	logged := buf.String()
	if !strings.Contains(logged, `reasoning="thinking it over"`) {
		t.Errorf("debug log missing the model's reasoning text, got: %s", logged)
	}
	if !strings.Contains(logged, "finish_reason=stop") {
		t.Errorf("debug log missing finish_reason, got: %s", logged)
	}
	if !strings.Contains(logged, "context_tokens=") {
		t.Errorf("debug log missing context_tokens, got: %s", logged)
	}
}

func TestRunInjectsPitfall(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "calc", ErrorPattern: "unknown function",
		Context: "using calc", ProvenFix: "only use whitelisted functions",
	})
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"foo(2)"}}`),
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, reg, kb, nil, "", 5)

	tr, _ := a.Run(context.Background(), "compute foo(2)")
	if got := toolObservation(tr.Messages); len(got) == 0 ||
		!strings.Contains(got[0].Content, "only use whitelisted functions") {
		t.Errorf("pitfall hint not injected: %+v", got)
	}
}

// trimIfNearWindow must never touch the system prompt, the original goal, or
// assistant text — only large, OLD tool-result content — and must never touch
// the most recent trimKeepRecent messages regardless of size, since the model
// is actively working with those right now.
func TestTrimIfNearWindowProtectsRecentAndNonToolMessages(t *testing.T) {
	big := strings.Repeat("x", trimMinResultSize*4)
	msgs := []llm.Message{
		{Role: "system", Content: big},          // never touched
		{Role: "user", Content: "do the thing"}, // never touched
		{Role: "assistant", Content: big},       // never touched — not a tool result
		{Role: "tool", Content: big},            // old + big + tool → eligible
		{Role: "tool", Content: "tiny"},         // too small to bother trimming
		{Role: "assistant", Content: ""},        // recent — protected regardless
		{Role: "tool", Content: big},            // recent — protected regardless
		{Role: "assistant", Content: ""},        // recent
		{Role: "tool", Content: big},            // recent
		{Role: "assistant", Content: ""},        // recent
		{Role: "tool", Content: big},            // recent (index 10, within last 6 of 11)
	}
	// Small enough to force trimming, but not so extreme that the main pass
	// alone can't get under budget — that "protected zone still doesn't fit"
	// case is the overflow-fallback pass's job, covered separately by
	// TestTrimIfNearWindowOverflowFallback (which does reach into this zone).
	freed := trimIfNearWindow(msgs, 3500)
	if freed <= 0 {
		t.Fatal("expected trimIfNearWindow to free something")
	}
	if msgs[0].Content != big {
		t.Error("system prompt must never be trimmed")
	}
	if msgs[1].Content != "do the thing" {
		t.Error("the user goal must never be trimmed")
	}
	if msgs[2].Content != big {
		t.Error("assistant text must never be trimmed")
	}
	if msgs[3].Content == big {
		t.Error("the old, large, eligible tool result should have been trimmed")
	}
	if msgs[4].Content != "tiny" {
		t.Error("a tool result under trimMinResultSize must be left alone")
	}
	for i := 6; i <= 10; i += 2 { // the recent tool messages (protected zone)
		if msgs[i].Content != big {
			t.Errorf("recent tool message at index %d must be protected, got %q", i, msgs[i].Content)
		}
	}
}

// Below inTaskTrimRatio of the window, trimIfNearWindow must be a no-op —
// it's a rare backstop, not everyday routine.
func TestTrimIfNearWindowNoopWhenUnderThreshold(t *testing.T) {
	msgs := []llm.Message{
		{Role: "system", Content: "short"},
		{Role: "tool", Content: strings.Repeat("x", trimMinResultSize*2)},
	}
	before := append([]llm.Message(nil), msgs...)
	if freed := trimIfNearWindow(msgs, 1_000_000); freed != 0 {
		t.Errorf("freed = %d, want 0 (well under threshold)", freed)
	}
	for i := range msgs {
		if msgs[i].Content != before[i].Content {
			t.Errorf("message %d changed when nothing should have been trimmed", i)
		}
	}
}

// A single task with many large tool results must actually complete (not
// error or hang) even when its own tool-call trail alone would organically
// exceed a small context window — and the request sent for the FINAL turn
// must show early results trimmed, proving the mid-task mechanism actually
// engaged during a real Run(), not just in isolation.
func TestRunTrimsOldLargeToolResultsMidTask(t *testing.T) {
	blobSize := trimMinResultSize * 6
	bigBlob := func(ctx context.Context, a tool.Args) tool.Result { return tool.Ok(strings.Repeat("x", blobSize)) }
	reg := tool.NewRegistry(tool.NewDomain(tool.DomainSpec{
		Name: "big", Summary: "test-only: returns a large fixed blob",
		Actions: []tool.Action{{Name: "get", Run: bigBlob}},
	}))

	const rounds = 8
	var replies []llm.Message
	for i := 0; i < rounds; i++ {
		// Each call's params differ (n=i) so the loop-detector's "repeating the
		// exact same call" check doesn't trip — this test is isolating the
		// context-window trim mechanism, not the (separate, already-correct)
		// stuck-loop nudge.
		args := fmt.Sprintf(`{"action":"get","params":{"n":%d}}`, i)
		replies = append(replies, toolCallReply(fmt.Sprintf("c%d", i), "big", args))
	}
	replies = append(replies, llm.Message{Role: "assistant", Content: "done"})
	fake := &scriptLLM{replies: replies}

	a := New(fake, reg, nil, nil, "", rounds+2)
	a.SetContextWindow(1000) // tiny relative to rounds*blobSize of raw tool output

	tr, err := a.Run(context.Background(), "fetch the big thing repeatedly")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "done" {
		t.Fatalf("final = %q, the task must still complete cleanly", tr.Final)
	}

	trimmedCount := 0
	for _, m := range fake.lastMsgs {
		if m.Role == "tool" && strings.Contains(m.Content, "trimmed") {
			trimmedCount++
		}
	}
	if trimmedCount == 0 {
		t.Error("expected at least one early tool result to have been trimmed by the time of the final request")
	}
}

// Bug repro: a tool call's Arguments (e.g. a file.write's large embedded
// "content" param) gets resent verbatim on every subsequent request — see
// client.go's toWire(), which copies ToolCall.Arguments into the wire
// message's Function.Arguments — so estimateMsgTokens must count it, not
// just m.Content, or a huge write/edit/run payload is an invisible blind spot.
func TestEstimateMsgTokensCountsToolCallArguments(t *testing.T) {
	bigArgs := fmt.Sprintf(`{"action":"write","params":{"path":"f.txt","content":%q}}`, strings.Repeat("x", 50_000))
	msgs := []llm.Message{
		{Role: "user", Content: "write a big file"},
		{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "c1", Name: "file", Arguments: bigArgs}}},
	}
	want := (len("write a big file") + len(bigArgs)) / 4
	if got := estimateMsgTokens(msgs); got != want {
		t.Errorf("estimateMsgTokens = %d, want %d — must count ToolCalls[].Arguments as well as Content", got, want)
	}
}

// Bug repro: when len(msgs) <= trimKeepRecent, protectFrom <= 0 and the main
// pass trims nothing at all — so a handful of "recent" messages that alone
// exceed the whole context window (e.g. a single large file.read result, no
// bigger than the file tool's own maxReadBytes cap) go completely unaddressed.
// The overflow-fallback pass must still make progress here, sparing only the
// single most recent message.
func TestTrimIfNearWindowOverflowFallback(t *testing.T) {
	huge := strings.Repeat("x", 200_000) // matches the file tool's maxReadBytes cap
	msgs := []llm.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "do the thing"},
		{Role: "assistant", Content: ""},
		{Role: "tool", Content: huge}, // oversized but "recent" — index 3
		{Role: "assistant", Content: ""},
		{Role: "tool", Content: "small final result"}, // the single most recent message
	}
	contextWindow := 8192
	if protectFrom := len(msgs) - trimKeepRecent; protectFrom > 0 {
		t.Fatalf("test setup: expected len(msgs) <= trimKeepRecent so the main pass is a no-op (protectFrom=%d)", protectFrom)
	}

	freed := trimIfNearWindow(msgs, contextWindow)
	if freed <= 0 {
		t.Fatal("expected the overflow fallback to free something")
	}
	if msgs[3].Content == huge {
		t.Error("the oversized 'recent' tool result must be trimmed by the overflow fallback")
	}
	if msgs[5].Content != "small final result" {
		t.Error("the single most recent message must still be spared")
	}
	if limit := int(float64(contextWindow) * inTaskTrimRatio); estimateMsgTokens(msgs) >= limit {
		t.Errorf("estimate still over limit after overflow fallback: %d >= %d", estimateMsgTokens(msgs), limit)
	}
}

// Bug repro: trimIfNearWindow's placeholder must preserve enough of a FAILED
// run result's content that runFailureReason — which actionsDigest (called by
// remember() at the end of Run(), to build the durable cross-task digest)
// uses to find the "— FAILED: ..." tag — still finds the failure after the
// trim. The placeholder alone has no "\n" at all, so runFailureReason's
// strings.Cut(content, "\n") failed and it silently returned "".
func TestTrimIfNearWindowPreservesFailureSignal(t *testing.T) {
	body := "go: cannot find module providing package foo\n" + strings.Repeat("more build output\n", 40)
	failContent := "exit 1\n" + body
	if len(failContent) < trimMinResultSize {
		t.Fatalf("test setup: failContent too small to be trim-eligible (%d bytes)", len(failContent))
	}
	msgs := []llm.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "fix the build"},
		toolCallReply("c0", "run", `{"action":"shell","params":{"command":"go test ./..."}}`),
		{Role: "tool", Content: failContent}, // old, big, FAILED — index 3, eligible for trimming
	}
	// Pad with enough small recent turns to push the failing pair (indices 2,3)
	// outside the protected trimKeepRecent window — 16 messages total, matching
	// the confirmed repro.
	for i := 0; i < trimKeepRecent; i++ {
		msgs = append(msgs, llm.Message{Role: "assistant", Content: fmt.Sprintf("thinking %d", i)})
		msgs = append(msgs, llm.Message{Role: "tool", Content: "ok"})
	}
	if protectFrom := len(msgs) - trimKeepRecent; protectFrom <= 3 {
		t.Fatalf("test setup: the failing result at index 3 must be outside the protected zone (protectFrom=%d)", protectFrom)
	}

	freed := trimIfNearWindow(msgs, 10) // tiny window forces trimming
	if freed <= 0 {
		t.Fatal("test setup: expected trimIfNearWindow to trim something")
	}
	if !strings.Contains(msgs[3].Content, "trimmed") {
		t.Fatalf("test setup: expected the old failing result at index 3 to have been trimmed, got %q", msgs[3].Content)
	}

	digest := actionsDigest(msgs)
	if !strings.Contains(digest, "FAILED") {
		t.Errorf("digest lost the FAILED marker after mid-task trim: %q", digest)
	}
}

func TestHintsRequireErrorPatternMatch(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "file", ErrorPattern: "missing required param(s): path",
		Context: "file: edit", ProvenFix: "include the path param",
	})
	a := New(&scriptLLM{}, tool.NewRegistry(tool.NewCalc()), kb, nil, "", 5)

	// An unrelated error (e.g. "no action") must NOT surface the path pitfall.
	if h := a.hints("file", "", `file: no action given — set "action" to one of: read, write`); h != "" {
		t.Errorf("irrelevant hint injected: %q", h)
	}
	// The same error recurring, for the SAME action it was learned on, does.
	if h := a.hints("file", "edit", "edit failed: missing required param(s): path"); !strings.Contains(h, "include the path param") {
		t.Errorf("relevant hint not injected: %q", h)
	}
}

// A domain often reuses the exact same generic validation text across
// different actions (file's "missing required param(s): path" fires for
// write, edit, AND append alike) — a lesson learned on one action must not
// surface as guidance for a different action's identical error, or it
// actively misleads (e.g. pointing a failed "write" at edit's find/replace
// params, which don't exist on write).
func TestHintsDontCrossActionsWithSharedErrorText(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "file", ErrorPattern: "missing required param(s): path",
		Context: "file: edit", ProvenFix: "include the path param in the edit action",
	})
	reg := tool.NewRegistry(tool.NewFile(nil, nil, nil))
	a := New(&scriptLLM{}, reg, kb, nil, "", 5)

	// A write hitting the identical generic error must NOT get the edit-specific hint.
	if h := a.hints("file", "write", "missing required param(s): path"); h != "" {
		t.Errorf("edit's hint leaked into a write failure: %q", h)
	}
	// The same lesson, for the action it was actually learned on, still fires.
	if h := a.hints("file", "edit", "missing required param(s): path"); !strings.Contains(h, "include the path param in the edit action") {
		t.Errorf("hint not shown for its own action: %q", h)
	}
}

// A pitfall keyed on a bare "exit N" pattern (the run tool's own generic
// wrapper prefix on every failed command) must never surface as a hint — it
// "matches" (and misleads on) any unrelated failure. Live case that surfaced
// this: a stale KB entry learned from a python dependency error kept showing
// up on an unrelated "go.mod already exists" failure.
func TestHintsNeverSurfacesGenericExitCodePattern(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "run", ErrorPattern: "exit 1",
		Context: "a python module missing a dependency", ProvenFix: "install the required system library",
	})
	reg := tool.NewRegistry(tool.NewCalc())
	a := New(&scriptLLM{}, reg, kb, nil, "", 5)

	if h := a.hints("run", "shell", "exit 1\ngo: /Users/roman220/rl_hero_go/go.mod already exists"); h != "" {
		t.Errorf("generic exit-code pitfall leaked into an unrelated go.mod failure: %q", h)
	}
}

// SetMaxHistory overrides the default trim cap (16) — memory "raw" callers
// raise it a lot so the FIFO cut, which breaks a local server's KV-cache
// prefix just like a summary compact does, stays a rare backstop instead of
// firing on every turn.
func TestSetMaxHistoryOverridesTrimCap(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "answer 1"},
		{Role: "assistant", Content: "answer 2"},
		{Role: "assistant", Content: "answer 3"},
		{Role: "assistant", Content: "answer 4"},
		{Role: "assistant", Content: "answer 5"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetMaxHistory(4) // 2 turns' worth (goal + final each)
	if a.MaxHistory() != 4 {
		t.Fatalf("MaxHistory() = %d, want 4", a.MaxHistory())
	}

	for i := range fake.replies {
		if _, err := a.Run(context.Background(), fake.replies[i].Content+" goal"); err != nil {
			t.Fatal(err)
		}
	}
	if a.SessionLen() != 4 {
		t.Errorf("SessionLen = %d, want 4 (cap enforced at the overridden value, not the default 16)", a.SessionLen())
	}
}

// remember()'s routine FIFO trim must count every message it drops into
// FrontTrimCount, but — unlike Reset/SetHistory/Compact — must NOT bump
// HistoryGen: a routine trim only shifts a prefix, it doesn't replace history
// wholesale, so a checkpoint indexing into it should be remappable (see
// cmd/agent's checkpointValid/effectiveHistLen) instead of invalidated
// outright.
func TestFrontTrimCountTracksRoutineTrimNotHistoryGen(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "answer 1"},
		{Role: "assistant", Content: "answer 2"},
		{Role: "assistant", Content: "answer 3"},
		{Role: "assistant", Content: "answer 4"},
		{Role: "assistant", Content: "answer 5"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetMaxHistory(4) // 2 turns' worth (goal + final each)
	startGen := a.HistoryGen()

	if a.FrontTrimCount() != 0 {
		t.Fatalf("FrontTrimCount() = %d before any trim, want 0", a.FrontTrimCount())
	}
	for i := range fake.replies {
		if _, err := a.Run(context.Background(), fake.replies[i].Content+" goal"); err != nil {
			t.Fatal(err)
		}
	}
	// Runs 1-2 fill the cap exactly (no trim); runs 3-5 each push 2 over the
	// cap and trim exactly 2 back off — 3 trims × 2 dropped = 6.
	if got := a.FrontTrimCount(); got != 6 {
		t.Errorf("FrontTrimCount() = %d after 5 runs at maxHistory=4, want 6", got)
	}
	if got := a.HistoryGen(); got != startGen {
		t.Errorf("HistoryGen() = %d after routine trims only, want unchanged at %d — routine trims must "+
			"not be conflated with a genuine reshape (Reset/SetHistory/Compact)", got, startGen)
	}
}

func TestSessionMemoryCarriesAcrossRuns(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "the answer is 4"},   // run 1 final
		{Role: "assistant", Content: "we computed 2+2=4"}, // run 2 final
	}}
	a := New(fake, reg, nil, nil, "", 5)

	if _, err := a.Run(context.Background(), "what is 2+2"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), "what did we just do?"); err != nil {
		t.Fatal(err)
	}

	// The second Run's prompt must contain the first goal AND its answer.
	var sawGoal, sawAnswer bool
	for _, m := range fake.lastMsgs {
		if strings.Contains(m.Content, "what is 2+2") {
			sawGoal = true
		}
		if strings.Contains(m.Content, "the answer is 4") {
			sawAnswer = true
		}
	}
	if !sawGoal || !sawAnswer {
		t.Errorf("session memory missing in 2nd run: goal=%v answer=%v msgs=%+v", sawGoal, sawAnswer, fake.lastMsgs)
	}

	a.Reset()
	if a.SessionLen() != 0 {
		t.Errorf("after Reset SessionLen = %d, want 0", a.SessionLen())
	}
}

func TestActionsDigest(t *testing.T) {
	msgs := []llm.Message{
		toolCallReply("c1", "file", `{"action":"write","params":{"path":"world.go","content":"package main"}}`),
		{Role: "tool", Content: "ok"},
		toolCallReply("c2", "file", `{"action":"read","params":{"path":"world.go"}}`), // read-only — must be skipped
		{Role: "tool", Content: "package main"},
		toolCallReply("c3", "run", `{"action":"shell","params":{"command":"go build ./..."}}`),
		{Role: "tool", Content: "ok"},
	}
	got := actionsDigest(msgs)
	if strings.Count(got, "world.go") != 1 {
		t.Errorf("digest should list the written file exactly once (read must be skipped): %q", got)
	}
	if !strings.Contains(got, "go build ./...") {
		t.Errorf("digest missing the command run: %q", got)
	}
	if got := actionsDigest([]llm.Message{{Role: "assistant", Content: "just talk, no tools"}}); got != "" {
		t.Errorf("no mutating tool calls → want empty digest, got %q", got)
	}
}

// A digest that only says a command was "run" — never that it failed, or why
// — gives the NEXT task no signal to avoid blindly repeating it; that's
// exactly what let a stuck-loop task's failure repeat itself one task later
// (live case: "go mod init" kept re-running across tasks because nothing ever
// told the model "go.mod already exists").
func TestActionsDigestIncludesFailureReason(t *testing.T) {
	msgs := []llm.Message{
		toolCallReply("c1", "run", `{"action":"shell","params":{"command":"go mod init rl_hero_go"}}`),
		llm.ToolResult("c1", "run", "exit 1\ngo: /Users/roman220/rl_hero_go/go.mod already exists"),
		toolCallReply("c2", "run", `{"action":"shell","params":{"command":"go build ./..."}}`),
		llm.ToolResult("c2", "run", "exit 0\nbuild ok"),
	}
	got := actionsDigest(msgs)
	if !strings.Contains(got, "go mod init rl_hero_go") || !strings.Contains(got, "FAILED") || !strings.Contains(got, "go.mod already exists") {
		t.Errorf("digest missing the failure reason for the failed command: %q", got)
	}
	if strings.Contains(got, "go build ./... — FAILED") {
		t.Errorf("a SUCCESSFUL command must not be tagged as failed: %q", got)
	}
}

// Caught live: the FAILED tag showed up on some retries of the exact same
// failure but not others. Root cause — a weak local model's OpenAI-compat
// endpoint often leaves ToolCall.ID empty, or reuses the same one across
// calls in a batch; the original ID-based result lookup silently found
// nothing (or the wrong result) whenever that happened. Correlation must work
// by POSITION — the message right after an assistant tool call is always its
// own result — regardless of what's in the ID field.
func TestActionsDigestFindsFailureReasonWithoutReliableToolCallIDs(t *testing.T) {
	msgs := []llm.Message{
		// Every call below shares the same (empty) ID, exactly like a weak
		// local model that never fills in tool_call ids.
		toolCallReply("", "run", `{"action":"shell","params":{"command":"go mod init rl_hero_go"}}`),
		llm.ToolResult("", "run", "exit 1\ngo: /Users/roman220/rl_hero_go/go.mod already exists"),
		toolCallReply("", "run", `{"action":"shell","params":{"command":"go mod init rl_hero_go"}}`), // repeated verbatim, same empty ID
		llm.ToolResult("", "run", "exit 1\ngo: /Users/roman220/rl_hero_go/go.mod already exists"),
	}
	got := actionsDigest(msgs)
	if !strings.Contains(got, "FAILED") || !strings.Contains(got, "go.mod already exists") {
		t.Errorf("digest missing the failure reason when tool-call IDs are all empty: %q", got)
	}
}

// Live case: "mkdir -p rl_hero_go && cd rl_hero_go && go mod init rl_hero_go"
// (62 bytes) was cut to "...go mod init rl_hero_" by the old 60-byte clip —
// losing the "go" that's the only thing distinguishing this command from any
// other "go mod init" call.
func TestActionsDigestDoesNotTruncateTheDistinguishingPartOfACommand(t *testing.T) {
	cmd := "mkdir -p rl_hero_go && cd rl_hero_go && go mod init rl_hero_go"
	msgs := []llm.Message{
		toolCallReply("c1", "run", fmt.Sprintf(`{"action":"shell","params":{"command":%q}}`, cmd)),
		llm.ToolResult("c1", "run", "exit 1\ngo: /Users/roman220/rl_hero_go/go.mod already exists"),
	}
	if got := actionsDigest(msgs); !strings.Contains(got, cmd) {
		t.Errorf("digest truncated the command before its distinguishing suffix: %q", got)
	}
}

// Live case: "go test ./..." fails on a compile error, the agent fixes the
// code, then runs the identical command again later in the SAME task and it
// passes. The dedup must keep the LATEST outcome, not freeze on the first
// (now-stale) failure — otherwise Compact strengthens that stale record into
// "never repeat this command", which is actively wrong once it's fixed.
func TestActionsDigestKeepsLatestOutcomeOnRetry(t *testing.T) {
	msgs := []llm.Message{
		toolCallReply("c1", "run", `{"action":"shell","params":{"command":"go test ./..."}}`),
		llm.ToolResult("c1", "run", "exit 1\n# foo\n./foo.go:1:1: syntax error"),
		toolCallReply("c2", "run", `{"action":"shell","params":{"command":"go test ./..."}}`),
		llm.ToolResult("c2", "run", "exit 0\nok  \tfoo\t0.002s"),
	}
	got := actionsDigest(msgs)
	if strings.Contains(got, "FAILED") {
		t.Errorf("digest still shows the stale FAILED outcome after a later success: %q", got)
	}
	if !strings.Contains(got, "go test ./...") {
		t.Errorf("digest missing the retried command entirely: %q", got)
	}
	if strings.Count(got, "go test ./...") != 1 {
		t.Errorf("digest should list the retried command exactly once (latest outcome only): %q", got)
	}
}

// The whole point of the digest: cross-run memory must know which files were
// actually created, not just whatever the model chose to say in its one-line
// final answer.
func TestActionsDigestCarriesAcrossRuns(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"write","params":{"path":"world.go","content":"package main"}}`),
		{Role: "assistant", Content: "created world.go"}, // run 1 final — doesn't restate the path
		{Role: "assistant", Content: "yes, it exists"},   // run 2 final
	}}
	a := New(fake, reg, nil, nil, "", 5)

	if _, err := a.Run(context.Background(), "build the world module"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), "did you create world.go?"); err != nil {
		t.Fatal(err)
	}

	var sawDigest bool
	for _, m := range fake.lastMsgs {
		if strings.Contains(m.Content, "world.go") && strings.Contains(m.Content, "actions this turn") {
			sawDigest = true
		}
	}
	if !sawDigest {
		t.Errorf("2nd run's prompt is missing the files-touched digest from run 1: %+v", fake.lastMsgs)
	}
}

type fakeArchiver struct{ goals, entries []string }

func (f *fakeArchiver) Archive(goal, entry string) {
	f.goals = append(f.goals, goal)
	f.entries = append(f.entries, entry)
}

// The archiver must see every entry remember() commits — including its
// digest — so a "history" tool built on it can recall what a later Compact
// has since folded away, and it must NOT fire for a detached (orphaned) agent.
func TestArchiverReceivesEveryRememberedTurn(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"write","params":{"path":"world.go","content":"package main"}}`),
		{Role: "assistant", Content: "created world.go"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	ar := &fakeArchiver{}
	a.SetArchiver(ar)

	if _, err := a.Run(context.Background(), "build the world module"); err != nil {
		t.Fatal(err)
	}
	if len(ar.goals) != 1 || ar.goals[0] != "build the world module" {
		t.Fatalf("archiver goals = %+v", ar.goals)
	}
	if !strings.Contains(ar.entries[0], "world.go") {
		t.Errorf("archived entry missing the digest: %q", ar.entries[0])
	}

	a.Detach()
	if _, err := a.Run(context.Background(), "second task"); err != nil {
		t.Fatal(err)
	}
	if len(ar.goals) != 1 {
		t.Errorf("a detached agent must not archive: goals = %+v", ar.goals)
	}
}

type recTracer struct {
	kinds        []string
	finalSuggest string
}

func (r *recTracer) Emit(kind string, f map[string]any) {
	r.kinds = append(r.kinds, kind)
	if kind == "final" {
		r.finalSuggest, _ = f["suggest"].(string)
	}
}

func (r *recTracer) has(kind string) bool {
	for _, k := range r.kinds {
		if k == kind {
			return true
		}
	}
	return false
}

func TestNoDuplicateFinalEmit(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	rt := &recTracer{}
	fake := &scriptLLM{replies: []llm.Message{{Role: "assistant", Content: "hi there"}}}
	a := New(fake, reg, nil, rt, "", 5)

	if _, err := a.Run(context.Background(), "say hi"); err != nil {
		t.Fatal(err)
	}
	var assistant, final int
	for _, k := range rt.kinds {
		switch k {
		case "assistant":
			assistant++
		case "final":
			final++
		}
	}
	if final != 1 || assistant != 0 {
		t.Errorf("emitted %v, want exactly 1 final and 0 assistant for a no-tool answer", rt.kinds)
	}
}

func TestCompactSummarizesSession(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "answer A"},
		{Role: "assistant", Content: "answer B"},
		{Role: "assistant", Content: "SUMMARY: we did A and B"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.Run(context.Background(), "task 1")
	a.Run(context.Background(), "task 2")
	if a.SessionLen() != 4 {
		t.Fatalf("SessionLen before compact = %d, want 4", a.SessionLen())
	}

	n, err := a.Compact(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Errorf("compacted %d messages, want 4", n)
	}
	if a.SessionLen() != 2 {
		t.Errorf("SessionLen after compact = %d, want 2 (summary pair)", a.SessionLen())
	}
	var found bool
	for _, m := range a.history {
		if strings.Contains(m.Content, "SUMMARY: we did A and B") {
			found = true
		}
	}
	if !found {
		t.Error("summary not stored in the compacted history")
	}
}

// Caught live: a weak model asked to summarize many retries of the same
// failing command blurred or dropped the specific failure reason — the
// compacted history read like a fresh start, and the very next task blindly
// repeated the exact command that had already failed every time. The
// deterministic actions-digest facts (see actionsDigest/TestActionsDigest*)
// must survive Compact verbatim, regardless of what the LLM's own summary
// says — even a summary that says nothing useful at all.
func TestCompactPreservesActionDigestsVerbatim(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "run", `{"action":"shell","params":{"command":"go mod init rl_hero_go"}}`),
		{Role: "assistant", Content: "Stopped — it kept repeating the same tool calls."}, // run 1 final
		{Role: "assistant", Content: "a summary that mentions nothing about go.mod"},     // Compact's own (lossy) summary
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.Run(context.Background(), "по плану идем")

	if _, err := a.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	var got string
	for _, m := range a.history {
		got += m.Content
	}
	if !strings.Contains(got, "go mod init rl_hero_go") {
		t.Errorf("compacted history lost the action digest entirely: %q", got)
	}
	if !strings.Contains(got, "a summary that mentions nothing about go.mod") {
		t.Errorf("compacted history should still include the LLM's own summary: %q", got)
	}
}

// Compact's OWN digest block (embedded in a "user"-role summary message) must
// survive a SECOND Compact call too, not just the first — otherwise it only
// ever survives one compaction, and a task two compactions later loses the
// exact command record entirely once the LLM's own prose drops it. The second
// compaction's scripted summarizer here deliberately omits the specific
// command detail, so the ONLY way it can survive is via the re-harvested
// digest block, not the summary text.
func TestCompactDigestSurvivesSecondCompaction(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "run", `{"action":"shell","params":{"command":"rm -rf build"}}`),
		{Role: "assistant", Content: "cleaned the build directory"},                            // run 1 final
		{Role: "assistant", Content: "first summary — mentions the cleanup"},                   // 1st Compact's summary
		{Role: "assistant", Content: "second summary — says nothing about any command at all"}, // 2nd Compact's (lossy) summary
	}}
	a := New(fake, reg, nil, nil, "", 5)
	if _, err := a.Run(context.Background(), "clean up the build dir"); err != nil {
		t.Fatal(err)
	}

	if _, err := a.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}

	var got string
	for _, m := range a.history {
		got += m.Content
	}
	if !strings.Contains(got, "rm -rf build") {
		t.Errorf("exact command record lost after a second compaction: %q", got)
	}
	if !strings.Contains(got, "second summary — says nothing about any command at all") {
		t.Errorf("compacted history should still include the 2nd Compact's own summary: %q", got)
	}
}

func TestSplitSuggestion(t *testing.T) {
	clean, sug := splitSuggestion("Wrote hello.sh and ran it.\nNEXT: add a test")
	if clean != "Wrote hello.sh and ran it." {
		t.Errorf("clean = %q", clean)
	}
	if sug != "add a test" {
		t.Errorf("suggestion = %q, want 'add a test'", sug)
	}
	if c, s := splitSuggestion("just an answer"); c != "just an answer" || s != "" {
		t.Errorf("no-NEXT case = %q,%q", c, s)
	}
	// A placeholder-shaped suggestion must be unwrapped, not shown with brackets.
	if _, s := splitSuggestion("done\nNEXT: <run the test script>"); s != "run the test script" {
		t.Errorf("bracketed suggestion = %q, want unwrapped", s)
	}
	// A "NEXT:" in the MIDDLE must stay part of the answer, not be extracted.
	mid := "Here is a script:\nNEXT: do X\nand it ends here"
	if c, s := splitSuggestion(mid); c != mid || s != "" {
		t.Errorf("mid-answer NEXT wrongly peeled: clean=%q sug=%q", c, s)
	}
	// Markdown/bullet-decorated NEXT must still be peeled (the reported bug).
	for _, in := range []string{
		"Done.\n**NEXT:** add a test",
		"Done.\n- NEXT: add a test",
		"Done.\n**NEXT: add a test**",
	} {
		c, s := splitSuggestion(in)
		if c != "Done." || s != "add a test" {
			t.Errorf("decorated NEXT %q → clean=%q sug=%q", in, c, s)
		}
	}
	// The prompt says to skip the line when nothing fits, but a model can fill
	// it with an honest "nothing to suggest" statement instead (reported live:
	// shown as a real Tab-acceptable suggestion, which it isn't). Any such
	// "no suggestion" phrasing must come back empty, not offered as one.
	for _, in := range []string{
		"Привет! Чем могу помочь?\nNEXT: — (no specific next step)",
		"done\nNEXT: none",
		"done\nNEXT: n/a",
		"done\nNEXT: -",
	} {
		if _, s := splitSuggestion(in); s != "" {
			t.Errorf("no-suggestion placeholder %q → suggestion=%q, want empty", in, s)
		}
	}
}

func TestParseArgsFoldsTopLevel(t *testing.T) {
	// Small models often omit the "params" wrapper.
	action, params := parseArgs(`{"action":"calculate","expression":"2+2"}`)
	if action != "calculate" {
		t.Errorf("action = %q, want calculate", action)
	}
	if params["expression"] != "2+2" {
		t.Errorf("params = %v, want folded expression", params)
	}
	if _, leaked := params["action"]; leaked {
		t.Error("action leaked into folded params")
	}
}

func TestParseArgsStringifiedParams(t *testing.T) {
	// The most common malformation: params double-encoded as a JSON string.
	action, params := parseArgs(`{"action":"write","params":"{\"path\":\"main.py\",\"content\":\"x\"}"}`)
	if action != "write" {
		t.Errorf("action = %q, want write", action)
	}
	if params["path"] != "main.py" || params["content"] != "x" {
		t.Errorf("params = %v, want decoded path+content", params)
	}
	// Action nested inside the stringified blob, none at top level.
	action, params = parseArgs(`{"params":"{\"action\":\"write\",\"path\":\"a.txt\",\"content\":\"hi\"}"}`)
	if action != "write" || params["path"] != "a.txt" {
		t.Errorf("nested-action: action=%q params=%v", action, params)
	}
	if _, leaked := params["action"]; leaked {
		t.Error("action leaked into decoded params")
	}
}

func TestLooksLikeRefusal(t *testing.T) {
	refusals := []string{
		"Here are the files:\n```\ncode\n```",
		"I don't have access to your files, copy them manually.",
		"Как языковая модель, я не имею доступа к файловой системе.",
	}
	for _, s := range refusals {
		if !looksLikeRefusal(s) {
			t.Errorf("looksLikeRefusal(%q) = false, want true", s)
		}
	}
	ok := []string{"Added a /health endpoint; tests pass.", "Done — wrote main.py and ran the tests."}
	for _, s := range ok {
		if looksLikeRefusal(s) {
			t.Errorf("looksLikeRefusal(%q) = true, want false", s)
		}
	}
}

// A chat model that dodges an action task (pastes files / "I can't access your
// filesystem") with no tool calls gets nudged once, then proceeds to use tools.
func TestRunNudgesRefusalThenActs(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "I don't have access to your filesystem. Here are the files:\n```py\nx=1\n```"},
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		{Role: "assistant", Content: "done — 4"},
	}}
	a := New(fake, reg, nil, nil, "", 6)
	tr, err := a.Run(context.Background(), "edit the files")
	if err != nil {
		t.Fatal(err)
	}
	if tr.Final != "done — 4" {
		t.Errorf("final = %q, want it to proceed past the refusal", tr.Final)
	}
	if len(toolObservation(tr.Messages)) != 1 {
		t.Error("expected the tool to run after the refusal nudge")
	}
}

// The refusal nudge fires at most once: a model that refuses twice has its second
// refusal accepted as the final answer (no infinite loop).
func TestRunAcceptsRefusalAfterOneNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	refusal := llm.Message{Role: "assistant", Content: "As a language model I cannot modify files. Copy:\n```\nx\n```"}
	fake := &scriptLLM{replies: []llm.Message{refusal, refusal}}
	a := New(fake, reg, nil, nil, "", 6)
	tr, err := a.Run(context.Background(), "edit the files")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(tr.Final, "cannot modify files") {
		t.Errorf("final = %q, want the 2nd refusal accepted", tr.Final)
	}
	if tr.Steps != 2 {
		t.Errorf("steps = %d, want 2 (refuse → nudge → refuse-accept)", tr.Steps)
	}
}

// A model that keeps making the exact same (succeeding) tool call makes no
// progress; the loop guard nudges once, then stops instead of running forever.
func TestRunStopsOnRepeatedIdenticalCalls(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	call := toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`)
	replies := make([]llm.Message, 12)
	for i := range replies {
		replies[i] = call
	}
	a := New(&scriptLLM{replies: replies}, reg, nil, nil, "", 20)
	tr, err := a.Run(context.Background(), "spin on the same call")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(tr.Final, "Stopped") {
		t.Errorf("final = %q, want the loop guard to stop it", tr.Final)
	}
	if tr.Steps >= 20 {
		t.Errorf("steps = %d, want the guard to stop well before maxSteps", tr.Steps)
	}
}

// Reported live: a task that burns its whole step budget on turns that keep
// succeeding at DIFFERENT things (so neither the "all failed" nor the
// "repeating" stuck-guard ever fires) but never write any chat content ended
// in TOTAL SILENCE in the TUI — nothing on screen, nothing in the log
// distinguishing it from any other ending, because tr.Final was simply "".
// The plain (one-shot) path already had its own fallback for this; the TUI
// path had none, since it only ever sees this same emitted "final" event.
func TestRunStepBudgetExhaustedWithNoContentGetsAFallbackMessage(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	replies := []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`),
		toolCallReply("c2", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		toolCallReply("c3", "calc", `{"action":"calculate","params":{"expression":"3+3"}}`),
	}
	a := New(&scriptLLM{replies: replies}, reg, nil, nil, "", 3) // maxSteps=3, exactly len(replies)

	tr, err := a.Run(context.Background(), "keep calculating")
	if err != nil {
		t.Fatal(err)
	}
	if !tr.Stopped {
		t.Error("want Stopped=true — ran out of steps")
	}
	if !strings.Contains(tr.Final, "step budget exhausted") {
		t.Errorf("final = %q, want a fallback message naming the step budget, not silence", tr.Final)
	}
}

func TestUnwrapEnvelope(t *testing.T) {
	// nested content.text (the reported leak)
	if got := unwrapEnvelope(`{"role":"assistant","content":{"text":"hello there"}}`); got != "hello there" {
		t.Errorf("nested = %q, want 'hello there'", got)
	}
	// content as a plain string
	if got := unwrapEnvelope(`{"role":"assistant","content":"hi"}`); got != "hi" {
		t.Errorf("string content = %q, want 'hi'", got)
	}
	// a normal answer is untouched, even one that contains JSON
	for _, s := range []string{
		"Here is the plan: do X then Y.",
		"```json\n{\"role\":\"assistant\"}\n```", // a JSON code block in prose
		`{"role":"user","content":"x"}`,          // not an assistant envelope
		`{"some":"object","without":"role"}`,     // no role
	} {
		if got := unwrapEnvelope(s); got != s {
			t.Errorf("unwrapEnvelope(%q) = %q, want unchanged", s, got)
		}
	}
}

func TestParseArgsNestedObject(t *testing.T) {
	action, params := parseArgs(`{"action":"edit","params":{"path":"a","find":"x","replace":"y"}}`)
	if action != "edit" || params["find"] != "x" || params["replace"] != "y" {
		t.Errorf("nested object: action=%q params=%v", action, params)
	}
	// Mixed shape: path at the top level, the rest under params — fold them together.
	action, params = parseArgs(`{"action":"edit","path":"main.go","params":{"find":"x","replace":"y"}}`)
	if action != "edit" || params["path"] != "main.go" || params["find"] != "x" {
		t.Errorf("mixed shape dropped a sibling: action=%q params=%v", action, params)
	}
}

// A model can wrap the params object in a needless singleton array — reported
// live (twice, real debug logs, LM Studio/nemotron): args was literally
// `{"action":"write","params":[{"path":"README.md","content":"..."}]}`. Before
// this fix, the params switch only matched map[string]any/string, so a []any
// fell through to the flattened-top-level branch and silently produced an
// EMPTY params map — action="write" survived but path/content vanished,
// dispatch failed with a missing-param error, and the model had to burn a
// whole extra step reasoning "I need to provide the path parameter properly.
// Let me try again..." (verbatim from the log) before self-correcting. A
// singleton array wrapping one object is unambiguous — there is exactly one
// way to read it — so unwrapping it is a safe shape fix, not a guess.
func TestParseArgsUnwrapsSingletonArrayParams(t *testing.T) {
	action, params := parseArgs(`{"action":"write","params":[{"path":"README.md","content":"x"}]}`)
	if action != "write" || params["path"] != "README.md" || params["content"] != "x" {
		t.Errorf("array-wrapped params: action=%q params=%v, want write/README.md/x", action, params)
	}
	// A multi-element array has no single unambiguous reading — must NOT guess,
	// leave params empty so the tool's own "missing param" error fires normally.
	action, params = parseArgs(`{"action":"write","params":[{"path":"a"},{"path":"b"}]}`)
	if action != "write" || len(params) != 0 {
		t.Errorf("multi-element array params: action=%q params=%v, want action=write and empty params", action, params)
	}
}

// A model can wrap the whole arguments string in its own tool-call convention
// instead of emitting bare JSON — reported live: the raw arguments string was
// literally "<parameter=params>\n{\"url\": \"...\"}\n</parameter>", which fails
// json.Unmarshal outright (doesn't even start with "{"). decodeObj falls back
// to the embedded object instead of giving up and losing the params entirely.
func TestParseArgsRecoversObjectWrappedInModelOwnTags(t *testing.T) {
	action, params := parseArgs("<parameter=params>\n{\"url\": \"https://example.com\"}\n</parameter>")
	if params["url"] != "https://example.com" {
		t.Errorf("action=%q params=%v, want url recovered from the tag-wrapped JSON", action, params)
	}
}

// A model can also leak the same "<parameter=NAME>value</parameter>" tag
// convention into just the "action" field's own value, inside otherwise
// perfectly valid JSON — reported live, twice: a bare action name
// ("<parameter=action>\nlist\n</parameter>") and a whole params object
// mislabeled as the action ("<parameter=params>\n{\"command\": \"ls -la\", ...}").
// decodeObj's own recovery doesn't help here, since the OUTER JSON parses
// just fine — the garbled text is a normal string value, not a parse failure.
// Before this fix the entire tag text became the action's value verbatim;
// for run (single action, single required param) garbledActionAsParam then
// ran the literal tag text as a shell command ("sh: syntax error"), and for
// file it surfaced as an unknown action.
func TestParseArgsRecoversActionFieldWrappedInModelOwnTags(t *testing.T) {
	action, params := parseArgs(`{"action":"<parameter=action>\nlist\n</parameter>","params":{"path":"."}}`)
	if action != "list" || params["path"] != "." {
		t.Errorf("bare tag: action=%q params=%v, want action=list params.path=.", action, params)
	}

	action, params = parseArgs(`{"action":"<parameter=params>\n{\"command\": \"ls -la\", \"cwd\": \"/Users/roman220/test\"}\n</parameter>"}`)
	if action != "" || params["command"] != "ls -la" || params["cwd"] != "/Users/roman220/test" {
		t.Errorf("object tag: action=%q params=%v, want action=\"\" command=\"ls -la\" cwd=\"/Users/roman220/test\"", action, params)
	}

	// Same object-tag shape, reported live for file (a multi-action domain):
	// no "action" key inside the recovered object, so parseArgs alone can't
	// pick which of file's 8 actions was meant — that's the registry's job
	// (inferAction, or the terse "no action given" error), not parseArgs'.
	// This just confirms the params (path=".") still come through clean.
	action, params = parseArgs(`{"action":"<parameter=params>\n{\"path\": \".\"}\n</parameter>"}`)
	if action != "" || params["path"] != "." {
		t.Errorf("file object tag: action=%q params=%v, want action=\"\" path=\".\"", action, params)
	}
}

// End-to-end version of the object-tag case above: parseArgs alone recovers
// the params but leaves action="" (no "action" key inside the mislabeled
// object); the registry's empty-action soleRequiredParam fallback (internal/
// tool) must then resolve that empty action to calc's one action
// ("calculate") for the whole pipeline to actually work, matching what a
// real Agent.Run does with a real Registry — not just parseArgs in isolation.
func TestRunRecoversActionTagWrappedAroundParamsEndToEnd(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	call := toolCallReply("c1", "calc", `{"action":"<parameter=params>\n{\"expression\": \"2+2\"}\n</parameter>"}`)
	fake := &scriptLLM{replies: []llm.Message{call, {Role: "assistant", Content: "done"}}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, err := a.Run(context.Background(), "what is 2+2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	obs := toolObservation(tr.Messages)
	if len(obs) != 1 || !strings.Contains(obs[0].Content, "4") {
		t.Errorf("observation = %+v, want a successful calculate of 2+2=4", obs)
	}
}

func TestRunConcurrentToolCallsStayOrdered(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	twoCalls := llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{
		{ID: "c1", Name: "calc", Arguments: `{"action":"calculate","params":{"expression":"2+2"}}`},
		{ID: "c2", Name: "calc", Arguments: `{"action":"calculate","params":{"expression":"10*10"}}`},
	}}
	fake := &scriptLLM{replies: []llm.Message{twoCalls, {Role: "assistant", Content: "done"}}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, _ := a.Run(context.Background(), "two sums")
	obs := toolObservation(tr.Messages)
	if len(obs) != 2 {
		t.Fatalf("observations = %d, want 2", len(obs))
	}
	if obs[0].ToolCallID != "c1" || !strings.Contains(obs[0].Content, "4") {
		t.Errorf("first observation = %+v, want c1=4", obs[0])
	}
	if obs[1].ToolCallID != "c2" || !strings.Contains(obs[1].Content, "100") {
		t.Errorf("second observation = %+v, want c2=100", obs[1])
	}
}

// A cancellation mid-batch (esc during a multi-call turn) must stop the REST
// of a SEQUENTIAL batch too — the loop only checked ctx on each individual
// execOne call, so a run cancelled after call 1 still dispatched every
// remaining mutation in that batch before returning.
func TestSequentialBatchStopsAfterCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ran := 0
	mutTool := tool.NewDomain(tool.DomainSpec{
		Name: "file", Summary: "files",
		Actions: []tool.Action{
			{Name: "write", Mutates: true, Run: func(context.Context, tool.Args) tool.Result {
				ran++
				cancel() // simulate esc landing right as call 1 finishes
				return tool.Ok("wrote")
			}},
		},
	})
	reg := tool.NewRegistry(mutTool)
	twoWrites := llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{
		{ID: "c1", Name: "file", Arguments: `{"action":"write","params":{}}`},
		{ID: "c2", Name: "file", Arguments: `{"action":"write","params":{}}`},
	}}
	fake := &scriptLLM{replies: []llm.Message{twoWrites, {Role: "assistant", Content: "done"}}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, _ := a.Run(ctx, "two writes")
	obs := toolObservation(tr.Messages)
	if len(obs) != 2 {
		t.Fatalf("observations = %d, want 2", len(obs))
	}
	if !strings.Contains(obs[1].Content, "cancelled") {
		t.Errorf("second observation = %+v, want it short-circuited as cancelled", obs[1])
	}
	if ran != 1 {
		t.Errorf("the mutating tool actually ran %d time(s), want exactly 1 (the second call must be skipped, not executed)", ran)
	}
}

// planFileTool is a minimal file-like tool with one read-only and one mutating
// action, for exercising the plan-mode gate.
func planFileTool() tool.Tool {
	return tool.NewDomain(tool.DomainSpec{
		Name: "file", Summary: "files",
		Actions: []tool.Action{
			{Name: "read", Run: func(context.Context, tool.Args) tool.Result { return tool.Ok("content-of-x") }},
			{Name: "write", Mutates: true, Run: func(context.Context, tool.Args) tool.Result { return tool.Ok("wrote") }},
		},
	})
}

func TestPlanModeBlocksMutationAndInjectsDirective(t *testing.T) {
	reg := tool.NewRegistry(planFileTool())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"write","params":{"path":"x","content":"y"}}`),
		{Role: "assistant", Content: "plan: 1. write x"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetPlanMode(true)

	tr, _ := a.Run(context.Background(), "make x")
	obs := toolObservation(tr.Messages)
	if len(obs) == 0 || !strings.Contains(obs[0].Content, "plan mode is ON") {
		t.Fatalf("write was not blocked in plan mode: %+v", obs)
	}
	var sawDirective bool
	for _, m := range fake.lastMsgs {
		if m.Role == "system" && strings.Contains(m.Content, "PLAN MODE is ON") {
			sawDirective = true
		}
	}
	if !sawDirective {
		t.Error("plan directive not injected into the prompt")
	}
}

func TestPlanModeAllowsReadOnly(t *testing.T) {
	reg := tool.NewRegistry(planFileTool())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "file", `{"action":"read","params":{"path":"x"}}`),
		{Role: "assistant", Content: "the file says X"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	a.SetPlanMode(true)

	tr, _ := a.Run(context.Background(), "read x")
	if obs := toolObservation(tr.Messages); len(obs) == 0 || !strings.Contains(obs[0].Content, "content-of-x") {
		t.Errorf("read-only call should run in plan mode: %+v", obs)
	}
}

// A model that keeps failing gets ONE rethink nudge; if it still fails, the run
// stops (bounded) and offers the user a steering suggestion.
func TestRunNudgesThenStops(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`) // empty action → always errors
	fake := &scriptLLM{replies: []llm.Message{bad, bad, bad, bad, bad, bad, bad, bad}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 20)

	tr, _ := a.Run(context.Background(), "do something")
	if tr.Steps > 2*maxStuckTurns+1 {
		t.Errorf("ran %d steps, want it bounded (~2x stuck, after one nudge)", tr.Steps)
	}
	if !strings.Contains(tr.Final, "Stopped") {
		t.Errorf("final = %q, want the stuck stop", tr.Final)
	}
	if !rt.has("nudge") {
		t.Error("expected one rethink nudge before stopping")
	}
	if rt.finalSuggest == "" {
		t.Error("the stop should offer the user a steering suggestion")
	}
}

// Reported live: a pasted transcript showed a failing run, a file write, then
// a SUCCESSFUL run, immediately followed by the "Stopped — it kept
// repeating..." message with no visible nudge line in between — looking like
// a false trigger from the outside. There was no way to tell, from the debug
// log alone, whether stuck/nudged carried over from turns earlier than
// whatever got pasted, or whether the reset-on-progress logic actually ran.
// This "stuck check" debug line must appear every turn (not just when a
// stuck/nudge/stop decision is made) so the counter's whole history —
// including a reset back to 0 on real progress — is reconstructable.
func TestStuckCheckDebugLogShowsCounterHistory(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(prev)

	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`) // empty action → always errors
	good := toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`)
	fake := &scriptLLM{replies: []llm.Message{bad, good, {Role: "assistant", Content: "done"}}}
	a := New(fake, reg, nil, nil, "", 5)

	if _, err := a.Run(context.Background(), "do something"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	logged := buf.String()
	if !strings.Contains(logged, `msg="stuck check" step=1 all_failed=true repeating=false stuck_before=0 nudged=false`) {
		t.Errorf("debug log missing the first (failing) turn's stuck state, got:\n%s", logged)
	}
	if !strings.Contains(logged, `msg="stuck check" step=2 all_failed=false repeating=false stuck_before=1 nudged=false`) {
		t.Errorf("debug log missing the second (successful) turn's stuck state (must show stuck_before=1 from the prior failure), got:\n%s", logged)
	}
}

// If the nudge unsticks the model (it answers), the run recovers instead of
// stopping.
func TestStuckNudgeRecovers(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	fake := &scriptLLM{replies: []llm.Message{
		bad, bad, bad, // 3 fails → nudge
		{Role: "assistant", Content: "I can't do that with calc — here's the answer in words."},
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 12)

	tr, _ := a.Run(context.Background(), "do x")
	if strings.Contains(tr.Final, "Stopped") {
		t.Errorf("should have recovered after the nudge, got: %q", tr.Final)
	}
	if !rt.has("nudge") {
		t.Error("expected a nudge before the recovery")
	}
	var injected bool
	for _, m := range fake.lastMsgs {
		if m.Role == "user" && strings.Contains(m.Content, "repeating the same tool") {
			injected = true
		}
	}
	if !injected {
		t.Error("the nudge message should be in the conversation sent to the model")
	}
}

// A model failing three DIFFERENT ways in a row (never repeating the same
// call) still trips the stuck counter (nErr == len(calls) each turn, with no
// repetition requirement) — but the nudge must not falsely tell it "you're
// repeating the same tool call" when it demonstrably isn't; that undermines
// the rest of the nudge (reported live: a model calling a hallucinated tool
// name, then a wrong action name, then the right call with empty params).
func TestStuckNudgeFramingForDistinctFailures(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"nope1","params":{}}`),
		toolCallReply("c2", "calc", `{"action":"nope2","params":{}}`),
		toolCallReply("c3", "calc", `{"action":"nope3","params":{}}`),
		{Role: "assistant", Content: "I can't do that with calc — here's the answer in words."},
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 12)

	a.Run(context.Background(), "do x")
	if !rt.has("nudge") {
		t.Fatal("expected a nudge")
	}
	var nudgeText string
	for _, m := range fake.lastMsgs {
		if m.Role == "user" && strings.Contains(m.Content, "tool call") {
			nudgeText = m.Content
		}
	}
	if strings.Contains(nudgeText, "repeating") {
		t.Errorf("nudge falsely claims repetition for three distinct failing calls: %q", nudgeText)
	}
	if !strings.Contains(nudgeText, "different tool calls") {
		t.Errorf("nudge doesn't name what actually happened: %q", nudgeText)
	}
}

type cancelLLM struct {
	calls  int
	cancel context.CancelFunc
}

func (c *cancelLLM) Chat(ctx context.Context, _ []llm.Message, _ []map[string]any) (llm.Message, error) {
	c.calls++
	if c.calls == 1 {
		c.cancel() // simulate the user pressing esc after the first action
		return toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`), nil
	}
	return llm.Message{}, ctx.Err()
}

// Cancelling (esc) mid-run keeps the partial work: the run ends cleanly (no error),
// is marked Cancelled, and is remembered so a follow-up can continue.
func TestRunCancelKeepsProgress(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	ctx, cancel := context.WithCancel(context.Background())
	a := New(&cancelLLM{cancel: cancel}, reg, nil, nil, "", 10)

	tr, err := a.Run(ctx, "do the thing")
	if err != nil {
		t.Fatalf("a cancel should end cleanly, got error: %v", err)
	}
	if !tr.Cancelled {
		t.Error("transcript should be marked Cancelled")
	}
	if a.SessionLen() == 0 {
		t.Error("a cancelled run with work done should be remembered so it can continue")
	}
}

// Progress between failure streaks earns a fresh nudge: a successful tool call
// clears the "already nudged" latch, so a couple of stumbles after good work don't
// insta-stop the run (the reported bug — a success didn't reset the error state).
func TestProgressResetsNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	good := toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`)
	fake := &scriptLLM{replies: []llm.Message{
		bad, bad, bad, // streak 1 → nudge #1
		good,          // progress → clears the nudged latch
		bad, bad, bad, // streak 2 → must nudge again, NOT insta-stop
		good,
		{Role: "assistant", Content: "done"},
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 30)

	tr, _ := a.Run(context.Background(), "do x")
	if strings.Contains(tr.Final, "Stopped") {
		t.Errorf("progress should reset the nudge latch, not stop: %q", tr.Final)
	}
	nudges := 0
	for _, k := range rt.kinds {
		if k == "nudge" {
			nudges++
		}
	}
	if nudges < 2 {
		t.Errorf("expected ≥2 nudges (one per streak, after a progress reset), got %d", nudges)
	}
}

func calcCall() llm.Message {
	return toolCallReply("c", "calc", `{"action":"calculate","params":{"expression":"1+1"}}`)
}

// ctxLLM wraps scriptLLM with a Context() reading that changes after every
// Chat call, mirroring OpenAIClient's real last-write-wins Context() field —
// so a test can prove Run() snapshots Transcript.PromptTokens from the MAIN
// turn's own Chat call, before a later same-iteration call (judgeGoal, which
// shares the same Chatter absent a dedicated judge model) gets a chance to
// overwrite that reading with its own, much shorter prompt.
type ctxLLM struct {
	scriptLLM
	ctxByCall []int // Context() reading to report after each Chat call, indexed by call order (0-based)
	ctx       int
}

func (c *ctxLLM) Chat(ctx context.Context, msgs []llm.Message, tools []map[string]any) (llm.Message, error) {
	msg, err := c.scriptLLM.Chat(ctx, msgs, tools)
	if i := c.scriptLLM.i - 1; i >= 0 && i < len(c.ctxByCall) {
		c.ctx = c.ctxByCall[i]
	}
	return msg, err
}

func (c *ctxLLM) Context() int { return c.ctx }

// Reproduces the bug: the goal judge (judgeGoal) calls a.llm.Chat again in the
// SAME iteration a final answer was produced, sharing the main client absent a
// dedicated judge model. Its own reply is much shorter, so it would otherwise
// leave the client's Context() reading far too low. Transcript.PromptTokens
// must reflect the MAIN turn's own reading (7000), not the judge's (1000).
func TestRunPromptTokensSnapshotBeforeJudgeClobbersContext(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &ctxLLM{
		scriptLLM: scriptLLM{replies: []llm.Message{
			calcCall(),                             // step 1: act — call index 0
			{Role: "assistant", Content: "did it"}, // step 2: finalize (MAIN turn) — call index 1
			{Role: "assistant", Content: "DONE"},   // judgeGoal's own call — call index 2
		}},
		ctxByCall: []int{500, 7000, 1000},
	}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3, false)

	tr, err := a.Run(context.Background(), "do it")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "did it" || !tr.GoalMet {
		t.Fatalf("final = %q, GoalMet = %v — want %q / true (judge said DONE)", tr.Final, tr.GoalMet, "did it")
	}
	if tr.PromptTokens != 7000 {
		t.Errorf("PromptTokens = %d, want 7000 (the MAIN turn's own Context(), not the judge's clobbered 1000)", tr.PromptTokens)
	}
}

// The goal loop re-feeds the goal when the judge says it isn't met, then accepts it
// once the judge says DONE. Returns counts the re-feeds; GoalMet records the verdict.
func TestRunGoalLoopRefeedsUntilJudgeSaysDone(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), // act
		{Role: "assistant", Content: "did part 1"},           // finalize #1
		{Role: "assistant", Content: "MORE: still need pt2"}, // judge: not met → re-feed
		calcCall(),                               // act again
		{Role: "assistant", Content: "all done"}, // finalize #2
		{Role: "assistant", Content: "DONE"},     // judge: met
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3, false)

	tr, err := a.Run(context.Background(), "do part 1 and part 2")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.Final != "all done" {
		t.Errorf("final = %q, want %q", tr.Final, "all done")
	}
	if tr.Returns != 1 {
		t.Errorf("returns = %d, want 1", tr.Returns)
	}
	if !tr.GoalMet {
		t.Error("GoalMet = false, want true (judge said DONE)")
	}
}

// A tool-less answer never triggers the judge (nothing was done to verify): the loop
// is gated on real progress, so a plain reply finalizes in one Chat call.
func TestRunGoalLoopSkipsToollessAnswer(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{{Role: "assistant", Content: "just an answer"}}}
	a := New(fake, reg, nil, nil, "", 10)
	a.SetGoalLoop(3, false)

	tr, _ := a.Run(context.Background(), "hi")
	if tr.Returns != 0 {
		t.Errorf("returns = %d, want 0 (no judge for a tool-less answer)", tr.Returns)
	}
	if fake.i != 1 {
		t.Errorf("Chat calls = %d, want 1 (the judge must not run)", fake.i)
	}
}

// When the judge never accepts, the loop stops at the TTL: the last finalize is taken
// as-is (no further judge call) and GoalMet stays false.
func TestRunGoalLoopStopsAtTTL(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	more := llm.Message{Role: "assistant", Content: "MORE: nope"}
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), {Role: "assistant", Content: "f1"}, more, // re-feed 1
		calcCall(), {Role: "assistant", Content: "f2"}, more, // re-feed 2
		calcCall(), {Role: "assistant", Content: "f3"}, // returns==TTL → accept, no judge
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(2, false)

	tr, _ := a.Run(context.Background(), "loop")
	if tr.Returns != 2 {
		t.Errorf("returns = %d, want 2 (TTL)", tr.Returns)
	}
	if tr.GoalMet {
		t.Error("GoalMet = true, want false (TTL exhausted, never judged done)")
	}
	// The last finalize stands, with the honest "not confirmed" note appended.
	if !strings.Contains(tr.Final, "f3") || !strings.Contains(tr.Final, "not confirmed complete") {
		t.Errorf("final = %q, want f3 + the not-confirmed note", tr.Final)
	}
}

// The judge must NOT rubber-stamp an unparseable reply as a met goal: it accepts
// the final (so it doesn't trap the loop) but leaves GoalMet false.
func TestRunGoalLoopUnclearJudgeDoesNotMarkMet(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(),
		{Role: "assistant", Content: "I think that's everything"}, // finalize
		{Role: "assistant", Content: "hmm, hard to say really"},   // judge: no DONE/MORE token
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3, false)

	tr, _ := a.Run(context.Background(), "do the thing")
	if tr.GoalMet {
		t.Error("GoalMet = true on an unparseable judge verdict, want false")
	}
	if tr.Returns != 0 {
		t.Errorf("returns = %d, want 0 (unclear verdict accepts, doesn't re-feed)", tr.Returns)
	}
	if tr.Final != "I think that's everything" {
		t.Errorf("final = %q", tr.Final)
	}
}

// With nudgeIdle OFF, a model that re-reads the goal after a re-feed but then
// finishes without doing any work is accepted immediately — and the final says the
// goal wasn't confirmed (not a misleading "done").
func TestRunGoalIdleNudgeOff(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), {Role: "assistant", Content: "did some"}, // act, finalize #1
		{Role: "assistant", Content: "MORE: not done"}, // judge → re-feed
		{Role: "assistant", Content: "I'll stop here"}, // idle finalize (no tool) after re-feed
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3, false) // idle-nudge off

	tr, _ := a.Run(context.Background(), "do it all")
	if tr.GoalMet {
		t.Error("GoalMet = true, want false (never confirmed)")
	}
	if !strings.Contains(tr.Final, "not confirmed complete") {
		t.Errorf("final = %q, want the goal-not-confirmed note", tr.Final)
	}
}

// With nudgeIdle ON, that same no-work finish gets ONE push; the model then does
// the work and the goal completes.
func TestRunGoalIdleNudgeOn(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		calcCall(), {Role: "assistant", Content: "did some"}, // act, finalize #1
		{Role: "assistant", Content: "MORE: not done"}, // judge → re-feed
		{Role: "assistant", Content: "I'll stop here"}, // idle finalize → nudged (no judge)
		calcCall(), {Role: "assistant", Content: "now done"}, // acts after the nudge, finalize #2
		{Role: "assistant", Content: "DONE"}, // judge: met
	}}
	a := New(fake, reg, nil, nil, "", 20)
	a.SetGoalLoop(3, true) // idle-nudge on

	tr, _ := a.Run(context.Background(), "do it all")
	// Reaching "now done" + DONE is only possible if the idle finish was nudged and
	// the model then worked (without the nudge it would stop at the stalled note).
	if !tr.GoalMet || tr.Final != "now done" {
		t.Errorf("idle nudge didn't recover: final=%q met=%v", tr.Final, tr.GoalMet)
	}
}

// A /btw side question is answered in its own no-tools turn between steps, and
// that answer does NOT become the task's final — the task keeps going.
func TestRunAnswersAside(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		{Role: "assistant", Content: "coverage is about 80%"}, // consumed by the aside answer
		{Role: "assistant", Content: "task finished"},         // the main task's finish
	}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 10)
	asked := false
	a.SetAsides(func() []string {
		if asked {
			return nil
		}
		asked = true
		return []string{"what's the test coverage?"}
	})

	tr, _ := a.Run(context.Background(), "do the thing")
	if !rt.has("aside") {
		t.Error("no aside event emitted for the /btw question")
	}
	if tr.Final != "task finished" {
		t.Errorf("final = %q — the aside answer must not become the task's final", tr.Final)
	}
}

// A model that double-encodes params as a JSON string AND puts a param at the top
// level (e.g. path) must not lose the top-level one.
func TestParseArgsStringParamsKeepsSiblings(t *testing.T) {
	action, params := parseArgs(`{"action":"write","path":"x.txt","params":"{\"content\":\"y\"}"}`)
	if action != "write" {
		t.Errorf("action = %q, want write", action)
	}
	if params["path"] != "x.txt" {
		t.Errorf("top-level path lost: %+v", params)
	}
	if params["content"] != "y" {
		t.Errorf("content = %v, want y", params["content"])
	}
}

func TestParseVerdict(t *testing.T) {
	cases := []struct {
		in       string
		want     judgeVerdict
		wantMiss string
	}{
		{"DONE", judgeDone, ""},
		{"The task is DONE.", judgeDone, ""},
		{"MORE: add the tests", judgeMore, "add the tests"},
		{"The plan needs MORE work: no tests yet", judgeMore, "work: no tests yet"},
		{"MOREOVER, it looks done", judgeDone, ""}, // MOREOVER must not match MORE
		{"yeah looks fine to me", judgeUnclear, ""},
		{"", judgeUnclear, ""},
	}
	for _, tc := range cases {
		got, miss := parseVerdict(tc.in)
		if got != tc.want || miss != tc.wantMiss {
			t.Errorf("parseVerdict(%q) = (%d,%q), want (%d,%q)", tc.in, got, miss, tc.want, tc.wantMiss)
		}
	}
}

// After the rethink nudge, a single further unproductive turn stops the run — the
// counter isn't reset on the nudge, so a degenerate model (which may think for
// minutes per turn) can't flail for another full budget before stopping.
func TestStuckStopsOneTurnAfterNudge(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	bad := toolCallReply("c", "calc", `{"action":"","params":{}}`)
	fake := &scriptLLM{replies: []llm.Message{bad, bad, bad, bad, bad, bad}}
	rt := &recTracer{}
	a := New(fake, reg, nil, rt, "", 30)

	tr, _ := a.Run(context.Background(), "do x")
	if !tr.Stopped || !strings.Contains(tr.Final, "Stopped") {
		t.Fatalf("want a stuck-stop, got stopped=%v final=%q", tr.Stopped, tr.Final)
	}
	if tr.Steps != 4 { // 3 turns to the nudge, 1 more dud turn to stop
		t.Errorf("steps = %d, want 4", tr.Steps)
	}
	nudges := 0
	for _, k := range rt.kinds {
		if k == "nudge" {
			nudges++
		}
	}
	if nudges != 1 {
		t.Errorf("nudges = %d, want 1", nudges)
	}
}

// An empty-action error must stay a single clean line — no full schema dump, no
// learned hints piled on (that buries the example for a weak model).
func TestEmptyActionErrorStaysTerse(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "k.json"))
	kb.Add(knowledge.Pitfall{Domain: "calc", ErrorPattern: "no action given", Context: "calc", ProvenFix: "provide an action"})
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c", "calc", `{"action":"","params":{}}`),
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, reg, kb, nil, "", 5)

	tr, _ := a.Run(context.Background(), "x")
	obs := toolObservation(tr.Messages)
	if len(obs) == 0 {
		t.Fatal("no observation")
	}
	c := obs[0].Content
	if !strings.Contains(c, "no action given") {
		t.Errorf("want the no-action message, got: %s", c)
	}
	if strings.Contains(c, "usage:") || strings.Contains(c, "Hints from past runs") {
		t.Errorf("empty-action error should stay terse, got:\n%s", c)
	}
}

func TestRunWrongToolHint(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc(), tool.NewWeb(nil, false))
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"search","params":{"query":"x"}}`),
		{Role: "assistant", Content: "ok"},
	}}
	a := New(fake, reg, nil, nil, "", 5)

	tr, _ := a.Run(context.Background(), "search for x")
	if got := toolObservation(tr.Messages); len(got) == 0 ||
		!strings.Contains(got[0].Content, `belongs to tool "web"`) {
		t.Errorf("wrong-tool correction not surfaced: %+v", got)
	}
}

// TestBeforeTurnInjectsBetweenTurns proves the /btw seam: a hook registered via
// SetBeforeTurn feeds messages into the working set BETWEEN turns of a live run,
// so a note dropped after the first turn reaches the model on the next one —
// without interrupting the loop.
func TestBeforeTurnInjectsBetweenTurns(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{replies: []llm.Message{
		toolCallReply("c1", "calc", `{"action":"calculate","params":{"expression":"2+2"}}`),
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, reg, nil, nil, "", 5)
	calls := 0
	a.SetBeforeTurn(func() []llm.Message {
		calls++
		if calls == 2 { // a note dropped after the first turn, mid-run
			return []llm.Message{llm.User("[by the way] use tabs")}
		}
		return nil
	})
	if _, err := a.Run(context.Background(), "compute"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	found := false
	for _, m := range fake.lastMsgs { // messages the model saw on the final (2nd) turn
		if m.Role == "user" && strings.Contains(m.Content, "[by the way] use tabs") {
			found = true
		}
	}
	if !found {
		t.Fatalf("beforeTurn note never reached the model on the next turn")
	}
	if calls < 2 {
		t.Errorf("beforeTurn called %d times, want it fired each turn", calls)
	}
}

// TestAnswerAsideSnapshotDoesNotRaceWithReset reproduces the idle /btw data
// race: the caller (mirroring tui.go's stIdle handler) launches AnswerAside on
// its own goroutine while the user is free to run Reset() (/clear) from the
// idle prompt. Before the fix, AnswerAside read a.history/a.system live from
// that goroutine, racing Reset()'s unsynchronized write. The fix has the
// caller capture base (System()+History()) synchronously, BEFORE starting the
// goroutine, so the goroutine never touches Agent fields at all. Run with
// -race: this must stay clean.
func TestAnswerAsideSnapshotDoesNotRaceWithReset(t *testing.T) {
	reg := tool.NewRegistry(tool.NewCalc())
	fake := &scriptLLM{}
	a := New(fake, reg, nil, nil, "", 10)
	a.SetHistory([]llm.Message{llm.User("g0"), {Role: "assistant", Content: "a0"}})

	// Captured synchronously, before the goroutine starts — the snapshot the
	// fix requires.
	base := append([]llm.Message{llm.System(a.System())}, a.History()...)

	done := make(chan struct{})
	go func() {
		a.AnswerAside(context.Background(), base, "what happened?")
		close(done)
	}()
	a.Reset() // the idle /clear equivalent, run concurrently with the aside
	<-done
}
