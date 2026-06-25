// Package e2e exercises the whole agent stack end to end: a real
// llm.OpenAIClient talking to a fake LM Studio over HTTP, driving the real agent
// loop, the real tools, and the real permission policy against a real temp
// filesystem. No mocks below the LLM boundary.
package e2e

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/llm"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
	"github.com/ipsupport-llc/ipsupport-code/internal/reflect"
	"github.com/ipsupport-llc/ipsupport-code/internal/tool"
)

// ---- fake LM Studio ----------------------------------------------------------

type fakeLM struct {
	mu    sync.Mutex
	queue []string // raw JSON response bodies, returned in order
	reqs  []string // captured request bodies
}

func (f *fakeLM) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.reqs = append(f.reqs, string(body))
	resp := contentResp("done")
	if len(f.queue) > 0 {
		resp = f.queue[0]
		f.queue = f.queue[1:]
	}
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, resp)
}

func (f *fakeLM) request(i int) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i < 0 || i >= len(f.reqs) {
		return ""
	}
	return f.reqs[i]
}

func msgResp(msg map[string]any) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": msg}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
	})
	return string(b)
}

func contentResp(text string) string {
	return msgResp(map[string]any{"role": "assistant", "content": text})
}

func toolResp(name string, args map[string]any) string {
	a, _ := json.Marshal(args)
	return msgResp(map[string]any{
		"role":    "assistant",
		"content": "",
		"tool_calls": []map[string]any{{
			"id": "c1", "type": "function",
			"function": map[string]any{"name": name, "arguments": string(a)},
		}},
	})
}

func call(action string, params map[string]any) map[string]any {
	return map[string]any{"action": action, "params": params}
}

// ---- real stack --------------------------------------------------------------

type allowApprover struct{}

func (allowApprover) Approve(_, _ string) bool { return true }

type recTracer struct {
	mu     sync.Mutex
	events []map[string]any
}

func (r *recTracer) Emit(kind string, fields map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e := map[string]any{"kind": kind}
	for k, v := range fields {
		e[k] = v
	}
	r.events = append(r.events, e)
}

// observationContains reports whether any observation event's content holds sub.
func (r *recTracer) observationContains(sub string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, e := range r.events {
		if e["kind"] == "observation" {
			if c, _ := e["content"].(string); strings.Contains(c, sub) {
				return true
			}
		}
	}
	return false
}

func buildStack(t *testing.T, url, ws string, kb *knowledge.KB) (*agent.Agent, *recTracer, *knowledge.KB) {
	t.Helper()
	if kb == nil {
		kb, _ = knowledge.Open(filepath.Join(t.TempDir(), "kb.json"))
	}
	c := config.Default() // includes the protective deny floor
	c.Workspace = ws
	c.LLM.BaseURL = url
	c.LLM.Model = "fake"
	c.Run.Default = "allow"
	c.File.Default = "allow"
	c.File.Jail = "."

	pol, err := policy.New(c)
	if err != nil {
		t.Fatal(err)
	}
	ap := allowApprover{}
	var reg *tool.Registry
	reg = tool.NewRegistry(
		tool.NewFile(pol, ap),
		tool.NewRun(pol, ap),
		tool.NewGit(pol, ap),
		tool.NewWeb(http.DefaultClient),
		tool.NewHelp(kb, func(d string) string { return reg.Usage(d) }),
		tool.NewCalc(),
	)
	tr := &recTracer{}
	ag := agent.New(llm.NewOpenAIClient(c.LLM), reg, kb, tr, "do tasks with tools", 8)
	return ag, tr, kb
}

func serve(t *testing.T, f *fakeLM) string {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return srv.URL
}

// ---- scenarios ---------------------------------------------------------------

func TestE2E_FileWriteThenRead(t *testing.T) {
	ws := t.TempDir()
	f := &fakeLM{queue: []string{
		toolResp("file", call("write", map[string]any{"path": "hello.txt", "content": "hi there"})),
		toolResp("file", call("read", map[string]any{"path": "hello.txt"})),
		contentResp("Wrote hello.txt; it reads: hi there"),
	}}
	ag, _, _ := buildStack(t, serve(t, f), ws, nil)

	tr, err := ag.Run(context.Background(), "create hello.txt saying 'hi there' and read it back")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(ws, "hello.txt"))
	if err != nil {
		t.Fatalf("file was not actually written: %v", err)
	}
	if string(data) != "hi there" {
		t.Errorf("file content = %q, want 'hi there'", data)
	}
	if !strings.Contains(tr.Final, "hi there") {
		t.Errorf("final = %q", tr.Final)
	}
}

func TestE2E_RunShell(t *testing.T) {
	ws := t.TempDir()
	f := &fakeLM{queue: []string{
		toolResp("run", call("shell", map[string]any{"command": "echo e2e-ok"})),
		contentResp("ran echo, output was e2e-ok"),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), ws, nil)

	if _, err := ag.Run(context.Background(), "echo e2e-ok"); err != nil {
		t.Fatal(err)
	}
	if !tr.observationContains("e2e-ok") {
		t.Error("shell output 'e2e-ok' did not flow back as an observation")
	}
}

func TestE2E_Calc(t *testing.T) {
	f := &fakeLM{queue: []string{
		toolResp("calc", call("calculate", map[string]any{"expression": "(1234*9)+2"})),
		contentResp("The result is 11108."),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), t.TempDir(), nil)

	if _, err := ag.Run(context.Background(), "compute (1234*9)+2"); err != nil {
		t.Fatal(err)
	}
	if !tr.observationContains("11108") {
		t.Error("calc did not produce 11108 in an observation")
	}
}

func TestE2E_PolicyDeniesDestructiveShell(t *testing.T) {
	ws := t.TempDir()
	sentinel := filepath.Join(ws, "victim.txt")
	if err := os.WriteFile(sentinel, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := &fakeLM{queue: []string{
		toolResp("run", call("shell", map[string]any{"command": "rm -rf " + sentinel})),
		contentResp("could not do that"),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), ws, nil)

	if _, err := ag.Run(context.Background(), "delete the victim file"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(sentinel); err != nil {
		t.Error("protective deny floor failed: rm -rf executed and removed the file")
	}
	if !tr.observationContains("denied") {
		t.Error("denial was not surfaced to the model")
	}
}

func TestE2E_PitfallInjectedOnError(t *testing.T) {
	kb, _ := knowledge.Open(filepath.Join(t.TempDir(), "kb.json"))
	kb.Add(knowledge.Pitfall{
		Domain: "calc", ErrorPattern: "unknown function",
		Context: "calc expression", ProvenFix: "use only whitelisted functions like sqrt/pow",
	})
	f := &fakeLM{queue: []string{
		toolResp("calc", call("calculate", map[string]any{"expression": "frobnicate(2)"})), // errors
		contentResp("I'll avoid that function"),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), t.TempDir(), kb)

	if _, err := ag.Run(context.Background(), "compute frobnicate(2)"); err != nil {
		t.Fatal(err)
	}
	if !tr.observationContains("use only whitelisted functions") {
		t.Error("matching pitfall was not injected into the tool error")
	}
}

func TestE2E_ReflectionPersistsLesson(t *testing.T) {
	f := &fakeLM{queue: []string{
		contentResp(`Here is what I learned:
[{"domain":"run","error_pattern":"permission denied","context":"writing system paths","proven_fix":"use sudo via the run tool"}]`),
	}}
	client := llm.NewOpenAIClient(config.LLM{BaseURL: serve(t, f), Model: "fake"})

	transcript := agent.Transcript{
		Messages: []llm.Message{
			llm.User("write to /etc/hosts"),
			{Role: "tool", Name: "run", Content: "exit 1 permission denied"},
		},
		Final: "could not write",
	}
	lessons, err := reflect.New(client).Reflect(context.Background(), transcript)
	if err != nil {
		t.Fatal(err)
	}
	if len(lessons) != 1 || lessons[0].Domain != "run" || !strings.Contains(lessons[0].ProvenFix, "sudo") {
		t.Fatalf("reflection lessons = %+v", lessons)
	}
}

func TestE2E_SessionMemoryAcrossTurns(t *testing.T) {
	f := &fakeLM{queue: []string{
		contentResp("the answer is 4"),     // turn 1
		contentResp("we computed 2+2 = 4"), // turn 2
	}}
	ag, _, _ := buildStack(t, serve(t, f), t.TempDir(), nil)

	if _, err := ag.Run(context.Background(), "what is 2+2"); err != nil {
		t.Fatal(err)
	}
	if _, err := ag.Run(context.Background(), "what did we just do?"); err != nil {
		t.Fatal(err)
	}
	// The second HTTP request must carry the first turn's goal and answer.
	second := f.request(1)
	if !strings.Contains(second, "what is 2+2") || !strings.Contains(second, "the answer is 4") {
		t.Errorf("second request lacks session memory:\n%s", second)
	}
}

// TestE2E_Binary builds and runs the real binary one-shot against the fake LM
// Studio — the truest end-to-end path: config load → wiring → agent → tool →
// reflect → stdout. Skipped with -short (it compiles the binary).
func TestE2E_Binary(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping binary build in -short mode")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not found")
	}
	bin := filepath.Join(t.TempDir(), "ipsupport-code")
	if out, err := exec.Command("go", "build", "-o", bin,
		"github.com/ipsupport-llc/ipsupport-code/cmd/agent").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	f := &fakeLM{queue: []string{
		toolResp("calc", call("calculate", map[string]any{"expression": "2+2"})),
		contentResp("The answer is 4."),
		contentResp("[]"), // reflection: no lessons
	}}
	url := serve(t, f)

	ws := t.TempDir()
	mustWrite(t, filepath.Join(ws, ".agent", "config.json"),
		`{"llm":{"base_url":"`+url+`/v1","model":"fake","max_steps":4},`+
			`"run":{"default":"allow"},"file":{"default":"allow","jail":"."}}`)

	cmd := exec.Command(bin, "-C", ws, "use calc to compute 2+2")
	cmd.Env = append(os.Environ(), "HOME="+t.TempDir()) // no global config
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "The answer is 4.") {
		t.Errorf("binary stdout = %q, want the final answer", out)
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestE2E_UsageGuidanceOnError(t *testing.T) {
	f := &fakeLM{queue: []string{
		toolResp("file", call("write", map[string]any{"content": "hi"})), // missing required "path"
		contentResp("ok, I see"),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), t.TempDir(), nil)

	if _, err := ag.Run(context.Background(), "write a file"); err != nil {
		t.Fatal(err)
	}
	if !tr.observationContains("missing required param(s): path") {
		t.Error("the precise missing-param error was not surfaced")
	}
	if !tr.observationContains("file usage") || !tr.observationContains("edit:") {
		t.Error("the tool schema was not injected to guide the model on a misuse error")
	}
}

func TestE2E_Git(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	ws := t.TempDir()
	for _, args := range [][]string{
		{"init"}, {"config", "user.email", "t@example.com"}, {"config", "user.name", "tester"},
	} {
		if out, err := exec.Command("git", append([]string{"-C", ws}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git setup %v: %v\n%s", args, err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(ws, "new.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := &fakeLM{queue: []string{
		toolResp("git", call("status", map[string]any{})),
		contentResp("there is one untracked file: new.txt"),
	}}
	ag, tr, _ := buildStack(t, serve(t, f), ws, nil)

	if _, err := ag.Run(context.Background(), "what's the git status?"); err != nil {
		t.Fatal(err)
	}
	if !tr.observationContains("new.txt") {
		t.Error("git status did not report the untracked file")
	}
}
