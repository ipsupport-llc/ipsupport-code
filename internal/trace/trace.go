// Package trace records the agent's decision path — goal, model turns, tool
// calls, observations, final answer, and learned lessons — as JSONL. One run is
// one trajectory; the file is the future training dataset (state → action →
// observation → outcome).
package trace

import (
	"bytes"
	"encoding/json"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
	"github.com/ipsupport-llc/ipsupport-code/internal/redact"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Tracer receives one structured event per agent step.
type Tracer interface {
	Emit(kind string, fields map[string]any)
}

// FileTracer appends JSONL records to a file. Emit is safe for concurrent use.
type FileTracer struct {
	mu    sync.Mutex
	f     *os.File
	enc   *json.Encoder
	runID string
}

// NewFileTracer opens (creating + appending) the trace file at path, tagging
// every record with runID.
func NewFileTracer(path, runID string) (*FileTracer, error) {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	// The trace file is global: every run on the machine appends to it. A record
	// is large enough that an O_APPEND write is not atomic, so two processes can
	// interleave mid-line and corrupt both records.
	if unlock, err := filelock.Lock(path); err == nil {
		defer unlock()
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	// The trace file records prompts, tool-call arguments, and file-read
	// observations — potentially sensitive content. O_CREATE's mode only
	// applies when the file doesn't already exist, so a pre-existing trace
	// file (e.g. from before this fix) would otherwise keep looser
	// permissions forever; tighten it on every open.
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return nil, err
	}
	return &FileTracer{f: f, enc: json.NewEncoder(f), runID: runID}, nil
}

// Emit writes one JSONL record: the standard time/run/kind fields plus the
// caller's fields, with credentials masked (internal/redact). The log keeps
// every command and tool output, and outlives the session: a token that once
// passed through the agent would otherwise go wherever the file goes next.
func (t *FileTracer) Emit(kind string, fields map[string]any) {
	if t == nil {
		return
	}
	rec := map[string]any{
		"time": time.Now().UTC().Format(time.RFC3339Nano),
		"run":  t.runID,
		"kind": kind,
	}
	for k, v := range fields {
		rec[k] = v
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	_ = t.enc.Encode(masked(rec)) // Encode appends the newline → JSONL
}

// masked is rec as plain JSON values with every string redacted. Going
// through JSON first reaches strings inside any type a caller passes (a named
// map, a struct), not only the ones a type switch knows; UseNumber keeps
// numbers exact.
func masked(rec map[string]any) any {
	b, err := json.Marshal(rec)
	if err != nil {
		return rec
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var plain any
	if dec.Decode(&plain) != nil {
		return rec
	}
	return redact.Value(plain)
}

// Close closes the underlying file.
func (t *FileTracer) Close() error {
	if t == nil {
		return nil
	}
	return t.f.Close()
}
