package main

import "sync"

// uiBridge is the seam between the agent (running in a background goroutine) and
// the Bubble Tea UI loop. It implements both trace.Tracer (live step events) and
// tool.Approver (interactive permission prompts), turning each into a channel
// message the UI can handle one at a time — so concurrent tool-call approvals
// can't deadlock on a shared stdin reader the way the plain prompt could.
type uiBridge struct {
	events    chan uiEvent
	approvals chan approvalReq

	mu    sync.Mutex
	abort chan struct{} // closed to deny all in-flight approvals on task cancel
}

type uiEvent struct {
	kind   string
	fields map[string]any
}

type approvalReq struct {
	kind, detail string
	reply        chan bool
}

func newBridge() *uiBridge {
	return &uiBridge{
		events:    make(chan uiEvent, 512),
		approvals: make(chan approvalReq),
		abort:     make(chan struct{}),
	}
}

// Emit (trace.Tracer) hands a step event to the UI. Buffered, so the agent only
// blocks if the UI falls far behind.
func (b *uiBridge) Emit(kind string, fields map[string]any) {
	b.events <- uiEvent{kind: kind, fields: fields}
}

// Approve (tool.Approver) blocks the calling tool until the UI answers, or
// returns false (deny) if the task is cancelled while it's waiting — so esc
// never leaves a tool goroutine blocked forever.
func (b *uiBridge) Approve(kind, detail string) bool {
	b.mu.Lock()
	abort := b.abort
	b.mu.Unlock()

	reply := make(chan bool, 1)
	select {
	case b.approvals <- approvalReq{kind: kind, detail: detail, reply: reply}:
	case <-abort:
		return false
	}
	select {
	case ok := <-reply:
		return ok
	case <-abort:
		return false
	}
}

// Abort denies every in-flight and pending approval (the task was cancelled) by
// closing the current signal. Closing — rather than swapping — means a goroutine
// that reads b.abort at any moment gets the same channel, so there's no window
// where a waiter misses the abort.
func (b *uiBridge) Abort() {
	b.mu.Lock()
	defer b.mu.Unlock()
	select {
	case <-b.abort: // already closed
	default:
		close(b.abort)
	}
}

// arm installs a fresh abort signal for a new task, so a previous cancel doesn't
// keep denying. Called at task start, before any tool can ask for approval.
func (b *uiBridge) arm() {
	b.mu.Lock()
	b.abort = make(chan struct{})
	b.mu.Unlock()
}
