package main

// uiBridge is the seam between the agent (running in a background goroutine) and
// the Bubble Tea UI loop. It implements both trace.Tracer (live step events) and
// tool.Approver (interactive permission prompts), turning each into a channel
// message the UI can handle one at a time — so concurrent tool-call approvals
// can't deadlock on a shared stdin reader the way the plain prompt could.
type uiBridge struct {
	events    chan uiEvent
	approvals chan approvalReq
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
	}
}

// Emit (trace.Tracer) hands a step event to the UI. Buffered, so the agent only
// blocks if the UI falls far behind.
func (b *uiBridge) Emit(kind string, fields map[string]any) {
	b.events <- uiEvent{kind: kind, fields: fields}
}

// Approve (tool.Approver) blocks the calling tool until the UI answers.
func (b *uiBridge) Approve(kind, detail string) bool {
	reply := make(chan bool, 1)
	b.approvals <- approvalReq{kind: kind, detail: detail, reply: reply}
	return <-reply
}
