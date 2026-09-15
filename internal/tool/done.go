package tool

import "context"

// NewDone returns the done tool: a bare, parameter-less signal a model can
// call, instead of a real action, to end the run. Reported live: a weak/
// tool-trained local model can be so habituated to always calling SOMETHING
// that it never produces the plain no-tool-call reply the loop treats as a
// clean finish — instead it invents a workaround, like repeatedly running a
// shell echo of "TASK COMPLETE" or "EXIT 0". The model was genuinely trying
// to say it was done, just through the wrong channel. This gives it a real,
// correct one: the agent loop (see Agent.Run's isDoneOnly) treats a turn whose
// ONLY tool call is done exactly like a plain no-tool-call finalize, using the
// same reply's own Content as the answer — done itself is never actually
// dispatched. It does NOT confirm the goal met on its own: the goal judge
// (a separate, independent mechanism) still decides that regardless of how
// the model signaled it thought it was finished.
func NewDone() Tool {
	return NewDomain(DomainSpec{
		Name:    "done",
		Summary: "Finish the task: call this alongside a short summary of what you did, written as your normal reply text — or with nothing more to say, just this.",
		Details: "e.g. {\"action\":\"done\",\"params\":{}}",
		NotHere: "NOT here — this only ends the run; it does nothing else. You can also just reply with no tool call at all — either way works.",
		Actions: []Action{{
			Name: "done",
			Run: func(context.Context, Args) Result {
				// Never actually reached: the agent loop intercepts a done-only
				// turn before any tool call is dispatched. Only exists so a
				// call the loop somehow doesn't intercept (done arriving
				// alongside other real tool calls, deliberately left
				// unintercepted — see isDoneOnly) still gets a harmless answer
				// instead of an "unknown tool" error.
				return Ok("(noted — finish your reply with no further tool calls to end the run)")
			},
		}},
	})
}
