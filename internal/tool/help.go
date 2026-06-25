package tool

import (
	"context"
	"fmt"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

type helpTool struct{ kb *knowledge.KB }

// NewHelp returns the help tool: an explicit escape hatch the model can call to
// recall learned lessons for a domain. (The agent loop also injects matching
// lessons into tool errors automatically.)
func NewHelp(kb *knowledge.KB) Tool { return &helpTool{kb: kb} }

func (*helpTool) Name() string      { return "help" }
func (*helpTool) Actions() []string { return []string{"lessons"} }

func (*helpTool) Description() string {
	return strings.TrimSpace(`Recall hard-won lessons (past errors and their proven fixes) for a tool domain when something keeps failing.
Actions:
  - lessons: {"domain": str}   one of: file, run, web, calc
NOT here — to actually run a command use run; to read files use file.`)
}

func (h *helpTool) Call(_ context.Context, action string, params map[string]any) Result {
	if action != "lessons" {
		return Err("help: unknown action " + action)
	}
	if err := Require(params, "domain"); err != nil {
		return Err(err.Error())
	}
	domain := Str(params, "domain")
	lessons := h.kb.Query(domain, "", 20)
	if len(lessons) == 0 {
		return Ok("no lessons recorded for " + domain + " yet")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Lessons for %s:\n", domain)
	for _, p := range lessons {
		fmt.Fprintf(&b, "- when you saw %q while %s, this worked: %s\n", p.ErrorPattern, p.Context, p.ProvenFix)
	}
	return Ok(strings.TrimRight(b.String(), "\n"))
}
