package tool

import (
	"context"
	"fmt"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
)

type helpTool struct {
	kb    *knowledge.KB
	usage func(domain string) string // real tool contract lookup (may be nil)
}

// NewHelp returns the help tool: an explicit escape hatch the model can call to
// see a domain's REAL usage (its current schema) plus the lessons learned for
// it. usage is the live contract lookup (e.g. Registry.Usage); it may be nil.
func NewHelp(kb *knowledge.KB, usage func(domain string) string) Tool {
	return &helpTool{kb: kb, usage: usage}
}

func (*helpTool) Name() string      { return "help" }
func (*helpTool) Actions() []string { return []string{"lessons"} }

func (*helpTool) Description() string {
	return strings.TrimSpace(`See a tool domain's real usage (its current actions + params) and the lessons learned for it. Consult this when a tool keeps failing.
Actions:
  - lessons: {"domain": str}   one of: file, run, git, web, calc
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

	var b strings.Builder
	if h.usage != nil {
		if u := strings.TrimSpace(h.usage(domain)); u != "" {
			fmt.Fprintf(&b, "Usage of %s:\n%s\n\n", domain, u)
		}
	}
	lessons := h.kb.Query(domain, "", 20)
	if len(lessons) == 0 {
		b.WriteString("No lessons recorded for " + domain + " yet.")
	} else {
		fmt.Fprintf(&b, "Lessons for %s:\n", domain)
		for _, p := range lessons {
			fmt.Fprintf(&b, "- when you saw %q while %s, this worked: %s\n", p.ErrorPattern, p.Context, p.ProvenFix)
		}
	}
	return Ok(strings.TrimRight(b.String(), "\n"))
}
