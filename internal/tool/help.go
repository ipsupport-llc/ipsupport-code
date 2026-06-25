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
	h := &helpTool{kb: kb, usage: usage}
	return NewDomain(DomainSpec{
		Name:    "help",
		Summary: "See a tool domain's real usage (its current actions + params) and the lessons learned for it. Consult this when a tool keeps failing.",
		NotHere: "NOT here — to actually run a command use run; to read files use file.",
		Actions: []Action{
			{Name: "lessons", Params: []Param{Req("domain", "str")}, Note: "(one of: file, run, git, web, calc)", Run: h.lessons},
		},
	})
}

func (h *helpTool) lessons(_ context.Context, a Args) Result {
	domain := a.Str("domain")

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
