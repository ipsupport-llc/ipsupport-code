package tool

import (
	"encoding/json"
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/knowledge"
	"github.com/ipsupport-llc/ipsupport-code/internal/policy"
)

// TestCatalogTokenBudget guards the per-turn token cost: the whole tool catalog
// ships in EVERY request to a small local model, so it must stay lean.
func TestCatalogTokenBudget(t *testing.T) {
	c := config.Default()
	c.Workspace = "."
	pol, _ := policy.New(c)
	kb, _ := knowledge.Open("")
	reg := NewRegistry(
		NewFile(pol, nil, nil), NewRun(pol, nil, 0, 0), NewGit(pol, nil, 0, false),
		NewWeb(nil, false), NewHelp(kb, nil), NewCalc(),
	)
	b, _ := json.Marshal(reg.OpenAITools())
	t.Logf("tool catalog: %d bytes (~%d tokens)", len(b), len(b)/4)
	// Raised from 4200 when git gained clone/fetch/pull/push/remote: five actions
	// with their params cost 327 bytes even with every note trimmed to the few
	// words that carry weight ("(ff-only)", "(PUBLISHES; never force)"). That is
	// the price of the capability, not description sprawl — so the ceiling moves
	// by what it actually cost and no more. The headroom is deliberately thin:
	// the next tool to grow should have to justify itself the same way.
	const maxBytes = 4600 // ~1150 tokens
	if len(b) > maxBytes {
		t.Errorf("catalog = %d bytes (~%d tok), over the %d-byte budget — trim a tool description", len(b), len(b)/4, maxBytes)
	}
}
