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
		NewFile(pol, nil, nil), NewRun(pol, nil, 0), NewGit(pol, nil),
		NewWeb(nil, false), NewHelp(kb, nil), NewCalc(),
	)
	b, _ := json.Marshal(reg.OpenAITools())
	t.Logf("tool catalog: %d bytes (~%d tokens)", len(b), len(b)/4)
	const maxBytes = 4200 // ~1050 tokens
	if len(b) > maxBytes {
		t.Errorf("catalog = %d bytes (~%d tok), over the %d-byte budget — trim a tool description", len(b), len(b)/4, maxBytes)
	}
}
