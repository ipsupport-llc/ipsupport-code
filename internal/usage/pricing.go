package usage

import (
	"sort"
	"strings"
)

// Price is a model's cost in USD per MILLION tokens, split by direction.
type Price struct{ In, Out float64 }

// DefaultPrices are rough list prices (USD / 1M tokens) for common models,
// matched by the longest substring of the model id. They drift over time and are
// approximate — /usage labels the figure "est." and a `prices` config map
// overrides any entry. A ":free" model (OpenRouter) is always $0.
var DefaultPrices = map[string]Price{
	"gpt-4o-mini":       {0.15, 0.60},
	"gpt-4o":            {2.50, 10.00},
	"chatgpt-4o-latest": {5.00, 15.00},
	"claude-3-5-haiku":  {0.80, 4.00},
	"claude-3-5-sonnet": {3.00, 15.00},
	"claude-3-7-sonnet": {3.00, 15.00},
	"claude-3-opus":     {15.00, 75.00},
	"sonnet":            {3.00, 15.00},
	"haiku":             {0.80, 4.00},
	"opus":              {15.00, 75.00},
	"grok-2":            {2.00, 10.00},
	"grok-4":            {3.00, 15.00},
	"llama-3.3-70b":     {0.59, 0.79},
	"llama-3.1-8b":      {0.05, 0.08},
}

// PriceFor returns the price for a model id: overrides first (longest substring
// match), then DefaultPrices, then zero (unknown / free). A ":free" id is $0.
func PriceFor(model string, overrides map[string]Price) Price {
	m := strings.ToLower(model)
	if strings.Contains(m, ":free") {
		return Price{}
	}
	if p, ok := longestMatch(m, overrides); ok {
		return p
	}
	if p, ok := longestMatch(m, DefaultPrices); ok {
		return p
	}
	return Price{}
}

// longestMatch finds the table key that is the longest substring of m.
func longestMatch(m string, table map[string]Price) (Price, bool) {
	keys := make([]string, 0, len(table))
	for k := range table {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return len(keys[i]) > len(keys[j]) })
	for _, k := range keys {
		if strings.Contains(m, strings.ToLower(k)) {
			return table[k], true
		}
	}
	return Price{}, false
}

// CostUSD estimates the dollar cost of prompt+completion tokens for a model.
func CostUSD(model string, prompt, completion int, overrides map[string]Price) float64 {
	p := PriceFor(model, overrides)
	return float64(prompt)/1e6*p.In + float64(completion)/1e6*p.Out
}

// CostSince sums the estimated cost of all entries on or after cutoff, pricing
// each by its own model. cutoff "" totals everything.
func (s *Store) CostSince(cutoff string, overrides map[string]Price) float64 {
	var total float64
	for _, e := range s.entries {
		if e.Date >= cutoff {
			total += CostUSD(e.Model, e.Prompt, e.Completion, overrides)
		}
	}
	return total
}
