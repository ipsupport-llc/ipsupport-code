package main

import (
	"testing"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// A built-in provider whose key comes from api_key_cmd counts as configured,
// with no api_key or env var behind it.
func TestAKeyCommandCountsAsAKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	cfg := config.Default()
	cfg.Provider = "openai"
	cfg.Providers = map[string]config.LLM{"openai": {APIKeyCmd: []string{"get-key"}}}
	if err := checkActiveProvider(cfg); err != nil {
		t.Errorf("checkActiveProvider: %v", err)
	}
	a := &app{cfg: cfg}
	if _, why := a.providerConn("openai"); why != "" {
		t.Errorf("providerConn: %s", why)
	}
	found := false
	for _, n := range a.configuredProviderNames() {
		found = found || n == "openai"
	}
	if !found {
		t.Errorf("configured providers %v lack openai", a.configuredProviderNames())
	}
	cfg.Providers = nil
	a = &app{cfg: cfg}
	if _, why := a.providerConn("openai"); why == "" {
		t.Error("no key and no command still counted as configured")
	}
}
