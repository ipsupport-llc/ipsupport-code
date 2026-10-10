package main

import (
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// Editing the active connection from inside the app: its address, key, type,
// model and context size, plus adding and removing providers. /config used to
// offer none of the first two for the local server — not even a hint that
// worked — so a server moved to another port meant quitting and hand-editing
// config.json.

// updateActive applies edit to the active connection — the local one (llm) or
// the named provider's saved preset — and writes it to the global file. A
// write that fails takes the edit back, so the screen never shows a value that
// is not on disk.
func (a *app) updateActive(edit func(*config.LLM)) error {
	if a.isLocal() {
		before := a.cfg.LLM
		edit(&a.cfg.LLM)
		if err := config.SaveGlobal(a.cfg.Name, a.cfg.LLM); err != nil {
			a.cfg.LLM = before
			return err
		}
		return nil
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	before, had := a.cfg.Providers[a.cfg.Provider]
	p := before
	edit(&p)
	a.cfg.Providers[a.cfg.Provider] = p
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		if had {
			a.cfg.Providers[a.cfg.Provider] = before
		} else {
			delete(a.cfg.Providers, a.cfg.Provider)
		}
		return err
	}
	return nil
}

// reconnect re-wires after a connection change, with a fresh window probe:
// the old one was for another endpoint or model.
func (a *app) reconnect() error {
	a.windowDetected = a.activeLLM().ContextWindowManual
	a.modelEpoch.Add(1)
	return a.wire()
}

// parseBaseURL accepts an http(s) URL with a host — "http://" alone, or a bare
// host, is not an address anyone can reach.
func parseBaseURL(raw string) (string, error) {
	s := strings.TrimRight(strings.TrimSpace(raw), "/")
	u, err := url.Parse(s)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("not an address: %q — e.g. http://localhost:1234/v1", raw)
	}
	if p := u.Port(); p != "" {
		if n, err := strconv.Atoi(p); err != nil || n < 1 || n > 65535 {
			return "", fmt.Errorf("port %s is out of range", p)
		}
	}
	return s, nil
}

// setBaseURL changes where the active connection points. For a built-in
// provider an empty value goes back to its default address.
func (a *app) setBaseURL(raw string) (string, error) {
	var v string
	if strings.TrimSpace(raw) == "" && !a.isLocal() && !config.IsCustomProvider(a.cfg, a.cfg.Provider) {
		v = "" // back to the template's address
	} else {
		var err error
		if v, err = parseBaseURL(raw); err != nil {
			return "", err
		}
	}
	if err := a.updateActive(func(l *config.LLM) { l.BaseURL = v }); err != nil {
		return "", err
	}
	if err := a.reconnect(); err != nil {
		return "", err
	}
	return "address → " + a.activeLLM().BaseURL, nil
}

// setActiveKey stores the active connection's API key; clear removes it.
func (a *app) setActiveKey(token string, clear bool) (string, error) {
	token = strings.TrimSpace(token)
	if !clear && token == "" {
		return "key unchanged", nil
	}
	if err := a.updateActive(func(l *config.LLM) { l.APIKey = token }); err != nil {
		return "", err
	}
	if err := a.wire(); err != nil {
		return "", err
	}
	if clear {
		return "key removed for " + a.providerName(), nil
	}
	return "key saved for " + a.providerName(), nil
}

// cycleLocalType switches the local connection between LM Studio's native API
// (model list, context detection) and plain OpenAI-compatible.
func (a *app) cycleLocalType() (string, error) {
	if !a.isLocal() {
		return "", fmt.Errorf("%s is OpenAI-compatible; the type applies to the local server", a.providerName())
	}
	// "openai", not "": an empty type is not written to the file, and the
	// default (lmstudio) came back at the next launch.
	next := "lmstudio"
	if a.cfg.LLM.Type == "lmstudio" {
		next = "openai"
	}
	if err := a.updateActive(func(l *config.LLM) { l.Type = next }); err != nil {
		return "", err
	}
	if err := a.reconnect(); err != nil {
		return "", err
	}
	return "server type → " + localTypeLabel(next), nil
}

func localTypeLabel(t string) string {
	if t == "lmstudio" {
		return "LM Studio"
	}
	return "OpenAI-compatible"
}

// setContextWindowValue parses and sets the active connection's context size;
// 0 goes back to auto-detection.
func (a *app) setContextWindowValue(raw string) (string, error) {
	s := strings.ReplaceAll(strings.TrimSpace(raw), "_", "")
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 10_000_000 {
		return "", fmt.Errorf("not a size: %q — tokens, e.g. 32768, or 0 for auto-detect", raw)
	}
	if err := a.setContextWindow(n); err != nil {
		return "", err
	}
	if err := a.wire(); err != nil {
		return "", err
	}
	return "context window → " + ctxLabel(n), nil
}

// validProviderName: a word a command can carry — no spaces, not "local".
func validProviderName(name string) error {
	if name == "" {
		return fmt.Errorf("a name is required")
	}
	if name == "local" {
		return fmt.Errorf(`"local" is the local server — select it as the provider and edit its address above`)
	}
	for _, r := range name {
		if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.') {
			return fmt.Errorf("a name is letters, digits, - _ . — got %q", name)
		}
	}
	return nil
}

// addProviderFields adds or edits a custom provider from separate fields — the
// panel form passes them as they were typed, never re-split by spaces. An
// empty key keeps the one on file. Editing the provider in use re-wires.
func (a *app) addProviderFields(name, rawURL, model, key string) (string, error) {
	name = strings.TrimSpace(name)
	if err := validProviderName(name); err != nil {
		return "", err
	}
	if _, ok := config.ProviderTemplates[name]; ok {
		return "", fmt.Errorf("%q is built in — select it as the provider, then set its key (and, for a proxy, its address) above", name)
	}
	u, err := parseBaseURL(rawURL)
	if err != nil {
		return "", err
	}
	if a.cfg.Providers == nil {
		a.cfg.Providers = map[string]config.LLM{}
	}
	before, had := a.cfg.Providers[name]
	p := before
	p.BaseURL = u
	p.Model = strings.TrimSpace(model)
	if k := strings.TrimSpace(key); k != "" {
		p.APIKey = k
	}
	a.cfg.Providers[name] = p
	if err := config.SaveProviders(a.cfg.Provider, a.cfg.Providers); err != nil {
		if had {
			a.cfg.Providers[name] = before
		} else {
			delete(a.cfg.Providers, name)
		}
		return "", err
	}
	if a.cfg.Provider == name {
		if err := a.reconnect(); err != nil {
			return "", err
		}
		return fmt.Sprintf("%q updated → %s (in use)", name, u), nil
	}
	verb := "added"
	if had {
		verb = "updated"
	}
	return fmt.Sprintf("%s %q → %s — select it under provider", verb, name, u), nil
}

// removeProvider forgets a provider: a custom one entirely; a built-in one's
// saved key and overrides (it stays available, back on its defaults). The
// provider in use goes back to local first.
func (a *app) removeProvider(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == "local" {
		return "", fmt.Errorf("the local server cannot be removed — edit its address instead")
	}
	_, saved := a.cfg.Providers[name]
	_, builtin := config.ProviderTemplates[name]
	if !saved {
		if builtin {
			return "", fmt.Errorf("%q has nothing saved to remove", name)
		}
		return "", fmt.Errorf("no provider %q — saved: %s", name, strings.Join(a.savedProviderNames(), ", "))
	}
	wasActive := a.cfg.Provider == name
	provider := a.cfg.Provider
	if wasActive {
		provider = "local"
	}
	rest := make(map[string]config.LLM, len(a.cfg.Providers))
	for k, v := range a.cfg.Providers {
		if k != name {
			rest[k] = v
		}
	}
	if err := config.SaveProviders(provider, rest); err != nil {
		return "", err
	}
	a.cfg.Providers, a.cfg.Provider = rest, provider
	if wasActive {
		if err := a.reconnect(); err != nil {
			return "", err
		}
	}
	msg := fmt.Sprintf("removed %q", name)
	if builtin {
		msg = fmt.Sprintf("%q is back on its defaults — its saved key is gone", name)
	}
	if wasActive {
		msg += " — now using local"
	}
	if users := a.profilesUsing(name); len(users) > 0 {
		msg += " · profiles still naming it: " + strings.Join(users, ", ")
	}
	return msg, nil
}

// savedProviderNames are the providers with anything saved, sorted.
func (a *app) savedProviderNames() []string {
	var out []string
	for n := range a.cfg.Providers {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// profilesUsing names the sub-agent profiles that point at provider.
func (a *app) profilesUsing(provider string) []string {
	var out []string
	for n, p := range a.cfg.Agents {
		if p.Provider == provider {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

// numberRows are the tuning rows typed as a number, with what each accepts.
// 0 everywhere means "the server's default".
var numberRows = map[string]struct {
	float    bool
	min, max float64
	example  string
}{
	"temperature":       {true, 0, 2, "0.7"},
	"top_p":             {true, 0, 1, "0.95"},
	"max_output_tokens": {false, 0, 10_000_000, "16000"},
	"idle_timeout":      {false, 0, 86_400, "300 (seconds)"},
	"retry_attempts":    {false, 0, 100, "8"},
}

// numberValue is a tuning row's current value, as the editor starts with it.
func (a *app) numberValue(key string) string {
	l := a.activeLLM()
	switch key {
	case "temperature":
		return strconv.FormatFloat(l.Temperature, 'g', -1, 64)
	case "top_p":
		return strconv.FormatFloat(l.TopP, 'g', -1, 64)
	case "max_output_tokens":
		return strconv.Itoa(l.MaxOutputTokens)
	case "idle_timeout":
		return strconv.Itoa(l.IdleTimeoutSeconds)
	case "retry_attempts":
		return strconv.Itoa(l.RetryAttempts)
	}
	return ""
}

// setNumber parses and saves a tuning row's value for the active connection.
func (a *app) setNumber(key, raw string) (string, error) {
	spec, ok := numberRows[key]
	if !ok {
		return "", fmt.Errorf("%s is not a number setting", key)
	}
	s := strings.ReplaceAll(strings.TrimSpace(raw), "_", "")
	if s == "" {
		s = "0"
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < spec.min || f > spec.max || (!spec.float && f != float64(int(f))) {
		return "", fmt.Errorf("%q: want %g to %g, e.g. %s — 0 = server default", raw, spec.min, spec.max, spec.example)
	}
	n := int(f)
	switch key {
	case "temperature":
		err = a.setTemperature(f)
	case "top_p":
		err = a.setTopP(f)
	case "max_output_tokens":
		err = a.setMaxOutputTokens(n)
	case "idle_timeout":
		err = a.setIdleTimeout(n)
	case "retry_attempts":
		err = a.setRetryAttempts(n)
	}
	if err != nil {
		return "", err
	}
	if err := a.wire(); err != nil {
		return "", err
	}
	if f == 0 {
		return key + " → server default", nil
	}
	return fmt.Sprintf("%s → %s", key, strconv.FormatFloat(f, 'g', -1, 64)), nil
}
