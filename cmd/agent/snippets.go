package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// Prompt snippets — reusable prompt templates. `/snip save <name> [text]` stores
// one (the text, or your last prompt when omitted); `/snip <name>` pulls it back
// into the input to edit and send; `/snip list` / `/snip rm <name>` manage them.
// Stored globally (~/.config/ipsupport-code/snippets.json) so templates follow
// you across projects.

// reservedSnip are the /snip subcommands — they shadow recall, so a snippet
// can't be named one of them.
var reservedSnip = map[string]bool{"save": true, "list": true, "rm": true, "remove": true, "delete": true}

// snipAction is the result of a /snip command: recall != "" means "put this in
// the input box" (TUI) or "print it" (REPL); otherwise lines are messages to show.
type snipAction struct {
	recall string
	lines  []string
}

// snip dispatches a /snip invocation (the text after "/snip ").
func (a *app) snip(rest string) snipAction {
	sub, arg := splitCommand(rest)
	switch sub {
	case "", "list":
		return snipAction{lines: a.snipList()}
	case "save":
		name, text := splitCommand(arg)
		return snipAction{lines: a.snipSave(name, text)}
	case "rm", "remove", "delete":
		return snipAction{lines: a.snipRemove(arg)}
	default: // recall by name
		text, ok := a.snippets[sub]
		if !ok {
			return snipAction{lines: []string{"no such snippet: " + sub + " — /snip list"}}
		}
		return snipAction{recall: text}
	}
}

// snipSave stores text (or, when text is empty, the last prompt you sent) under
// name, persisting to disk.
func (a *app) snipSave(name, text string) []string {
	name = strings.TrimSpace(name)
	if name == "" {
		return []string{"usage: /snip save <name> [text]  — omit text to save your last prompt"}
	}
	if reservedSnip[name] {
		return []string{"'" + name + "' is a reserved word — pick another name"}
	}
	if text = strings.TrimSpace(text); text == "" {
		if text = a.lastTaskPrompt(); text == "" {
			return []string{"no previous prompt to save — /snip save " + name + " <text>"}
		}
	}
	if a.snippets == nil {
		a.snippets = map[string]string{}
	}
	a.snippets[name] = text
	if err := a.saveSnippets(); err != nil {
		return []string{"snippet not saved: " + err.Error()}
	}
	return []string{"saved snippet: " + name + "  (/snip " + name + " to recall)"}
}

// snipRemove deletes a snippet.
func (a *app) snipRemove(name string) []string {
	name = strings.TrimSpace(name)
	if name == "" {
		return []string{"usage: /snip rm <name>"}
	}
	if _, ok := a.snippets[name]; !ok {
		return []string{"no such snippet: " + name}
	}
	delete(a.snippets, name)
	if err := a.saveSnippets(); err != nil {
		return []string{"snippet not removed: " + err.Error()}
	}
	return []string{"removed snippet: " + name}
}

// snipList lists snippets (sorted), each with a one-line preview.
func (a *app) snipList() []string {
	if len(a.snippets) == 0 {
		return []string{"no snippets yet — /snip save <name> [text] to add one",
			"  then /snip <name> pulls it into the input to edit & send"}
	}
	out := []string{"snippets:"}
	for _, n := range a.snipNames() {
		out = append(out, fmt.Sprintf("  %-14s %s", n, oneLine(a.snippets[n], 60)))
	}
	return append(out, "  /snip <name> to recall · /snip rm <name>")
}

// snipNames returns the snippet names, sorted (for listing and Tab completion).
func (a *app) snipNames() []string {
	names := make([]string, 0, len(a.snippets))
	for n := range a.snippets {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// lastTaskPrompt returns the most recent submitted line that was an actual task
// (not a /command or a !shell line) — what `/snip save <name>` captures.
func (a *app) lastTaskPrompt() string {
	for i := len(a.promptHist) - 1; i >= 0; i-- {
		p := strings.TrimSpace(a.promptHist[i])
		if p == "" || strings.HasPrefix(p, "/") || strings.HasPrefix(p, "!") {
			continue
		}
		return a.promptHist[i]
	}
	return ""
}

// loadSnippets reads the persisted snippets (best-effort).
func (a *app) loadSnippets() {
	data, err := os.ReadFile(config.SnippetsPath())
	if err != nil {
		return
	}
	m := map[string]string{}
	if json.Unmarshal(data, &m) == nil {
		a.snippets = m
	}
}

// saveSnippets writes the snippets store atomically (0600 — a template may hold
// anything the user pasted).
func (a *app) saveSnippets() error {
	data, err := json.MarshalIndent(a.snippets, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(config.SnippetsPath(), data, 0o600)
}
