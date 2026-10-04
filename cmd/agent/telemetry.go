package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/agent"
	"github.com/ipsupport-llc/ipsupport-code/internal/config"
	"github.com/ipsupport-llc/ipsupport-code/internal/feedback"
	"github.com/ipsupport-llc/ipsupport-code/internal/telemetry"
)

// Usage statistics and ratings (adr/0016). Telemetry is on by default for a
// NEW install — the first-run setup writes it on and says so — and an update
// never turns it on for an existing one: settleTelemetryDefault writes it off
// at the first launch of a build that knows about it. Either way the first
// report goes only after the first day ends, so it can be turned off before
// anything has been sent.

// telemetryNotice is what first-run setup says, and /telemetry repeats.
const telemetryNotice = "Anonymous usage statistics are on: once a day, the version, OS, and counts of what was used (tasks, tools, sub-agents) — never code, prompts, commands, file names or model names. Turn off: ipsupport-code config set telemetry false · details: /telemetry"

func telemetryPath() string {
	return filepath.Join(filepath.Dir(config.GlobalPath()), "telemetry.json")
}
func ratePath() string { return filepath.Join(filepath.Dir(config.GlobalPath()), "rate.json") }

// apiBase is where reports and reviews go; IPS_API_BASE points a development
// build at a local server.
func apiBase() (string, bool) {
	if v := strings.TrimRight(strings.TrimSpace(os.Getenv("IPS_API_BASE")), "/"); v != "" {
		return v, true
	}
	return telemetry.DefaultAPIBase, false
}

// userAgent names the program, not the system.
func userAgent() string { return "ipsupport-code/" + telemetry.AppVersion(version) }

// settleTelemetryDefault decides, once, for an install that existed before
// this build: off. An update never turns telemetry on.
func settleTelemetryDefault(existed bool) {
	if !existed {
		return // a first run: setup decides (on), or nothing is decided (off) when there is no setup
	}
	cfg, err := config.Load(".")
	if err != nil || cfg.Telemetry != nil {
		return
	}
	if err := config.SaveTelemetry(false); err != nil {
		slog.Debug("telemetry default not saved", "err", err)
	}
}

// telemetryStatus says whether reports are recorded and sent, and if not, why.
func (a *app) telemetryStatus() (enabled, active bool, why string) {
	enabled = a.cfg.Telemetry != nil && *a.cfg.Telemetry
	_, custom := apiBase()
	switch {
	case !enabled:
		return false, false, "off"
	case doNotTrack():
		return true, false, "DO_NOT_TRACK is set"
	case a.cfg.Offline:
		return true, false, "offline mode is on"
	case !telemetry.IsRelease(version) && !custom:
		return true, false, "a development build sends nothing (set IPS_API_BASE to test against a server)"
	}
	return true, true, ""
}

// doNotTrack honors the DO_NOT_TRACK convention (consoledonottrack.com).
func doNotTrack() bool {
	v := strings.TrimSpace(os.Getenv("DO_NOT_TRACK"))
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}

// startTelemetry runs at launch: it forgets everything when telemetry is
// off, and otherwise records that the program ran today and sends finished
// days — now, and every three hours while it keeps running.
func (a *app) startTelemetry(ctx context.Context) {
	_, _ = feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) {
		if s.FirstLaunch.IsZero() {
			s.FirstLaunch = time.Now()
		}
	})
	enabled, active, _ := a.telemetryStatus()
	path := telemetryPath()
	if !enabled {
		if _, err := os.Stat(path); err == nil {
			_ = telemetry.Disable(path) // off means the install ID and unsent counters go too
		}
		return
	}
	if !active {
		return
	}
	if err := telemetry.Enable(path); err != nil {
		slog.Debug("telemetry state", "err", err)
		return
	}
	a.recordTelemetry(telemetry.Counts{}) // a day the program ran on is reported
	go func() {
		t := time.NewTicker(3 * time.Hour)
		defer t.Stop()
		for {
			a.sendTelemetry(ctx)
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
		}
	}()
}

func (a *app) sendTelemetry(ctx context.Context) {
	base, _ := apiBase()
	res, err := telemetry.Send(ctx, telemetry.SendOptions{
		Path: telemetryPath(), Endpoint: base + "/telemetry", Version: version,
		Info: telemetry.CollectInfo(), UserAgent: userAgent(),
	})
	if err != nil || res.Sent > 0 || res.Dropped > 0 || res.Deferred {
		slog.Debug("telemetry send", "sent", res.Sent, "dropped", res.Dropped, "deferred", res.Deferred, "err", err)
	}
}

// recordTelemetry adds to today's counters, when reports are being kept.
func (a *app) recordTelemetry(c telemetry.Counts) {
	if _, active, _ := a.telemetryStatus(); !active {
		return
	}
	if err := telemetry.Record(telemetryPath(), time.Now().Format(time.DateOnly), c); err != nil {
		slog.Debug("telemetry record", "err", err)
	}
}

// countTask records one finished task: that it ran, the tools it called,
// whether it was a goal or plan-mode run, and the kind and family of model.
func (a *app) countTask(tr agent.Transcript) {
	f := map[string]int{"tasks": 1}
	for _, m := range tr.Messages {
		if m.Role != "tool" {
			continue
		}
		f["tool_calls"]++
		switch m.Name {
		case "mcp":
			f["mcp"]++
		case "skill":
			f["skills"]++
		}
	}
	if a.goal.Status == "active" || tr.Returns > 0 || tr.GoalMet {
		f["goals"] = 1
	}
	if a.ag != nil && a.ag.PlanMode() {
		f["plan_mode"] = 1
	}
	if a.isLocal() {
		f["local_model"] = 1
	} else {
		f["cloud_model"] = 1
	}
	a.recordTelemetry(telemetry.Counts{Features: f, Families: []string{telemetry.Family(a.activeLLM().Model)}})
}

// countSpawn records a delegated task: an LLM sub-agent and its model's
// family, or an external CLI agent (no model of ours to name).
func (a *app) countSpawn(external bool, model string) {
	if external {
		a.recordTelemetry(telemetry.Counts{Features: map[string]int{"external_agents": 1}})
		return
	}
	a.recordTelemetry(telemetry.Counts{Features: map[string]int{"subagents": 1}, Families: []string{telemetry.Family(model)}})
}

// telemetryCommand backs /telemetry: status, exactly what would be sent, and
// on / off / reset.
func (a *app) telemetryCommand(rest string) []string {
	path := telemetryPath()
	switch strings.ToLower(strings.TrimSpace(rest)) {
	case "on":
		if err := config.SaveTelemetry(true); err != nil {
			return []string{"error: " + err.Error()}
		}
		t := true
		a.cfg.Telemetry = &t
		if _, active, _ := a.telemetryStatus(); active {
			_ = telemetry.Enable(path)
		}
		return append([]string{"usage statistics on — a new anonymous install ID"}, a.telemetryStatusLines()...)
	case "off":
		if err := config.SaveTelemetry(false); err != nil {
			return []string{"error: " + err.Error()}
		}
		f := false
		a.cfg.Telemetry = &f
		if err := telemetry.Disable(path); err != nil {
			return []string{"error: " + err.Error()}
		}
		return []string{"usage statistics off — the install ID and every unsent counter are deleted"}
	case "reset":
		if err := telemetry.ResetID(path); err != nil {
			return []string{"error: " + err.Error()}
		}
		return []string{"new anonymous install ID — this machine now reports as a new install"}
	case "", "status", "show":
		return a.telemetryStatusLines()
	}
	return []string{"usage: /telemetry [show] · /telemetry on|off · /telemetry reset"}
}

func (a *app) telemetryStatusLines() []string {
	enabled, active, why := a.telemetryStatus()
	if !enabled {
		return []string{
			"usage statistics: off — nothing is recorded or sent",
			"  turn on: /telemetry on (anonymous: version, OS, counts of what was used — never code, prompts, commands, paths or model names)",
		}
	}
	out := []string{"usage statistics: on — " + map[bool]string{true: "recording; finished days are sent once a day", false: "but paused: " + why}[active]}
	s, err := telemetry.Load(telemetryPath())
	if err != nil {
		return append(out, "  error: "+err.Error())
	}
	if s.InstallID != "" {
		out = append(out, "  install ID  "+s.InstallID[:8]+"… (random; /telemetry reset for a new one)")
	}
	days := make([]string, 0, len(s.Days))
	for d := range s.Days {
		days = append(days, d)
	}
	sort.Strings(days)
	if len(days) == 0 {
		return append(out, "  nothing waiting to be sent")
	}
	info := telemetry.CollectInfo()
	out = append(out, "  waiting to be sent — exactly these reports (today's goes after today ends):")
	for _, d := range days {
		b, _ := json.Marshal(telemetry.BuildReport(s, d, version, info))
		out = append(out, "    "+string(b))
	}
	return append(out, "  never sent: code, prompts, commands, tool arguments or output, file names or paths, model names, keys, provider URLs")
}

// rateCommand backs /rate: "<1-5> <a few words> [--name <name>]", or
// "later" / "never" for the suggestion. It returns the lines to show now
// and, for a review, the send to run off the UI goroutine.
func (a *app) rateCommand(rest string) (lines []string, send func(context.Context) []string) {
	rest = strings.TrimSpace(rest)
	switch strings.ToLower(rest) {
	case "":
		return []string{
			"rate ipsupport-code: /rate <1-5> <a few words> [--name <your name>]",
			"  e.g. /rate 5 works great with my local model --name Ana",
			"  it appears on the site once it has been moderated; without --name it is anonymous",
		}, nil
	case "later":
		_, _ = feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) { feedback.Snooze(s, time.Now()) })
		return []string{"ok — I'll ask again in two weeks"}, nil
	case "never":
		_, _ = feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) { s.Never = true })
		return []string{"ok — I won't suggest it again (/rate still works any time)"}, nil
	}
	if a.cfg.Offline {
		return []string{"offline mode is on — /rate needs the internet. Run /offline off first."}, nil
	}
	stars, text, _ := strings.Cut(rest, " ")
	n, err := strconv.Atoi(strings.TrimSuffix(stars, "/5"))
	if err != nil {
		return []string{"start with the rating, 1 to 5: /rate 4 <a few words>"}, nil
	}
	author := ""
	if i := strings.LastIndex(text, "--name"); i >= 0 {
		text, author = text[:i], strings.TrimSpace(text[i+len("--name"):])
	}
	r := feedback.Review{Product: telemetry.Product, Rating: n, Text: strings.TrimSpace(text),
		Version: telemetry.AppVersion(version), Author: author}
	if !telemetry.IsRelease(version) {
		r.Version = "" // a dev build's describe string isn't a version anyone can install
	}
	if err := r.Validate(); err != nil {
		return []string{err.Error()}, nil
	}
	base, _ := apiBase()
	return []string{"sending your rating…"}, func(ctx context.Context) []string {
		if err := feedback.Submit(ctx, base+"/reviews", userAgent(), r, http.DefaultClient); err != nil {
			return []string{"rating not sent: " + err.Error()}
		}
		_, _ = feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) { s.Reviewed = true })
		return []string{fmt.Sprintf("thank you — %d★ sent; it appears on the site once it has been moderated", n)}
	}
}

// rateHint is the one-line suggestion after a finished task, or "". At most
// once per two weeks: showing it is itself the snooze, so it never repeats on
// every task. Not on a development build, and not offline.
func (a *app) rateHint() string {
	if a.cfg.Offline || !telemetry.IsRelease(version) {
		return ""
	}
	now := time.Now()
	show := false
	_, _ = feedback.UpdatePrompt(ratePath(), func(s *feedback.PromptState) {
		if s.ShouldPrompt(now) {
			show = true
			feedback.Snooze(s, now)
		}
	})
	if !show {
		return ""
	}
	return "✦ enjoying ipsupport-code? rate it: /rate 1-5 <a few words> · /rate never"
}
