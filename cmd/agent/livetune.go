package main

import (
	"encoding/json"
	"maps"
	"slices"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// Tuning changed while a task runs.
//
// The task's own goroutine reads a.cfg — a sub-agent's connection is resolved
// from it mid-task — so the UI thread does not write a.cfg until the task
// ends: a write racing that read is a fatal "concurrent map read and map
// write". Rebuilding the agent (wire) mid-task is worse still: the running
// task keeps writing its history into the old one. What each request is sent
// with lives in the client, though, and is swapped there atomically
// (llm SetTuning). So a tuning change made during a task goes into the clients
// at once — the model's next reply already uses it — and is staged in
// cfgPending as "tune:<key>=<value>", saved to a.cfg and the config file when
// the task ends.

const tunePrefix = "tune:"

// liveTuneRows are the /config rows whose change takes effect during a task.
// Not the idle timeout: it is built into the connection's transport.
var liveTuneRows = map[string]bool{
	"temperature": true, "top_p": true, "max_output_tokens": true, "retry_attempts": true,
	"judge_max_output_tokens": true, "reasoning": true, "judge_reasoning": true, "loop_detection": true,
}

type tuneEdit struct{ key, value string }

func parseTune(s string) (tuneEdit, bool) {
	rest, ok := strings.CutPrefix(s, tunePrefix)
	if !ok {
		return tuneEdit{}, false
	}
	k, v, _ := strings.Cut(rest, "=")
	return tuneEdit{k, v}, true
}

// tuneEdits are the tuning changes staged so far, in order.
func (m *tuiModel) tuneEdits() []tuneEdit {
	var out []tuneEdit
	for _, p := range m.cfgPending {
		if e, ok := parseTune(p); ok {
			out = append(out, e)
		}
	}
	return out
}

// tunedValue is the value a row has in the running task, when one was staged.
func (m *tuiModel) tunedValue(key string) (string, bool) {
	v, ok := "", false
	for _, e := range m.tuneEdits() {
		if e.key == key {
			v, ok = e.value, true
		}
	}
	return v, ok
}

// tunedState is the active connection, the reasoning map and the judge's
// output cap with edits applied — copies: a.cfg is not touched.
func (a *app) tunedState(edits []tuneEdit) (config.LLM, map[string]json.RawMessage, int) {
	l, r, budget := a.activeLLM(), maps.Clone(a.cfg.Reasoning), a.cfg.JudgeMaxOutputTokens
	if r == nil {
		r = map[string]json.RawMessage{}
	}
	prov := a.providerName()
	for _, e := range edits {
		f, _ := strconv.ParseFloat(e.value, 64)
		switch e.key {
		case "temperature":
			l.Temperature = f
		case "top_p":
			l.TopP = f
		case "max_output_tokens":
			l.MaxOutputTokens = int(f)
		case "retry_attempts":
			l.RetryAttempts = int(f)
		case "judge_max_output_tokens":
			budget = int(f)
		case "loop_detection":
			l.DisableLoopDetection = e.value == "off"
		case "reasoning", "judge_reasoning":
			k := prov + "/" + l.Model
			if e.key == "judge_reasoning" {
				k = "judge:" + k
			}
			if shape, _ := reasoningShape(prov, e.value); shape != nil {
				r[k] = shape
			} else {
				delete(r, k)
			}
		}
	}
	return l, r, budget
}

// pushTuning puts the staged edits into the clients the running task uses.
// It reports whether the goal judge got them too: a judge sharing the task's
// connection has no client of its own to retune, and gets one when the task
// ends.
func (a *app) pushTuning(edits []tuneEdit) (judgeToo bool) {
	l, r, budget := a.tunedState(edits)
	prov := a.providerName()
	task := l
	task.Extra = reasoningParamsIn(r, prov, l.Model, "")
	if a.client != nil {
		a.client.SetTuning(task)
	}
	if a.judgeClient == nil {
		return false
	}
	j := l
	j.Extra = reasoningParamsIn(r, prov, l.Model, "judge")
	if budget > 0 {
		j.MaxOutputTokens = budget
	}
	a.judgeClient.SetTuning(j)
	return true
}

// tuneNow stages one change for the end of the task and puts it into effect.
func (m *tuiModel) tuneNow(key, value string) {
	m.cfgPending = append(m.cfgPending, tunePrefix+key+"="+value)
	judgeToo := m.app.pushTuning(m.tuneEdits())
	msg := key + " → " + value + " — from the model's next reply; saved when the task ends"
	if strings.HasPrefix(key, "judge_") && !judgeToo {
		msg = key + " → " + value + " — the goal judge picks it up when the task ends"
	}
	m.push(cDim.Render("  " + msg))
}

// tuneLive is enter on a live tuning row while a task runs.
func (m *tuiModel) tuneLive(key string) (tea.Model, tea.Cmd) {
	l, r, _ := m.app.tunedState(m.tuneEdits())
	prov := m.app.providerName()
	switch key {
	case "reasoning":
		if _, ok := reasoningShape(prov, "high"); !ok {
			m.push(cDim.Render("  " + prov + " reasoning must be set raw in config.json (key " + prov + "/" + l.Model + ")"))
			return m, nil
		}
		cur, set := scopedLevelIn(r, "", prov, l.Model)
		if !set {
			cur = "default"
		}
		m.tuneNow(key, nextReasoning(prov, cur))
	case "judge_reasoning":
		cur, set := scopedLevelIn(r, "judge:", prov, l.Model)
		if !set {
			cur = "default"
		}
		m.tuneNow(key, nextLevel(cur))
	case "loop_detection":
		m.tuneNow(key, onOff(l.DisableLoopDetection)) // off now → on, and back
	default: // a number: typed in the row's editor, committed by tuneTyped
		v, ok := m.tunedValue(key)
		if !ok {
			v = m.app.numberValue(key)
		}
		m.cfgEdit = &cfgEditor{key: key, f: newTextField(v, false)}
	}
	return m, nil
}

// tuneTyped commits a number typed during a task.
func (m *tuiModel) tuneTyped(key, raw string) (tea.Model, tea.Cmd) {
	f, err := parseNumber(key, raw)
	if err != nil {
		m.cfgEdit.err = err.Error()
		return m, nil
	}
	m.cfgEdit = nil
	m.tuneNow(key, strconv.FormatFloat(f, 'g', -1, 64))
	return m, nil
}

// applyTune saves a staged change once the task is over.
func (m *tuiModel) applyTune(e tuneEdit) (tea.Model, tea.Cmd) {
	switch e.key {
	case "reasoning":
		m.pushLines(m.app.reasoningCommand(e.value))
	case "judge_reasoning":
		m.pushLines(m.app.reasoningCommand("judge " + e.value))
	case "loop_detection":
		off := e.value == "off"
		if err := m.app.updateActive(func(l *config.LLM) { l.DisableLoopDetection = off }); err != nil {
			m.push(cErr.Render("  could not persist: " + err.Error()))
		} else if err := m.app.wire(); err != nil {
			m.push(cErr.Render("  " + err.Error()))
		}
	default:
		if _, err := m.app.setNumber(e.key, e.value); err != nil {
			m.push(cErr.Render("  " + e.key + " not saved: " + err.Error()))
		}
	}
	return m, nil
}

// reasoningWhileBusy is /reasoning <level> or /reasoning judge <level> during
// a task: in effect from the next reply. It reports false for what still
// waits for the task to end (the learning pass's own setting, anything else).
func (m *tuiModel) reasoningWhileBusy(rest string) bool {
	f := strings.Fields(strings.ToLower(rest))
	key, level := "reasoning", ""
	switch {
	case len(f) == 1:
		level = f[0]
	case len(f) == 2 && f[0] == "judge":
		key, level = "judge_reasoning", f[1]
	default:
		return false
	}
	if !slices.Contains(reasoningLevels, level) {
		return false
	}
	prov := m.app.providerName()
	if shape, ok := reasoningShape(prov, level); !ok || (shape == nil && level != "off") {
		return false // usage or "set it raw" — the queued command says which
	}
	m.tuneNow(key, level)
	return true
}
