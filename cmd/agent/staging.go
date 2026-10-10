package main

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ipsupport-llc/ipsupport-code/internal/config"
)

// /config while a task runs.
//
// The task's own goroutine reads a.cfg — a sub-agent's connection is resolved
// from it mid-task — so the UI thread does not write a.cfg until the task
// ends: a write racing that read is a fatal "concurrent map read and map
// write". Rebuilding the agent (wire) mid-task is worse still: the running
// task keeps writing its history into the old one.
//
// So a change made during a task is staged, and staged as WHERE IT LANDS —
// a value, not a keypress: a row pressed 34 times used to stage 34 cycle
// steps and show only "×34", never the result. Connection settings carry the
// provider and model they were made for, so a staged provider switch can't
// carry them to another connection. What each request is sent with lives in
// the client and is swapped there atomically (llm SetTuning): those settings
// also go into the running task's clients at once.

// stagedEdit is one /config change waiting for the task to end.
type stagedEdit struct {
	key   string // the /config row
	value string // where it lands: a number, a level, on/off, a name, a cycle row's raw value
	prov  string // connection-bound edits: the provider (and model) they were made for;
	model string // removeprovider: the provider to remove
	steps int    // cycle rows: presses from the live value
}

func (e stagedEdit) sameTarget(o stagedEdit) bool {
	return e.key == o.key && e.prov == o.prov && e.model == o.model
}

// stage records e for the task's end, in place of an earlier edit to the same
// row and connection.
func (m *tuiModel) stage(e stagedEdit) {
	for i, p := range m.cfgPending {
		if p.sameTarget(e) {
			m.cfgPending[i] = e
			return
		}
	}
	m.cfgPending = append(m.cfgPending, e)
}

func (m *tuiModel) unstage(e stagedEdit) {
	m.cfgPending = slices.DeleteFunc(m.cfgPending, e.sameTarget)
}

// staged is the edit waiting for a row, for the connection in use.
func (m *tuiModel) staged(key string) (stagedEdit, bool) {
	prov, model := m.app.providerName(), m.app.activeLLM().Model
	for _, e := range m.cfgPending {
		if e.key != key {
			continue
		}
		switch {
		case connectionBound[key] && (e.prov != prov || e.model != model),
			key == "model" && e.prov != prov:
			continue
		}
		return e, true
	}
	return stagedEdit{}, false
}

// liveTuneRows are the rows whose change also goes into the running task.
// Not the idle timeout: it is built into the connection's transport.
var liveTuneRows = map[string]bool{
	"temperature": true, "top_p": true, "max_output_tokens": true, "retry_attempts": true,
	"judge_max_output_tokens": true, "reasoning": true, "judge_reasoning": true, "loop_detection": true,
}

// connectionBound are the live rows saved to one provider/model; the judge's
// output cap is global.
var connectionBound = map[string]bool{
	"temperature": true, "top_p": true, "max_output_tokens": true, "retry_attempts": true,
	"reasoning": true, "judge_reasoning": true, "loop_detection": true,
}

// tunedState is the active connection, the reasoning map and the judge's
// output cap with the staged edits for this connection applied — copies:
// a.cfg is not touched.
func (m *tuiModel) tunedState() (config.LLM, map[string]json.RawMessage, int) {
	a := m.app
	l, r, budget := a.activeLLM(), maps.Clone(a.cfg.Reasoning), a.cfg.JudgeMaxOutputTokens
	if r == nil {
		r = map[string]json.RawMessage{}
	}
	prov := a.providerName()
	for _, e := range m.cfgPending {
		if e.key == "judge_max_output_tokens" {
			budget, _ = strconv.Atoi(e.value)
			continue
		}
		if !connectionBound[e.key] || e.prov != prov || e.model != l.Model {
			continue
		}
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
		case "loop_detection":
			l.DisableLoopDetection = e.value == "off"
		case "reasoning", "judge_reasoning":
			k := prov + "/" + l.Model
			if e.key == "judge_reasoning" {
				k = "judge:" + k
			}
			if shape, known := reasoningShape(prov, e.value); known && shape != nil {
				r[k] = shape
			} else if known {
				delete(r, k)
			}
		}
	}
	return l, r, budget
}

// pushTuning puts the staged tuning into the clients the running task uses.
// It reports whether the goal judge got it too: a judge sharing the task's
// connection has no client of its own to retune, and gets one when the task
// ends.
func (m *tuiModel) pushTuning() (judgeToo bool) {
	a := m.app
	l, r, budget := m.tunedState()
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

// tuneNow stages a tuning change and puts it into the running task.
func (m *tuiModel) tuneNow(key, value string) {
	e := stagedEdit{key: key, value: value}
	if connectionBound[key] {
		e.prov, e.model = m.app.providerName(), m.app.activeLLM().Model
	}
	m.stage(e)
	judgeToo := m.pushTuning()
	msg := key + " → " + value + " — from the model's next reply; saved when the task ends"
	if strings.HasPrefix(key, "judge_") && !judgeToo {
		msg = key + " → " + value + " — the goal judge picks it up when the task ends"
	}
	m.push(cDim.Render("  " + msg))
}

// tuneLive is enter on a live tuning row while a task runs.
func (m *tuiModel) tuneLive(key string) (tea.Model, tea.Cmd) {
	l, r, _ := m.tunedState()
	prov := m.app.providerName()
	switch key {
	case "reasoning", "judge_reasoning":
		if _, ok := reasoningShape(prov, "high"); !ok {
			m.push(cDim.Render("  " + prov + " reasoning must be set raw in config.json (key " + prov + "/" + l.Model + ")"))
			return m, nil
		}
		scope := ""
		if key == "judge_reasoning" {
			scope = "judge:"
		}
		cur, set := scopedLevelIn(r, scope, prov, l.Model)
		if !set {
			cur = "default"
		}
		if key == "reasoning" {
			m.tuneNow(key, nextReasoning(prov, cur))
		} else {
			m.tuneNow(key, nextLevel(cur))
		}
	case "loop_detection":
		m.tuneNow(key, onOff(l.DisableLoopDetection)) // off now → on, and back
	default: // a number: typed in the row's editor, committed by tuneTyped
		v := m.app.numberValue(key)
		if e, ok := m.staged(key); ok {
			v = e.value
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

// shadow is the part of the settings a cycle row moves — a copy, so where a
// row would land can be worked out without touching a.cfg.
type shadow struct {
	cfg       config.Config
	plan      bool
	accentIdx int
}

func (m *tuiModel) liveShadow() shadow {
	return shadow{cfg: m.app.cfg, plan: m.app.planMode, accentIdx: m.accentIdx}
}

// cycleRow is a row that steps through its values on enter: get reads where
// it is, step moves it one press — the same moves its activation makes.
type cycleRow struct {
	get  func(s *shadow) string
	step func(s *shadow)
}

func flip(b *bool) { *b = !*b }

var cycleRows = map[string]cycleRow{
	"mode":       {func(s *shadow) string { return strconv.FormatBool(s.plan) }, func(s *shadow) { flip(&s.plan) }},
	"perm_files": {func(s *shadow) string { return s.cfg.File.Default }, func(s *shadow) { s.cfg.File.Default = nextStr(s.cfg.File.Default, permCycle) }},
	"perm_run":   {func(s *shadow) string { return s.cfg.Run.Default }, func(s *shadow) { s.cfg.Run.Default = nextStr(s.cfg.Run.Default, permCycle) }},
	"timeout": {func(s *shadow) string { return strconv.Itoa(s.cfg.Run.TimeoutSeconds) },
		func(s *shadow) { s.cfg.Run.TimeoutSeconds = nextInt(s.cfg.Run.TimeoutSeconds, timeoutCycle) }},
	"offline": {func(s *shadow) string { return strconv.FormatBool(s.cfg.Offline) }, func(s *shadow) { flip(&s.cfg.Offline) }},
	"memory": {func(s *shadow) string { return s.cfg.Memory }, func(s *shadow) {
		if s.cfg.Memory == "raw" {
			s.cfg.Memory = ""
		} else {
			s.cfg.Memory = "raw"
		}
	}},
	"compact_threshold": {func(s *shadow) string { return fmt.Sprint(compactThreshold(s.cfg.CompactThreshold)) },
		func(s *shadow) {
			s.cfg.CompactThreshold = nextFloat(compactThreshold(s.cfg.CompactThreshold), compactThresholdCycle)
		}},
	"max_steps": {func(s *shadow) string { return strconv.Itoa(s.cfg.GoalMaxSteps) },
		func(s *shadow) { s.cfg.GoalMaxSteps = nextInt(s.cfg.GoalMaxSteps, maxStepsCycle) }},
	"max_history": {func(s *shadow) string { return strconv.Itoa(s.cfg.MaxHistory) },
		func(s *shadow) { s.cfg.MaxHistory = nextInt(s.cfg.MaxHistory, maxHistoryCycle) }},
	"max_stuck_turns": {func(s *shadow) string { return strconv.Itoa(s.cfg.MaxStuckTurns) },
		func(s *shadow) { s.cfg.MaxStuckTurns = nextInt(s.cfg.MaxStuckTurns, maxStuckTurnsCycle) }},
	"goal_ttl": {func(s *shadow) string { return strconv.Itoa(s.cfg.GoalMaxReturns) },
		func(s *shadow) { s.cfg.GoalMaxReturns = nextInt(s.cfg.GoalMaxReturns, goalTTLCycle) }},
	"goal_nudge": {func(s *shadow) string { return strconv.FormatBool(s.cfg.GoalNudge) }, func(s *shadow) { flip(&s.cfg.GoalNudge) }},
	"reflection": {func(s *shadow) string { return strconv.FormatBool(s.cfg.ReflectDisabled) },
		func(s *shadow) { flip(&s.cfg.ReflectDisabled) }},
	"knowledge_retention": {func(s *shadow) string { return strconv.Itoa(s.cfg.KnowledgeRetentionDays) },
		func(s *shadow) {
			s.cfg.KnowledgeRetentionDays = nextInt(s.cfg.KnowledgeRetentionDays, knowledgeRetentionCycle)
		}},
	"spawn": {func(s *shadow) string { return s.cfg.Spawn.Default }, func(s *shadow) {
		if s.cfg.Spawn.Default == "allow" {
			s.cfg.Spawn.Default = "ask"
		} else {
			s.cfg.Spawn.Default = "allow"
		}
	}},
	"subexec": {func(s *shadow) string { return strconv.FormatBool(s.cfg.Spawn.Exec) }, func(s *shadow) { flip(&s.cfg.Spawn.Exec) }},
	"channel": {func(s *shadow) string { return channelOf(s.cfg) }, func(s *shadow) {
		if channelOf(s.cfg) == "nightly" {
			s.cfg.Channel = "stable"
		} else {
			s.cfg.Channel = "nightly"
		}
	}},
	"color": {func(s *shadow) string { return strconv.Itoa(s.accentIdx) },
		func(s *shadow) { s.accentIdx = (s.accentIdx + 1) % len(colorCycle) }},
}

// shadowLabel is how a row shows the shadow's value — the row's own wording.
func (m *tuiModel) shadowLabel(key string, s shadow) string {
	v := &tuiModel{app: &app{cfg: s.cfg, planMode: s.plan, workspace: m.app.workspace},
		accent: lipgloss.Color(colorCycle[s.accentIdx]), accentIdx: s.accentIdx}
	_, value, _ := v.configRowView(key)
	return value
}

// stageCycle is enter on a cycle row during a task: one more step from where
// it was going, shown as where it now lands.
func (m *tuiModel) stageCycle(key string) {
	c := cycleRows[key]
	e := stagedEdit{key: key, steps: 1}
	if prev, ok := m.staged(key); ok {
		e.steps = prev.steps + 1
	}
	live := m.liveShadow()
	s := m.liveShadow()
	for range e.steps {
		c.step(&s)
	}
	e.value = c.get(&s)
	label := m.shadowLabel(key, s)
	if e.value == c.get(&live) {
		m.unstage(e)
		m.push(cDim.Render("  " + key + " back to " + label + " — nothing staged"))
		return
	}
	m.stage(e)
	m.push(cDim.Render("  staged — " + key + " → " + label + " when the task ends"))
}

// applyStaged saves one staged edit once the task is over. It reports
// whether the connection must be rebuilt for the change to take.
func (m *tuiModel) applyStaged(e stagedEdit) (cmd tea.Cmd, rewire bool) {
	a := m.app
	fail := func(err error) { m.push(cErr.Render("  " + e.key + " not saved: " + err.Error())) }
	switch e.key {
	case "provider":
		m.pushLines(a.setProvider(e.value))
		return m.detectWindowCmd(), false
	case "removeprovider":
		if msg, err := a.removeProvider(e.prov); err != nil {
			m.push(cErr.Render("  " + e.prov + " not removed: " + err.Error()))
		} else {
			m.push(cDim.Render("  " + msg))
		}
		return m.detectWindowCmd(), false
	case "model":
		if err := a.updateConnection(e.prov, func(l *config.LLM) {
			if l.Model != e.value {
				l.ContextWindowManual = false
			}
			l.Model = e.value
		}); err != nil {
			fail(err)
			return nil, true
		}
		if e.prov == a.providerName() {
			a.windowDetected = false
			a.modelEpoch.Add(1)
		}
		m.push(cDim.Render("  model → " + e.value + " (" + e.prov + ")"))
		return nil, true
	case "reasoning", "judge_reasoning":
		k := e.prov + "/" + e.model
		if e.key == "judge_reasoning" {
			k = "judge:" + k
		}
		if _, ok, err := a.applyReasoning(k, e.prov, e.value); err != nil {
			fail(err)
		} else if !ok {
			fail(fmt.Errorf("%s has no known reasoning setting", e.prov))
		} else {
			m.push(cDim.Render("  " + e.key + " → " + e.value + " (" + e.prov + " · " + e.model + ")"))
		}
		return nil, true
	case "loop_detection":
		if err := a.updateConnection(e.prov, func(l *config.LLM) { l.DisableLoopDetection = e.value == "off" }); err != nil {
			fail(err)
		}
		return nil, true
	case "judge_max_output_tokens":
		n, _ := strconv.Atoi(e.value)
		if err := a.setJudgeMaxOutput(n); err != nil {
			fail(err)
		}
		return nil, true
	case "temperature", "top_p", "max_output_tokens", "retry_attempts":
		f, _ := strconv.ParseFloat(e.value, 64)
		if err := a.updateConnection(e.prov, func(l *config.LLM) {
			switch e.key {
			case "temperature":
				l.Temperature = f
			case "top_p":
				l.TopP = f
			case "max_output_tokens":
				l.MaxOutputTokens = int(f)
			case "retry_attempts":
				l.RetryAttempts = int(f)
			}
		}); err != nil {
			fail(err)
		} else {
			m.push(cDim.Render("  " + e.key + " → " + e.value + " (" + e.prov + " · " + e.model + ")"))
		}
		return nil, true
	}
	if c, ok := cycleRows[e.key]; ok {
		// The row's own activation, pressed until it lands — it saves, says
		// what it did, and rolls back on a failed save (then it stops: no
		// progress).
		for range len(colorCycle) + 8 {
			s := m.liveShadow()
			if c.get(&s) == e.value {
				return cmd, false
			}
			_, cmd = m.activateConfigRow(e.key)
			if after := m.liveShadow(); c.get(&after) == c.get(&s) {
				break
			}
		}
		m.push(cErr.Render("  " + e.key + " did not reach the staged value — see above"))
		return cmd, false
	}
	return nil, false
}

// stagedRowView is how a row with a staged change shows it: the value it
// lands on, and a hint saying when — or "" when nothing is staged for it.
func (m *tuiModel) stagedRowView(key, liveValue string) (value, hint string) {
	if key == "removeprovider" {
		var names []string
		for _, e := range m.cfgPending {
			if e.key == key {
				names = append(names, e.prov)
			}
		}
		if len(names) == 0 {
			return "", ""
		}
		return liveValue, "staged: remove " + strings.Join(names, ", ") + " when the task ends"
	}
	e, ok := m.staged(key)
	if !ok {
		return "", ""
	}
	switch {
	case liveTuneRows[key]:
		if strings.HasPrefix(key, "judge_") && m.app.judgeClient == nil {
			return e.value, "staged — now " + liveValue + " · saved when the task ends"
		}
		return e.value, "in use now · saved when the task ends"
	case key == "provider", key == "model":
		return e.value, "staged — now " + liveValue + " · switches when the task ends"
	}
	if _, ok := cycleRows[key]; ok {
		s := m.liveShadow()
		for range e.steps {
			cycleRows[key].step(&s)
		}
		return m.shadowLabel(key, s), "staged — now " + liveValue + " · applies when the task ends"
	}
	return "", ""
}
