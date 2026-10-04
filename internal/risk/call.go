package risk

import (
	"fmt"
	"sort"
	"strings"
)

const (
	// maxValue caps one parameter's contribution. A file write carries the whole
	// new file in "content"; without a cap, one call's body would supply more
	// n-grams than everything else put together and the score would be about the
	// prose, not the action.
	maxValue = 400
	// maxText caps the whole rendered call, so a pathological argument list
	// cannot turn a microsecond of scoring into a millisecond of it.
	maxText = 2000
)

// CallText renders a tool call as the one string the model sees. It is a wire
// format in the same sense the hash is: scripts/train_risk.py builds it
// identically from the dataset, and any divergence trains weights against text
// that inference never produces.
//
// Parameters are sorted by name so the same call always renders the same way —
// Go map iteration is randomized, and without this the features for one call
// would differ run to run.
func CallText(tool, action string, params map[string]any) string {
	// Caps count characters, not bytes: Python's len and slices do, and a byte
	// cut can split a rune in two.
	var b []rune
	b = append(b, []rune(tool)...)
	if action != "" {
		b = append(b, ' ')
		b = append(b, []rune(action)...)
	}
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := strings.TrimSpace(fmt.Sprint(params[k]))
		if v == "" {
			continue
		}
		r := []rune(v)
		if len(r) > maxValue {
			r = r[:maxValue]
		}
		b = append(b, ' ')
		b = append(b, []rune(k)...)
		b = append(b, '=')
		b = append(b, r...)
		if len(b) > maxText {
			break
		}
	}
	if len(b) > maxText {
		b = b[:maxText]
	}
	return string(b)
}

// LabelSafe is the label that means "none of the risky ones". It is scored like
// any other label; Assess only uses it for reporting, never to decide the
// headline number — a model that is unsure will happily put 0.6 on both "safe"
// and "destructive", and taking the max over the risky labels says the useful
// thing in that case while 1-safe does not.
const LabelSafe = "safe"

// Assessment is one call's score: a single headline number to threshold on,
// which label produced it, and the full breakdown for the log.
//
// The two halves answer different questions. Scores say what the call DOES —
// the base model's word, which local learning never touches. Risk says whether
// it deserves attention HERE, and that is what a workspace's own answers adjust
// (see Tuned). An approved `git push` still has an external side effect; what
// the approvals teach is that in this project it is not worth a warning.
type Assessment struct {
	Risk     float32            // strongest risky label's probability, 0..1, after this workspace's corrections
	Top      string             // the label that produced Risk ("" if the model has only "safe")
	BaseRisk float32            // Risk as the base model alone put it
	Scores   map[string]float32 // every label the model carries, as the base model scored it
}

// Assess scores one tool call.
func (m *Model) Assess(tool, action string, params map[string]any) Assessment {
	scores := m.Score(CallText(tool, action, params))
	a := Assessment{Scores: make(map[string]float32, len(scores))}
	for i, l := range m.Labels {
		a.Scores[l] = scores[i]
		if l == LabelSafe || m.informational(i) {
			continue
		}
		if scores[i] > a.Risk {
			a.Risk, a.Top = scores[i], l
		}
	}
	a.BaseRisk = a.Risk
	return a
}

func (m *Model) informational(i int) bool {
	return i < len(m.Informational) && m.Informational[i]
}

// IsInformational reports whether a label describes the call rather than
// warning about it — reported in the log, excluded from the headline risk.
func (m *Model) IsInformational(label string) bool {
	for i, l := range m.Labels {
		if l == label {
			return m.informational(i)
		}
	}
	return false
}

// Above reports the labels at or over t, strongest first — what a log line
// wants to name when a call scores high. Informational labels are included:
// "this reached the network" is worth reading next to the score even though it
// is not part of it.
func (a Assessment) Above(t float32) []string {
	type kv struct {
		k string
		v float32
	}
	var out []kv
	for k, v := range a.Scores {
		if k != LabelSafe && v >= t {
			out = append(out, kv{k, v})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].v > out[j].v })
	names := make([]string, len(out))
	for i, e := range out {
		names[i] = fmt.Sprintf("%s=%.2f", e.k, e.v)
	}
	return names
}
