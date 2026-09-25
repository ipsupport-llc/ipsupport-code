package risk

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
)

// Learning from real use, rather than from scripts/risk_dataset.jsonl.
//
// The base weights are trained offline on synthetic data and are never
// modified. What a run learns goes into a sparse DELTA beside them: the
// workspace's own corrections, from the only source of ground truth the agent
// has for free — the approval prompt. A human answering it is labelling the
// call, and the two answers worth learning from are the two the shadow log
// already counts as disagreements:
//
//	the model flagged it and the human approved   -> a false alarm
//	the model stayed quiet and the human refused  -> a miss
//
// Everything else teaches nothing: an approval of a call the model already
// thought was fine confirms only that both agreed.
//
// A deny can mean "not now" or "I'll do it myself" as easily as "that is
// dangerous", so a single one must not move the model much — and it does not.
// The step is normalized by the number of features in the call, so one
// correction shifts the logit by about learnRate, and the total contribution a
// delta may ever make is hard-capped (maxShift). Several consistent corrections
// move the score; one bad label is noise that decays.
const (
	// learnRate is how far one correction moves the logit — exactly that, now
	// that Featurize returns a unit-length vector: the update's shift works out
	// to learnRate * (p - target) * Sum(v^2), and Sum(v^2) is 1.
	//
	// Both constants are measured against the shipped model, not chosen. It puts
	// a confident flagged call between logit 4 and 9. 1.5 per correction means
	// one answer visibly moves a borderline score and four to six consistent ones
	// overturn a confident wrong one — responsive to a pattern, deaf to a
	// one-off.
	//
	// They were 8.0 and 90 against the previous model, whose logits ran to 45
	// because the features were un-normalized. Retraining into a differently
	// scaled model wants these re-measured again; the tests find a flagged call
	// rather than naming one, for the same reason.
	learnRate = 1.5
	// decay shrinks every stored adjustment on each update, so corrections that
	// stop being repeated fade instead of accumulating forever.
	decay = 0.002
	// maxShift is a runaway stop, not the safety mechanism, and it has to clear
	// the model's own confidence — an earlier value sat BELOW it and made a
	// confidently wrong score impossible to correct, which is exactly the case
	// local learning exists for. Roughly three times the observed maximum of 9.
	// What keeps the corrections honest is the rate above, the decay below, and
	// /risk reset.
	maxShift = 30.0
	// pruneBelow drops adjustments too small to matter, keeping the file bounded.
	pruneBelow = 1e-4
)

// Delta is the sparse, per-label weight adjustment learned locally. Row i lines
// up with the base model's Labels[i].
// A delta is only meaningful against the exact model it was learned from: the
// row order is the base model's label order, and the indices within a row are
// positions in its FEATURE SPACE. Both are checked on load, and the second is
// the one that fails silently if it is not — retrain with a different hash seed
// or feature count and every stored index means a different feature, so the
// corrections still apply, still look plausible in /risk, and adjust entirely
// the wrong things.
//
// A rejected delta is not a loss. It is a cache; the durable record of what
// taught it is risk-feedback.jsonl, which is raw (tool, action, params, labels)
// and independent of the model, the feature space and the label set. Whatever
// changes, the corrections refit from that.
type Delta struct {
	Cfg    FeatureConfig // the feature space these indices are positions in
	Labels []string      // the base model's labels, in order
	Rows   []map[uint32]float32
}

// Tuned is a base model plus the local corrections. The base is shared and
// immutable; only the delta changes, under the lock, while tool calls on other
// goroutines are scoring.
type Tuned struct {
	mu      sync.RWMutex
	base    *Model
	d       *Delta
	changed bool
}

// NewTuned pairs a base model with a delta (nil for none).
func NewTuned(base *Model, d *Delta) *Tuned {
	if base == nil {
		return nil
	}
	if d == nil || len(d.Rows) != len(base.Labels) {
		d = &Delta{Labels: base.Labels, Rows: make([]map[uint32]float32, len(base.Labels))}
	}
	return &Tuned{base: base, d: d}
}

// Base is the underlying model — its labels, config and informational flags.
func (t *Tuned) Base() *Model { return t.base }

// Assess scores a call through the base model plus the local corrections.
func (t *Tuned) Assess(tool, action string, params map[string]any) Assessment {
	if t == nil {
		return Assessment{}
	}
	text := CallText(tool, action, params)
	vec := Featurize(t.base.Cfg, text)
	dim := int(t.base.Cfg.Dim)

	t.mu.RLock()
	defer t.mu.RUnlock()

	a := Assessment{Scores: make(map[string]float32, len(t.base.Labels))}
	for li, l := range t.base.Labels {
		row := t.base.W[li*dim : (li+1)*dim]
		var z, dz float32
		z = t.base.Bias[li]
		for idx, val := range vec {
			z += row[idx] * val
			if adj := t.d.Rows[li]; adj != nil {
				dz += adj[idx] * val
			}
		}
		// The cap is the whole safety story for local learning: whatever the
		// corrections say, they move this by at most maxShift logits.
		if dz > maxShift {
			dz = maxShift
		} else if dz < -maxShift {
			dz = -maxShift
		}
		s := sigmoid(z + dz)
		a.Scores[l] = s
		if l == LabelSafe || t.base.informational(li) {
			continue
		}
		if s > a.Risk {
			a.Risk, a.Top = s, l
		}
	}
	return a
}

// Correction is what a human's answer taught about one call.
type Correction struct {
	Tool, Action string
	Params       map[string]any
	// Risky is the human's verdict: true when they refused a call the model had
	// not flagged, false when they approved one it had.
	Risky bool
	// Labels are the model's own labels being corrected. On a false alarm these
	// are the ones that fired, which is exactly known. On a miss the fact of risk
	// comes from the human but the KIND does not, so this carries the model's own
	// strongest hypothesis — reinforcing its guess about which sort of risk,
	// while the human supplies only that there is one.
	Labels []string
}

// Learn applies one correction. Returns how many adjustments the delta now
// holds, so a caller can log that the model actually moved.
func (t *Tuned) Learn(c Correction) int {
	if t == nil || len(c.Labels) == 0 {
		return 0
	}
	vec := Featurize(t.base.Cfg, CallText(c.Tool, c.Action, c.Params))
	if len(vec) == 0 {
		return 0
	}
	target := float32(0)
	if c.Risky {
		target = 1
	}
	// No further normalization: Featurize already returns a unit-length vector,
	// so the shift this update produces is exactly
	//
	//	(p - target) * learnRate * Sum(v^2)  ==  (p - target) * learnRate
	//
	// Dividing by len(vec) here used to be what counteracted un-normalized
	// features accumulating counts. With the vector normalized it is no longer a
	// correction but a bug: it made the step a hundredth of what the constant
	// says, and six consistent answers moved a confident score by 0.09.
	scale := float32(learnRate)

	t.mu.Lock()
	defer t.mu.Unlock()
	dim := int(t.base.Cfg.Dim)
	for _, label := range c.Labels {
		li := t.labelIndex(label)
		if li < 0 {
			continue
		}
		if t.d.Rows[li] == nil {
			t.d.Rows[li] = map[uint32]float32{}
		}
		adj := t.d.Rows[li]
		row := t.base.W[li*dim : (li+1)*dim]
		z := t.base.Bias[li]
		for idx, val := range vec {
			z += (row[idx] + adj[idx]) * val
		}
		g := (sigmoid(z) - target) * scale
		for idx, val := range vec {
			adj[idx] -= g * val
		}
		for idx, v := range adj {
			v *= 1 - decay
			if v > -pruneBelow && v < pruneBelow {
				delete(adj, idx)
				continue
			}
			adj[idx] = v
		}
	}
	t.changed = true
	n := 0
	for _, r := range t.d.Rows {
		n += len(r)
	}
	return n
}

func (t *Tuned) labelIndex(label string) int {
	for i, l := range t.base.Labels {
		if l == label {
			return i
		}
	}
	return -1
}

// ResetDelta drops every local correction, in place. In place because the
// scorer is shared: other goroutines may be scoring a call right now, and
// swapping the whole thing out from under them is a data race.
func (t *Tuned) ResetDelta() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.d.Rows {
		t.d.Rows[i] = nil
	}
	t.changed = false
}

// Adjustments is how many weights the local delta currently holds, for /risk.
func (t *Tuned) Adjustments() int {
	if t == nil {
		return 0
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	n := 0
	for _, r := range t.d.Rows {
		n += len(r)
	}
	return n
}

// deltaMagic identifies the delta file. It is paired with a label list, so a
// delta learned against one base model is refused rather than silently applied
// to another whose rows mean something else.
var deltaMagic = [8]byte{'I', 'P', 'S', 'R', 'D', 'L', 'T', 0x01}

// SaveDelta writes the corrections, if any changed since they were loaded.
func (t *Tuned) SaveDelta(path string) error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.changed {
		return nil
	}
	var buf []byte
	buf = append(buf, deltaMagic[:]...)
	c := t.base.Cfg
	buf = binary.LittleEndian.AppendUint32(buf, c.Dim)
	buf = binary.LittleEndian.AppendUint32(buf, c.Seed)
	buf = append(buf, c.WordMin, c.WordMax, c.CharMin, c.CharMax)
	var fl uint8
	if c.Lowercase {
		fl = 1
	}
	buf = append(buf, fl)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(t.base.Labels)))
	for _, l := range t.base.Labels {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(l)))
		buf = append(buf, l...)
	}
	for _, r := range t.d.Rows {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(r)))
		idxs := make([]uint32, 0, len(r))
		for i := range r {
			idxs = append(idxs, i)
		}
		sort.Slice(idxs, func(a, b int) bool { return idxs[a] < idxs[b] }) // stable file for a stable diff
		for _, i := range idxs {
			buf = binary.LittleEndian.AppendUint32(buf, i)
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(r[i]))
		}
	}
	if err := atomicfile.Write(path, buf, 0o644); err != nil {
		return err
	}
	t.changed = false
	return nil
}

// LoadDelta reads corrections for base. A delta whose labels do not match the
// base model's is refused: the rows are positional, so applying one to a
// different label set would adjust the wrong things.
func LoadDelta(path string, base *Model) (*Delta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	r := &reader{b: data}
	var m [8]byte
	r.read(m[:])
	if r.err == nil && m != deltaMagic {
		return nil, fmt.Errorf("risk: %s is not a delta file", path)
	}
	var cfg FeatureConfig
	cfg.Dim = r.u32()
	cfg.Seed = r.u32()
	cfg.WordMin, cfg.WordMax = r.u8(), r.u8()
	cfg.CharMin, cfg.CharMax = r.u8(), r.u8()
	cfg.Lowercase = r.u8()&1 == 1
	n := int(r.u16())
	if r.err != nil {
		return nil, r.err
	}
	// The feature space first: this is the mismatch that would otherwise pass
	// unnoticed — same labels, same row count, indices meaning something else.
	if cfg != base.Cfg {
		return nil, fmt.Errorf("risk: delta was learned against a different feature space (%+v, model has %+v) — dropping the local corrections; they refit from the feedback log", cfg, base.Cfg)
	}
	d := &Delta{Cfg: cfg, Labels: make([]string, n), Rows: make([]map[uint32]float32, n)}
	for i := range d.Labels {
		d.Labels[i] = r.str()
	}
	for i := range d.Rows {
		cnt := int(r.u32())
		if r.err != nil {
			return nil, r.err
		}
		if cnt > 1<<22 {
			return nil, fmt.Errorf("risk: delta row %d claims %d adjustments", i, cnt)
		}
		row := make(map[uint32]float32, cnt)
		for j := 0; j < cnt; j++ {
			idx := r.u32()
			row[idx] = r.f32()
		}
		d.Rows[i] = row
	}
	if r.err != nil {
		return nil, r.err
	}
	if len(d.Labels) != len(base.Labels) {
		return nil, fmt.Errorf("risk: delta has %d labels, model has %d — retrained model, dropping the local corrections", len(d.Labels), len(base.Labels))
	}
	for i := range d.Labels {
		if d.Labels[i] != base.Labels[i] {
			return nil, fmt.Errorf("risk: delta label %d is %q, model has %q — dropping the local corrections", i, d.Labels[i], base.Labels[i])
		}
	}
	return d, nil
}

// ── the feedback log ────────────────────────────────────────────────────────

// FeedbackRow is one correction, in the same shape scripts/risk_dataset.jsonl
// uses. That is the point: the examples a run collects append straight onto the
// synthetic dataset and retrain the BASE model offline, starting from the
// existing weights rather than from zero. Nothing has to be exported, converted
// or handed over — the dataset assembles itself out of real use.
type FeedbackRow struct {
	Tool   string         `json:"tool"`
	Action string         `json:"action,omitempty"`
	Params map[string]any `json:"params,omitempty"`
	Labels []string       `json:"labels"`
	// Source and When are ours, not the trainer's; it ignores unknown keys.
	Source string `json:"source"`
	When   string `json:"when"`
}

// AppendFeedback records one correction for later offline retraining.
func AppendFeedback(path string, c Correction) error {
	labels := c.Labels
	if !c.Risky {
		labels = []string{LabelSafe}
	}
	row := FeedbackRow{
		Tool: c.Tool, Action: c.Action, Params: c.Params, Labels: labels,
		Source: "approval", When: time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	// Append-only and shared: two sessions on one checkout both correct the same
	// model, and a feedback record is long enough that an O_APPEND write is not
	// atomic — two writers can interleave mid-line and leave neither example
	// parseable. Same reasoning as the session archive and the trace log.
	if unlock, err := filelock.Lock(path); err == nil {
		defer unlock()
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}
