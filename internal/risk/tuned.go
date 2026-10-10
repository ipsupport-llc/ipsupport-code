package risk

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/ipsupport-llc/ipsupport-code/internal/atomicfile"
	"github.com/ipsupport-llc/ipsupport-code/internal/filelock"
	"github.com/ipsupport-llc/ipsupport-code/internal/shellsplit"
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
// What an answer teaches is whether a call deserves attention HERE — never what
// the call does. The delta moves the headline risk and leaves every label score
// as the base model gave it (see Assess, and adr/0014): approving `git push` is
// "this is fine in this project", not "this has no external side effect".
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
// The same goes for the WEIGHTS. A retrained model can keep the feature space
// and the labels and still be a different model: the delta's corrections were
// sized against the old weights' mistakes, and applied on top of a base that
// has since learned the same lesson they push it twice. So the delta records a
// fingerprint of the exact base it was learned against, and is refused on any
// other.
//
// A refused delta is dropped, not rebuilt: nothing replays the corrections, and
// this workspace relearns from the answers that follow. What is kept is
// risk-feedback.jsonl, the raw record of every correction, for retraining the
// base offline.
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
	fp      [8]byte // base.Fingerprint(), computed once
	d       *Delta
	synced  []map[uint32]float32 // the rows as last read from or written to disk
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
	return &Tuned{base: base, fp: base.Fingerprint(), d: d, synced: cloneRows(d.Rows)}
}

func cloneRows(rows []map[uint32]float32) []map[uint32]float32 {
	out := make([]map[uint32]float32, len(rows))
	for i, r := range rows {
		out[i] = make(map[uint32]float32, len(r))
		for k, v := range r {
			out[i][k] = v
		}
	}
	return out
}

// Fingerprint identifies a model's exact contents — header, labels and every
// weight. Two models that differ only in their weights have different ones, which
// is what a delta needs to know and what Cfg and Labels cannot tell it.
func (m *Model) Fingerprint() [8]byte {
	h := sha256.New()
	_ = m.Write(h) // a hash.Hash never returns a write error
	var fp [8]byte
	copy(fp[:], h.Sum(nil))
	return fp
}

// Scope is where a call is made: the workspace whose paths are the project's
// own, and the shell its run.shell commands are cut by. It belongs to the
// observer of one agent — a delegate keeps the shell it was started with
// when the host's run.shell changes.
type Scope struct {
	Workspace string
	Shell     shellsplit.Dialect
}

// maxParts bounds what one call costs to score. Past it the rest is not
// scored, and the assessment says so (Incomplete).
const maxParts = 256

// parts is what a call is scored as. A shell line is a program of several
// commands, and scored as one text a long harmless tail dilutes a destructive
// head: `rm -rf x && cat > report.md <<EOF …` came out at 0.07, `rm -rf x`
// alone at 0.88. So the line's code (its commands and operators, without
// heredoc bodies, here-strings or comments — data, not commands) is scored,
// each group of commands joined by | && || (a pipeline's danger can be in the
// joining: `curl … | sh`, whatever comes before it), and each command on its
// own; the call's risk is the highest.
//
// The line is cut as written and each part then localized: localizing first
// rewrote a heredoc's word and the cut swallowed what followed. Every part is
// a copy: what Assess hands back must not change with the caller's map.
func (t *Tuned) parts(sc Scope, tool, action string, params map[string]any) (out []map[string]any, incomplete bool) {
	with := func(c string, set bool) map[string]any {
		p := make(map[string]any, len(params))
		for k, v := range params {
			if s, ok := v.(string); ok {
				v = LocalizePath(s, sc.Workspace)
			}
			p[k] = v
		}
		if set {
			p["command"] = Localize(c, sc.Workspace)
		}
		return p
	}
	cmd, ok := params["command"].(string)
	if tool != "run" || action != "shell" || !ok {
		return []map[string]any{with("", false)}, false
	}
	parsed := shellsplit.Parse(sc.Shell, cmd)
	// The whole, as code. A line that is all comment runs nothing: its code is
	// empty and scored as such, not as the comment's words.
	out = []map[string]any{with(parsed.Code, true)}
	seen := map[string]bool{parsed.Code: true}
	for _, c := range append(append([]string{}, parsed.Groups...), parsed.Commands...) {
		if seen[c] {
			continue
		}
		if len(out) == maxParts {
			return out, true
		}
		seen[c] = true
		out = append(out, with(c, true))
	}
	return out, parsed.Incomplete
}

// Base is the underlying model — its labels, config and informational flags.
func (t *Tuned) Base() *Model { return t.base }

// Assess scores a call through the base model plus the local corrections —
// each of its parts (see parts), the call's risk being the highest. No
// workspace, and sh: see AssessIn.
func (t *Tuned) Assess(tool, action string, params map[string]any) Assessment {
	return t.AssessIn(Scope{}, tool, action, params)
}

// AssessIn is Assess for a call made in sc: its shell cuts the line, and
// paths into its workspace are read as the project's own (see Localize).
func (t *Tuned) AssessIn(sc Scope, tool, action string, params map[string]any) Assessment {
	if t == nil {
		return Assessment{}
	}
	parts, incomplete := t.parts(sc, tool, action, params)
	var out Assessment
	for i, p := range parts {
		a := t.assessText(CallText(tool, action, p))
		a.Params, a.PartScores = p, a.Scores
		if i == 0 {
			out = a
			out.Scores = maps.Clone(a.Scores) // pooled below; PartScores keeps the part's own
			continue
		}
		for l, v := range a.Scores { // what the call does: the most any part does —
			// and it is safe only as far as every part is
			if (l == LabelSafe) == (v < out.Scores[l]) {
				out.Scores[l] = v
			}
		}
		out.BaseRisk = max(out.BaseRisk, a.BaseRisk)
		if a.Risk > out.Risk {
			out.Risk, out.Top, out.Params, out.PartScores = a.Risk, a.Top, p, a.PartScores
		}
	}
	if tool == "run" && action == "shell" {
		out.Shell = sc.Shell.String()
	}
	out.Incomplete = incomplete
	return out
}

// assessText scores one rendered call.
func (t *Tuned) assessText(text string) Assessment {
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
		// What the call does stays the base model's word; the corrections only
		// decide how much attention it gets here. Applying them to the label
		// itself turned "approved git push" into "git push has no external side
		// effect" — true of nothing, and written to the log as if it were.
		base := sigmoid(z)
		a.Scores[l] = base
		if l == LabelSafe || t.base.informational(li) {
			continue
		}
		if base > a.BaseRisk {
			a.BaseRisk = base
		}
		if s := sigmoid(z + dz); s > a.Risk {
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
	// Shell is the dialect the call's shell line was cut by (see Assessment).
	Shell string
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
		// An informational label never reaches the headline, which is all a
		// correction adjusts — learning one would only grow the file.
		if li < 0 || t.base.informational(li) {
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

// Reset drops every local correction, in memory and on disk at path. In place
// because the scorer is shared: other goroutines may be scoring a call right
// now, and swapping the whole thing out from under them is a data race.
//
// The file goes under the same locks SaveDelta writes it under — the in-process
// one and the file lock another session on this workspace saves under. Removing
// it outside (as /risk reset used to) raced a save already in flight — learned
// just before the reset, written just after the remove — and the corrections
// the user had just cleared came back on the next start.
func (t *Tuned) Reset(path string) error {
	if t == nil {
		return nil
	}
	unlock, err := filelock.Lock(path)
	if err != nil {
		return err
	}
	defer unlock()
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.d.Rows {
		t.d.Rows[i] = nil
	}
	t.synced = cloneRows(t.d.Rows)
	t.changed = false
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
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

// deltaMagic identifies the delta file. It is followed by the fingerprint of
// the base model it was learned against, its feature config and its label
// list, so a delta learned against one base is refused rather than silently
// applied to another. Version 2 added the fingerprint; a version-1 file cannot
// say which weights it belongs to and is refused.
var (
	deltaMagic   = [8]byte{'I', 'P', 'S', 'R', 'D', 'L', 'T', 0x02}
	deltaMagicV1 = [8]byte{'I', 'P', 'S', 'R', 'D', 'L', 'T', 0x01}
)

// SaveDelta writes the corrections, if any changed since they were loaded.
//
// Another session on the same workspace may have saved since this one last
// read the file, or reset it. So the save is a merge under a file lock: what is
// on disk now, plus what THIS session learned since it last synced. Writing the
// in-memory rows wholesale lost the other session's corrections, and brought
// back the ones a reset in the other session had cleared.
func (t *Tuned) SaveDelta(path string) error {
	if t == nil {
		return nil
	}
	unlock, err := filelock.Lock(path)
	if err != nil {
		return err
	}
	defer unlock()
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.changed {
		return nil
	}
	rows := make([]map[uint32]float32, len(t.d.Rows))
	if disk, err := LoadDelta(path, t.base); err == nil {
		rows = disk.Rows // missing, or refused for another model: this session's learning alone
	}
	for i, r := range t.d.Rows {
		if rows[i] == nil {
			rows[i] = map[uint32]float32{}
		}
		for idx, v := range r {
			rows[i][idx] += v - t.synced[i][idx]
		}
		for idx, v := range t.synced[i] {
			if _, ok := r[idx]; !ok {
				rows[i][idx] -= v // pruned here since the last sync
			}
		}
		for idx, v := range rows[i] {
			if v > -pruneBelow && v < pruneBelow {
				delete(rows[i], idx)
			}
		}
	}
	var buf []byte
	buf = append(buf, deltaMagic[:]...)
	buf = append(buf, t.fp[:]...)
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
	for _, r := range rows {
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
	t.d.Rows, t.synced = rows, cloneRows(rows)
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
	var m, fp [8]byte
	r.read(m[:])
	if r.err == nil && m == deltaMagicV1 {
		return nil, fmt.Errorf("risk: %s was written by an older build and does not say which model it was learned against — dropping the local corrections", path)
	}
	if r.err == nil && m != deltaMagic {
		return nil, fmt.Errorf("risk: %s is not a delta file", path)
	}
	r.read(fp[:])
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
		return nil, fmt.Errorf("risk: delta was learned against a different feature space (%+v, model has %+v) — dropping the local corrections", cfg, base.Cfg)
	}
	d := &Delta{Cfg: cfg, Labels: make([]string, n), Rows: make([]map[uint32]float32, n)}
	for i := range d.Labels {
		d.Labels[i] = r.str()
	}
	for i := range d.Rows {
		cnt := uint64(r.u32())
		if r.err != nil {
			return nil, r.err
		}
		// Each adjustment is 8 bytes; a count the rest of the file cannot hold is
		// corrupt, and is refused before the map is sized from it.
		if left := uint64(len(data) - r.i); cnt*8 > left {
			return nil, fmt.Errorf("risk: delta row %d claims %d adjustments, the file has room for %d", i, cnt, left/8)
		}
		row := make(map[uint32]float32, cnt)
		for j := uint64(0); j < cnt; j++ {
			idx := r.u32()
			v := r.f32()
			if f := float64(v); math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("risk: delta row %d holds a %v adjustment", i, v)
			}
			if idx >= cfg.Dim {
				return nil, fmt.Errorf("risk: delta row %d adjusts feature %d, the model has %d", i, idx, cfg.Dim)
			}
			row[idx] = v
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
	if r.i != len(data) {
		return nil, fmt.Errorf("risk: %d trailing bytes after the delta", len(data)-r.i)
	}
	// Last, because it is the least specific: every check above names what
	// changed, and this one can only say that SOMETHING in the weights did.
	if fp != base.Fingerprint() {
		return nil, fmt.Errorf("risk: delta was learned against a different build of the model (same labels and features, different weights) — dropping the local corrections")
	}
	return d, nil
}

// ── the feedback log ────────────────────────────────────────────────────────

// FeedbackRow is one answer at an approval prompt, in the shape
// scripts/risk_dataset.jsonl uses — but deliberately WITHOUT labels.
//
// An answer is a verdict on whether the call was acceptable, not a statement of
// what it does, and the dataset's labels are the second thing. Recording an
// approval as labels ["safe"] taught the next base model that `git push` has no
// external side effect and `rm -rf build/` deletes nothing. So the row carries
// the Verdict and what the model Scored, and the trainer skips approval rows
// unless a person has labelled one (set "labels" and "source": "manual") — which
// is the only way a verdict becomes a fact about the call.
type FeedbackRow struct {
	Tool    string         `json:"tool"`
	Action  string         `json:"action,omitempty"`
	Params  map[string]any `json:"params,omitempty"`
	Labels  []string       `json:"labels,omitempty"`
	Shell   string         `json:"shell,omitempty"`  // sh, powershell or cmd: how the trainer cuts it
	Verdict string         `json:"verdict"`          // "approved" or "refused"
	Scored  []string       `json:"scored,omitempty"` // the labels the model put on it — its claim, not the answer
	Source  string         `json:"source"`
	When    string         `json:"when"`
}

// AppendFeedback records one correction for later offline retraining.
func AppendFeedback(path string, c Correction) error {
	verdict := "approved"
	if c.Risky {
		verdict = "refused"
	}
	row := FeedbackRow{
		Tool: c.Tool, Action: c.Action, Params: c.Params, Shell: c.Shell, Verdict: verdict, Scored: c.Labels,
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
	// Private: a row holds the call's full arguments, secrets included. Chmod
	// too, for a log an older build created world-readable.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_ = f.Chmod(0o600)
	if _, err := f.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}
