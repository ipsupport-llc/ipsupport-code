package risk

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// aFlaggedCall finds a command the CURRENT shipped model flags, rather than
// hardcoding one. Retraining moves individual scores — "rm -rf Debug" was a
// false alarm until the build-output vocabulary started coming from
// github/gitignore, which has Debug in it — and a fixture pinned to one string
// turns every retrain into a test failure that says nothing.
func aFlaggedCall(t *testing.T, tn *Tuned) string {
	t.Helper()
	for _, c := range []string{
		"truncate -s 0 install_manifest.txt", "shred -u dkms.conf",
		"rm -rf src/app.ts", "cat ~/.ssh/id_rsa", "cat /etc/shadow",
	} {
		if tn.Assess("run", "shell", map[string]any{"command": c}).Risk >= Threshold {
			return c
		}
	}
	t.Skip("the shipped model flags none of the candidates — pick new ones")
	return ""
}

func tunedModel(t *testing.T) *Tuned {
	t.Helper()
	m, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	return NewTuned(m, nil)
}

// A false alarm the human waved through has to come down — and nothing else may
// move with it. The corrections are the only thing standing between a model
// trained on synthetic data and the way one particular project actually works.
func TestACorrectionMovesOnlyWhatItWasAbout(t *testing.T) {
	tn := tunedModel(t)
	target := aFlaggedCall(t, tn)
	const other = "go test ./..." // must stay exactly as unremarkable as it was

	before := tn.Assess("run", "shell", map[string]any{"command": target})
	otherBefore := tn.Assess("run", "shell", map[string]any{"command": other}).Risk

	// Keep answering the same way, as a person who keeps approving the same call
	// would, and count how many it takes. The property is the point, not a
	// number: one answer must not overturn a confident score (that is
	// TestOneCorrectionNudgesRatherThanFlips), and a handful must.
	const limit = 12
	n := 0
	for ; n < limit; n++ {
		cur := tn.Assess("run", "shell", map[string]any{"command": target})
		if cur.Risk < Threshold {
			break
		}
		c, ok := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
			map[string]any{"command": target}, cur), true)
		if !ok {
			t.Fatalf("iteration %d: a flagged call approved by a human should be a correction", n)
		}
		if tn.Learn(c) == 0 {
			t.Fatalf("iteration %d: the correction stored no adjustments", n)
		}
	}

	after := tn.Assess("run", "shell", map[string]any{"command": target})
	if after.Risk >= Threshold {
		t.Errorf("risk is still %.2f after %d corrections — learning is too slow to be useful", after.Risk, limit)
	}
	if after.Risk >= before.Risk {
		t.Errorf("risk went %.2f -> %.2f; repeated corrections should bring a false alarm down", before.Risk, after.Risk)
	}
	if n < 2 {
		t.Errorf("one answer overturned a confident score (%.2f -> %.2f); a refusal can mean "+
			"\"not now\", so a single one must only nudge", before.Risk, after.Risk)
	}
	t.Logf("%d consistent answers took it from %.2f to %.2f", n, before.Risk, after.Risk)

	if got := tn.Assess("run", "shell", map[string]any{"command": other}).Risk; got > otherBefore+0.05 {
		t.Errorf("an unrelated call moved %.2f -> %.2f; corrections must be local to what they corrected", otherBefore, got)
	}
}

// One refusal must not flip the model: a person saying "no" often means "not
// now" or "I'll do it myself", and only a repeated pattern is a judgement about
// the call itself.
func TestOneCorrectionNudgesRatherThanFlips(t *testing.T) {
	tn := tunedModel(t)
	const safe = "go test ./..."
	before := tn.Assess("run", "shell", map[string]any{"command": safe})
	if before.Risk >= Threshold {
		t.Fatalf("the base model already flags %q (%.2f)", safe, before.Risk)
	}
	c, ok := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
		map[string]any{"command": safe}, before), false) // refused
	if !ok {
		t.Fatal("an unflagged call refused by a human should be a correction")
	}
	tn.Learn(c)
	if after := tn.Assess("run", "shell", map[string]any{"command": safe}); after.Risk >= Threshold {
		t.Errorf("one refusal took %q from %.2f to %.2f — a single answer must not flip the model",
			safe, before.Risk, after.Risk)
	}
}

// Only the two disagreements carry information. Learning from an agreement
// would be learning from the model's own output.
func TestOnlyDisagreementsTeach(t *testing.T) {
	tn := tunedModel(t)
	risky := tn.Assess("run", "shell", map[string]any{"command": aFlaggedCall(t, tn)})
	safe := tn.Assess("run", "shell", map[string]any{"command": "go test ./..."})
	if risky.Risk < Threshold || safe.Risk >= Threshold {
		t.Fatalf("the fixtures no longer hold: risky=%.2f safe=%.2f", risky.Risk, safe.Risk)
	}
	for _, tc := range []struct {
		name     string
		a        Assessment
		approved bool
		want     bool
	}{
		{"flagged and approved is a false alarm", risky, true, true},
		{"quiet and refused is a miss", safe, false, true},
		{"flagged and refused: they agreed", risky, false, false},
		{"quiet and approved: they agreed", safe, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := CorrectionFrom(WithAssessment(context.Background(), "run", "shell", map[string]any{"command": "x"}, tc.a), tc.approved)
			if ok != tc.want {
				t.Errorf("teaches = %v, want %v", ok, tc.want)
			}
		})
	}
	// And with nothing scored on the context there is nothing to learn from.
	if _, ok := CorrectionFrom(context.Background(), false); ok {
		t.Error("a context carrying no assessment should teach nothing")
	}
}

func TestDeltaSurvivesARoundTrip(t *testing.T) {
	tn := tunedModel(t)
	target := aFlaggedCall(t, tn)
	for i := 0; i < 3; i++ {
		c, _ := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
			map[string]any{"command": target}, tn.Assess("run", "shell", map[string]any{"command": target})), true)
		tn.Learn(c)
	}
	want := tn.Assess("run", "shell", map[string]any{"command": target}).Risk

	path := filepath.Join(t.TempDir(), "risk-delta.bin")
	if err := tn.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	base, _ := Default()
	d, err := LoadDelta(path, base)
	if err != nil {
		t.Fatal(err)
	}
	// Not exact equality: the dot product sums a map, so Go's randomized
	// iteration order changes the float32 rounding between two runs of the same
	// arithmetic. What has to survive the round trip is the value, not the bits.
	got := NewTuned(base, d).Assess("run", "shell", map[string]any{"command": target}).Risk
	if diff := got - want; diff > 1e-4 || diff < -1e-4 {
		t.Errorf("risk %.6f after a round trip, want %.6f", got, want)
	}
	if n := NewTuned(base, d).Adjustments(); n == 0 {
		t.Error("the reloaded delta holds no adjustments")
	}
}

// The delta's rows are positional, so pairing one with a model whose labels
// differ would adjust the wrong things. It must be refused, not applied.
func TestDeltaRefusesAMismatchedModel(t *testing.T) {
	tn := tunedModel(t)
	target := aFlaggedCall(t, tn)
	c, _ := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
		map[string]any{"command": target}, tn.Assess("run", "shell", map[string]any{"command": target})), true)
	tn.Learn(c)
	path := filepath.Join(t.TempDir(), "d.bin")
	if err := tn.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	other := &Model{
		Cfg: tn.Base().Cfg, Labels: []string{"danger", "safe"}, Bias: []float32{0, 0},
		Informational: []bool{false, false},
		W:             make([]float32, 2*int(tn.Base().Cfg.Dim)),
	}
	if _, err := LoadDelta(path, other); err == nil {
		t.Error("loaded a delta against a different label set")
	} else if !strings.Contains(err.Error(), "labels") && !strings.Contains(err.Error(), "label") {
		t.Errorf("error = %q, want it to name the label mismatch", err)
	}
}

// Answers are recorded in the dataset's own shape — but as VERDICTS, with no
// labels. An approval says the call was acceptable, not what it does: written
// as labels ["safe"] it taught the next base model that an approved `rm -rf`
// deletes nothing. The trainer skips these rows until a person labels one.
func TestFeedbackIsWrittenInTheDatasetsOwnShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "risk-feedback.jsonl")
	for _, c := range []Correction{
		{Tool: "run", Action: "shell", Params: map[string]any{"command": "rm -rf Debug"}, Risky: false, Labels: []string{"destructive"}},
		{Tool: "run", Action: "shell", Params: map[string]any{"command": "xxd ca.key"}, Risky: true, Labels: []string{"credential_access"}},
	} {
		if err := AppendFeedback(path, c); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("wrote %d line(s), want 2", len(lines))
	}
	for i, want := range [][]string{
		{`"verdict":"approved"`, `"scored":["destructive"]`},
		{`"verdict":"refused"`, `"scored":["credential_access"]`},
	} {
		for _, w := range want {
			if !strings.Contains(lines[i], w) {
				t.Errorf("line %d is missing %s:\n%s", i, w, lines[i])
			}
		}
		if strings.Contains(lines[i], `"labels"`) {
			t.Errorf("line %d claims labels — an answer is a verdict, not a label:\n%s", i, lines[i])
		}
	}
	for _, want := range []string{`"tool":"run"`, `"action":"shell"`, `"command":"xxd ca.key"`} {
		if !strings.Contains(lines[1], want) {
			t.Errorf("missing %s — the trainer reads these keys:\n%s", want, lines[1])
		}
	}
}

// The label check is the obvious one. The FEATURE SPACE check is the one that
// matters: same labels, same row count, but a different hash seed or feature
// count means every stored index points at a different feature — so the
// corrections would still load, still look plausible, and adjust the wrong
// things. Silent, and only visible as a model that got quietly worse.
func TestDeltaRefusesADifferentFeatureSpace(t *testing.T) {
	tn := tunedModel(t)
	target := aFlaggedCall(t, tn)
	c, _ := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
		map[string]any{"command": target}, tn.Assess("run", "shell", map[string]any{"command": target})), true)
	tn.Learn(c)
	path := filepath.Join(t.TempDir(), "d.bin")
	if err := tn.SaveDelta(path); err != nil {
		t.Fatal(err)
	}

	base := tn.Base()
	for _, tc := range []struct {
		name string
		mut  func(*Model)
	}{
		{"a different hash seed", func(m *Model) { m.Cfg.Seed++ }},
		{"a different feature count", func(m *Model) { m.Cfg.Dim /= 2 }},
		{"a different n-gram range", func(m *Model) { m.Cfg.CharMax++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			other := &Model{Cfg: base.Cfg, Labels: base.Labels, Bias: base.Bias,
				Informational: base.Informational, W: base.W}
			tc.mut(other)
			if _, err := LoadDelta(path, other); err == nil {
				t.Error("loaded a delta learned against a different feature space")
			} else if !strings.Contains(err.Error(), "feature space") {
				t.Errorf("error = %q, want it to name the feature space", err)
			}
		})
	}
	// …and it still loads against the model it was actually learned from.
	if _, err := LoadDelta(path, base); err != nil {
		t.Errorf("refused its own model: %v", err)
	}
}

// learnedDelta learns a few corrections and saves them, returning the file.
func learnedDelta(t *testing.T) (*Tuned, string) {
	t.Helper()
	tn := tunedModel(t)
	target := aFlaggedCall(t, tn)
	for i := 0; i < 3; i++ {
		c, _ := CorrectionFrom(WithAssessment(context.Background(), "run", "shell",
			map[string]any{"command": target}, tn.Assess("run", "shell", map[string]any{"command": target})), true)
		tn.Learn(c)
	}
	path := filepath.Join(t.TempDir(), "risk-delta.bin")
	if err := tn.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	return tn, path
}

// A retrained model can keep the labels and the feature space and still be a
// different model. The corrections were sized against the OLD weights'
// mistakes; on a base that has since learned the same lesson they push it twice.
func TestDeltaRefusesDifferentWeights(t *testing.T) {
	tn, path := learnedDelta(t)
	b := tn.Base()
	retrained := &Model{Cfg: b.Cfg, Labels: b.Labels, Informational: b.Informational,
		Bias: append([]float32{}, b.Bias...), W: append([]float32{}, b.W...)}
	retrained.W[12345] += 0.5 // one weight: same labels, same features, another model

	if _, err := LoadDelta(path, b); err != nil {
		t.Fatalf("the delta no longer loads against its own base: %v", err)
	}
	if _, err := LoadDelta(path, retrained); err == nil {
		t.Error("applied a delta to a base with different weights")
	} else if !strings.Contains(err.Error(), "different build") {
		t.Errorf("error = %q, want it to say the weights changed", err)
	}
}

// A delta from before the fingerprint existed cannot say which weights it
// belongs to, so it is refused rather than trusted.
func TestDeltaRefusesTheOldFormat(t *testing.T) {
	tn, path := learnedDelta(t)
	data, _ := os.ReadFile(path)
	old := append(append([]byte{}, deltaMagicV1[:]...), data[16:]...) // v1: no fingerprint
	os.WriteFile(path, old, 0o644)
	if _, err := LoadDelta(path, tn.Base()); err == nil || !strings.Contains(err.Error(), "older build") {
		t.Errorf("error = %v, want the old format refused by name", err)
	}
}

// The delta file is read back on every start, so a corrupt one must fail
// loudly and cheaply — not size a map from a garbage count, and not load a NaN
// that turns every score it touches into "safe".
func TestDeltaRejectsCorruptRows(t *testing.T) {
	tn, path := learnedDelta(t)
	good, _ := os.ReadFile(path)
	base := tn.Base()
	// The first row's count sits right after the header.
	hdr := 8 + 8 + 4 + 4 + 5 + 2
	for _, l := range base.Labels {
		hdr += 2 + len(l)
	}
	firstCount := binary.LittleEndian.Uint32(good[hdr:])
	if firstCount == 0 {
		t.Skip("the first row is empty; nothing to corrupt")
	}
	entry := hdr + 4 // first (index, value) pair of the first row

	for _, tc := range []struct {
		name  string
		patch func(b []byte)
		want  string
	}{
		{"count past the end", func(b []byte) { binary.LittleEndian.PutUint32(b[hdr:], 1<<30) }, "room for"},
		{"NaN adjustment", func(b []byte) { binary.LittleEndian.PutUint32(b[entry+4:], math.Float32bits(float32(math.NaN()))) }, "NaN"},
		{"index outside the model", func(b []byte) { binary.LittleEndian.PutUint32(b[entry:], base.Cfg.Dim+7) }, "adjusts feature"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := append([]byte{}, good...)
			tc.patch(b)
			os.WriteFile(path, b, 0o644)
			if _, err := LoadDelta(path, base); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

// An approval is a verdict on whether a call is acceptable HERE, not a claim
// about what it does. Approving `git push` over and over must stop the warning
// in this workspace — and must not teach that git push has no external side
// effect. The label scores are the base model's word and stay exactly that.
func TestApprovalsQuietTheWarningNotTheLabels(t *testing.T) {
	tn := tunedModel(t)
	target := map[string]any{"command": aFlaggedCall(t, tn)}
	before := tn.Assess("run", "shell", target)

	for i := 0; i < 12 && tn.Assess("run", "shell", target).Risk >= Threshold; i++ {
		as := tn.Assess("run", "shell", target)
		c, ok := CorrectionFrom(WithAssessment(context.Background(), "run", "shell", target, as), true)
		if !ok {
			t.Fatal("an approved flagged call produced no correction")
		}
		tn.Learn(c)
	}
	after := tn.Assess("run", "shell", target)

	if after.Risk >= Threshold {
		t.Errorf("still flagged after a dozen approvals: risk %.2f", after.Risk)
	}
	if !near(after.BaseRisk, before.BaseRisk) {
		t.Errorf("base risk moved %.4f -> %.4f; only the workspace's view may", before.BaseRisk, after.BaseRisk)
	}
	for l, v := range before.Scores {
		if !near(after.Scores[l], v) {
			t.Errorf("label %s moved %.4f -> %.4f — an approval rewrote what the call does", l, v, after.Scores[l])
		}
	}
}

// And the other direction: refusing a call the model was quiet about raises the
// warning here, without inventing a label the base model did not give it.
func TestRefusalsRaiseTheWarningNotTheLabels(t *testing.T) {
	tn := tunedModel(t)
	quiet := map[string]any{"command": "go test ./..."}
	before := tn.Assess("run", "shell", quiet)
	if before.Risk >= Threshold {
		t.Skip("the shipped model flags go test")
	}
	for i := 0; i < 3; i++ {
		as := tn.Assess("run", "shell", quiet)
		c, ok := CorrectionFrom(WithAssessment(context.Background(), "run", "shell", quiet, as), false)
		if !ok {
			break // flagged now: a refusal of a flagged call teaches nothing more
		}
		tn.Learn(c)
	}
	after := tn.Assess("run", "shell", quiet)
	if after.Risk <= before.Risk {
		t.Errorf("risk %.4f -> %.4f after refusals; want it raised", before.Risk, after.Risk)
	}
	for l, v := range before.Scores {
		if !near(after.Scores[l], v) {
			t.Errorf("label %s moved %.4f -> %.4f on a refusal", l, v, after.Scores[l])
		}
	}
}

// near compares two scores of the same call. Not ==: the dot product sums a
// map, and Go's randomized iteration order changes the float32 rounding
// between two evaluations of identical arithmetic.
func near(a, b float32) bool { return a-b < 1e-5 && b-a < 1e-5 }

// Two sessions on one workspace share the delta file. Each must add its own
// corrections to what is on disk, not overwrite the other's; and a reset in
// one must not be undone by the other's next save.
func TestTwoSessionsShareTheDelta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "risk-delta.bin")
	base, _ := Default()
	learn := func(tn *Tuned, cmd string) {
		tn.Learn(Correction{Tool: "run", Action: "shell", Params: map[string]any{"command": cmd}, Labels: []string{"destructive"}})
	}
	// sum adds (sign=1) or subtracts (sign=-1) y into x.
	sum := func(x, y []map[uint32]float32, sign float32) []map[uint32]float32 {
		out := cloneRows(x)
		for i, r := range y {
			for k, v := range r {
				out[i][k] += sign * v
			}
		}
		return out
	}
	fileIs := func(want []map[uint32]float32, what string) {
		t.Helper()
		d, err := LoadDelta(path, base)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			for k, v := range want[i] {
				if diff := d.Rows[i][k] - v; (v > pruneBelow || v < -pruneBelow) && (diff > 1e-5 || diff < -1e-5) {
					t.Fatalf("%s: row %d feature %d is %v on disk, want %v", what, i, k, d.Rows[i][k], v)
				}
			}
			for k, v := range d.Rows[i] {
				if w := want[i][k]; w == 0 && v != 0 {
					t.Fatalf("%s: row %d feature %d is %v on disk, want none", what, i, k, v)
				}
			}
		}
	}

	a, b := NewTuned(base, nil), NewTuned(base, nil)
	learn(a, "rm -rf build")
	fromA := cloneRows(a.d.Rows)
	if err := a.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	learn(b, "git clean -fdx")
	fromB := cloneRows(b.d.Rows)
	if err := b.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	fileIs(sum(fromA, fromB, 1), "after both saved")

	if err := a.Reset(path); err != nil {
		t.Fatal(err)
	}
	before := cloneRows(b.d.Rows)
	learn(b, "docker system prune -af")
	after := cloneRows(b.d.Rows)
	if err := b.SaveDelta(path); err != nil {
		t.Fatal(err)
	}
	fileIs(sum(after, before, -1), "after a reset elsewhere")
}

// The feedback log carries the full, unredacted arguments of approved calls,
// so other local users must not be able to read it — including a log an
// older build already created world-readable.
func TestFeedbackIsPrivate(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no POSIX modes")
	}
	path := filepath.Join(t.TempDir(), "risk-feedback.jsonl")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := AppendFeedback(path, Correction{Tool: "run", Action: "shell", Params: map[string]any{"command": "ls"}}); err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(path); fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v, want 0600", fi.Mode().Perm())
	}
}

// A call the workspace's own corrections pushed over the threshold, while every
// base label stayed under it, must still be unlearnable by approving it — or a
// false alarm the delta created could only be cleared by /risk reset.
func TestAnApprovalUndoesAFalseAlarmTheDeltaCreated(t *testing.T) {
	as := Assessment{Risk: 0.7, Top: "destructive", BaseRisk: 0.3,
		Scores: map[string]float32{"destructive": 0.3, "safe": 0.6}}
	ctx := WithAssessment(context.Background(), "run", "shell", map[string]any{"command": "make clean"}, as)
	c, ok := CorrectionFrom(ctx, true)
	if !ok || len(c.Labels) != 1 || c.Labels[0] != "destructive" || c.Risky {
		t.Fatalf("correction = %+v, %v; want an approval teaching %q down", c, ok, "destructive")
	}
}
