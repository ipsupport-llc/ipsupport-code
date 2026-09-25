package risk

import (
	"context"
	"os"
	"path/filepath"
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

// The corrections are recorded in the dataset's own shape, so they concatenate
// onto the synthetic set and fine-tune the base offline. Nothing has to be
// exported or converted — that is the answer to "how do I hand over a dataset".
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
	// An approved false alarm is recorded as safe; a refused miss keeps the label.
	if !strings.Contains(lines[0], `"labels":["safe"]`) {
		t.Errorf("a false alarm should be recorded as safe:\n%s", lines[0])
	}
	if !strings.Contains(lines[1], `"labels":["credential_access"]`) {
		t.Errorf("a miss should keep the label:\n%s", lines[1])
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
