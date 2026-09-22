package risk

import (
	"bytes"
	"strings"
	"testing"
)

func tinyModel() *Model {
	m := &Model{
		Cfg:    FeatureConfig{Dim: 64, Seed: 7, WordMin: 1, WordMax: 1, Lowercase: true},
		Labels: []string{"destructive", "safe"},
		Bias:   []float32{-1, 0.5},
	}
	m.W = make([]float32, len(m.Labels)*int(m.Cfg.Dim))
	i, sign := hashIdx(m.Cfg, "w1:rm")
	m.W[int(i)] = 8 * sign // "rm" pushes the destructive row hard
	return m
}

func TestModelRoundTrip(t *testing.T) {
	want := tinyModel()
	var buf bytes.Buffer
	if err := want.Write(&buf); err != nil {
		t.Fatal(err)
	}
	got, err := Load(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if got.Cfg != want.Cfg {
		t.Errorf("config = %+v, want %+v", got.Cfg, want.Cfg)
	}
	if strings.Join(got.Labels, ",") != strings.Join(want.Labels, ",") {
		t.Errorf("labels = %v, want %v", got.Labels, want.Labels)
	}
	for i := range want.W {
		if got.W[i] != want.W[i] {
			t.Fatalf("weight %d = %v, want %v", i, got.W[i], want.W[i])
		}
	}
	// And it scores: the whole point of the round trip is that the numbers
	// survive it, not just the header.
	if s := got.Assess("run", "shell", map[string]any{"command": "rm x"}); s.Risk < 0.9 {
		t.Errorf("risk = %.2f after a round trip, want > 0.9", s.Risk)
	}
}

// A model file is meant to be swapped by hand. Every way that can go wrong must
// fail loudly: a file read as the wrong layout still scores, and scores wrongly.
func TestLoadRejectsBadFiles(t *testing.T) {
	var good bytes.Buffer
	if err := tinyModel().Write(&good); err != nil {
		t.Fatal(err)
	}
	b := good.Bytes()

	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"empty", nil, "unexpected EOF"},
		{"not a model", []byte("hello there, this is not a model at all"), "bad magic"},
		{"truncated mid-weights", b[:len(b)-8], "unexpected EOF"},
		{"trailing bytes", append(append([]byte{}, b...), 0, 0, 0, 0), "trailing bytes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(tc.data)
			if err == nil {
				t.Fatalf("loaded %s without error", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to mention %q", err, tc.want)
			}
		})
	}

	// A future format must be refused, not guessed at.
	bumped := append([]byte{}, b...)
	bumped[8] = 99
	if _, err := Load(bumped); err == nil || !strings.Contains(err.Error(), "format v99") {
		t.Errorf("error = %v, want a refusal naming the version", err)
	}
}

// The header declares the allocation size, so a corrupt one must not be able to
// ask for gigabytes before failing.
func TestLoadRefusesAnAbsurdHeader(t *testing.T) {
	m := tinyModel()
	m.Cfg.Dim = 1 << 30
	var buf bytes.Buffer
	_ = m.Write(&buf) // writes a header claiming far more weights than follow
	if _, err := Load(buf.Bytes()); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Errorf("error = %v, want a size refusal", err)
	}
}
