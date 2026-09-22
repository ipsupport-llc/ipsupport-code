package risk

import (
	"strings"
	"testing"
)

// Shadow mode's whole product is the disagreement column: which calls the
// policy waved through that the model would have stopped, and which it gated
// that the model finds unremarkable. The first says what a risk gate could add;
// the second says how much friction it would cost.
func TestDisagreementIsClassifiedBothWays(t *testing.T) {
	m, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	risky := map[string]any{"command": "cat ~/.ssh/id_rsa"}
	routine := map[string]any{"command": "go test ./..."}

	for _, tc := range []struct {
		name          string
		params        map[string]any
		verdict       PolicyVerdict
		wantDisagreed int64
	}{
		{"risky call the policy allowed", risky, VerdictAllow, 1},
		{"risky call the policy asked about", risky, VerdictAsk, 0},
		{"routine call the policy allowed", routine, VerdictAllow, 0},
		{"routine call the policy denied", routine, VerdictDeny, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := NewShadow(NewTuned(m, nil))
			s.Observe("run", "shell", tc.params, tc.verdict)
			if _, _, d := s.Stats(); d != tc.wantDisagreed {
				t.Errorf("disagreed = %d, want %d", d, tc.wantDisagreed)
			}
		})
	}
}

func TestShadowCountsAndSummarises(t *testing.T) {
	m, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	s := NewShadow(NewTuned(m, nil))
	s.Observe("run", "shell", map[string]any{"command": "go build ./..."}, VerdictAllow)
	s.Observe("run", "shell", map[string]any{"command": "rm -rf /opt/data"}, VerdictAllow)
	c, f, d := s.Stats()
	if c != 2 || f != 1 || d != 1 {
		t.Errorf("stats = %d scored / %d flagged / %d disagreed, want 2/1/1", c, f, d)
	}
	if sum := s.Summary(); !strings.Contains(sum, "scored 2") || !strings.Contains(sum, "1 disagreed") {
		t.Errorf("summary = %q", sum)
	}
}

// Shadow mode must be exactly that: the return value carries the score, and
// nothing in the type offers a way to stop a call.
func TestObserveReturnsAScoreAndNothingElse(t *testing.T) {
	m, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	a := NewShadow(NewTuned(m, nil)).Observe("run", "shell", map[string]any{"command": "rm -rf /"}, VerdictAllow)
	if a.Risk < Threshold {
		t.Errorf("risk = %.2f for `rm -rf /`, want >= %.2f", a.Risk, Threshold)
	}
	if len(a.Above(Threshold)) == 0 {
		t.Error("Above() named no labels for a call that scored over the threshold")
	}
}

func TestVerdictStrings(t *testing.T) {
	for v, want := range map[PolicyVerdict]string{
		VerdictAllow: "allow", VerdictAsk: "ask", VerdictDeny: "deny", VerdictUnknown: "unknown",
	} {
		if got := v.String(); got != want {
			t.Errorf("%d.String() = %q, want %q", v, got, want)
		}
	}
}
