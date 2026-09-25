// Package risk scores a tool call for how dangerous it looks, as a signal
// alongside the permission policy rather than a replacement for it.
//
// The whole model is a hashed-feature linear classifier: text in, one
// probability per label out. That shape is deliberate. It has no runtime
// dependency (no Python, no CGO, no ONNX), it is a few hundred kilobytes of
// float32 that go:embed can carry, and scoring is a handful of dot products
// over the features a call actually has — microseconds, on the hot path of
// every tool call. A bigger model would buy accuracy the harness cannot
// currently justify paying for in latency, binary size, or build complexity.
package risk

import (
	"math"
	"strings"
	"unicode"
)

// FeatureConfig is how text becomes feature indices. It is read FROM the model
// file, never hardcoded here: a retrained model that uses different n-gram
// ranges or a different feature space must be droppable in as a new file, and
// that only works if the extractor takes its parameters from the file too.
type FeatureConfig struct {
	Dim       uint32 // feature space size; indices are taken modulo this
	Seed      uint32 // hash seed, so two models can disagree about collisions
	WordMin   uint8  // word n-gram range, inclusive; 0 disables word n-grams
	WordMax   uint8
	CharMin   uint8 // character n-gram range, inclusive; 0 disables char n-grams
	CharMax   uint8
	Lowercase bool
}

// hashIdx is FNV-1a over the seed then the feature's bytes, folded into the
// feature space. The exact arithmetic is a wire format: the Python trainer
// computes indices the same way, and a one-bit difference silently trains the
// weights for a feature that inference will never look up. TestHashVectors
// pins the values, and scripts/train_risk.py asserts the same ones.
func hashIdx(cfg FeatureConfig, s string) (idx uint32, sign float32) {
	const (
		offset32 = 2166136261
		prime32  = 16777619
	)
	h := uint32(offset32)
	for _, b := range []byte{byte(cfg.Seed), byte(cfg.Seed >> 8), byte(cfg.Seed >> 16), byte(cfg.Seed >> 24)} {
		h = (h ^ uint32(b)) * prime32
	}
	for i := 0; i < len(s); i++ {
		h = (h ^ uint32(s[i])) * prime32
	}
	// Sign hashing: one bit of the same hash decides whether the feature adds or
	// subtracts. Two colliding features then cancel as often as they reinforce,
	// which keeps a collision from reading as "this feature, twice as strongly".
	sign = 1
	if h&1 == 1 {
		sign = -1
	}
	return (h >> 1) % cfg.Dim, sign
}

// Tokenize splits call text into words the way both sides of the model agree
// on: runs of letters, digits and the punctuation that carries meaning in a
// command line. "rm -rf /" is three tokens, not one; "~/.aws/credentials" keeps
// its slashes and dots, because those are exactly what makes it recognizable.
func Tokenize(cfg FeatureConfig, text string) []string {
	if cfg.Lowercase {
		text = strings.ToLower(text)
	}
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			out = append(out, b.String())
			b.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			b.WriteRune(r)
		case r == '/', r == '.', r == '-', r == '_', r == '~', r == ':', r == '\\':
			b.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return out
}

// Featurize turns call text into the sparse vector the classifier scores:
// index → accumulated value. Word n-grams carry the phrasing ("rm -rf"), char
// n-grams carry the shape of things a tokenizer would otherwise split or a
// model would otherwise have to have seen verbatim (a path fragment, a
// base64-looking blob, a typo'd flag).
func Featurize(cfg FeatureConfig, text string) map[uint32]float32 {
	if cfg.Dim == 0 {
		return nil
	}
	vec := make(map[uint32]float32, 256)
	add := func(feature string) {
		i, sign := hashIdx(cfg, feature)
		vec[i] += sign
	}
	toks := Tokenize(cfg, text)
	if cfg.WordMin > 0 {
		for n := int(cfg.WordMin); n <= int(cfg.WordMax); n++ {
			for i := 0; i+n <= len(toks); i++ {
				add("w" + string(rune('0'+n)) + ":" + strings.Join(toks[i:i+n], " "))
			}
		}
	}
	if cfg.CharMin > 0 {
		// Over the normalized text, not the tokens: the separators between
		// tokens are part of what a character n-gram is for ("&& rm", "/../").
		t := text
		if cfg.Lowercase {
			t = strings.ToLower(t)
		}
		r := []rune(t)
		for n := int(cfg.CharMin); n <= int(cfg.CharMax); n++ {
			for i := 0; i+n <= len(r); i++ {
				add("c" + string(rune('0'+n)) + ":" + string(r[i:i+n]))
			}
		}
	}
	return l2(vec)
}

// l2 scales the vector to unit length, which is the difference between a score
// that means something and one that depends on how long the command was.
//
// Without it a repeated feature accumulates: "rm -rf /opt/<380 chars>" produced
// a vector with Sum(v^2) = 426490 against 104 for the same command with a short
// path, and the dot product grew with it — logit 1523 against 14.5. Every
// scale-dependent constant then stops meaning what it says: the 0.50 threshold
// sits at a different confidence for a long call than a short one, and the cap
// on local corrections (maxShift) went from 619% of the logit to 5.9%, which
// made a long call impossible to correct at all.
//
// Normalized, Sum(v^2) is 1 by construction: logits are comparable across
// inputs, one correction moves the logit by exactly learnRate, and the cap is
// a real bound again.
func l2(vec map[uint32]float32) map[uint32]float32 {
	var sq float64
	for _, v := range vec {
		sq += float64(v) * float64(v)
	}
	if sq == 0 {
		return vec
	}
	inv := float32(1 / math.Sqrt(sq))
	for i, v := range vec {
		vec[i] = v * inv
	}
	return vec
}
