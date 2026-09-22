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
	return vec
}
