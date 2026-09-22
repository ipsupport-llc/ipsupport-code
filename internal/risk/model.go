package risk

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// magic identifies the weights file and pins its layout version. A model is
// meant to be replaced by dropping in a new file, so the file has to carry
// everything inference needs to interpret it — dimensions, hash seed, n-gram
// ranges, label names — and say so loudly when it doesn't.
var magic = [8]byte{'I', 'P', 'S', 'R', 'I', 'S', 'K', 0x01}

// FormatVersion is the layout below. A reader refuses anything else rather than
// guessing: silently mis-reading float32 weights produces a model that scores,
// and scores wrongly, which is worse than not loading.
const FormatVersion uint16 = 1

// Model is a multi-label linear classifier: one weight row and one bias per
// label, scored as sigmoid(Wx + b). Immutable once loaded, so a single instance
// is safe to share across goroutines.
type Model struct {
	Cfg    FeatureConfig
	Labels []string
	Bias   []float32
	// Informational marks a label that describes the call without being a reason
	// for concern, and so does not contribute to the headline risk number.
	// "network" is the case this exists for: reaching the internet is a property,
	// not a danger — the risky half of it is external_side_effect. Counted as
	// risk, every fetch of a documentation page scored 1.00, which is how a
	// signal becomes noise. Declared per label IN THE FILE so a replacement model
	// decides for its own label set.
	Informational []bool
	// W is row-major, len(Labels) rows of Cfg.Dim. One flat slice rather than a
	// [][]float32: it is the difference between one allocation and one per label,
	// and the row stride is the only thing a reader needs to know.
	W []float32
}

// Load reads a model from the binary format. Everything it needs comes from the
// file — that is what lets a retrained model with a different feature space or
// a different label set replace this one without touching any Go code.
func Load(data []byte) (*Model, error) {
	r := &reader{b: data}
	var m [8]byte
	r.read(m[:])
	if r.err == nil && m != magic {
		return nil, fmt.Errorf("risk: not a model file (bad magic %q)", m[:])
	}
	if v := r.u16(); r.err == nil && v != FormatVersion {
		return nil, fmt.Errorf("risk: model format v%d, this build reads v%d", v, FormatVersion)
	}
	var mo Model
	mo.Cfg.Dim = r.u32()
	mo.Cfg.Seed = r.u32()
	mo.Cfg.WordMin, mo.Cfg.WordMax = r.u8(), r.u8()
	mo.Cfg.CharMin, mo.Cfg.CharMax = r.u8(), r.u8()
	mo.Cfg.Lowercase = r.u8()&1 == 1
	n := int(r.u16())
	if r.err != nil {
		return nil, r.err
	}
	if mo.Cfg.Dim == 0 || n == 0 {
		return nil, errors.New("risk: model declares no features or no labels")
	}
	// The weight matrix is the whole file; a bad header would otherwise make this
	// try to allocate gigabytes before failing.
	if want := int(mo.Cfg.Dim) * n; want > (1<<28) || want <= 0 {
		return nil, fmt.Errorf("risk: model too large (%d labels x %d features)", n, mo.Cfg.Dim)
	}
	mo.Labels = make([]string, n)
	mo.Informational = make([]bool, n)
	for i := range mo.Labels {
		mo.Labels[i] = r.str()
		mo.Informational[i] = r.u8()&1 == 1
	}
	mo.Bias = make([]float32, n)
	for i := range mo.Bias {
		mo.Bias[i] = r.f32()
	}
	mo.W = make([]float32, n*int(mo.Cfg.Dim))
	for i := range mo.W {
		mo.W[i] = r.f32()
	}
	if r.err != nil {
		return nil, r.err
	}
	if r.i != len(data) {
		return nil, fmt.Errorf("risk: %d trailing bytes after the weights", len(data)-r.i)
	}
	return &mo, nil
}

// Write emits the format Load reads. Kept beside the reader so the two cannot
// drift, and used by the tests to build models without shelling out to Python.
func (m *Model) Write(w io.Writer) error {
	var buf []byte
	buf = append(buf, magic[:]...)
	buf = binary.LittleEndian.AppendUint16(buf, FormatVersion)
	buf = binary.LittleEndian.AppendUint32(buf, m.Cfg.Dim)
	buf = binary.LittleEndian.AppendUint32(buf, m.Cfg.Seed)
	buf = append(buf, m.Cfg.WordMin, m.Cfg.WordMax, m.Cfg.CharMin, m.Cfg.CharMax)
	var flags uint8
	if m.Cfg.Lowercase {
		flags |= 1
	}
	buf = append(buf, flags)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(m.Labels)))
	for i, l := range m.Labels {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(l)))
		buf = append(buf, l...)
		var f uint8
		if i < len(m.Informational) && m.Informational[i] {
			f = 1
		}
		buf = append(buf, f)
	}
	for _, b := range m.Bias {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(b))
	}
	for _, v := range m.W {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
	}
	_, err := w.Write(buf)
	return err
}

// Score returns one probability per label, in Labels order. The dot product
// runs over the features the text actually has — a few hundred — not over the
// whole feature space, which is what keeps this in the microseconds.
func (m *Model) Score(text string) []float32 {
	vec := Featurize(m.Cfg, text)
	out := make([]float32, len(m.Labels))
	dim := int(m.Cfg.Dim)
	for li := range m.Labels {
		row := m.W[li*dim : (li+1)*dim]
		z := m.Bias[li]
		for idx, val := range vec {
			z += row[idx] * val
		}
		out[li] = sigmoid(z)
	}
	return out
}

// ScoreOf is Score for one label, by name; ok is false for a label this model
// does not have — which is normal, since a replacement model may carry a
// different set.
func (m *Model) ScoreOf(scores []float32, label string) (float32, bool) {
	for i, l := range m.Labels {
		if l == label && i < len(scores) {
			return scores[i], true
		}
	}
	return 0, false
}

func sigmoid(z float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(z))))
}

// reader is a little-endian cursor that latches its first error, so the parse
// above reads as a straight list of fields instead of an error check per line.
type reader struct {
	b   []byte
	i   int
	err error
}

func (r *reader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if r.i+n > len(r.b) {
		r.err = io.ErrUnexpectedEOF
		return nil
	}
	s := r.b[r.i : r.i+n]
	r.i += n
	return s
}

func (r *reader) read(dst []byte) {
	if s := r.take(len(dst)); s != nil {
		copy(dst, s)
	}
}

func (r *reader) u8() uint8 {
	s := r.take(1)
	if s == nil {
		return 0
	}
	return s[0]
}

func (r *reader) u16() uint16 {
	s := r.take(2)
	if s == nil {
		return 0
	}
	return binary.LittleEndian.Uint16(s)
}

func (r *reader) u32() uint32 {
	s := r.take(4)
	if s == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(s)
}

func (r *reader) f32() float32 {
	return math.Float32frombits(r.u32())
}

func (r *reader) str() string {
	n := int(r.u16())
	s := r.take(n)
	if s == nil {
		return ""
	}
	return string(s)
}
