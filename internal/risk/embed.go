package risk

import (
	_ "embed"
	"log/slog"
	"os"
	"sync"
)

// modelBin is the trained weights, built by scripts/train_risk.py. Embedded
// rather than loaded from disk so the agent stays one file with nothing to
// install — the whole reason the model is a few hundred KB of float32 and not
// a runtime.
//
//go:embed model.bin
var modelBin []byte

// EnvModelPath overrides the embedded weights with a file on disk. This is what
// makes a model replaceable without a rebuild: point it at a newly trained
// file, run, compare the logs. The file carries its own feature config, so a
// model with a different feature space or label set loads unchanged.
const EnvModelPath = "IPS_RISK_MODEL"

var (
	once      sync.Once
	def       *Model
	defErr    error
	defSource string
)

// Default returns the process-wide model: the file named by IPS_RISK_MODEL if
// set, else the embedded weights. Parsed once; the result is immutable and safe
// to share.
func Default() (*Model, error) {
	once.Do(func() {
		if p := os.Getenv(EnvModelPath); p != "" {
			data, err := os.ReadFile(p)
			if err != nil {
				defErr = err
				return
			}
			def, defErr, defSource = nil, nil, p
			def, defErr = Load(data)
			return
		}
		defSource = "embedded"
		def, defErr = Load(modelBin)
	})
	return def, defErr
}

// DefaultOrNil is Default for callers that treat an unusable model as "no risk
// signal" rather than an error — which is every caller on the hot path, since a
// scoring problem must never stop a tool call the policy already allowed.
func DefaultOrNil() *Model {
	m, err := Default()
	if err != nil {
		slog.Warn("risk model unusable — scoring disabled for this run", "source", defSource, "err", err)
		return nil
	}
	return m
}
