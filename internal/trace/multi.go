package trace

// Multi fans one event out to several tracers (e.g. the JSONL file plus the live
// TUI). Nil tracers are dropped, so Multi(file, nil) is safe.
func Multi(tracers ...Tracer) Tracer {
	var live []Tracer
	for _, t := range tracers {
		if t != nil {
			live = append(live, t)
		}
	}
	if len(live) == 1 {
		return live[0]
	}
	return multi(live)
}

type multi []Tracer

func (m multi) Emit(kind string, fields map[string]any) {
	for _, t := range m {
		t.Emit(kind, fields)
	}
}
