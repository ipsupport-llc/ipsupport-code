package tool

import (
	"context"
	"strings"
	"testing"
)

func calc(expr string) Result {
	return NewCalc().Call(context.Background(), "calculate", map[string]any{"expression": expr})
}

func TestCalcInteger(t *testing.T) {
	r := calc("3847*29")
	if r.IsError || r.Content != "111563" {
		t.Errorf("3847*29 = %+v, want 111563", r)
	}
}

func TestCalcFuncs(t *testing.T) {
	r := calc("sqrt(2)+1")
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.HasPrefix(r.Content, "2.414") {
		t.Errorf("sqrt(2)+1 = %q, want ~2.414...", r.Content)
	}
}

func TestCalcUnsafeRejected(t *testing.T) {
	if r := calc("os.Exit(1)"); !r.IsError {
		t.Errorf("os.Exit(1) should be rejected, got %+v", r)
	}
	if r := calc("foo(2)"); !r.IsError {
		t.Errorf("unknown function should be rejected, got %+v", r)
	}
}

func TestCalcMissingParam(t *testing.T) {
	r := NewCalc().Call(context.Background(), "calculate", map[string]any{})
	if !r.IsError || !strings.Contains(r.Content, "expression") {
		t.Errorf("missing expression = %+v, want error naming 'expression'", r)
	}
}
