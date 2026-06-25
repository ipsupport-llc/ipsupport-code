package tool

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"strconv"
	"strings"
)

// NewCalc returns the calc tool: safe arithmetic evaluation, so small models
// don't have to do unreliable mental math.
func NewCalc() Tool {
	return NewDomain(DomainSpec{
		Name:    "calc",
		Summary: "Evaluate arithmetic safely (AST, not eval) — use for ANY math.",
		Details: "Operators + - * / % and parens; functions sqrt cbrt pow abs floor ceil round log log2 log10 exp sin cos tan hypot min max; constants pi e tau.\ne.g. {\"action\":\"calculate\",\"params\":{\"expression\":\"sqrt(2)+pi\"}}",
		NotHere: "NOT here — files → file; shell → run.",
		Actions: []Action{{
			Name:   "calculate",
			Params: []Param{Req("expression", "str")},
			Run: func(_ context.Context, a Args) Result {
				v, err := evalArith(a.Str("expression"))
				if err != nil {
					return Err("calc error: " + err.Error())
				}
				return Ok(formatNum(v))
			},
		}},
	})
}

func formatNum(f float64) string {
	if f == math.Trunc(f) && math.Abs(f) < 1e15 {
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

func evalArith(expr string) (float64, error) {
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("parse: %w", err)
	}
	return evalNode(node)
}

var calcConsts = map[string]float64{"pi": math.Pi, "e": math.E, "tau": 2 * math.Pi}

func evalNode(n ast.Expr) (float64, error) {
	switch e := n.(type) {
	case *ast.ParenExpr:
		return evalNode(e.X)
	case *ast.BasicLit:
		if e.Kind == token.INT || e.Kind == token.FLOAT {
			return strconv.ParseFloat(e.Value, 64)
		}
		return 0, fmt.Errorf("unsupported literal %q", e.Value)
	case *ast.Ident:
		if v, ok := calcConsts[strings.ToLower(e.Name)]; ok {
			return v, nil
		}
		return 0, fmt.Errorf("unknown identifier %q", e.Name)
	case *ast.UnaryExpr:
		x, err := evalNode(e.X)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.SUB:
			return -x, nil
		case token.ADD:
			return x, nil
		}
		return 0, fmt.Errorf("unsupported unary operator %s", e.Op)
	case *ast.BinaryExpr:
		x, err := evalNode(e.X)
		if err != nil {
			return 0, err
		}
		y, err := evalNode(e.Y)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return x + y, nil
		case token.SUB:
			return x - y, nil
		case token.MUL:
			return x * y, nil
		case token.QUO:
			if y == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return x / y, nil
		case token.REM:
			return math.Mod(x, y), nil
		}
		return 0, fmt.Errorf("unsupported operator %s", e.Op)
	case *ast.CallExpr:
		id, ok := e.Fun.(*ast.Ident)
		if !ok {
			return 0, fmt.Errorf("unsupported function call")
		}
		fn, ok := calcFuncs[strings.ToLower(id.Name)]
		if !ok {
			return 0, fmt.Errorf("unknown function %q", id.Name)
		}
		args := make([]float64, len(e.Args))
		for i, a := range e.Args {
			v, err := evalNode(a)
			if err != nil {
				return 0, err
			}
			args[i] = v
		}
		return fn(args)
	}
	return 0, fmt.Errorf("unsupported expression")
}

var calcFuncs = map[string]func([]float64) (float64, error){
	"sqrt":  un(math.Sqrt),
	"cbrt":  un(math.Cbrt),
	"abs":   un(math.Abs),
	"floor": un(math.Floor),
	"ceil":  un(math.Ceil),
	"round": un(math.Round),
	"log":   un(math.Log),
	"log2":  un(math.Log2),
	"log10": un(math.Log10),
	"exp":   un(math.Exp),
	"sin":   un(math.Sin),
	"cos":   un(math.Cos),
	"tan":   un(math.Tan),
	"pow":   bin(math.Pow),
	"hypot": bin(math.Hypot),
	"min":   bin(math.Min),
	"max":   bin(math.Max),
}

func un(f func(float64) float64) func([]float64) (float64, error) {
	return func(a []float64) (float64, error) {
		if len(a) != 1 {
			return 0, fmt.Errorf("expected 1 argument, got %d", len(a))
		}
		return f(a[0]), nil
	}
}

func bin(f func(float64, float64) float64) func([]float64) (float64, error) {
	return func(a []float64) (float64, error) {
		if len(a) != 2 {
			return 0, fmt.Errorf("expected 2 arguments, got %d", len(a))
		}
		return f(a[0], a[1]), nil
	}
}
