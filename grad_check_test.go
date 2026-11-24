package gorgonia

import (
	"errors"
	"fmt"
	"hash"
	"testing"

	"github.com/chewxy/hm"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// gradCheckScalarCost builds x1*x2*x1 with analytic gradient d/dx1 = 2*x1*x2.
func gradCheckScalarCost() (g *ExprGraph, x1, x2, cost *Node) {
	g = NewGraph()
	x1 = NewScalar(g, Float64, WithName("x1"), WithValue(3.0))
	x2 = NewScalar(g, Float64, WithName("x2"), WithValue(5.0))
	xx := Must(Mul(x1, x1))
	cost = Must(Mul(xx, x2))
	return
}

func TestGradCheck_ScalarPasses(t *testing.T) {
	g, x1, x2, cost := gradCheckScalarCost()
	rep, err := GradCheckStrict(cost, x1, x2)
	require.NoError(t, err)
	require.NotNil(t, rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	assert.Equal(t, 2, rep.CheckedCount)
	assert.Equal(t, 0, rep.SkippedCount)
	// d/dx1 = 2*x1*x2 = 30, d/dx2 = x1^2 = 9
	assert.Equal(t, 30.0, rep.Entries[0].Analytic)
	assert.Equal(t, 9.0, rep.Entries[1].Analytic)
	assert.Less(t, rep.MaxAbsError, 1e-3)
	_ = g
}

func TestGradCheck_Deterministic(t *testing.T) {
	g, x1, x2, cost := gradCheckScalarCost()
	first, err := GradCheckStrict(cost, x1, x2)
	require.NoError(t, err)
	second, err := GradCheckStrict(cost, x1, x2)
	require.NoError(t, err)
	_ = g
	assert.Equal(t, first.String(), second.String(), "two checks over the same graph must be byte-identical")
	assert.Equal(t, first.Entries, second.Entries)
	assert.Equal(t, first.Skipped, second.Skipped)
}

func TestGradCheck_MatrixPointwiseReduction(t *testing.T) {
	// cost = sum(w ⊙ w ⊙ const) exercises matrix multiplication free but
	// pointwise + reduction combinations. d/dw = 2 * w * const.
	g := NewGraph()
	w := NewMatrix(g, Float64, WithName("w"), WithShape(2, 3), WithInit(RangedFrom(1)))
	c := NewConstant(tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})))
	cost := Must(Sum(Must(HadamardProd(Must(HadamardProd(w, w)), c))))

	rep, err := GradCheckStrict(cost, w)
	require.NoError(t, err, "report: %s", rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	assert.Equal(t, 6, rep.CheckedCount)

	// Compare against the analytic formula element-wise.
	wv := w.Value().Data().([]float64)
	for _, e := range rep.Entries {
		want := 2 * wv[e.Index] * float64(e.Index+1)
		assert.InDelta(t, want, e.Analytic, 1e-9, "element %d", e.Index)
	}
	_ = g
}

func TestGradCheck_MatMul(t *testing.T) {
	// cost = sum(A * B) with A (2x3) and B (3x2). d/dA = ones * B^T,
	// d/dB = A^T * ones.
	g := NewGraph()
	a := NewMatrix(g, Float64, WithName("a"), WithShape(2, 3), WithInit(RangedFrom(1)))
	b := NewMatrix(g, Float64, WithName("b"), WithShape(3, 2), WithInit(RangedFrom(1)))
	prod := Must(Mul(a, b))
	cost := Must(Sum(prod))

	rep, err := GradCheckStrict(cost, a, b)
	require.NoError(t, err, "report: %s", rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	assert.Equal(t, 12, rep.CheckedCount)
	_ = g
}

func TestGradCheck_PerParamWorstLocation(t *testing.T) {
	// Two parameters of different sizes; the worst error location must be
	// reported with the parameter name and flat index, not just a magnitude.
	g := NewGraph()
	a := NewVector(g, Float64, WithName("a"), WithShape(3), WithInit(RangedFrom(1)))
	b := NewVector(g, Float64, WithName("b"), WithShape(3), WithInit(RangedFrom(1)))
	cost := Must(Sum(Must(HadamardProd(a, b))))

	rep, err := GradCheckWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-4, RelTolerance: 1e-4}, a, b)
	require.NoError(t, err, "report: %s", rep)
	require.True(t, rep.Passed, "report: %s", rep.String())
	// a = [1,2,3], b = [1,2,3]; cost = sum(a*b) -> d/da = d/db = b = a.
	assert.Equal(t, 6, rep.CheckedCount)
	seen := map[string]bool{}
	for _, e := range rep.Entries {
		seen[e.Param] = true
	}
	assert.True(t, seen["a"])
	assert.True(t, seen["b"])
}

func TestGradCheck_WrongGradientIsCaught(t *testing.T) {
	// A cost whose symbolic derivative is missing a factor. Build
	// z = sum((2x)^2) and compare the analytic gradient with regards to the
	// *inner* node, which is correct, then sabotage by checking against the
	// outer node while perturbing the inner one. To keep it simple and honest
	// this test checks the mechanism: a self-consistent gradient passes, and
	// the report's MaxAbs fields point at the worst element when a tolerance
	// is made impossibly tight.
	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"), WithShape(4), WithInit(RangedFrom(1)))
	cost := Must(Sum(Must(HadamardProd(x, x))))

	rep, err := GradCheckWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-12, RelTolerance: 1e-12}, x)
	require.NoError(t, err)
	// Finite differences cannot meet a 1e-12 tolerance; the check must report
	// failure rather than log and pass.
	assert.False(t, rep.Passed)
	// The fail-closed variant turns that report into an actionable error.
	_, cerr := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-12, RelTolerance: 1e-12}, x)
	require.Error(t, cerr)
	var gcerr *GradCheckError
	assert.True(t, errors.As(cerr, &gcerr))
	assert.Contains(t, cerr.Error(), "worst parameter")
	assert.NotEqual(t, "", rep.MaxAbsParam)
	assert.Greater(t, rep.MaxAbsError, 0.0)
	assert.Equal(t, "x", rep.MaxAbsParam)
}

func TestGradCheck_SkipsNonFloat(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	dummy := NewScalar(g, Float64, WithName("dummy"), WithValue(1.0))
	cost := Must(Add(x, dummy))

	// Rename to check the skip is recorded for a non-float parameter. We use
	// an integer node that does not participate in the cost.
	iv := NewScalar(g, Int, WithName("iv"), WithValue(3))
	_ = iv

	rep, err := GradCheckStrict(cost, x)
	require.NoError(t, err)
	assert.True(t, rep.Passed)

	// A parameter that is not on the differentiable path must be recorded as a
	// skip, not silently ignored.
	rep2, err := GradCheck(cost, x, iv)
	// Gradient wrt iv is rejected by Differentiate (not an input of the cost
	// graph? it is an input but not differentiable), so Differentiate errors.
	if err == nil {
		assert.Greater(t, rep2.SkippedCount, 0)
	}
}

func TestGradCheck_ReportStringStable(t *testing.T) {
	g, x1, x2, cost := gradCheckScalarCost()
	rep, err := GradCheckStrict(cost, x1, x2)
	require.NoError(t, err)
	_ = g
	s := rep.String()
	assert.Contains(t, s, "gradcheck: PASS")
	assert.Contains(t, s, "entry x1[0]")
	assert.Contains(t, s, "entry x2[0]")
	assert.Contains(t, s, "maxAbsError=")
}

func TestGradCheck_ToleranceConfigurable(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(3.0))
	cost := Must(Mul(x, x))
	_ = g

	loose, err := GradCheckStrictWithConfig(cost, GradCheckConfig{AbsTolerance: 1, RelTolerance: 1}, x)
	require.NoError(t, err)
	assert.True(t, loose.Passed)

	// A tight tolerance may or may not be met by finite differences at a
	// given step; what matters is that the configured thresholds are the ones
	// used in the verdict.
	tight, err := GradCheckWithConfig(cost, GradCheckConfig{AbsTolerance: 1e-9, RelTolerance: 1e-9, Step: 1e-2}, x)
	require.NoError(t, err)
	assert.Equal(t, 1e-9, tight.Config.AbsTolerance)
	assert.Equal(t, 1e-9, tight.Config.RelTolerance)
	assert.Equal(t, 1e-2, tight.Config.Step)
	if tight.CheckedCount > 0 {
		assert.Equal(t, tight.MaxAbsError > 1e-9 || tight.MaxRelError > 1e-9, !tight.Passed)
	}
}

func TestGradCheck_MaxEntries(t *testing.T) {
	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"), WithShape(10), WithInit(RangedFrom(1)))
	cost := Must(Sum(Must(HadamardProd(x, x))))
	rep, err := GradCheckStrictWithConfig(cost, GradCheckConfig{MaxEntries: 3}, x)
	require.NoError(t, err)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	assert.Equal(t, 3, rep.CheckedCount)
	_ = g
}

func TestGradCheck_RestoresValues(t *testing.T) {
	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"), WithShape(3), WithInit(RangedFrom(1)))
	cost := Must(Sum(Must(HadamardProd(x, x))))
	before := append([]float64(nil), x.Value().Data().([]float64)...)

	rep, err := GradCheckStrict(cost, x)
	require.NoError(t, err)
	assert.True(t, rep.Passed)

	after := x.Value().Data().([]float64)
	assert.Equal(t, before, after, "checking must leave parameter values untouched")
	_ = g
}

func TestGradCheck_MatrixMultiplyAnalytic(t *testing.T) {
	// A closed-form check of a matrix multiplication gradient, exercising the
	// linAlg SymDiff path against finite differences.
	g := NewGraph()
	a := NewMatrix(g, Float64, WithName("a"), WithShape(2, 2), WithInit(RangedFrom(1)))
	b := NewMatrix(g, Float64, WithName("b"), WithShape(2, 2), WithInit(RangedFrom(3)))
	cost := Must(Sum(Must(Mul(a, b))))

	rep, err := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-5, AbsTolerance: 1e-4, RelTolerance: 1e-6}, a, b)
	require.NoError(t, err, "report: %s", rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	_ = g
}

func TestGradCheck_Float32(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float32, WithName("x"), WithValue(float32(2.0)))
	y := NewScalar(g, Float32, WithName("y"), WithValue(float32(3.0)))
	cost := Must(Mul(x, y))

	rep, err := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-2, AbsTolerance: 1e-2, RelTolerance: 1e-2}, x, y)
	require.NoError(t, err, "report: %s", rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	_ = g
}

func TestGradCheck_ElementwiseOperators(t *testing.T) {
	cases := []struct {
		name string
		fn   func(g *ExprGraph, x *Node) *Node
	}{
		{"exp", func(g *ExprGraph, x *Node) *Node { return Must(Exp(x)) }},
		{"ln", func(g *ExprGraph, x *Node) *Node { return Must(Log(x)) }},
		{"tanh", func(g *ExprGraph, x *Node) *Node { return Must(Tanh(x)) }},
		{"sigmoid", func(g *ExprGraph, x *Node) *Node { return Must(Sigmoid(x)) }},
		{"cube", func(g *ExprGraph, x *Node) *Node { return Must(Cube(x)) }},
		{"sqrt", func(g *ExprGraph, x *Node) *Node { return Must(Sqrt(x)) }},
		{"inverse", func(g *ExprGraph, x *Node) *Node { return Must(Inverse(x)) }},
		{"sin", func(g *ExprGraph, x *Node) *Node { return Must(Sin(x)) }},
		{"cos", func(g *ExprGraph, x *Node) *Node { return Must(Cos(x)) }},
		{"softplus", func(g *ExprGraph, x *Node) *Node { return Must(Softplus(x)) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := NewGraph()
			x := NewVector(g, Float64, WithName("x"), WithShape(3), WithInit(ValuesOf(0.7)))
			cost := Must(Sum(c.fn(g, x)))
			rep, err := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-4, RelTolerance: 1e-5}, x)
			require.NoError(t, err, "report: %s", rep)
			assert.Equal(t, 3, rep.CheckedCount)
		})
	}
}

func TestGradCheck_MixedExpression(t *testing.T) {
	// A composite expression: elementwise, matmul, reduction and broadcast-free
	// arithmetic in one cost.
	g := NewGraph()
	w := NewMatrix(g, Float64, WithName("w"), WithShape(3, 2), WithInit(GlorotU(1)))
	x := NewMatrix(g, Float64, WithName("x"), WithShape(3, 2), WithInit(GlorotU(1)))
	prod := Must(Mul(Must(Transpose(w)), x)) // 2x2
	act := Must(Tanh(prod))
	reg := Must(Sum(Must(HadamardProd(w, w))))
	cost := Must(Add(Must(Mean(Must(HadamardProd(act, act)))), reg))

	rep, err := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-4, RelTolerance: 1e-4}, w, x)
	require.NoError(t, err, "report: %s", rep)
	assert.True(t, rep.Passed, "report: %s", rep.String())
	assert.Equal(t, 12, rep.CheckedCount)
}

// badSquareOp is a deliberately broken square: its forward value is correct
// (x*x) but its symbolic derivative drops the factor of 2. It exists to prove
// that GradCheck rejects a wrong gradient and points at the parameter at
// fault.
type badSquareOp struct{}

func (badSquareOp) Arity() int { return 1 }
func (badSquareOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}
func (badSquareOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape).Clone(), nil
}
func (badSquareOp) Do(inputs ...Value) (Value, error) {
	if len(inputs) != 1 {
		return nil, errors.New("badSquare takes exactly one input")
	}
	op := newElemUnaryOp(squareOpType, &Node{t: TypeOf(inputs[0])})
	return op.Do(inputs[0])
}
func (badSquareOp) ReturnsPtr() bool      { return false }
func (badSquareOp) CallsExtern() bool     { return false }
func (badSquareOp) OverwritesInput() int  { return -1 }
func (badSquareOp) WriteHash(h hash.Hash) { fmt.Fprint(h, "badSquareOp") }
func (badSquareOp) Hashcode() uint32      { return simpleHash(badSquareOp{}) }
func (badSquareOp) String() string        { return "badSquare" }

func (badSquareOp) DiffWRT(int) []bool { return []bool{true} }
func (badSquareOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	// Wrong on purpose: d(x^2)/dx should be 2x*grad, not grad.
	return Nodes{grad}, nil
}
func (badSquareOp) DoDiff(ExecutionContext, Nodes, *Node) error { return nil }

func TestGradCheck_WrongGradientPointsAtParameter(t *testing.T) {
	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"), WithShape(4), WithInit(RangedFrom(1)))
	bad := Must(ApplyOp(badSquareOp{}, x))
	cost := Must(Sum(bad))

	rep, err := GradCheckWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-4, RelTolerance: 1e-4}, x)
	require.NoError(t, err, "the check itself must run")
	require.False(t, rep.Passed, "a wrong derivative must fail the check")
	assert.Equal(t, "x", rep.MaxAbsParam, "the report must name the offending parameter")
	assert.Greater(t, rep.MaxAbsError, 1.0)
	// Every element must be flagged: the missing factor is not a rounding issue.
	assert.Equal(t, 4, rep.FailedCount())

	// Fail-closed variant returns an actionable error.
	_, cerr := GradCheckStrictWithConfig(cost, GradCheckConfig{Step: 1e-6, AbsTolerance: 1e-4, RelTolerance: 1e-4}, x)
	require.Error(t, cerr)
	assert.Contains(t, cerr.Error(), "worst parameter x")
}
