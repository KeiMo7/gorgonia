package gorgonia

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// runTopKOnDense is a small convenience used across the op-level tests. The
// values op takes (x, indices), so the helper runs the indices op first.
func runTopKOnDense(t *testing.T, op Op, in tensor.Tensor) Value {
	v, _, _, err := anyToValue(in)
	require.NoError(t, err)
	if vop, ok := op.(*topKValuesOp); ok {
		iop, err := newTopKIndicesOp(vop.axis, vop.k, vop.direction, vop.d)
		require.NoError(t, err)
		idx, err := iop.Do(v)
		require.NoError(t, err)
		out, err := vop.Do(v, idx)
		require.NoError(t, err)
		return out
	}
	out, err := op.Do(v)
	require.NoError(t, err)
	return out
}

func TestTopKOpDo_Basic2D(t *testing.T) {
	assert := assert.New(t)
	// 2x4 matrix, select top-3 along the last axis.
	in := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{
		1, 4, 3, 4, // tie at 4 (indices 1 and 3): ascending index must win
		9, 2, 8, 7,
	}))

	valuesOp, err := newTopKValuesOp(-1, 3, topKMaximum, 2)
	require.NoError(t, err)
	indicesOp, err := newTopKIndicesOp(-1, 3, topKMaximum, 2)
	require.NoError(t, err)

	idxOut := runTopKOnDense(t, indicesOp, in).(*tensor.Dense)
	valOut := runTopKOnDense(t, valuesOp, in).(*tensor.Dense)

	assert.Equal(tensor.Shape{2, 3}, idxOut.Shape())
	assert.Equal(tensor.Shape{2, 3}, valOut.Shape())
	assert.Equal(tensor.Int, idxOut.Dtype(), "indices must always be tensor.Int")
	assert.Equal(tensor.Float64, valOut.Dtype())

	assert.Equal([]int{1, 3, 2, 0, 2, 3}, idxOut.Ints(), "ties broken by ascending original index")
	assert.Equal([]float64{4, 4, 3, 9, 8, 7}, valOut.Float64s())
}

func TestTopKOpDo_Batched3D(t *testing.T) {
	assert := assert.New(t)
	// shape (2, 2, 3), select top-2 along axis 1.
	in := tensor.New(tensor.WithShape(2, 2, 3), tensor.WithBacking([]float64{
		1, 2, 3,
		4, 5, 6,

		7, 8, 9,
		1, 1, 1,
	}))

	indicesOp, err := newTopKIndicesOp(1, 2, topKMaximum, 3)
	require.NoError(t, err)
	valuesOp, err := newTopKValuesOp(1, 2, topKMaximum, 3)
	require.NoError(t, err)

	idxOut := runTopKOnDense(t, indicesOp, in).(*tensor.Dense)
	valOut := runTopKOnDense(t, valuesOp, in).(*tensor.Dense)

	assert.Equal(tensor.Shape{2, 2, 3}, idxOut.Shape())
	// batch 0: row 1 beats row 0 everywhere; batch 1: row 0 beats row 1.
	assert.Equal([]int{1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1}, idxOut.Ints())
	assert.Equal([]float64{4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 1, 1}, valOut.Float64s())
}

func TestTopKOpDo_NegativeAxes(t *testing.T) {
	assert := assert.New(t)
	in := tensor.New(tensor.WithShape(2, 3, 4),
		tensor.WithBacking(tensor.Range(tensor.Float64, 0, 24)))

	for _, axis := range []int{-1, -2, -3} {
		canon, err := canonicalizeAxis(axis, 3)
		require.NoError(t, err)

		opNeg, err := newTopKIndicesOp(axis, 2, topKMaximum, 3)
		require.NoError(t, err)
		opPos, err := newTopKIndicesOp(canon, 2, topKMaximum, 3)
		require.NoError(t, err)

		neg := runTopKOnDense(t, opNeg, in).(*tensor.Dense)
		pos := runTopKOnDense(t, opPos, in).(*tensor.Dense)
		assert.Equal(pos.Ints(), neg.Ints(), "axis %d must equal axis %d", axis, canon)
	}
}

func TestTopKOpDo_KClampedAndZero(t *testing.T) {
	assert := assert.New(t)
	in := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float64{3, 1, 2}))

	// k larger than the axis length: clamped, no error, full stable ordering.
	opClamp, err := newTopKIndicesOp(-1, 10, topKMaximum, 1)
	require.NoError(t, err)
	out := runTopKOnDense(t, opClamp, in).(*tensor.Dense)
	assert.Equal(tensor.Shape{3}, out.Shape())
	assert.Equal([]int{0, 2, 1}, out.Ints())

	// k == 0: legal, empty output tensor with the correct rank.
	opZero, err := newTopKIndicesOp(0, 0, topKMaximum, 1)
	require.NoError(t, err)
	zeroOut := runTopKOnDense(t, opZero, in).(*tensor.Dense)
	assert.Equal(tensor.Shape{0}, zeroOut.Shape())
	assert.Equal(0, zeroOut.Size())

	in2d := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 6)))
	opZero2, err := newTopKValuesOp(1, 0, topKMaximum, 2)
	require.NoError(t, err)
	zero2d := runTopKOnDense(t, opZero2, in2d).(*tensor.Dense)
	assert.Equal(tensor.Shape{2, 0}, zero2d.Shape())
	assert.Equal(0, zero2d.Size())
}

func TestTopKOpDo_EmptyInput(t *testing.T) {
	assert := assert.New(t)
	// An empty axis: nothing to select regardless of k.
	in := tensor.New(tensor.WithShape(0, 3), tensor.WithBacking([]float64{}))
	op, err := newTopKIndicesOp(0, 2, topKMaximum, 2)
	require.NoError(t, err)
	out := runTopKOnDense(t, op, in).(*tensor.Dense)
	assert.Equal(tensor.Shape{0, 3}, out.Shape())

	opV, err := newTopKValuesOp(1, 2, topKMaximum, 2)
	require.NoError(t, err)
	vout := runTopKOnDense(t, opV, in).(*tensor.Dense)
	assert.Equal(tensor.Shape{0, 2}, vout.Shape())
}

func TestTopKOpDo_StabilityAcrossRuns(t *testing.T) {
	assert := assert.New(t)
	// all-equal input: the full ordering must come back as 0..n-1 every time.
	in := tensor.New(tensor.WithShape(5), tensor.WithBacking([]float64{7, 7, 7, 7, 7}))
	op, err := newTopKIndicesOp(0, 5, topKMaximum, 1)
	require.NoError(t, err)
	for run := 0; run < 10; run++ {
		out := runTopKOnDense(t, op, in).(*tensor.Dense)
		assert.Equal([]int{0, 1, 2, 3, 4}, out.Ints())
	}
}

func TestTopKOpDo_InfAndNaN(t *testing.T) {
	assert := assert.New(t)
	// Documented rule for maximum selection: every non-NaN value (including
	// +Inf and -Inf) is ordered before NaN.
	in := tensor.New(tensor.WithShape(6), tensor.WithBacking([]float64{
		math.Inf(-1), 2, math.Inf(1), math.NaN(), 5, math.NaN(),
	}))

	opMax, err := newTopKIndicesOp(0, 6, topKMaximum, 1)
	require.NoError(t, err)
	maxOut := runTopKOnDense(t, opMax, in).(*tensor.Dense)
	// +Inf, 5, 2, -Inf, then the NaNs tie-broken by ascending index.
	assert.Equal([]int{2, 4, 1, 0, 3, 5}, maxOut.Ints())

	opMin, err := newTopKIndicesOp(0, 4, topKMinimum, 1)
	require.NoError(t, err)
	minOut := runTopKOnDense(t, opMin, in).(*tensor.Dense)
	// smallest non-NaN values first; the NaNs never make it into k=4.
	assert.Equal([]int{0, 1, 4, 2}, minOut.Ints())
}

func TestTopKOpDo_Dtypes(t *testing.T) {
	assert := assert.New(t)

	// float32
	f32 := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, 5, 3, 5}))
	opF32, err := newTopKIndicesOp(0, 3, topKMaximum, 1)
	require.NoError(t, err)
	f32Out := runTopKOnDense(t, opF32, f32).(*tensor.Dense)
	assert.Equal(tensor.Int, f32Out.Dtype())
	assert.Equal([]int{1, 3, 2}, f32Out.Ints())

	// int input: values stay int, indices are still tensor.Int.
	ints := tensor.New(tensor.WithShape(4), tensor.WithBacking([]int{-2, 9, 9, 0}))
	opII, err := newTopKIndicesOp(0, 3, topKMaximum, 1)
	require.NoError(t, err)
	iOut := runTopKOnDense(t, opII, ints).(*tensor.Dense)
	assert.Equal(tensor.Int, iOut.Dtype())
	assert.Equal([]int{1, 2, 3}, iOut.Ints())

	opIV, err := newTopKValuesOp(0, 3, topKMaximum, 1)
	require.NoError(t, err)
	vOut := runTopKOnDense(t, opIV, ints).(*tensor.Dense)
	assert.Equal(tensor.Int, vOut.Dtype())
	assert.Equal([]int{9, 9, 0}, vOut.Ints())
}

func TestTopKOpDo_DoesNotMutateInput(t *testing.T) {
	assert := assert.New(t)
	backing := []float64{3, 1, 4, 1, 5, 9}
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(backing))
	op, err := newTopKIndicesOp(1, 2, topKMaximum, 2)
	require.NoError(t, err)
	_, err = op.Do(in)
	require.NoError(t, err)
	assert.Equal([]float64{3, 1, 4, 1, 5, 9}, in.Float64s(), "input backing must be untouched")
}

func TestTopKOpDo_InputNotMutatedByGather(t *testing.T) {
	assert := assert.New(t)
	backing := []float64{3, 1, 4, 1, 5, 9}
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(backing))
	idx := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]int{2, 0, 2, 1}))
	op, err := newTopKValuesOp(1, 2, topKMaximum, 2)
	require.NoError(t, err)
	out, err := op.Do(in, idx)
	require.NoError(t, err)
	assert.Equal([]float64{4, 3, 9, 5}, out.Data())
	assert.Equal([]float64{3, 1, 4, 1, 5, 9}, in.Float64s())
}

func TestTopKOpDo_TransposedInput(t *testing.T) {
	assert := assert.New(t)
	// A transposed view is not contiguous; the op must materialize a copy
	// rather than index the flat backing incorrectly, and must not modify the
	// original tensor.
	orig := tensor.New(tensor.WithShape(2, 3),
		tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	tr, err := tensor.T(orig) // shape becomes (3, 2)
	require.NoError(t, err)
	trDense := tr.(*tensor.Dense)
	require.True(t, trDense.IsMaterializable())

	op, err := newTopKIndicesOp(1, 1, topKMaximum, 2)
	require.NoError(t, err)
	out := runTopKOnDense(t, op, tr).(*tensor.Dense)

	// transposed rows are [1,4], [2,5], [3,6]; the max along axis 1 is the
	// second element (index 1) in every row.
	assert.Equal(tensor.Shape{3, 1}, out.Shape())
	assert.Equal([]int{1, 1, 1}, out.Ints())
	assert.Equal([]float64{1, 2, 3, 4, 5, 6}, orig.Float64s(), "original tensor must be untouched")
}

func TestTopKOp_ConstructionErrors(t *testing.T) {
	_, err := newTopKIndicesOp(0, -1, topKMaximum, 2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "k must be non-negative")

	_, err = newTopKIndicesOp(2, 1, topKMaximum, 2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "axis out of range")

	_, err = newTopKIndicesOp(-3, 1, topKMaximum, 2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "axis out of range")

	_, err = newTopKIndicesOp(0, 1, topKMaximum, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rank")

	// rank mismatch surfaces during shape inference.
	op, err := newTopKIndicesOp(0, 1, topKMaximum, 2)
	require.NoError(t, err)
	_, err = op.InferShape(tensor.Shape{3})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rank mismatch")
}

func TestTopKOp_UnsupportedDtype(t *testing.T) {
	in := tensor.New(tensor.WithShape(3), tensor.WithBacking([]int32{1, 2, 3}))
	op, err := newTopKIndicesOp(0, 1, topKMaximum, 1)
	require.NoError(t, err)
	v, _, _, err := anyToValue(in)
	require.NoError(t, err)
	_, err = op.Do(v)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet supported")
}

func TestTopKOp_HashAndEquality(t *testing.T) {
	assert := assert.New(t)

	op1, err := newTopKIndicesOp(1, 2, topKMaximum, 3)
	require.NoError(t, err)
	op2, err := newTopKIndicesOp(1, 2, topKMaximum, 3)
	require.NoError(t, err)
	opOtherAxis, err := newTopKIndicesOp(0, 2, topKMaximum, 3)
	require.NoError(t, err)
	opOtherK, err := newTopKIndicesOp(1, 1, topKMaximum, 3)
	require.NoError(t, err)
	opMin, err := newTopKIndicesOp(1, 2, topKMinimum, 3)
	require.NoError(t, err)
	valOp, err := newTopKValuesOp(1, 2, topKMaximum, 3)
	require.NoError(t, err)

	assert.Equal(op1.Hashcode(), op2.Hashcode(), "identical params must hash equal")
	assert.NotEqual(op1.Hashcode(), opOtherAxis.Hashcode(), "different axis must not merge")
	assert.NotEqual(op1.Hashcode(), opOtherK.Hashcode(), "different k must not merge")
	assert.NotEqual(op1.Hashcode(), opMin.Hashcode(), "different direction must not merge")
	assert.NotEqual(op1.Hashcode(), valOp.Hashcode(), "values and indices ops must not merge")
}

func TestTopKOp_StringAndInterfaces(t *testing.T) {
	assert := assert.New(t)
	op, err := newTopKValuesOp(0, 2, topKMaximum, 2)
	require.NoError(t, err)
	assert.Contains(op.String(), "TopK")
	assert.Equal(-1, op.OverwritesInput(), "selection ops must never overwrite their input")
	assert.False(op.CallsExtern())
	assert.True(op.ReturnsPtr())
}

/*
 * Graph-level tests: node construction, shape/type inference.
 */

func TestTopKGraph_ShapeAndType(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	x := NewTensor(g, tensor.Float64, 3, WithName("x"), WithShape(2, 3, 4))

	values, indices, err := TopK(x, 2, -1)
	require.NoError(t, err)

	assert.Equal(tensor.Shape{2, 3, 2}, values.Shape())
	assert.Equal(tensor.Shape{2, 3, 2}, indices.Shape())
	assert.Equal(tensor.Float64, values.Dtype())
	assert.Equal(tensor.Int, indices.Dtype())
	assert.Equal(3, values.Dims(), "rank-dependent inference must preserve rank")
	assert.Equal(3, indices.Dims())

	// integer input: index dtype remains Int and does not participate in the
	// element type inference.
	g2 := NewGraph()
	xi := NewTensor(g2, tensor.Int, 2, WithName("xi"), WithShape(4, 5))
	_, ii, err := TopK(xi, 3, 1)
	require.NoError(t, err)
	assert.Equal(tensor.Int, ii.Dtype())
	assert.Equal(tensor.Shape{4, 3}, ii.Shape())
}

func TestTopKGraph_BadConstructionFailsAtBuild(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, tensor.Float64, 2, WithShape(3, 4))

	// out-of-range axis must be rejected while building the graph, not at run.
	_, _, err := TopK(x, 1, 5)
	require.Error(t, err)

	_, _, err = TopK(x, -2, 0)
	require.Error(t, err)
}

/*
 * Both execution engines must agree.
 */

func TestTopKEngines_Agree(t *testing.T) {
	backing := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8}
	shape := tensor.Shape{3, 4}

	build := func() (*ExprGraph, *Node, *Node, *Node) {
		g := NewGraph()
		x := NewMatrix(g, tensor.Float64, WithName("x"), WithShape(shape...),
			WithValue(tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))))
		v, i, err := TopK(x, 2, 1)
		require.NoError(t, err)
		return g, x, v, i
	}

	// Tape machine (compiling engine).
	g1, _, v1, i1 := build()
	tape := NewTapeMachine(g1)
	require.NoError(t, tape.RunAll())
	require.NoError(t, tape.Close())

	// Lisp machine (interpreted engine). Forward-only: when a graph also
	// carries backward nodes the lisp machine requires every root to be a
	// scalar, whereas the selection outputs are matrices by design.
	g2, _, v2, i2 := build()
	lisp := NewLispMachine(g2, ExecuteFwdOnly())
	require.NoError(t, lisp.RunAll())
	require.NoError(t, lisp.Close())

	assert.Equal(t, v1.Value().Data(), v2.Value().Data())
	assert.Equal(t, i1.Value().Data(), i2.Value().Data())

	// spot-check the actual numbers.
	assert.Equal(t, []float64{4, 3, 9, 6, 8, 5}, v1.Value().Data())
	assert.Equal(t, []int{2, 0, 1, 3, 3, 0}, i1.Value().Data())
}

/*
 * Differentiation: symbolic (tape) and automatic (lisp) paths, numeric
 * finite-difference check, and accumulation on repeated selection.
 */

func TestTopKGrad_TapeMachine(t *testing.T) {
	assert := assert.New(t)

	g := NewGraph()
	x := NewMatrix(g, tensor.Float64, WithName("x"), WithShape(2, 3),
		WithValue(tensor.New(tensor.WithShape(2, 3),
			tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))))
	v, _, err := TopK(x, 2, 1)
	require.NoError(t, err)
	cost := Must(Mean(v))
	_, err = Grad(cost, x)
	require.NoError(t, err)

	tape := NewTapeMachine(g)
	require.NoError(t, tape.RunAll())
	require.NoError(t, tape.Close())

	grad, err := x.Grad()
	require.NoError(t, err)

	// top-2 per row of 3 means each selected position gets 1/(2*2) = 0.25;
	// unselected positions get zero.
	assert.Equal([]float64{0, 0.25, 0.25, 0, 0.25, 0.25}, grad.Data())
}

// TestTopKGrad_LispMachine exercises the automatic differentiation (ADOp
// DoDiff) path of the lisp/interpreted engine directly. The selection node is
// the root and its gradient is seeded through Backpropagate, which isolates the
// op's own backward contract from upstream reductions.
func TestTopKGrad_LispMachine(t *testing.T) {
	assert := assert.New(t)

	g := NewGraph()
	x := NewMatrix(g, tensor.Float64, WithName("x"), WithShape(2, 3),
		WithValue(tensor.New(tensor.WithShape(2, 3),
			tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))))
	v, _, err := TopK(x, 2, 1)
	require.NoError(t, err)

	seed := tensor.New(tensor.WithShape(2, 2),
		tensor.WithBacking([]float64{1, 1, 1, 1}))
	seedNode := NewMatrix(g, tensor.Float64, WithShape(2, 2), WithValue(seed))
	_, err = Backpropagate(Nodes{v}, Nodes{seedNode}, Nodes{x})
	require.NoError(t, err)

	lisp := NewLispMachine(g, WithManualGradient())
	require.NoError(t, lisp.RunAll())
	require.NoError(t, lisp.Close())

	grad, err := x.Grad()
	require.NoError(t, err)
	// an all-ones seed is scattered onto the two selected positions of each row.
	assert.Equal([]float64{0, 1, 1, 0, 1, 1}, grad.Data())
}

func TestTopKGrad_NumericCheck(t *testing.T) {
	assert := assert.New(t)
	const eps = 1e-6
	const tol = 1e-5

	shape := tensor.Shape{2, 4}
	backing := []float64{0.5, -1.2, 3.4, 0.2, 2.2, -0.7, 1.1, 2.2}

	// scalar cost built from the top-k values, so finite differences apply.
	cost := func(data []float64) float64 {
		in := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(append([]float64(nil), data...)))
		op, err := newTopKValuesOp(1, 2, topKMaximum, 2)
		if err != nil {
			t.Fatal(err)
		}
		idxOp, err := newTopKIndicesOp(1, 2, topKMaximum, 2)
		if err != nil {
			t.Fatal(err)
		}
		idxV, err := idxOp.Do(in)
		require.NoError(t, err)
		v, err := op.Do(in, idxV)
		require.NoError(t, err)
		var sum float64
		for _, f := range v.Data().([]float64) {
			sum += f
		}
		return sum / float64(v.Size())
	}

	// analytic gradient through the graph (tape machine).
	g := NewGraph()
	x := NewMatrix(g, tensor.Float64, WithName("x"), WithShape(shape...),
		WithValue(tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))))
	v, _, err := TopK(x, 2, 1)
	require.NoError(t, err)
	c := Must(Mean(v))
	_, err = Grad(c, x)
	require.NoError(t, err)
	m := NewTapeMachine(g)
	require.NoError(t, m.RunAll())
	require.NoError(t, m.Close())
	grad, err := x.Grad()
	require.NoError(t, err)
	analytic := grad.Data().([]float64)

	numeric := make([]float64, len(backing))
	for i := range backing {
		up := append([]float64(nil), backing...)
		dn := append([]float64(nil), backing...)
		up[i] += eps
		dn[i] -= eps
		numeric[i] = (cost(up) - cost(dn)) / (2 * eps)
	}

	for i := range analytic {
		assert.InDelta(numeric[i], analytic[i], tol, "gradient mismatch at element %d", i)
	}
}

func TestTopKGrad_NumericCheckFloat32(t *testing.T) {
	assert := assert.New(t)
	const eps = float32(1e-3)
	const tol = float32(2e-2)

	shape := tensor.Shape{2, 4}
	backing := []float32{0.5, -1.2, 3.4, 0.2, 2.2, -0.7, 1.1, 2.2}

	cost := func(data []float32) float32 {
		in := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(append([]float32(nil), data...)))
		op, err := newTopKValuesOp(1, 2, topKMaximum, 2)
		require.NoError(t, err)
		idxOp, err := newTopKIndicesOp(1, 2, topKMaximum, 2)
		require.NoError(t, err)
		idxV, err := idxOp.Do(in)
		require.NoError(t, err)
		v, err := op.Do(in, idxV)
		require.NoError(t, err)
		var sum float32
		for _, f := range v.Data().([]float32) {
			sum += f
		}
		return sum / float32(v.Size())
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithName("x"), WithShape(shape...),
		WithValue(tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))))
	v, _, err := TopK(x, 2, 1)
	require.NoError(t, err)
	c := Must(Mean(v))
	_, err = Grad(c, x)
	require.NoError(t, err)
	m := NewTapeMachine(g)
	require.NoError(t, m.RunAll())
	require.NoError(t, m.Close())
	grad, err := x.Grad()
	require.NoError(t, err)
	analytic := grad.Data().([]float32)

	numeric := make([]float32, len(backing))
	for i := range backing {
		up := append([]float32(nil), backing...)
		dn := append([]float32(nil), backing...)
		up[i] += eps
		dn[i] -= eps
		numeric[i] = (cost(up) - cost(dn)) / (2 * eps)
	}

	for i := range analytic {
		assert.InDelta(float64(numeric[i]), float64(analytic[i]), float64(tol), "f32 gradient mismatch at element %d", i)
	}
}

func TestTopKGrad_AccumulatesRepeatedSelection(t *testing.T) {
	assert := assert.New(t)
	// A row where one value dominates: with k > 1 it still wins only one
	// slot, so build a case where gather legitimately repeats an index:
	// feed a duplicated index via the values op on an input where two top
	// values are equal — ties select distinct indices, so instead verify
	// accumulation directly via the diff op with a repeated-index tensor.
	g := NewGraph()
	x := NewVector(g, tensor.Float64, WithName("x"), WithShape(4),
		WithValue(tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{1, 2, 3, 4}))))
	v, idx, err := TopK(x, 3, 0)
	require.NoError(t, err)
	cost := Must(Sum(v))
	_, err = Grad(cost, x)
	require.NoError(t, err)
	m := NewTapeMachine(g)
	require.NoError(t, m.RunAll())
	require.NoError(t, m.Close())
	_ = idx

	// direct diff-op check: index 2 selected twice must accumulate twice.
	op, err := newTopKValuesOp(0, 3, topKMaximum, 1)
	require.NoError(t, err)
	diff := &topKValuesDiffOp{op.topKBase}

	xin := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{1, 2, 3, 4}))
	dy := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float64{0.5, 0.7, 0.9}))
	repeated := tensor.New(tensor.WithShape(3), tensor.WithBacking([]int{2, 2, 0}))

	out, err := diff.Do(xin, dy, repeated)
	require.NoError(t, err)
	assert.Equal([]float64{0.9, 0, 1.2, 0}, out.Data(), "repeated index 2 must accumulate 0.5+0.7")
}

/*
 * Differentiating the indices op must be refused explicitly, never silently
 * zeroed.
 */

func TestTopKIndices_NotDifferentiable(t *testing.T) {
	g := NewGraph()
	x := NewVector(g, tensor.Float64, WithName("x"), WithShape(4),
		WithValue(tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{1, 2, 3, 4}))))
	idx, err := ArgMax(x, 0)
	require.NoError(t, err)
	// Chain a second non-differentiable selection onto the first so the
	// discrete index node becomes an internal node with a non-differentiable
	// sole parent. Backpropagation then cannot route around it and the graph
	// construction rejects the request explicitly rather than silently
	// producing a zero gradient.
	idx2, err := ArgMax(idx, 0)
	require.NoError(t, err)
	selected, err := ByIndices(x, idx2, 0)
	require.NoError(t, err)
	cost := Must(Sum(selected))

	_, ok := idx.op.(SDOp)
	assert.False(t, ok, "indices op must not advertise symbolic differentiation")
	_, ok = idx.op.(ADOp)
	assert.False(t, ok, "indices op must not advertise automatic differentiation")

	_, err = Grad(cost, x)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unable to differentiate")
}

func TestArgMaxArgMin_K1Consistency(t *testing.T) {
	assert := assert.New(t)
	in := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{
		1, 4, 3, 4,
		9, 2, 8, 7,
	}))

	amax, err := newTopKIndicesOp(1, 1, topKMaximum, 2)
	require.NoError(t, err)
	amin, err := newTopKIndicesOp(1, 1, topKMinimum, 2)
	require.NoError(t, err)

	amOut := runTopKOnDense(t, amax, in).(*tensor.Dense)
	aminOut := runTopKOnDense(t, amin, in).(*tensor.Dense)

	assert.Equal(tensor.Shape{2, 1}, amOut.Shape(), "k=1 keeps the leading dimension and collapses the axis to size 1")
	assert.Equal([]int{1, 0}, amOut.Ints(), "ArgMax tie picks the lowest index")
	assert.Equal([]int{0, 1}, aminOut.Ints())
}

func TestTopK_DeviceRejection(t *testing.T) {
	// Simulate device-backed memory by linking the tensor to an engine whose
	// allocations are not Go-accessible, exactly as the CUDA engine flags its
	// allocations (WithEngine sets NativelyInaccessible in that case).
	in := tensor.New(
		tensor.WithShape(4),
		tensor.WithBacking([]float64{1, 2, 3, 4}),
		tensor.WithEngine(inaccessibleEngine{}),
	)
	require.False(t, in.IsNativelyAccessible())

	op, err := newTopKIndicesOp(0, 2, topKMaximum, 1)
	require.NoError(t, err)
	v, _, _, err := anyToValue(in)
	require.NoError(t, err)
	_, err = op.Do(v)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only implemented for CPU")

	opV, err := newTopKValuesOp(0, 2, topKMaximum, 1)
	require.NoError(t, err)
	idx := tensor.New(tensor.WithShape(2), tensor.WithBacking([]int{3, 2}))
	_, err = opV.Do(in, idx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only implemented for CPU")
}

// inaccessibleEngine is a minimal tensor.Engine that reports device-style
// (non-Go-accessible) memory. Its allocation methods are never called by the
// op under test because the op rejects the tensor before touching memory.
type inaccessibleEngine struct{}

func (inaccessibleEngine) AllocAccessible() bool                               { return false }
func (inaccessibleEngine) Alloc(size int64) (tensor.Memory, error)             { return nil, nil }
func (inaccessibleEngine) Free(mem tensor.Memory, size int64) error            { return nil }
func (inaccessibleEngine) Memset(mem tensor.Memory, val interface{}) error     { return nil }
func (inaccessibleEngine) Memclr(mem tensor.Memory)                            {}
func (inaccessibleEngine) Memcpy(dst, src tensor.Memory) error                 { return nil }
func (inaccessibleEngine) Accessible(mem tensor.Memory) (tensor.Memory, error) { return mem, nil }
func (inaccessibleEngine) WorksWith(order tensor.DataOrder) bool               { return true }
