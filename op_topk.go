package gorgonia

/*
This file contains the TopK family of selection ops:

  - topKValuesOp: selects the k greatest (or smallest) values along an axis,
    keeping the leading dimensions of the input.
  - topKIndicesOp: returns the indices of those values. k == 1 yields the
    ArgMax/ArgMin semantics.

Both ops are pure with respect to their input: the input's underlying data is
never modified and every output is freshly allocated. Only the CPU execution
path is supported; running either op on a non-CPU device returns an
identifiable error rather than computing a wrong result.

Numerical rules (documented and enforced):

  - Ties are broken by ascending original index. The ordering is deterministic
    so repeated runs of the same input produce identical results.
  - NaNs sort after every non-NaN value (including +Inf), matching the total
    ordering used by Go's math.IsNaN convention for maximum selection. For
    minimum selection NaNs therefore end up last and are only selected when
    there are not enough finite values to fill k.
  - +Inf and -Inf are otherwise ordinary ordered values.
  - k larger than the axis length is clamped to the axis length.
  - k == 0 is legal and produces empty output tensors of the expected rank.
*/

import (
	"encoding/binary"
	"fmt"
	"hash"
	"math"
	"sort"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// selection directions
const (
	topKMaximum = 1  // keep the k greatest elements (TopK / ArgMax)
	topKMinimum = -1 // keep the k smallest elements (TopKMin / ArgMin)
)

// topKBase holds the fields shared by the values and indices variants.
type topKBase struct {
	// axis is the canonicalized (non-negative) reduction axis.
	axis int
	// k is the number of elements to select. It is never negative after
	// construction; k larger than the axis length is clamped at execution.
	k int
	// direction is either topKMaximum or topKMinimum.
	direction int
	// d is the input rank, captured at construction so type inference does not
	// depend on a fixed dimensionality.
	d int
}

// canonicalizeAxis maps a possibly negative axis onto a non-negative one given
// the input rank. The rule mirrors how negative axes are documented for the
// existing reduction ops: axis -1 refers to the last dimension.
func canonicalizeAxis(axis, dims int) (int, error) {
	if dims <= 0 {
		return 0, errors.Errorf("cannot resolve axis %d for a %d-dimensional input: expected rank >= 1, got %d", axis, dims, dims)
	}
	if axis < 0 {
		axis += dims
	}
	if axis < 0 || axis >= dims {
		return 0, errors.Errorf("axis out of range: axis %d is not valid for a %d-dimensional input (valid range: [%d, %d])",
			axis, dims, -dims, dims-1)
	}
	return axis, nil
}

func newTopKBase(axis, k, direction, dims int) (topKBase, error) {
	if k < 0 {
		return topKBase{}, errors.Errorf("k must be non-negative: got k = %d", k)
	}
	ax, err := canonicalizeAxis(axis, dims)
	if err != nil {
		return topKBase{}, err
	}
	return topKBase{
		axis:      ax,
		k:         k,
		direction: direction,
		d:         dims,
	}, nil
}

// topKShape computes the output shape from the input shape. The selected axis
// shrinks to min(k, axis length); every other dimension is preserved.
func (op topKBase) topKShape(in tensor.Shape) (tensor.Shape, error) {
	if len(in) != op.d {
		return nil, errors.Errorf("rank mismatch: %v was constructed for a %d-dimensional input but got shape %v (%d dimensions)",
			op, op.d, in, len(in))
	}
	if op.axis >= len(in) {
		return nil, errors.Errorf("axis out of range: axis %d is not valid for shape %v", op.axis, in)
	}
	out := in.Clone()
	axisLen := in[op.axis]
	selectCount := op.k
	if selectCount > axisLen {
		selectCount = axisLen
	}
	out[op.axis] = selectCount
	return out, nil
}

func (op topKBase) String() string {
	kind := "TopK"
	if op.direction == topKMinimum {
		kind = "TopKMin"
	}
	return fmt.Sprintf("%s{axis=%d, k=%d, d=%d}", kind, op.axis, op.k, op.d)
}

// writeTopKHash serializes every field that distinguishes two selections. It is
// shared by the values and indices variants, which additionally write their own
// discriminator prefix so a values node and an indices node with identical
// parameters never hash equal.
func (op topKBase) writeTopKHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, int32(op.axis)); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, int32(op.k)); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, int32(op.direction)); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, int32(op.d)); err != nil {
		panic(err)
	}
}

// checkDevice enforces the CPU-only execution constraint. Both execution
// engines route through Do, so this rejects unsupported devices at runtime
// with a self-describing error. A Dense tensor backed by device memory is
// flagged as not natively accessible, which is true under both build tags and
// therefore does not require a cuda-specific code path.
func (op topKBase) checkDevice(inputs []Value) error {
	for i, in := range inputs {
		if dv, ok := in.(*dualValue); ok {
			in = dv.Value
		}
		t, ok := in.(*tensor.Dense)
		if !ok {
			return errors.Errorf("%v only supports *tensor.Dense inputs on CPU: input %d is %T", op, i, in)
		}
		if !t.IsNativelyAccessible() {
			return errors.Errorf("%v is only implemented for CPU; input %d is backed by non-CPU (device) memory", op, i)
		}
	}
	return nil
}

// supportedTopKDtype reports whether the input dtype is one of the three
// supported element types: the two floating point types and machine ints.
func supportedTopKDtype(dt tensor.Dtype) bool {
	switch dt {
	case tensor.Float64, tensor.Float32, tensor.Int:
		return true
	}
	return false
}

func (op topKBase) checkDtype(dt tensor.Dtype) error {
	if !supportedTopKDtype(dt) {
		return errors.Errorf(unsupportedDtype+": %v supports Float64, Float32 and Int inputs only", dt, op)
	}
	return nil
}

/*
 * ordering helper: produces a stable (value, original-index) ordering of one
 * axis-aligned slice.
 */

// topKSortF64 returns the original indices of the n elements accessible at
// base, base+stride, ... sorted by the selection direction. Ties are always
// broken by ascending original index, which makes the result deterministic.
func topKSortF64(data []float64, base, stride, n int, direction int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		va, vb := data[base+ia*stride], data[base+ib*stride]
		na, nb := math.IsNaN(va), math.IsNaN(vb)
		// NaNs are ordered after every finite value (and after Inf). A valid
		// strict weak ordering: if either side is NaN, "a before b" holds only
		// when b (not a) is the NaN.
		if na || nb {
			return !na && nb
		}
		if va != vb {
			if direction == topKMaximum {
				return va > vb
			}
			return va < vb
		}
		return ia < ib
	})
	return idx
}

func topKSortF32(data []float32, base, stride, n int, direction int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		va, vb := data[base+ia*stride], data[base+ib*stride]
		na, nb := math.IsNaN(float64(va)), math.IsNaN(float64(vb))
		if na || nb {
			return !na && nb
		}
		if va != vb {
			if direction == topKMaximum {
				return va > vb
			}
			return va < vb
		}
		return ia < ib
	})
	return idx
}

func topKSortInt(data []int, base, stride, n int, direction int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		va, vb := data[base+ia*stride], data[base+ib*stride]
		if va != vb {
			if direction == topKMaximum {
				return va > vb
			}
			return va < vb
		}
		return ia < ib
	})
	return idx
}

// topKIteration describes one pass over the axis for a single combination of
// the leading and trailing dimensions.
type topKIteration struct {
	base    int // flat offset of the first element of the slice, in the input
	outBase int // flat offset of the matching slice in the k-sized output
	stride  int // flat distance between consecutive axis elements
	n       int // length of the axis
}

// topKIterations enumerates axis-aligned slices of a dense tensor. Dense
// tensors produced by Materialize use row-major (C-order) strides, so the
// iteration pattern only depends on the shape. outAxisLen is the length of the
// selected axis in the output (min(k, input axis length)).
func topKIterations(shape tensor.Shape, axis, outAxisLen int) []topKIteration {
	rank := len(shape)
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}
	axisLen := shape[axis]
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	iters := make([]topKIteration, 0, outer*inner)
	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			iters = append(iters, topKIteration{
				base:    o*axisLen*inner + i,
				outBase: o*outAxisLen*inner + i,
				stride:  inner,
				n:       axisLen,
			})
		}
	}
	return iters
}

// materializeDense ensures the input is a contiguous, row-major *tensor.Dense
// whose backing slice can be addressed flat. Views and transposed tensors are
// copied; the original is left untouched.
func materializeDense(v Value) (*tensor.Dense, error) {
	t, ok := v.(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("expected *tensor.Dense, got %T", v)
	}
	if t.IsMaterializable() {
		mt := t.Materialize()
		d, ok := mt.(*tensor.Dense)
		if !ok {
			return nil, errors.Errorf("materialized tensor is not *tensor.Dense: got %T", mt)
		}
		return d, nil
	}
	return t, nil
}

// runTopK performs the selection on a dense input and returns freshly
// allocated values and indices tensors. The output shape is derived from
// op.topKShape so k clamping is applied consistently.
func runTopK(op topKBase, in *tensor.Dense) (values, indices *tensor.Dense, err error) {
	if err = op.checkDtype(in.Dtype()); err != nil {
		return nil, nil, err
	}
	in, err = materializeDense(in)
	if err != nil {
		return nil, nil, err
	}

	shape := in.Shape()
	outShape, err := op.topKShape(shape)
	if err != nil {
		return nil, nil, err
	}
	values = tensor.New(tensor.WithShape(outShape...), tensor.Of(in.Dtype()))
	indices = tensor.New(tensor.WithShape(outShape...), tensor.Of(tensor.Int))

	iters := topKIterations(shape, op.axis, outShape[op.axis])
	inner := 1
	for i := op.axis + 1; i < len(shape); i++ {
		inner *= shape[i]
	}
	k := outShape[op.axis]

	switch in.Dtype() {
	case tensor.Float64:
		src := in.Float64s()
		dst := values.Float64s()
		idst := indices.Ints()
		for _, it := range iters {
			order := topKSortF64(src, it.base, it.stride, it.n, op.direction)
			for j := 0; j < k; j++ {
				srcIdx := it.base + order[j]*it.stride
				dstIdx := it.outBase + j*inner
				dst[dstIdx] = src[srcIdx]
				idst[dstIdx] = order[j]
			}
		}
	case tensor.Float32:
		src := in.Float32s()
		dst := values.Float32s()
		idst := indices.Ints()
		for _, it := range iters {
			order := topKSortF32(src, it.base, it.stride, it.n, op.direction)
			for j := 0; j < k; j++ {
				srcIdx := it.base + order[j]*it.stride
				dstIdx := it.outBase + j*inner
				dst[dstIdx] = src[srcIdx]
				idst[dstIdx] = order[j]
			}
		}
	case tensor.Int:
		src := in.Ints()
		dst := values.Ints()
		idst := indices.Ints()
		for _, it := range iters {
			order := topKSortInt(src, it.base, it.stride, it.n, op.direction)
			for j := 0; j < k; j++ {
				srcIdx := it.base + order[j]*it.stride
				dstIdx := it.outBase + j*inner
				dst[dstIdx] = src[srcIdx]
				idst[dstIdx] = order[j]
			}
		}
	default:
		return nil, nil, errors.Errorf(unsupportedDtype+": %v", in.Dtype(), op)
	}
	return values, indices, nil
}

/*
 * topKValuesOp
 */

type topKValuesOp struct {
	topKBase
}

func newTopKValuesOp(axis, k, direction, dims int) (*topKValuesOp, error) {
	base, err := newTopKBase(axis, k, direction, dims)
	if err != nil {
		return nil, err
	}
	return &topKValuesOp{base}, nil
}

// Arity is 2: the data tensor x and the integer indices tensor produced by the
// sibling topKIndicesOp. The indices are an input (rather than recomputed) so
// that symbolic differentiation can thread them into the backward op, and so
// the forward values and indices always agree.
func (op *topKValuesOp) Arity() int { return 2 }

// Type is rank-dependent: a rank-d element tensor and a rank-d Int tensor map
// to another rank-d tensor of the same element type. The rank is captured at
// construction rather than fixed.
func (op *topKValuesOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := makeTensorType(op.d, a)
	idx := makeTensorType(op.d, tensor.Int)
	return hm.NewFnType(t, idx, t)
}

func (op *topKValuesOp) InferShape(dimsizers ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(dimsizers)); err != nil {
		return nil, err
	}
	in, ok := dimsizers[0].(tensor.Shape)
	if !ok {
		return nil, errors.Errorf("expected a tensor.Shape, got %T", dimsizers[0])
	}
	idxShape, ok := dimsizers[1].(tensor.Shape)
	if !ok {
		return nil, errors.Errorf("expected indices to have a tensor.Shape, got %T", dimsizers[1])
	}
	out, err := op.topKShape(in)
	if err != nil {
		return nil, err
	}
	if !idxShape.Eq(out) {
		return nil, errors.Errorf("indices shape mismatch: %v expects indices of shape %v but got %v", op, out, idxShape)
	}
	return out, nil
}

func (op *topKValuesOp) Do(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	if err := op.checkDevice(inputs); err != nil {
		return nil, err
	}
	in, err := materializeDense(inputs[0])
	if err != nil {
		return nil, err
	}
	indices, err := materializeDense(inputs[1])
	if err != nil {
		return nil, err
	}
	if indices.Dtype() != tensor.Int {
		return nil, errors.Errorf("indices must have dtype tensor.Int, got %v", indices.Dtype())
	}
	return gatherTopKValues(op.topKBase, in, indices)
}

func (op *topKValuesOp) ReturnsPtr() bool     { return true }
func (op *topKValuesOp) CallsExtern() bool    { return false }
func (op *topKValuesOp) OverwritesInput() int { return -1 }
func (op *topKValuesOp) IsBinary() bool       { return true }
func (op *topKValuesOp) Hashcode() uint32     { return simpleHash(op) }
func (op *topKValuesOp) WriteHash(h hash.Hash) {
	h.Write([]byte("topKValues"))
	op.writeTopKHash(h)
}

// DiffWRT: only the data tensor carries gradients; differentiating with
// respect to discrete indices is undefined, so the second input reports false.
func (op *topKValuesOp) DiffWRT(inputs int) []bool {
	if inputs != op.Arity() {
		panic(fmt.Sprintf("%v expects %d inputs, got %d", op, op.Arity(), inputs))
	}
	return []bool{true, false}
}

func (op *topKValuesOp) SymDiff(inputs Nodes, output, gradNode *Node) (Nodes, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	diff := &topKValuesDiffOp{op.topKBase}
	dx, err := ApplyOp(diff, inputs[0], gradNode, inputs[1])
	if err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	return Nodes{dx, nil}, nil
}

// DoDiff is the automatic-differentiation counterpart of SymDiff. It scatters
// the output gradient back onto the selected input positions; repeated
// selection of the same position accumulates.
func (op *topKValuesOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 2 {
		return errors.Errorf("%v DoDiff expects 2 inputs, got %d", op, len(inputs))
	}
	diff := &topKValuesDiffOp{op.topKBase}
	xdv, ydv := getDV(inputs[0], output)
	indices, err := materializeDense(inputs[1].boundTo.(*dualValue).Value)
	if err != nil {
		return err
	}
	dx, err := diff.Do(xdv.Value, ydv.d, indices)
	if err != nil {
		return err
	}
	return xdv.SetDeriv(dx)
}

/*
 * topKIndicesOp
 */

type topKIndicesOp struct {
	topKBase
}

func newTopKIndicesOp(axis, k, direction, dims int) (*topKIndicesOp, error) {
	base, err := newTopKBase(axis, k, direction, dims)
	if err != nil {
		return nil, err
	}
	return &topKIndicesOp{base}, nil
}

func (op *topKIndicesOp) Arity() int { return 1 }

// Type returns the signature Tensor d a -> Tensor d Int: the dtype of the
// indices does not participate in the inference of the element type, so the
// integer output type is fixed regardless of the input dtype (including
// integer inputs).
func (op *topKIndicesOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	in := makeTensorType(op.d, a)
	out := makeTensorType(op.d, tensor.Int)
	return hm.NewFnType(in, out)
}

func (op *topKIndicesOp) InferShape(dimsizers ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(dimsizers)); err != nil {
		return nil, err
	}
	in, ok := dimsizers[0].(tensor.Shape)
	if !ok {
		return nil, errors.Errorf("expected a tensor.Shape, got %T", dimsizers[0])
	}
	return op.topKShape(in)
}

func (op *topKIndicesOp) Do(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	if err := op.checkDevice(inputs); err != nil {
		return nil, err
	}
	in, err := materializeDense(inputs[0])
	if err != nil {
		return nil, err
	}
	_, indices, err := runTopK(op.topKBase, in)
	return indices, err
}

func (op *topKIndicesOp) ReturnsPtr() bool     { return true }
func (op *topKIndicesOp) CallsExtern() bool    { return false }
func (op *topKIndicesOp) OverwritesInput() int { return -1 }
func (op *topKIndicesOp) isUnary() bool        { return true }
func (op *topKIndicesOp) Hashcode() uint32     { return simpleHash(op) }
func (op *topKIndicesOp) WriteHash(h hash.Hash) {
	h.Write([]byte("topKIndices"))
	op.writeTopKHash(h)
}

// The indices op deliberately does NOT implement SDOp/ADOp: the gradient with
// respect to discrete indices is undefined. Requiring a gradient through it
// therefore fails graph construction with an explicit "Not a SymDifOp" error
// instead of silently producing a zero gradient.

/*
 * topKValuesDiffOp: scatters the gradient of the selected values back onto the
 * input positions that were selected. It is the selection analogue of
 * byIndicesOpDiffOp.
 */

type topKValuesDiffOp struct {
	// topKBase is a value (not a pointer embedding) on purpose: embedding
	// *topKValuesOp would promote DoDiff and make this symbolic-gradient node
	// satisfy the ADOp interface, causing the lisp machine to execute it a
	// second time during its autodiff pass.
	topKBase
}

// Arity is 3: the data tensor x, the output gradient dy and the integer
// selection indices.
func (op *topKValuesDiffOp) Arity() int { return 3 }

func (op *topKValuesDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := makeTensorType(op.d, a)
	idx := makeTensorType(op.d, tensor.Int)
	return hm.NewFnType(t, t, idx, t)
}

func (op *topKValuesDiffOp) InferShape(dimsizers ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(dimsizers)); err != nil {
		return nil, err
	}
	in, ok := dimsizers[0].(tensor.Shape)
	if !ok {
		return nil, errors.Errorf("expected a tensor.Shape, got %T", dimsizers[0])
	}
	return in.Clone(), nil
}

func (op *topKValuesDiffOp) ReturnsPtr() bool     { return true }
func (op *topKValuesDiffOp) CallsExtern() bool    { return false }
func (op *topKValuesDiffOp) OverwritesInput() int { return -1 }
func (op *topKValuesDiffOp) Hashcode() uint32     { return simpleHash(op) }
func (op *topKValuesDiffOp) WriteHash(h hash.Hash) {
	h.Write([]byte("topKValuesDiff"))
	op.writeTopKHash(h)
}

func (op *topKValuesDiffOp) String() string {
	return fmt.Sprintf("TopKValuesDiff{axis=%d, k=%d, d=%d}", op.axis, op.k, op.d)
}

// Do takes (x, dy, indices). x is only used for shape/dtype metadata; the
// scatter only reads dy and indices.
func (op *topKValuesDiffOp) Do(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	x, err := materializeDense(unwrapDV(inputs[0]))
	if err != nil {
		return nil, err
	}
	grad, err := materializeDense(unwrapDV(inputs[1]))
	if err != nil {
		return nil, err
	}
	indices, err := materializeDense(unwrapDV(inputs[2]))
	if err != nil {
		return nil, err
	}
	if indices.Dtype() != tensor.Int {
		return nil, errors.Errorf("indices must have dtype tensor.Int, got %v", indices.Dtype())
	}
	if grad.Dtype() != x.Dtype() {
		return nil, errors.Errorf("gradient dtype %v does not match input dtype %v", grad.Dtype(), x.Dtype())
	}

	out := tensor.New(tensor.WithShape(x.Shape().Clone()...), tensor.Of(x.Dtype()))
	if err = scatterTopKGradient(op.topKBase, x.Shape(), grad, indices, out); err != nil {
		return nil, err
	}
	return out, nil
}

// scatterTopKGradient writes grad back onto the selected input positions.
// When the same input position is selected multiple times the contributions
// are summed.
func scatterTopKGradient(base topKBase, inShape tensor.Shape, grad, indices, out *tensor.Dense) error {
	outShape, err := base.topKShape(inShape)
	if err != nil {
		return err
	}
	if !grad.Shape().Eq(outShape) {
		return errors.Errorf("gradient shape mismatch: expected %v, got %v", outShape, grad.Shape())
	}
	if !indices.Shape().Eq(outShape) {
		return errors.Errorf("indices shape mismatch: expected %v, got %v", outShape, indices.Shape())
	}

	inner := 1
	for i := base.axis + 1; i < len(inShape); i++ {
		inner *= inShape[i]
	}
	k := outShape[base.axis]
	iters := topKIterations(inShape, base.axis, k)
	idx := indices.Ints()

	switch out.Dtype() {
	case tensor.Float64:
		g := grad.Float64s()
		dx := out.Float64s()
		for _, it := range iters {
			for j := 0; j < k; j++ {
				selected := idx[it.outBase+j*inner]
				dx[it.base+selected*inner] += g[it.outBase+j*inner]
			}
		}
	case tensor.Float32:
		g := grad.Float32s()
		dx := out.Float32s()
		for _, it := range iters {
			for j := 0; j < k; j++ {
				selected := idx[it.outBase+j*inner]
				dx[it.base+selected*inner] += g[it.outBase+j*inner]
			}
		}
	default:
		return errors.Errorf("gradient scatter requires a floating point input, got %v", out.Dtype())
	}
	return nil
}

// gatherTopKValues assembles the values output from x and the precomputed
// integer indices, reusing the same iteration layout as the selection.
func gatherTopKValues(base topKBase, x, indices *tensor.Dense) (Value, error) {
	outShape, err := base.topKShape(x.Shape())
	if err != nil {
		return nil, err
	}
	if !indices.Shape().Eq(outShape) {
		return nil, errors.Errorf("indices shape mismatch: expected %v, got %v", outShape, indices.Shape())
	}
	out := tensor.New(tensor.WithShape(outShape...), tensor.Of(x.Dtype()))

	inner := 1
	for i := base.axis + 1; i < len(x.Shape()); i++ {
		inner *= x.Shape()[i]
	}
	k := outShape[base.axis]
	iters := topKIterations(x.Shape(), base.axis, k)
	idx := indices.Ints()

	switch x.Dtype() {
	case tensor.Float64:
		src, dst := x.Float64s(), out.Float64s()
		for _, it := range iters {
			for j := 0; j < k; j++ {
				selected := idx[it.outBase+j*inner]
				dst[it.outBase+j*inner] = src[it.base+selected*inner]
			}
		}
	case tensor.Float32:
		src, dst := x.Float32s(), out.Float32s()
		for _, it := range iters {
			for j := 0; j < k; j++ {
				selected := idx[it.outBase+j*inner]
				dst[it.outBase+j*inner] = src[it.base+selected*inner]
			}
		}
	case tensor.Int:
		src, dst := x.Ints(), out.Ints()
		for _, it := range iters {
			for j := 0; j < k; j++ {
				selected := idx[it.outBase+j*inner]
				dst[it.outBase+j*inner] = src[it.base+selected*inner]
			}
		}
	default:
		return nil, errors.Errorf(unsupportedDtype+": %v", x.Dtype(), base)
	}
	return out, nil
}

func unwrapDV(v Value) Value {
	if dv, ok := v.(*dualValue); ok {
		return dv.Value
	}
	return v
}

/*
 * Public node-level constructors
 */

// TopK returns two nodes. The first holds the k greatest values of x along
// axis; the second holds the corresponding indices into the axis. The leading
// dimensions of x are preserved and the selected axis has length min(k, axis
// length). Negative axes are interpreted relative to the input rank, matching
// the reduction ops.
func TopK(x *Node, k int, axis int) (values, indices *Node, err error) {
	return topK(x, k, axis, topKMaximum, "TopK")
}

// TopKMin is the minimum-selection variant of TopK: it returns the k smallest
// values and their indices.
func TopKMin(x *Node, k int, axis int) (values, indices *Node, err error) {
	return topK(x, k, axis, topKMinimum, "TopKMin")
}

func topK(x *Node, k, axis, direction int, name string) (values, indices *Node, err error) {
	opI, err := newTopKIndicesOp(axis, k, direction, x.Dims())
	if err != nil {
		return nil, nil, errors.Wrap(err, name)
	}
	if indices, err = ApplyOp(opI, x); err != nil {
		return nil, nil, errors.Wrapf(err, "%s indices", name)
	}
	opV, err := newTopKValuesOp(axis, k, direction, x.Dims())
	if err != nil {
		return nil, nil, errors.Wrap(err, name)
	}
	if values, err = ApplyOp(opV, x, indices); err != nil {
		return nil, nil, errors.Wrapf(err, "%s values", name)
	}
	return values, indices, nil
}

// ArgMax returns the index of the greatest value along axis. It is exactly
// TopK with k == 1, keeping the same tie-breaking (ascending original index)
// and NaN rules.
func ArgMax(x *Node, axis int) (*Node, error) {
	op, err := newTopKIndicesOp(axis, 1, topKMaximum, x.Dims())
	if err != nil {
		return nil, errors.Wrap(err, "ArgMax")
	}
	return ApplyOp(op, x)
}

// ArgMin returns the index of the smallest value along axis. It is exactly
// TopKMin with k == 1.
func ArgMin(x *Node, axis int) (*Node, error) {
	op, err := newTopKIndicesOp(axis, 1, topKMinimum, x.Dims())
	if err != nil {
		return nil, errors.Wrap(err, "ArgMin")
	}
	return ApplyOp(op, x)
}

// interface compliance
var (
	_ Op   = &topKValuesOp{}
	_ Op   = &topKIndicesOp{}
	_ SDOp = &topKValuesOp{}
	_ ADOp = &topKValuesOp{}
	_ Op   = &topKValuesDiffOp{}
)
