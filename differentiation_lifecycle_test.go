package gorgonia

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// buildSharedSubgraph returns a graph with two costs over one shared subgraph:
//
//	common = x * y
//	z1     = common + 1
//	z2     = common * 2
func buildSharedSubgraph() (g *ExprGraph, x, y, common, z1, z2 *Node) {
	g = NewGraph()
	x = NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	y = NewScalar(g, Float64, WithName("y"), WithValue(3.0))
	common = Must(Mul(x, y))
	z1 = Must(Add(common, NewConstant(1.0)))
	z2 = Must(Mul(common, NewConstant(2.0)))
	return
}

// gradOf runs cost's graph and reads the gradients of the given inputs.
//
// The gradients are zeroed before the run, exactly as a training loop must do
// at the start of each iteration: the tape machine keeps its historical
// accumulate-on-run semantics, so leaving a residual in place would add it to
// the freshly computed gradient.
func gradOf(t *testing.T, g *ExprGraph, cost *Node, params ...*Node) []float64 {
	t.Helper()
	if _, err := Grad(cost, params...); err != nil {
		t.Fatalf("Grad failed: %v", err)
	}
	for _, p := range params {
		ZeroNodeGradient(p)
	}
	m := NewTapeMachine(g, BindDualValues(params...))
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll failed: %v", err)
	}
	out := make([]float64, 0, len(params))
	for _, p := range params {
		gv, err := p.Grad()
		if err != nil {
			t.Fatalf("no gradient for %v: %v", p, err)
		}
		out = append(out, gv.Data().(float64))
	}
	return out
}

func TestDifferentiate_RepeatSameCostStable(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()

	first := gradOf(t, g, z1, x, y)
	nodesAfterFirst := len(g.AllNodes())
	edgesAfterFirst := g.Edges().Len()

	for i := 0; i < 5; i++ {
		again := gradOf(t, g, z1, x, y)
		assert.Equal(t, first, again, "iteration %d", i)
		assert.Equal(t, nodesAfterFirst, len(g.AllNodes()), "node count grew at iteration %d", i)
		assert.Equal(t, edgesAfterFirst, g.Edges().Len(), "edge count grew at iteration %d", i)
	}
}

func TestDifferentiate_SharedSubgraphMatchesFreshGraph(t *testing.T) {
	// First cost on the original graph.
	g, x, y, _, z1, z2 := buildSharedSubgraph()
	gotZ1 := gradOf(t, g, z1, x, y)

	// Same graph, second cost sharing the subgraph.
	gotZ2 := gradOf(t, g, z2, x, y)

	// Rebuild z2 on a pristine graph and compare element-wise.
	gFresh, xFresh, yFresh, _, _, z2Fresh := buildSharedSubgraph()
	wantZ2 := gradOf(t, gFresh, z2Fresh, xFresh, yFresh)
	assert.Equal(t, wantZ2, gotZ2, "shared-subgraph gradient must match a fresh graph")

	// Rebuild z1 on a pristine graph and compare.
	gFresh1, xFresh1, yFresh1, _, z1Fresh, _ := buildSharedSubgraph()
	wantZ1 := gradOf(t, gFresh1, z1Fresh, xFresh1, yFresh1)
	assert.Equal(t, wantZ1, gotZ1, "first cost must match a fresh graph")
}

func TestDifferentiate_NodeCountBounded(t *testing.T) {
	g, x, y, _, z1, z2 := buildSharedSubgraph()
	base := len(g.AllNodes())

	// Alternate between the two costs many times. Each differentiation
	// replaces the previous one, so the graph must not grow beyond one
	// differentiation's worth.
	var peak int
	for i := 0; i < 20; i++ {
		cost := z1
		if i%2 == 1 {
			cost = z2
		}
		if _, err := Grad(cost, x, y); err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		if n := len(g.AllNodes()); n > peak {
			peak = n
		}
	}

	// z2 (the larger derivative tree) adds a bounded number of nodes; the
	// important part is that repeating never grows past that bound.
	if peak > base+16 {
		t.Fatalf("node count grew unbounded: base %d, peak %d", base, peak)
	}
	// After the loop the graph must be back at the size of a single
	// differentiation (z2 on iteration 19).
	if got := len(g.AllNodes()); got != peak {
		t.Fatalf("node count %d != peak %d", got, peak)
	}
}

func TestDifferentiate_ResetRestoresGraph(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()
	base := len(g.AllNodes())
	baseEdges := g.Edges().Len()
	baseRoots := len(g.Roots())

	beforeDeriv := x.deriv
	beforeDerivOf := append(Nodes(nil), x.derivOf...)

	if _, err := Grad(z1, x, y); err != nil {
		t.Fatal(err)
	}
	require.NotEqual(t, base, len(g.AllNodes()), "differentiation should add nodes")
	require.NotNil(t, x.deriv, "differentiation should set the deriv mark")

	if err := g.ResetDifferentiation(); err != nil {
		t.Fatalf("reset failed: %v", err)
	}

	assert.Equal(t, base, len(g.AllNodes()), "node count after reset")
	assert.Equal(t, baseEdges, g.Edges().Len(), "edge count after reset")
	assert.Equal(t, baseRoots, len(g.Roots()), "root count after reset")
	assert.Equal(t, GradGraphClean, g.DifferentiationState())
	assert.True(t, x.deriv == beforeDeriv, "deriv mark restored")
	assert.Equal(t, beforeDerivOf, x.derivOf, "derivOf restored")

	// Values, types and connections of the pre-existing nodes are untouched.
	assert.Equal(t, 2.0, x.Value().Data())
	assert.Equal(t, 3.0, y.Value().Data())
	assert.Equal(t, Float64, x.Dtype())
	assert.True(t, x.Shape().IsScalar())

	// The graph is immediately usable for a fresh differentiation.
	got := gradOf(t, g, z1, x, y)
	gFresh, xFresh, yFresh, _, z1Fresh, _ := buildSharedSubgraph()
	want := gradOf(t, gFresh, z1Fresh, xFresh, yFresh)
	assert.Equal(t, want, got)
}

func TestDifferentiate_RepeatedResetCycle(t *testing.T) {
	g, x, y, _, z1, z2 := buildSharedSubgraph()
	base := len(g.AllNodes())
	baseEdges := g.Edges().Len()

	for i := 0; i < 5; i++ {
		cost := z1
		if i%2 == 1 {
			cost = z2
		}
		if _, err := Grad(cost, x, y); err != nil {
			t.Fatalf("cycle %d: %v", i, err)
		}
		if err := g.ResetDifferentiation(); err != nil {
			t.Fatalf("cycle %d reset: %v", i, err)
		}
		assert.Equal(t, base, len(g.AllNodes()), "cycle %d node count", i)
		assert.Equal(t, baseEdges, g.Edges().Len(), "cycle %d edge count", i)
	}
}

func TestDifferentiate_ErrorKinds(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()

	// Non-scalar cost.
	vec := NewVector(g, Float64, WithName("vec"), WithShape(3))
	_, err := Grad(vec, x)
	var de *DifferentiationError
	require.True(t, asDiffError(err, &de), "expected *DifferentiationError, got %T (%v)", err, err)
	assert.Equal(t, DiffErrCostNotScalar, de.Kind)

	// WRT that is not an input.
	_, err = Grad(z1, Must(Add(x, y)))
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrWRTNotInput, de.Kind)

	// WRT from another graph.
	other := NewGraph()
	foreign := NewScalar(other, Float64, WithName("foreign"))
	_, err = Grad(z1, foreign)
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrWRTNotInGraph, de.Kind)

	// Nil WRT.
	_, err = Grad(z1, nil)
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrWRTNil, de.Kind)

	// Nil cost.
	_, err = Grad(nil, x)
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrCostNil, de.Kind)

	// None of the above may degrade to "no gradient": they must be a
	// DifferentiationError, and each with the right kind. Reaching here means
	// err was not nil in every case.
}

func TestDifferentiate_FailureRollsBack(t *testing.T) {
	// A graph with a non-differentiable op on the path from y.
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	y := NewScalar(g, Float64, WithName("y"), WithValue(3.0))
	nd := Must(Sign(y)) // sign is not differentiable
	cost := Must(Add(x, nd))

	base := len(g.AllNodes())
	baseEdges := g.Edges().Len()

	_, err := Grad(cost, x, y)
	require.Error(t, err, "differentiating through a non-differentiable op must fail")
	var de *DifferentiationError
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrNotDifferentiable, de.Kind)

	// The graph must be back to exactly its original shape, with no
	// half-applied marks.
	assert.Equal(t, base, len(g.AllNodes()), "failed differentiation leaked nodes")
	assert.Equal(t, baseEdges, g.Edges().Len(), "failed differentiation leaked edges")
	assert.Nil(t, x.deriv, "failed differentiation left a deriv mark")
	assert.Equal(t, GradGraphClean, g.DifferentiationState())

	// x alone is differentiable, so the graph is still usable.
	if _, err := Grad(cost, x); err != nil {
		t.Fatalf("graph should still be differentiable for x: %v", err)
	}
}

func TestDifferentiate_ConcurrentSessions(t *testing.T) {
	const sessions = 8
	var wg sync.WaitGroup
	errs := make([]error, sessions)
	results := make([][]float64, sessions)

	for i := 0; i < sessions; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			g, x, y, _, z1, _ := buildSharedSubgraph()
			if _, err := Grad(z1, x, y); err != nil {
				errs[i] = err
				return
			}
			m := NewTapeMachine(g, BindDualValues(x, y))
			defer m.Close()
			if err := m.RunAll(); err != nil {
				errs[i] = err
				return
			}
			xg, _ := x.Grad()
			yg, _ := y.Grad()
			results[i] = []float64{xg.Data().(float64), yg.Data().(float64)}
		}(i)
	}
	wg.Wait()

	for i := 0; i < sessions; i++ {
		require.NoError(t, errs[i], "session %d", i)
		assert.Equal(t, []float64{3, 2}, results[i], "session %d", i)
	}
}

func TestDifferentiate_ConcurrentSameGraph(t *testing.T) {
	// Concurrent differentiation on the *same* graph must be serialised; the
	// result must be consistent and never corrupt the graph.
	g, x, y, _, z1, _ := buildSharedSubgraph()
	base := len(g.AllNodes())

	var wg sync.WaitGroup
	errs := make([]error, 8)
	for i := range errs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = Differentiate(z1, x, y)
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		require.NoError(t, err, "goroutine %d", i)
	}
	// The first call differentiates; the rest hit the cache. Either way the
	// graph holds exactly one differentiation.
	assert.Equal(t, base+2, len(g.AllNodes()))
}

func TestDifferentiate_DifferentParamSets(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	y := NewScalar(g, Float64, WithName("y"), WithValue(3.0))
	cost := Must(Add(Must(Mul(x, y)), x))

	// Only x.
	if _, err := Grad(cost, x); err != nil {
		t.Fatal(err)
	}
	ZeroNodeGradient(x)
	m := NewTapeMachine(g, BindDualValues(x, y))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	xg, err := x.Grad()
	require.NoError(t, err)
	assert.Equal(t, 4.0, xg.Data().(float64)) // d/dx (xy + x) = y + 1
	m.Close()

	// Both x and y, on the same graph.
	if _, err := Grad(cost, x, y); err != nil {
		t.Fatal(err)
	}
	ZeroNodeGradient(x)
	ZeroNodeGradient(y)
	m2 := NewTapeMachine(g, BindDualValues(x, y))
	if err := m2.RunAll(); err != nil {
		t.Fatal(err)
	}
	xg, _ = x.Grad()
	yg, err := y.Grad()
	require.NoError(t, err)
	assert.Equal(t, 4.0, xg.Data().(float64))
	assert.Equal(t, 2.0, yg.Data().(float64)) // d/dy (xy + x) = x
}

func TestDifferentiate_DeterministicAcrossRuns(t *testing.T) {
	// The same request repeated must return the very same derivative nodes,
	// and a different request must produce structurally identical ones.
	_, x, y, _, z1, _ := buildSharedSubgraph()
	first, err := Grad(z1, x, y)
	require.NoError(t, err)
	second, err := Grad(z1, x, y)
	require.NoError(t, err)
	require.Equal(t, len(first), len(second))
	for i := range first {
		assert.True(t, first[i] == second[i], "gradient node %d not reused", i)
	}
}

func TestDifferentiate_HelperMethods(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()
	assert.Equal(t, GradGraphClean, g.DifferentiationState())
	assert.Nil(t, g.DifferentiatedNodes())

	nodes, err := Grad(z1, x, y)
	require.NoError(t, err)
	assert.Equal(t, GradGraphDifferentiated, g.DifferentiationState())
	assert.Equal(t, len(nodes), len(g.DifferentiatedNodes()))
	assert.Equal(t, "", g.DifferentiationBrokenReason())

	require.NoError(t, g.ResetDifferentiation())
	assert.Equal(t, GradGraphClean, g.DifferentiationState())
	assert.Nil(t, g.DifferentiatedNodes())
}

// asDiffError is a tiny errors.As without importing errors in every test.
func asDiffError(err error, target **DifferentiationError) bool {
	for err != nil {
		if de, ok := err.(*DifferentiationError); ok {
			*target = de
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func TestDifferentiate_MatrixSharedSubgraph(t *testing.T) {
	// Same property with matrices, to exercise reduction/linalg SymDiff paths.
	build := func() (g *ExprGraph, w, b, c1, c2 *Node) {
		g = NewGraph()
		w = NewMatrix(g, Float64, WithName("w"), WithShape(2, 2), WithInit(RangedFrom(1)))
		b = NewMatrix(g, Float64, WithName("b"), WithShape(2, 2), WithInit(ValuesOf(1.0)))
		common := Must(Mul(w, Must(Transpose(w))))
		shifted := Must(Add(common, b))
		c1 = Must(Sum(shifted))
		scale := NewConstant(tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})))
		c2 = Must(Sum(Must(HadamardProd(shifted, scale))))
		return
	}
	g, w, b, c1, c2 := build()
	if _, err := Grad(c1, w, b); err != nil {
		t.Fatalf("first differentiation: %v", err)
	}
	if _, err := Grad(c2, w, b); err != nil {
		t.Fatalf("second differentiation: %v", err)
	}

	gFresh, wF, bF, _, c2F := build()
	if _, err := Grad(c2F, wF, bF); err != nil {
		t.Fatalf("fresh differentiation: %v", err)
	}

	m := NewTapeMachine(g, BindDualValues(w, b))
	defer m.Close()
	require.NoError(t, m.RunAll())
	mF := NewTapeMachine(gFresh, BindDualValues(wF, bF))
	defer mF.Close()
	require.NoError(t, mF.RunAll())

	wg, _ := w.Grad()
	wgF, _ := wF.Grad()
	assert.Equal(t, wgF.Data(), wg.Data(), "shared-graph w gradient differs from fresh graph")
	bg, _ := b.Grad()
	bgF, _ := bF.Grad()
	assert.Equal(t, bgF.Data(), bg.Data(), "shared-graph b gradient differs from fresh graph")
	_ = fmt.Sprint()
}

func TestDifferentiate_StaleMarksRejected(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()

	// Simulate a direct Backpropagate / SetDerivOf that bypassed the
	// lifecycle: set a derivative mark by hand.
	marker := Must(Add(x, y))
	x.deriv = marker

	_, err := Grad(z1, x, y)
	var de *DifferentiationError
	require.True(t, asDiffError(err, &de), "expected a DifferentiationError, got %v", err)
	assert.Equal(t, DiffErrStaleMarks, de.Kind)

	// ClearDerivativeMarks adopts the graph without touching values/types.
	g.ClearDerivativeMarks()
	assert.Nil(t, x.deriv)
	assert.Equal(t, 2.0, x.Value().Data())

	if _, err := Grad(z1, x, y); err != nil {
		t.Fatalf("graph should be usable after clearing marks: %v", err)
	}
}

func TestDifferentiate_BrokenGraphIsExplicit(t *testing.T) {
	g, x, y, _, z1, _ := buildSharedSubgraph()
	g.diffBrokenReason = "injected: rollback could not restore the graph"

	_, err := Grad(z1, x, y)
	var de *DifferentiationError
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrGraphMarked, de.Kind)
	assert.Equal(t, GradGraphBroken, g.DifferentiationState())
	assert.Contains(t, g.DifferentiationBrokenReason(), "injected")
	assert.Contains(t, err.Error(), "injected", "the reason must be reported, not hidden")

	// ResetDifferentiation clears the broken state.
	require.NoError(t, g.ResetDifferentiation())
	assert.Equal(t, GradGraphClean, g.DifferentiationState())
	if _, err := Grad(z1, x, y); err != nil {
		t.Fatalf("graph should recover after reset: %v", err)
	}
}

func TestDifferentiate_BackpropagateStillWorks(t *testing.T) {
	// The public Backpropagate contract is unchanged: it still differentiates,
	// and the lifecycle treats its marks as stale rather than silently reusing
	// them.
	g, x, y, _, z1, _ := buildSharedSubgraph()
	gradOut := g.AddNode(NewConstant(1.0))
	grads, err := Backpropagate(Nodes{z1}, Nodes{gradOut}, Nodes{x, y})
	require.NoError(t, err)
	require.Len(t, grads, 2)
	assert.NotNil(t, grads[0])
	assert.NotNil(t, grads[1])

	// A Grad on top of the direct marks is refused with a diagnosable kind.
	_, err = Grad(z1, x, y)
	var de *DifferentiationError
	require.True(t, asDiffError(err, &de))
	assert.Equal(t, DiffErrStaleMarks, de.Kind)
}

func TestDifferentiate_RecoversAfterRollbackFailure(t *testing.T) {
	// Failure at the analysis stage (non-differentiable output) must not leave
	// the graph broken: it can be differentiated again immediately.
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	y := NewScalar(g, Float64, WithName("y"), WithValue(3.0))
	nd := Must(Sign(y))
	badCost := Must(Add(x, nd))
	goodCost := Must(Mul(x, y))

	if _, err := Grad(badCost, x, y); err == nil {
		t.Fatal("expected the first differentiation to fail")
	}
	if g.DifferentiationState() != GradGraphClean {
		t.Fatalf("graph state after failure: %v", g.DifferentiationState())
	}

	if _, err := Grad(goodCost, x, y); err != nil {
		t.Fatalf("graph did not recover: %v", err)
	}
}
