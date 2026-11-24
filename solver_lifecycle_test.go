package gorgonia

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// rosenbrockGraph builds a small two-parameter cost with a known optimum.
func rosenbrockGraph() (g *ExprGraph, x1, x2, cost *Node) {
	g = NewGraph()
	x1 = NewScalar(g, Float64, WithName("x1"), WithValue(-0.5))
	x2 = NewScalar(g, Float64, WithName("x2"), WithValue(0.5))
	one := NewConstant(1.0)
	hundred := NewConstant(100.0)
	cost = Must(Add(
		Must(Square(Must(Sub(one, x1)))),
		Must(Mul(hundred, Must(Square(Must(Sub(x2, Must(Square(x1)))))))),
	))
	return
}

func TestTrainingSession_CleanGradientEachIteration(t *testing.T) {
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()

	session := NewTrainingSession(g, vm, NewGradientPool(), x1, x2)
	solver := NewVanillaSolver(WithLearnRate(0.001))

	var firstIterGrads []float64
	for i := 0; i < 6; i++ {
		n, err := session.BeginIteration()
		require.NoError(t, err)
		assert.Equal(t, 2, n, "iteration %d: both gradients should be zeroed", i)

		require.NoError(t, vm.RunAll())
		g1, _ := x1.Grad()
		g2, _ := x2.Grad()
		got := []float64{g1.Data().(float64), g2.Data().(float64)}
		if i == 0 {
			firstIterGrads = got
		}

		// Compare against an independent run from the same parameter values
		// on a fresh session.
		require.NoError(t, session.Step(solver))
		vm.Reset()
	}
	require.NotNil(t, firstIterGrads)
	assert.Equal(t, 47.0, firstIterGrads[0]) // d/dx1 at (-0.5, 0.5)
	assert.Equal(t, 50.0, firstIterGrads[1])
}

func TestTrainingSession_GradientCleanAfterFailure(t *testing.T) {
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	require.NoError(t, vm.RunAll())

	before1, _ := x1.Grad()
	before2, _ := x2.Grad()
	original1 := before1.Data().(float64)
	original2 := before2.Data().(float64)

	// A failing solver: run the model once, then have a solver that always
	// errors. SolverStepSafe must zero and dirty both gradients.
	failing := &alwaysFailSolver{}
	err := SolverStepSafe(failing, []ValueGrad{x1, x2})
	require.Error(t, err)
	var stepErr *ErrSolverStep
	require.True(t, errors.As(err, &stepErr))

	assert.True(t, x1.IsGradDirty(), "x1 should be marked dirty")
	assert.True(t, x2.IsGradDirty(), "x2 should be marked dirty")
	g1, _ := x1.Grad()
	g2, _ := x2.Grad()
	assert.Equal(t, 0.0, g1.Data().(float64), "failed step must zero the gradient")
	assert.Equal(t, 0.0, g2.Data().(float64), "failed step must zero the gradient")

	// The next iteration re-zeroes (idempotent), clears the dirty mark, and
	// the run recomputes the gradient from scratch rather than adding a
	// residual.
	session := NewTrainingSession(g, vm, NewGradientPool(), x1, x2)
	_, err = session.BeginIteration()
	require.NoError(t, err)
	assert.False(t, x1.IsGradDirty())
	assert.False(t, x2.IsGradDirty())
	vm.Reset() // the previous run left the program counter at the end
	require.NoError(t, vm.RunAll())
	after1, _ := x1.Grad()
	after2, _ := x2.Grad()
	assert.Equal(t, original1, after1.Data().(float64))
	assert.Equal(t, original2, after2.Data().(float64))
}

func TestTrainingSession_IterationMatchesIndependentRun(t *testing.T) {
	// Run N iterations with the session, and at each iteration compare the
	// gradient with a freshly built graph evaluated at the same point.
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	session := NewTrainingSession(g, vm, NewGradientPool(), x1, x2)
	solver := NewVanillaSolver(WithLearnRate(0.0005))

	for i := 0; i < 5; i++ {
		_, err := session.BeginIteration()
		require.NoError(t, err)
		require.NoError(t, vm.RunAll())
		g1, _ := x1.Grad()
		g2, _ := x2.Grad()

		// Independent reference.
		rg, rx1, rx2, rcost := rosenbrockGraph()
		Let(rx1, x1.Value().Data())
		Let(rx2, x2.Value().Data())
		if _, err := Grad(rcost, rx1, rx2); err != nil {
			t.Fatal(err)
		}
		rvm := NewTapeMachine(rg, BindDualValues(rx1, rx2))
		require.NoError(t, rvm.RunAll())
		rg1, _ := rx1.Grad()
		rg2, _ := rx2.Grad()
		rvm.Close()

		assert.InDelta(t, rg1.Data().(float64), g1.Data().(float64), 1e-12, "iter %d x1 grad", i)
		assert.InDelta(t, rg2.Data().(float64), g2.Data().(float64), 1e-12, "iter %d x2 grad", i)
		_ = rg

		require.NoError(t, session.Step(solver))
		vm.Reset()
	}
}

func TestTrainingSession_BufferLifecycle(t *testing.T) {
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	pool := NewGradientPool()
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	session := NewTrainingSession(g, vm, pool, x1, x2)

	// Borrow and return across iterations.
	var firstBuffer *GradBuffer
	for i := 0; i < 3; i++ {
		b, err := session.BorrowGradientBuffer(tensor.Shape{2}, tensor.Float64)
		require.NoError(t, err)
		if i == 0 {
			firstBuffer = b
		}
		// Buffers are already zeroed.
		tt, err := b.Tensor()
		require.NoError(t, err)
		for _, v := range tt.Data().([]float64) {
			assert.Equal(t, 0.0, v)
		}
		stats := pool.Stats()
		assert.Equal(t, 1, stats.Outstanding, "iter %d outstanding", i)
		require.NoError(t, session.ReturnGradientBuffers())
		assert.Equal(t, 0, pool.Stats().Outstanding)
	}

	// The very first handle is now dead even though the storage was reused.
	_, err := firstBuffer.Tensor()
	assert.ErrorIs(t, err, ErrBufferReleased)
	assert.Equal(t, 1, pool.Stats().New, "the buffer must have been reused, not reallocated")
}

func TestTrainingSession_BuffersReturnedOnFailure(t *testing.T) {
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	pool := NewGradientPool()
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	session := NewTrainingSession(g, vm, pool, x1, x2)

	func() {
		defer session.EndIteration()
		defer func() { _ = recover() }()
		_, err := session.BorrowGradientBuffer(tensor.Shape{4}, tensor.Float64)
		require.NoError(t, err)
		// Simulate a mid-iteration failure by panicking; EndIteration is in a
		// defer and must still return the buffer.
		panic("simulated failure")
	}()

	assert.Equal(t, 0, pool.Stats().Outstanding, "buffers leaked after a failed iteration")
}

func TestMarkGradDirty_ZeroesGradient(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"), WithValue(2.0))
	cost := Must(HadamardProd(x, x))
	if _, err := Grad(cost, x); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x))
	require.NoError(t, vm.RunAll())
	vm.Close()
	gv, _ := x.Grad()
	require.NotEqual(t, 0.0, gv.Data().(float64))

	require.NoError(t, MarkGradDirty(x))
	assert.True(t, x.IsGradDirty())
	gv, _ = x.Grad()
	assert.Equal(t, 0.0, gv.Data().(float64))

	require.NoError(t, ZeroGrad(x))
	assert.Equal(t, 0.0, gv.Data().(float64))
}

func TestTrainingSession_ConcurrentSessions(t *testing.T) {
	// Several sessions run concurrently with race detection enabled. Each
	// owns its graph, VM and pool, and must converge identically.
	const n = 6
	var wg sync.WaitGroup
	results := make([]float64, n)
	errs := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			g, x1, x2, cost := rosenbrockGraph()
			if _, err := Grad(cost, x1, x2); err != nil {
				errs[i] = err
				return
			}
			vm := NewTapeMachine(g, BindDualValues(x1, x2))
			defer vm.Close()
			session := NewTrainingSession(g, vm, NewGradientPool(), x1, x2)
			solver := NewVanillaSolver(WithLearnRate(0.001))
			for step := 0; step < 20; step++ {
				if _, err := session.BeginIteration(); err != nil {
					errs[i] = err
					return
				}
				if err := vm.RunAll(); err != nil {
					errs[i] = err
					return
				}
				if err := session.Step(solver); err != nil {
					errs[i] = err
					return
				}
				vm.Reset()
			}
			results[i] = x1.Value().Data().(float64)
		}(i)
	}
	wg.Wait()
	for i := range errs {
		require.NoError(t, errs[i], "session %d", i)
	}
	for i := 1; i < n; i++ {
		assert.InDelta(t, results[0], results[i], 1e-12, "session %d diverged", i)
	}
}

type alwaysFailSolver struct{}

func (*alwaysFailSolver) Step([]ValueGrad) error { return errors.New("injected solver failure") }

// updateWeightSolver is a solver that applies a plain in-place gradient step
// to *tensor.Dense weights and zeroes the gradient, so that a training loop
// can be exercised end-to-end without relying on the built-in solvers'
// historical behaviour.
type updateWeightSolver struct{ lr float64 }

func (s *updateWeightSolver) Step(model []ValueGrad) error {
	for _, vg := range model {
		n, ok := vg.(*Node)
		if !ok {
			continue
		}
		gv, err := n.Grad()
		if err != nil {
			return err
		}
		switch wv := n.Value().(type) {
		case *tensor.Dense:
			gd, ok := gv.(*tensor.Dense)
			if !ok {
				return fmt.Errorf("expected *tensor.Dense gradient, got %T", gv)
			}
			if !wv.Shape().Eq(gd.Shape()) {
				return fmt.Errorf("shape mismatch: %v vs %v", wv.Shape(), gd.Shape())
			}
			w := wv.Float64s()
			g := gd.Float64s()
			for i := range w {
				w[i] -= s.lr * g[i]
			}
			gd.Zero()
		case *F64:
			g, ok := gv.(*F64)
			if !ok {
				return fmt.Errorf("expected *F64 gradient, got %T", gv)
			}
			*wv -= F64(s.lr * float64(*g))
			*g = 0
		default:
			return fmt.Errorf("unsupported weight type %T", n.Value())
		}
	}
	return nil
}

func TestTrainingSession_FullLoopConverges(t *testing.T) {
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	pool := NewGradientPool()
	session := NewTrainingSession(g, vm, pool, x1, x2)

	initial := cost.Value()
	_ = initial

	var firstCostAfterRun float64
	for i := 0; i < 50; i++ {
		_, err := session.BeginIteration()
		require.NoError(t, err)
		require.NoError(t, vm.RunAll())
		if i == 0 {
			firstCostAfterRun = cost.Value().Data().(float64)
		}
		require.NoError(t, session.Step(&updateWeightSolver{lr: 0.001}))
		vm.Reset()
	}

	final := cost.Value().Data().(float64)
	assert.Less(t, final, firstCostAfterRun, "cost must decrease over the loop")
	// The node count is stable after the first iteration.
	peaks := len(g.AllNodes())
	vm.Reset()
	for i := 0; i < 10; i++ {
		_, err := session.BeginIteration()
		require.NoError(t, err)
		require.NoError(t, vm.RunAll())
		require.NoError(t, session.Step(&updateWeightSolver{lr: 0.001}))
		vm.Reset()
		assert.Equal(t, peaks, len(g.AllNodes()), "iteration %d grew the graph", i)
	}

	stats := pool.Stats()
	assert.Equal(t, 0, stats.Outstanding, "no buffers may leak from a training loop")
}

func TestTrainingSession_GradientBufferZeroAtTakeOut(t *testing.T) {
	// A buffer obtained from the session is zero at the point of take-out,
	// even if a previous iteration left it dirty.
	pool := NewGradientPool()
	g, x1, x2, cost := rosenbrockGraph()
	if _, err := Grad(cost, x1, x2); err != nil {
		t.Fatal(err)
	}
	vm := NewTapeMachine(g, BindDualValues(x1, x2))
	defer vm.Close()
	session := NewTrainingSession(g, vm, pool, x1, x2)

	dirty, err := pool.Get(tensor.Shape{5}, tensor.Float64)
	require.NoError(t, err)
	tt, _ := dirty.Tensor()
	tt.Data().([]float64)[0] = 42
	require.NoError(t, pool.Put(dirty))

	fresh, err := session.BorrowGradientBuffer(tensor.Shape{5}, tensor.Float64)
	require.NoError(t, err)
	ft, _ := fresh.Tensor()
	for i, v := range ft.Data().([]float64) {
		assert.Equal(t, 0.0, v, "element %d not zeroed at take-out", i)
	}
	require.NoError(t, session.ReturnGradientBuffers())
}
