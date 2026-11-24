package gorgonia

/*
This file integrates differentiation with a training loop.

Two problems are solved:

 1. Gradient buffers are not automatically clean at the start of each
    iteration. Historically each solver zeroed its parameters' gradients at
    the *end* of its Step. If the step (or the run before it) failed, the
    residual of the previous iteration was added to the next one. A
    TrainingSession owns a pool and guarantees that every parameter's
    gradient is zeroed before the forward/backward pass begins.

 2. A solver that fails half-way through the model leaves the parameters it
    had not reached with an unwritten gradient. SolverStepSafe marks those
    parameters dirty and zeroes whatever gradient value is reachable, so the
    next iteration starts from zero rather than from a stale residual.
    SolverStep keeps its historical behaviour for existing call sites.

The "zero at take-out" rule is honoured here: the session zeroes gradients at
the point it borrows the buffers for an iteration, never after the solver has
consumed them.
*/

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// ErrSolverStep marks a solver failure that was made safe with
// SolverStepSafe.
type ErrSolverStep struct {
	Solver string
	Node   *Node
	Err    error
}

func (e *ErrSolverStep) Error() string {
	name := e.Solver
	if name == "" {
		name = "solver"
	}
	if e.Node != nil {
		return errors.Wrapf(e.Err, "%s failed on %v; gradient marked dirty", name, e.Node).Error()
	}
	return errors.Wrapf(e.Err, "%s failed; gradients marked dirty", name).Error()
}

func (e *ErrSolverStep) Unwrap() error { return e.Err }

// IsGradDirty reports whether n's gradient has been invalidated by a failed
// solver step.
func (n *Node) IsGradDirty() bool { return n != nil && n.gradDirty }

// MarkGradDirty marks n's gradient as stale and zeroes whatever gradient
// value is currently reachable. The next differentiation will start from a
// zeroed gradient, so a residual can never be accumulated.
func MarkGradDirty(n *Node) error {
	if n == nil {
		return errors.New("cannot mark a nil node's gradient dirty")
	}
	n.gradDirty = true
	return zeroGradValue(n)
}

// ZeroGrad zeros the gradient of n in place. It returns an error if n has no
// gradient value at all; that is not treated as a failure by
// SolverStepSafe, because a dirty mark is enough.
func ZeroGrad(n *Node) error {
	if n == nil {
		return errors.New("cannot zero a nil node's gradient")
	}
	return zeroGradValue(n)
}

// ZeroNodeGradient zeroes only the derivative half of a node's dualValue.
// Unlike ZeroGrad it never allocates and never fails when the node has no
// derivative yet. It is the building block used by TrainingSession to clean
// gradients at the take-out point.
func ZeroNodeGradient(n *Node) {
	if n == nil {
		return
	}
	dv, ok := n.boundTo.(*dualValue)
	if !ok || dv == nil || dv.d == nil {
		return
	}
	switch d := dv.d.(type) {
	case Zeroer:
		d.Zero()
	case *F64:
		*d = 0
	case *F32:
		*d = 0
	case tensor.Tensor:
		d.Zero()
	}
}

var errNoGradValue = errors.New("no gradient value")

func zeroGradValue(n *Node) error {
	v, err := n.Grad()
	if err != nil {
		return errNoGradValue
	}
	if v == nil {
		return errNoGradValue
	}
	switch vt := v.(type) {
	case Zeroer:
		vt.Zero()
		return nil
	case *F64:
		*vt = 0
		return nil
	case *F32:
		*vt = 0
		return nil
	case tensor.Tensor:
		vt.Zero()
		return nil
	default:
		return errors.Errorf("cannot zero a gradient of type %T", v)
	}
}

// SolverStepSafe runs step and, if it fails, marks the gradient of every
// parameter the solver did not finish as dirty (and zeroes it where possible).
//
// The error is returned wrapped in *ErrSolverStep so callers can diagnose
// which parameter failed. Parameters that the solver has already updated keep
// their (valid, consumed) gradient; parameters it never reached are dirty.
func SolverStepSafe(step Solver, model []ValueGrad) error {
	if step == nil {
		return errors.New("nil solver supplied")
	}
	if len(model) == 0 {
		return nil
	}
	err := step.Step(model)
	if err == nil {
		return nil
	}

	// We do not know exactly where Step stopped, so be conservative: every
	// parameter in the model gets its gradient zeroed; the ones that were not
	// reached are additionally marked dirty.
	var first *Node
	for _, vg := range model {
		var n *Node
		if nn, ok := vg.(*Node); ok {
			n = nn
		}
		if first == nil && n != nil {
			first = n
		}
		_ = zeroGradValueNode(vg)
		if n != nil {
			n.gradDirty = true
		}
	}
	return &ErrSolverStep{Solver: solverName(step), Node: first, Err: err}
}

func zeroGradValueNode(vg ValueGrad) error {
	if n, ok := vg.(*Node); ok {
		return zeroGradValue(n)
	}
	g, err := vg.Grad()
	if err != nil || g == nil {
		return err
	}
	if z, ok := g.(Zeroer); ok {
		z.Zero()
	}
	return nil
}

func solverName(s Solver) string {
	if n, ok := s.(Namer); ok {
		return n.Name()
	}
	return ""
}

// TrainingSession ties a graph, a VM and a gradient pool together so that the
// invariant "gradients are clean at the start of every iteration" holds.
type TrainingSession struct {
	Graph  *ExprGraph
	VM     VM
	Pool   *GradientPool
	Params Nodes

	// buffers are held for the duration of one iteration and returned at
	// IterationEnd. They are retained while bound so the pool cannot reclaim
	// them mid-run.
	buffers []*GradBuffer
}

// NewTrainingSession creates a session. If pool is nil the process-wide
// default pool is used.
func NewTrainingSession(g *ExprGraph, vm VM, pool *GradientPool, params ...*Node) *TrainingSession {
	if pool == nil {
		pool = DefaultGradientPool()
	}
	return &TrainingSession{
		Graph:  g,
		VM:     vm,
		Pool:   pool,
		Params: append(Nodes(nil), params...),
	}
}

// BeginIteration prepares the session for a training iteration:
//   - any parameter whose gradient was marked dirty is zeroed and the mark is
//     cleared,
//   - every parameter's gradient is zeroed (this is the authoritative point;
//     never zero after the backward pass).
//
// It returns the number of parameters whose gradient was zeroed.
func (s *TrainingSession) BeginIteration() (int, error) {
	if s == nil {
		return 0, errors.New("nil training session")
	}
	zeroed := 0
	var firstErr error
	for _, p := range s.Params {
		if p == nil {
			continue
		}
		// A parameter with no gradient value yet has nothing to inherit, so
		// there is nothing to zero. This happens on the very first
		// iteration, before the VM has bound the dual values.
		if err := zeroGradValue(p); err != nil && !errors.Is(err, errNoGradValue) {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		p.gradDirty = false
		zeroed++
	}
	return zeroed, firstErr
}

// BorrowGradientBuffer borrows a zeroed buffer for a parameter's gradient
// shape. The buffer is retained until ReturnGradientBuffers is called.
func (s *TrainingSession) BorrowGradientBuffer(shape tensor.Shape, dt tensor.Dtype) (*GradBuffer, error) {
	if s == nil || s.Pool == nil {
		return nil, errors.New("nil training session or pool")
	}
	b, err := s.Pool.Get(shape, dt)
	if err != nil {
		return nil, err
	}
	if err := b.Retain(); err != nil {
		_ = s.Pool.Put(b)
		return nil, err
	}
	s.buffers = append(s.buffers, b)
	return b, nil
}

// ReturnGradientBuffers returns every buffer borrowed during the iteration.
// This is the authoritative release point; buffers cannot be returned while
// retained, so a double-return surfaces as an error rather than a corruption.
func (s *TrainingSession) ReturnGradientBuffers() error {
	if s == nil {
		return nil
	}
	var firstErr error
	for _, b := range s.buffers {
		if err := b.Release(); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := s.Pool.Put(b); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.buffers = nil
	return firstErr
}

// EndIteration zeroes the gradients the next iteration must not inherit and
// returns the borrowed buffers. Call it in a defer so a failed solver step
// does not leak buffers.
func (s *TrainingSession) EndIteration() error {
	if s == nil {
		return nil
	}
	var firstErr error
	for _, p := range s.Params {
		if p == nil {
			continue
		}
		if err := zeroGradValue(p); err != nil {
			p.gradDirty = true
			if firstErr == nil {
				firstErr = err
			}
		}
	}
	if err := s.ReturnGradientBuffers(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// Step runs the solver within the session, making a failure safe.
func (s *TrainingSession) Step(solver Solver) error {
	if s == nil {
		return errors.New("nil training session")
	}
	model := make([]ValueGrad, 0, len(s.Params))
	for _, p := range s.Params {
		if p != nil {
			model = append(model, p)
		}
	}
	return SolverStepSafe(solver, model)
}

// GradientStats returns the pool statistics for observability.
func (s *TrainingSession) GradientStats() GradientPoolStats {
	if s == nil || s.Pool == nil {
		return GradientPoolStats{}
	}
	return s.Pool.Stats()
}
