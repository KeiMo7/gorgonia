package gorgonia

/*
This file implements the lifecycle of symbolic differentiation on an
*ExprGraph.

Historically Gorgonia memoised the derivative of every node in
Node.deriv and never invalidated it. That made a second Grad() call on the
same graph return the first call's derivative, made Grad() of a different
cost sharing a subgraph fail to find gradients, and caused the graph to grow
by a fresh set of derivative nodes on every call.

The model implemented here is:

  - Grad() (and Differentiate()) are idempotent per (cost, wrt-set). Asking
    for the same pair twice returns the very same derivative nodes and does
    not touch the graph at all.
  - A differentiation is recorded as a "slot". Slots with disjoint footprints
    (the non-input nodes the cost depends on) may coexist; this is what keeps
    the multi-head TestMisha pattern working, where one VM run evaluates
    several independent costs. A request whose footprint or WRT set overlaps
    an existing slot evicts *all* slots first: the derivative nodes created by
    them are removed from the graph and the derivative marks are restored to
    their pre-differentiation values. This keeps the node/edge count bounded
    and guarantees the result is identical to differentiating on a freshly
    built graph.
  - Derivative marks that were not recorded by a slot (for example, marks left
    by a direct Backpropagate call) are treated as stale: Differentiate
    refuses with DiffErrStaleMarks and asks the caller to call
    ResetDifferentiation first, instead of silently consuming them.
  - Every DifferentiationError carries a machine-checkable Kind, so callers
    can tell "cost is not a scalar", "WRT is not an input", "WRT is not part
    of this graph" and "graph still carries marks from a previous, failed
    differentiation" apart. None of those degrade to "no gradient".

The locking discipline is simple: diffMu serialises all differentiation
lifecycle operations on one graph. It is held while the graph is mutated, so
concurrent Differentiate() calls on the same graph are serialised (the
per-session serialisation required by the gradient-buffer API is separate;
see gradient_pool.go).
*/

import (
	"fmt"
	"sort"
	"strings"
	"unsafe"

	"gorgonia.org/tensor"
)

// DiffErrorKind classifies a failure to differentiate. It exists so that
// callers can distinguish reasons without string matching, and specifically
// so that a failure never degrades into "no gradient".
type DiffErrorKind int

const (
	// DiffErrNone is the zero value; it never appears in an error.
	DiffErrNone DiffErrorKind = iota
	// DiffErrCostNil means a nil cost was supplied.
	DiffErrCostNil
	// DiffErrCostNotScalar means the cost is not a scalar node.
	DiffErrCostNotScalar
	// DiffErrCostNotInGraph means the cost is not attached to a graph.
	DiffErrCostNotInGraph
	// DiffErrWRTNil means a nil WRT node was supplied.
	DiffErrWRTNil
	// DiffErrWRTNotInput means the WRT node is not an input node.
	DiffErrWRTNotInput
	// DiffErrWRTNotInGraph means the WRT node does not belong to the cost's graph.
	DiffErrWRTNotInGraph
	// DiffErrUnsupportedDtype means the cost dtype cannot be differentiated.
	DiffErrUnsupportedDtype
	// DiffErrNotDifferentiable means symbolic differentiation itself failed.
	DiffErrNotDifferentiable
	// DiffErrGraphMarked means the graph is explicitly marked as
	// un-differentiable because a previous differentiation could not be
	// rolled back cleanly.
	DiffErrGraphMarked
	// DiffErrStaleMarks means the graph carries derivative marks that were not
	// created by Differentiate (typically a direct Backpropagate call).
	// Consuming them would return the previous differentiation's result, so
	// the caller must call ResetDifferentiation (or ClearDerivativeMarks)
	// first.
	DiffErrStaleMarks
)

func (k DiffErrorKind) String() string {
	switch k {
	case DiffErrNone:
		return "none"
	case DiffErrCostNil:
		return "cost-not-provided"
	case DiffErrCostNotScalar:
		return "cost-not-scalar"
	case DiffErrCostNotInGraph:
		return "cost-not-in-graph"
	case DiffErrWRTNil:
		return "wrt-not-provided"
	case DiffErrWRTNotInput:
		return "wrt-not-input"
	case DiffErrWRTNotInGraph:
		return "wrt-not-in-graph"
	case DiffErrUnsupportedDtype:
		return "unsupported-dtype"
	case DiffErrNotDifferentiable:
		return "not-differentiable"
	case DiffErrGraphMarked:
		return "graph-marked"
	case DiffErrStaleMarks:
		return "stale-marks"
	default:
		return fmt.Sprintf("DiffErrorKind(%d)", int(k))
	}
}

// DifferentiationError is the error type returned by Differentiate() and
// Grad() when differentiation cannot be carried out. Use
// errors.As(err, &derivErr) to inspect Kind, Cost and Nodes.
type DifferentiationError struct {
	Kind   DiffErrorKind
	Cost   *Node
	Nodes  Nodes
	Reason string
	Err    error
}

func (e *DifferentiationError) Error() string {
	var buf strings.Builder
	buf.WriteString("differentiation failed")
	if e.Kind != DiffErrNone {
		fmt.Fprintf(&buf, " (%s)", e.Kind)
	}
	if e.Reason != "" {
		fmt.Fprintf(&buf, ": %s", e.Reason)
	}
	if e.Err != nil {
		fmt.Fprintf(&buf, ": %v", e.Err)
	}
	if len(e.Nodes) > 0 {
		fmt.Fprintf(&buf, " [nodes: %v]", e.Nodes)
	}
	return buf.String()
}

// Unwrap allows errors.Is/errors.As to reach the underlying cause.
func (e *DifferentiationError) Unwrap() error { return e.Err }

func diffErr(kind DiffErrorKind, cost *Node, ns Nodes, err error, format string, args ...interface{}) *DifferentiationError {
	return &DifferentiationError{
		Kind:   kind,
		Cost:   cost,
		Nodes:  ns,
		Reason: fmt.Sprintf(format, args...),
		Err:    err,
	}
}

// GradGraphState describes the differentiation lifecycle state of a graph.
type GradGraphState int

const (
	// GradGraphClean means no differentiation is currently recorded on the graph.
	GradGraphClean GradGraphState = iota
	// GradGraphDifferentiated means one or more differentiation slots are
	// recorded and the graph can be differentiated again.
	GradGraphDifferentiated
	// GradGraphBroken means a differentiation failed and could not be rolled
	// back; the graph refuses further differentiation until
	// ResetDifferentiation() is called.
	GradGraphBroken
)

func (s GradGraphState) String() string {
	switch s {
	case GradGraphClean:
		return "clean"
	case GradGraphDifferentiated:
		return "differentiated"
	case GradGraphBroken:
		return "broken"
	default:
		return fmt.Sprintf("GradGraphState(%d)", int(s))
	}
}

// differentiationSlot records one successful differentiation so that it can
// be either re-activated (cache hit) or evicted without leaving marks behind.
type differentiationSlot struct {
	key       string
	cost      *Node
	wrt       Nodes
	grads     Nodes
	created   Nodes           // nodes appended to the graph by this slot
	derivs    map[*Node]*Node // full Node.deriv value after this slot differentiated
	derivOfs  map[*Node]Nodes // full Node.derivOf value after this slot differentiated
	footprint map[*Node]struct{}
}

// differentiationFootprint returns the set of non-input, non-constant nodes
// that differentiating cost reads or writes. Two differentiations whose
// footprints intersect cannot share a graph without cross-contaminating each
// other's gradients.
func differentiationFootprint(cost *Node, wrt Nodes) map[*Node]struct{} {
	fp := make(map[*Node]struct{})
	var walk func(*Node)
	walk = func(n *Node) {
		if n == nil || n.isInput() || n.isConstant() {
			return
		}
		if _, seen := fp[n]; seen {
			return
		}
		fp[n] = struct{}{}
		for _, c := range n.children {
			walk(c)
		}
	}
	walk(cost)
	return fp
}

// staleMarksLocked returns nodes carrying a derivative mark that is not
// accounted for by the recorded differentiation slots.
func (g *ExprGraph) staleMarksLocked() Nodes {
	if len(g.diffSlots) > 0 {
		res := make(map[*Node]struct{})
		for _, slot := range g.diffSlots {
			for n := range slot.derivs {
				res[n] = struct{}{}
			}
		}
		var stale Nodes
		for _, n := range g.all {
			if n.deriv == nil {
				continue
			}
			if _, ok := res[n]; ok {
				continue
			}
			stale = append(stale, n)
		}
		return stale
	}
	var stale Nodes
	for _, n := range g.all {
		if n.deriv != nil {
			stale = append(stale, n)
		}
	}
	return stale
}

// ClearDerivativeMarks drops every derivative mark on the graph without
// touching values, types or connections. Use it to adopt a graph on which a
// direct Backpropagate (or SetDerivOf) previously set marks.
func (g *ExprGraph) ClearDerivativeMarks() {
	if g == nil {
		return
	}
	g.diffMu.Lock()
	defer g.diffMu.Unlock()
	g.clearDerivsLocked()
	g.restoreDerivOfsLocked(nil)
}

func (g *ExprGraph) conflictLocked(wrt Nodes, footprint map[*Node]struct{}) bool {
	want := make(map[*Node]struct{}, len(wrt))
	for _, n := range wrt {
		want[n] = struct{}{}
	}
	for _, slot := range g.diffSlots {
		for _, n := range slot.wrt {
			if _, ok := want[n]; ok {
				return true
			}
		}
		for n := range slot.footprint {
			if _, ok := footprint[n]; ok {
				return true
			}
		}
	}
	return false
}

// Differentiate is the lifecycle-aware form of Grad. It computes the
// symbolic gradient of the scalar cost with regards to the given input nodes.
//
// Unlike a bare Backpropagate call it is safe to call repeatedly on the same
// graph: the result is memoised per (cost, wrt-set), obsolete derivative
// subtrees are removed when a conflicting differentiation is requested, and
// a failure rolls the graph back to a re-differentiable state.
func Differentiate(cost *Node, WRTs ...*Node) (retVal Nodes, err error) {
	if cost == nil {
		return nil, diffErr(DiffErrCostNil, nil, nil, nil, "nil cost supplied")
	}
	if !cost.IsScalar() {
		return nil, diffErr(DiffErrCostNotScalar, cost, nil, nil, "expected cost to be a scalar, got %v with shape %v", cost, cost.Shape())
	}
	g := cost.g
	if g == nil {
		return nil, diffErr(DiffErrCostNotInGraph, cost, nil, nil, "cost %v is not bound to a graph", cost)
	}

	// Validate WRTs before touching any graph state.
	wrt := make(Nodes, len(WRTs))
	copy(wrt, WRTs)
	for i, n := range wrt {
		switch {
		case n == nil:
			return nil, diffErr(DiffErrWRTNil, cost, nil, nil, "nil WRT at position %d", i)
		case n.g != g:
			return nil, diffErr(DiffErrWRTNotInGraph, cost, Nodes{n}, nil, "WRT %d (%v) does not belong to the cost's graph", i, n)
		case !n.isInput():
			return nil, diffErr(DiffErrWRTNotInput, cost, Nodes{n}, nil, "can only differentiate with regards to input nodes; WRT %d (%v) is not an input", i, n)
		}
	}

	var dt tensor.Dtype
	var ok bool
	if dt, ok = cost.t.(tensor.Dtype); !ok {
		return nil, diffErr(DiffErrUnsupportedDtype, cost, nil, nil, "expected a scalar dtype for cost, got %v", cost.t)
	}
	if dt != Float64 && dt != Float32 {
		return nil, diffErr(DiffErrUnsupportedDtype, cost, nil, nil, "%s is not yet supported by differentiation", dt)
	}

	g.diffMu.Lock()
	defer g.diffMu.Unlock()

	if g.diffBrokenReason != "" {
		return nil, diffErr(DiffErrGraphMarked, cost, nil, nil, "graph is marked un-differentiable: %s", g.diffBrokenReason)
	}

	if g.diffSlots == nil {
		g.diffSlots = make(map[string]*differentiationSlot)
	}
	key := differentiationKey(cost, wrt)
	if slot, ok := g.diffSlots[key]; ok {
		g.activateSlotLocked(slot)
		return slot.grads.clone(), nil
	}

	// Marks that this graph's lifecycle did not create are stale. Consuming
	// them is the historical bug (a second differentiation returns the first
	// one's derivative), so refuse and name the reason instead of silently
	// overwriting user state.
	if stale := g.staleMarksLocked(); len(stale) > 0 {
		return nil, diffErr(DiffErrStaleMarks, cost, stale, nil,
			"graph carries derivative marks not created by Differentiate; call ResetDifferentiation() first")
	}

	// Derivative marks (Node.deriv / Node.derivOf) and the gradient
	// accumulation performed by the tape machine are graph-global. Two
	// differentiations can therefore only coexist when their dependency
	// subgraphs are disjoint: otherwise a single VM run would add the
	// gradients of both costs together (and a second Grad() on a shared
	// subgraph would reuse the first one's marks). Disjoint heads (the
	// multi-output case, see TestMisha) remain supported because no node is
	// shared. Everything else evicts the previous differentiation first, so
	// the result is exactly what a freshly built graph would produce and the
	// node/edge count stays at one differentiation's worth. Repeating the
	// same request is a cache hit above and allocates nothing at all.
	footprint := differentiationFootprint(cost, wrt)
	if g.conflictLocked(wrt, footprint) {
		g.clearSlotsLocked()
	}
	if len(g.diffSlots) == 0 {
		g.captureBaseLocked()
	}

	startLen := len(g.all)
	startRoots := g.roots
	startDerivs := g.snapshotDerivsLocked()
	startDerivOfs := g.snapshotDerivOfsLocked()
	startActive := g.diffActive
	createdByThisCall := func() Nodes { return append(Nodes(nil), g.all[startLen:]...) }

	rollback := func() {
		defer func() {
			if r := recover(); r != nil {
				g.diffBrokenReason = fmt.Sprintf("rollback panicked: %v", r)
			}
		}()
		removed := createdByThisCall()
		g.removeCreatedLocked(removed)
		g.clearDerivsLocked()
		for n, d := range startDerivs {
			n.deriv = d
		}
		g.restoreDerivOfsLocked(startDerivOfs)
		g.roots = startRoots
		g.diffActive = startActive
	}

	g.clearDerivsLocked()

	var gradOut *Node
	switch dt {
	case Float64:
		gradOut = onef64
	case Float32:
		gradOut = onef32
	}
	gradOut = g.AddNode(gradOut)

	grads, berr := backpropagate(Nodes{cost}, Nodes{gradOut}, wrt, false)
	if berr != nil {
		rollback()
		if g.diffBrokenReason != "" {
			return nil, diffErr(DiffErrGraphMarked, cost, wrt, berr, "graph could not be rolled back: %s", g.diffBrokenReason)
		}
		return nil, diffErr(DiffErrNotDifferentiable, cost, wrt, berr, "symbolic differentiation failed")
	}

	// The returned gradient list must be complete. A missing entry here would
	// otherwise surface far away as "no gradient".
	if len(grads) != len(wrt) {
		rollback()
		return nil, diffErr(DiffErrNotDifferentiable, cost, wrt, nil, "differentiation returned %d gradients for %d WRT nodes", len(grads), len(wrt))
	}
	for i, gr := range grads {
		if gr == nil {
			rollback()
			return nil, diffErr(DiffErrNotDifferentiable, cost, Nodes{wrt[i]}, nil, "differentiation returned a nil gradient for WRT %v", wrt[i])
		}
	}

	slot := &differentiationSlot{
		key:       key,
		cost:      cost,
		wrt:       append(Nodes(nil), wrt...),
		grads:     append(Nodes(nil), grads...),
		created:   createdByThisCall(),
		derivs:    g.snapshotDerivsLocked(),
		derivOfs:  g.snapshotDerivOfsLocked(),
		footprint: footprint,
	}
	g.diffSlots[key] = slot
	g.diffActive = slot
	return slot.grads.clone(), nil
}

// ResetDifferentiation removes every differentiation recorded on the graph:
// all derivative nodes created by Differentiate are detached from the graph,
// the derivative marks of pre-existing nodes are restored to their values
// before the first differentiation, and the graph is left clean and ready to
// be differentiated again. Values, types and connections of pre-existing
// nodes are not modified.
func (g *ExprGraph) ResetDifferentiation() error {
	if g == nil {
		return nil
	}
	g.diffMu.Lock()
	defer g.diffMu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			g.diffBrokenReason = fmt.Sprintf("reset panicked: %v", r)
		}
	}()
	g.clearSlotsLocked()
	return nil
}

// DifferentiationState reports whether the graph currently carries a
// differentiation, and whether it is still safe to differentiate.
func (g *ExprGraph) DifferentiationState() GradGraphState {
	if g == nil {
		return GradGraphClean
	}
	g.diffMu.Lock()
	defer g.diffMu.Unlock()
	switch {
	case g.diffBrokenReason != "":
		return GradGraphBroken
	case len(g.diffSlots) > 0:
		return GradGraphDifferentiated
	default:
		return GradGraphClean
	}
}

// DifferentiatedNodes returns the gradient nodes of the most recently
// activated differentiation, or nil if the graph is clean.
func (g *ExprGraph) DifferentiatedNodes() Nodes {
	if g == nil {
		return nil
	}
	g.diffMu.Lock()
	defer g.diffMu.Unlock()
	if g.diffActive == nil {
		return nil
	}
	return g.diffActive.grads.clone()
}

// DifferentiationBrokenReason returns the reason the graph was marked
// un-differentiable, or "" if it is usable.
func (g *ExprGraph) DifferentiationBrokenReason() string {
	if g == nil {
		return ""
	}
	g.diffMu.Lock()
	defer g.diffMu.Unlock()
	return g.diffBrokenReason
}

/* internal lifecycle helpers (all called with g.diffMu held) */

func differentiationKey(cost *Node, wrt Nodes) string {
	ptrs := make([]uintptr, len(wrt))
	for i, n := range wrt {
		ptrs[i] = uintptr(unsafe.Pointer(n))
	}
	sort.Slice(ptrs, func(i, j int) bool { return ptrs[i] < ptrs[j] })
	var buf strings.Builder
	fmt.Fprintf(&buf, "%x|", uintptr(unsafe.Pointer(cost)))
	for _, p := range ptrs {
		fmt.Fprintf(&buf, "%x,", p)
	}
	return buf.String()
}

// captureBaseLocked records derivative marks and roots as they were before the
// first differentiation of this graph, so ResetDifferentiation can restore them.
func (g *ExprGraph) captureBaseLocked() {
	if g.diffHaveBase {
		return
	}
	g.diffBaseDerivs = g.snapshotDerivsLocked()
	g.diffBaseDerivOfs = g.snapshotDerivOfsLocked()
	g.diffBaseRoots = g.roots
	g.diffHaveBase = true
}

func (g *ExprGraph) snapshotDerivOfsLocked() map[*Node]Nodes {
	ret := make(map[*Node]Nodes)
	for _, n := range g.all {
		if len(n.derivOf) > 0 {
			ret[n] = append(Nodes(nil), n.derivOf...)
		}
	}
	return ret
}

// restoreDerivOfsLocked puts every node's derivOf back exactly as it was in
// the snapshot. Nodes that gained a derivOf entry and were not in the snapshot
// get it cleared.
func (g *ExprGraph) restoreDerivOfsLocked(snap map[*Node]Nodes) {
	for _, n := range g.all {
		if prev, ok := snap[n]; ok {
			n.derivOf = append(Nodes(nil), prev...)
			continue
		}
		if len(n.derivOf) > 0 {
			n.derivOf = nil
		}
	}
}

func (g *ExprGraph) snapshotDerivsLocked() map[*Node]*Node {
	ret := make(map[*Node]*Node)
	for _, n := range g.all {
		if n.deriv != nil {
			ret[n] = n.deriv
		}
	}
	return ret
}

func (g *ExprGraph) clearDerivsLocked() {
	for _, n := range g.all {
		n.deriv = nil
	}
	for _, slot := range g.diffSlots {
		for _, n := range slot.created {
			n.deriv = nil
		}
	}
}

func (g *ExprGraph) activateSlotLocked(slot *differentiationSlot) {
	g.clearDerivsLocked()
	for n, d := range slot.derivs {
		n.deriv = d
	}
	g.restoreDerivOfsLocked(slot.derivOfs)
	g.diffActive = slot
}

func (g *ExprGraph) clearSlotsLocked() {
	var allCreated Nodes
	for _, slot := range g.diffSlots {
		allCreated = append(allCreated, slot.created...)
	}
	// De-duplicate: a node can only be created by one slot, but be defensive.
	seen := make(map[*Node]struct{}, len(allCreated))
	created := allCreated[:0]
	for _, n := range allCreated {
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		created = append(created, n)
	}

	g.removeCreatedLocked(created)

	// Any derivOf entry pointing at a removed node is now dangling.
	if len(created) > 0 {
		removed := make(map[*Node]struct{}, len(created))
		for _, n := range created {
			removed[n] = struct{}{}
		}
		for _, n := range g.all {
			if len(n.derivOf) == 0 {
				continue
			}
			derivOf := n.derivOf[:0]
			for _, d := range n.derivOf {
				if _, gone := removed[d]; gone {
					continue
				}
				derivOf = append(derivOf, d)
			}
			if len(derivOf) == 0 {
				n.derivOf = nil
			} else {
				n.derivOf = derivOf
			}
		}
	}

	g.diffSlots = make(map[string]*differentiationSlot)
	g.diffActive = nil

	// Restore marks and roots to the pre-first-differentiation state.
	g.clearDerivsLocked()
	if g.diffHaveBase {
		for n, d := range g.diffBaseDerivs {
			n.deriv = d
		}
		g.restoreDerivOfsLocked(g.diffBaseDerivOfs)
		if g.roots != nil || g.diffBaseRoots != nil {
			g.roots = g.diffBaseRoots
		}
	}
	g.diffBaseDerivs = nil
	g.diffBaseDerivOfs = nil
	g.diffBaseRoots = nil
	g.diffHaveBase = false
	g.diffBrokenReason = ""
}

// removeCreatedLocked detaches nodes created by differentiation from the
// graph. It deliberately does not touch the value, type or children of any
// pre-existing node.
func (g *ExprGraph) removeCreatedLocked(created Nodes) {
	if len(created) == 0 {
		return
	}
	removed := make(map[*Node]struct{}, len(created))
	for _, n := range created {
		removed[n] = struct{}{}
	}
	for _, n := range created {
		// Remove this node from the parent lists of its children (the edge
		// from n to each child is stored as an entry in g.to[child]).
		for _, child := range n.children {
			if tos, ok := g.to[child]; ok {
				g.to[child] = tos.remove(n)
			}
		}
		delete(g.to, n)

		h := n.Hashcode()
		if cur, ok := g.byHash[h]; ok && cur == n {
			delete(g.byHash, h)
		}
		if ev, ok := g.evac[h]; ok {
			ev = ev.remove(n)
			switch len(ev) {
			case 0:
				delete(g.evac, h)
				// If the only remaining candidate was n, byHash must not be
				// left pointing at a removed node.
				if cur, ok := g.byHash[h]; ok && cur == n {
					delete(g.byHash, h)
				}
			case 1:
				// Undo the "collision" marker AddNode set when n was added.
				g.evac[h] = ev
				if cur, ok := g.byHash[h]; !ok || cur == nil {
					g.byHash[h] = ev[0]
					delete(g.evac, h)
				}
			default:
				g.evac[h] = ev
			}
		}
		g.all = g.all.remove(n)
		g.leaves = g.leaves.remove(n)
		g.constants = g.constants.remove(n)
	}
	// byID caches indexes into g.all; removal invalidates it.
	g.byID = make(map[int64]int)
}

func (ns Nodes) clone() Nodes {
	if ns == nil {
		return nil
	}
	ret := make(Nodes, len(ns))
	copy(ret, ns)
	return ret
}
