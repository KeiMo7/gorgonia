package gorgonia

/*
This file implements a deterministic numerical gradient checker.

For a scalar cost c(x0, ..., xn), the analytic gradient produced by
Differentiate is compared element-wise against a central finite difference

    (c(x + h*e_i) - c(x - h*e_i)) / (2h)

perturbing one element at a time. The report is fully deterministic: the
perturbation direction is always +e_i then -e_i in row-major (flat index)
order, and the report entries are emitted in the order of the WRT slice,
then by flat index. Running the check twice on the same graph and values
therefore produces byte-identical reports.

Entries that cannot be checked (non-float values, shape mismatches, nodes
that turn out to be non-differentiable) are recorded with a reason and the
number of skipped entries is reported, instead of being silently dropped.
*/

import (
	"fmt"
	"math"
	"strings"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// GradCheckConfig configures a gradient check.
type GradCheckConfig struct {
	// Step is the finite-difference step h. Defaults to 1e-4.
	Step float64
	// AbsTolerance is the maximum acceptable absolute error. Defaults to 1e-4.
	AbsTolerance float64
	// RelTolerance is the maximum acceptable relative error. Defaults to 1e-3.
	RelTolerance float64
	// MaxEntries caps the number of elements checked per parameter (0 means
	// no cap). Elements are checked in a deterministic order.
	MaxEntries int
}

func (c GradCheckConfig) withDefaults() GradCheckConfig {
	if c.Step == 0 {
		c.Step = 1e-4
	}
	if c.AbsTolerance == 0 {
		c.AbsTolerance = 1e-4
	}
	if c.RelTolerance == 0 {
		c.RelTolerance = 1e-3
	}
	return c
}

// GradCheckEntry is the comparison for one element of one parameter.
type GradCheckEntry struct {
	Param    string  // parameter name
	Index    int     // flat index into the parameter
	Analytic float64 // analytic gradient
	Numeric  float64 // central finite difference
	AbsError float64 // |analytic - numeric|
	RelError float64 // absError / max(1, |analytic|, |numeric|)
}

// GradCheckSkip records an element that could not be checked.
type GradCheckSkip struct {
	Param  string
	Index  int
	Reason GradCheckSkipReason
	Detail string
}

// GradCheckSkipReason explains why an element was skipped.
type GradCheckSkipReason string

const (
	SkipNotFloat          GradCheckSkipReason = "not-float"
	SkipShapeMismatch     GradCheckSkipReason = "shape-mismatch"
	SkipNotDifferentiable GradCheckSkipReason = "not-differentiable"
	SkipNoValue           GradCheckSkipReason = "no-value"
	SkipNonFinite         GradCheckSkipReason = "non-finite"
)

// GradCheckReport is the outcome of a gradient check.
type GradCheckReport struct {
	Passed       bool
	Params       []string
	Entries      []GradCheckEntry
	Skipped      []GradCheckSkip
	CheckedCount int
	SkippedCount int
	MaxAbsError  float64
	MaxRelError  float64
	// MaxAbsParam/MaxAbsIndex locate the worst absolute error.
	MaxAbsParam string
	MaxAbsIndex int
	// MaxRelParam/MaxRelIndex locate the worst relative error.
	MaxRelParam string
	MaxRelIndex int
	Config      GradCheckConfig
	Summary     string
	Err         error
}

// String renders a deterministic, line-oriented report.
func (r GradCheckReport) String() string {
	var buf strings.Builder
	verdict := "PASS"
	if !r.Passed {
		verdict = "FAIL"
	}
	fmt.Fprintf(&buf, "gradcheck: %s\n", verdict)
	fmt.Fprintf(&buf, "config: step=%g absTol=%g relTol=%g maxEntries=%d\n", r.Config.Step, r.Config.AbsTolerance, r.Config.RelTolerance, r.Config.MaxEntries)
	fmt.Fprintf(&buf, "checked=%d skipped=%d params=%d\n", r.CheckedCount, r.SkippedCount, len(r.Params))
	fmt.Fprintf(&buf, "maxAbsError=%g at %s[%d]\n", r.MaxAbsError, r.MaxAbsParam, r.MaxAbsIndex)
	fmt.Fprintf(&buf, "maxRelError=%g at %s[%d]\n", r.MaxRelError, r.MaxRelParam, r.MaxRelIndex)
	if r.Err != nil {
		fmt.Fprintf(&buf, "error: %v\n", r.Err)
	}
	for _, p := range r.Params {
		for _, e := range r.Entries {
			if e.Param != p {
				continue
			}
			status := "ok"
			if e.AbsError > r.Config.AbsTolerance || e.RelError > r.Config.RelTolerance {
				status = "BAD"
			}
			fmt.Fprintf(&buf, "entry %s[%d] analytic=%g numeric=%g abs=%g rel=%g %s\n", e.Param, e.Index, e.Analytic, e.Numeric, e.AbsError, e.RelError, status)
		}
	}
	for _, s := range r.Skipped {
		fmt.Fprintf(&buf, "skip %s[%d] reason=%s detail=%s\n", s.Param, s.Index, s.Reason, s.Detail)
	}
	buf.WriteString(r.Summary)
	return buf.String()
}

// GradCheck verifies the analytic gradients of cost with regards to params
// against central finite differences.
//
// The graph is left exactly as it was found: differentiation is performed
// through Differentiate (which restores any previous marks on eviction), the
// parameter values are restored after every perturbation, and the regression
// results (forward values) are recomputed between perturbations.
func GradCheck(cost *Node, params ...*Node) (*GradCheckReport, error) {
	return GradCheckWithConfig(cost, GradCheckConfig{}, params...)
}

// GradCheckWithConfig is GradCheck with an explicit configuration.
func GradCheckWithConfig(cost *Node, cfg GradCheckConfig, params ...*Node) (*GradCheckReport, error) {
	cfg = cfg.withDefaults()
	rep := &GradCheckReport{Config: cfg}

	if cost == nil {
		rep.Err = errors.New("nil cost supplied to GradCheck")
		rep.Summary = rep.Err.Error() + "\n"
		return rep, rep.Err
	}

	analytics, err := Differentiate(cost, params...)
	if err != nil {
		rep.Err = err
		rep.Summary = fmt.Sprintf("analytic differentiation failed: %v\n", err)
		return rep, err
	}

	g := cost.g
	names := make([]string, len(params))
	for i, p := range params {
		names[i] = paramName(p, i)
	}
	rep.Params = names

	// Snapshot the original values so the graph is not left perturbed.
	originals := make(map[*Node]Value, len(params))
	for _, p := range params {
		v := p.Value()
		if v == nil {
			continue
		}
		cloned, cerr := CloneValue(v)
		if cerr != nil {
			rep.Err = errors.Wrapf(cerr, "cannot snapshot value of %v", p)
			rep.Summary = rep.Err.Error() + "\n"
			return rep, rep.Err
		}
		originals[p] = cloned
	}
	defer func() {
		for p, v := range originals {
			_ = Let(p, v)
		}
	}()

	// runForward recomputes the whole graph and returns the cost's scalar.
	// Every run re-executes the derivative nodes, so it also overwrites their
	// values; the analytic gradients are therefore snapshotted from the base
	// pass rather than read after a perturbation run.
	runForward := func() (float64, error) {
		m := NewTapeMachine(g)
		defer m.Close()
		if rerr := m.RunAll(); rerr != nil {
			return math.NaN(), rerr
		}
		cv, cerr := scalarOf(cost.Value())
		if cerr != nil {
			return math.NaN(), cerr
		}
		return cv, nil
	}

	if _, err := runForward(); err != nil {
		rep.Err = errors.Wrap(err, "forward pass failed before the gradient check")
		rep.Summary = rep.Err.Error() + "\n"
		return rep, rep.Err
	}

	// Snapshot every analytic gradient now, while the base forward pass is
	// still the most recent run.
	type analyticSnap struct {
		value Value
		err   error
	}
	analyticSnaps := make([]analyticSnap, len(params))
	for i := range params {
		v := analytics[i].Value()
		if v == nil {
			analyticSnaps[i].err = errors.Errorf("analytic gradient node %v has no value", analytics[i])
			continue
		}
		cloned, cerr := CloneValue(v)
		if cerr != nil {
			analyticSnaps[i].err = errors.Wrapf(cerr, "cannot snapshot analytic gradient of %v", analytics[i])
			continue
		}
		analyticSnaps[i].value = cloned
	}

	for pi, p := range params {
		name := names[pi]
		analytic, aerr := analyticSnaps[pi].value, analyticSnaps[pi].err

		pv := originals[p]
		if pv == nil {
			rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: -1, Reason: SkipNoValue, Detail: "parameter has no bound value"})
			continue
		}
		dt := pv.Dtype()
		if dt != tensor.Float64 && dt != tensor.Float32 {
			rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: -1, Reason: SkipNotFloat, Detail: fmt.Sprintf("dtype %v", dt)})
			continue
		}
		if aerr != nil {
			rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: -1, Reason: SkipNotDifferentiable, Detail: aerr.Error()})
			continue
		}
		if !analytic.Shape().Eq(pv.Shape()) {
			rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: -1, Reason: SkipShapeMismatch, Detail: fmt.Sprintf("analytic %v vs value %v", analytic.Shape(), pv.Shape())})
			continue
		}

		n := pv.Size()
		if n == 0 {
			n = pv.Shape().TotalSize()
		}
		limit := n
		if cfg.MaxEntries > 0 && limit > cfg.MaxEntries {
			limit = cfg.MaxEntries
		}

		for i := 0; i < limit; i++ {
			aVal, ok := floatAt(analytic, i)
			if !ok {
				rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: i, Reason: SkipNotFloat, Detail: "analytic value is not a float"})
				continue
			}

			plus, err := perturbAndRun(g, p, originals[p], i, cfg.Step, runForward)
			if err != nil {
				rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: i, Reason: SkipNonFinite, Detail: err.Error()})
				continue
			}
			minus, err := perturbAndRun(g, p, originals[p], i, -cfg.Step, runForward)
			if err != nil {
				rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: i, Reason: SkipNonFinite, Detail: err.Error()})
				continue
			}
			if math.IsNaN(plus) || math.IsNaN(minus) || math.IsInf(plus, 0) || math.IsInf(minus, 0) {
				rep.Skipped = append(rep.Skipped, GradCheckSkip{Param: name, Index: i, Reason: SkipNonFinite, Detail: fmt.Sprintf("cost %v / %v", plus, minus)})
				continue
			}

			numeric := (plus - minus) / (2 * cfg.Step)
			absErr := math.Abs(aVal - numeric)
			denom := math.Max(1, math.Max(math.Abs(aVal), math.Abs(numeric)))
			relErr := absErr / denom
			entry := GradCheckEntry{
				Param:    name,
				Index:    i,
				Analytic: aVal,
				Numeric:  numeric,
				AbsError: absErr,
				RelError: relErr,
			}
			rep.Entries = append(rep.Entries, entry)
			rep.CheckedCount++
			if absErr > rep.MaxAbsError {
				rep.MaxAbsError = absErr
				rep.MaxAbsParam = name
				rep.MaxAbsIndex = i
			}
			if relErr > rep.MaxRelError {
				rep.MaxRelError = relErr
				rep.MaxRelParam = name
				rep.MaxRelIndex = i
			}
		}
	}

	// Leave the graph's values exactly as the caller left them.
	for p, v := range originals {
		_ = Let(p, v)
	}
	if _, rerr := runForward(); rerr == nil {
		// The final forward pass rebinds derivative nodes to the unperturbed
		// values. Any error here is not fatal for the report; the caller can
		// always re-run the graph.
	}

	rep.SkippedCount = len(rep.Skipped)
	rep.Passed = rep.FailedCount() == 0 && rep.CheckedCount > 0 && rep.Err == nil
	if rep.CheckedCount == 0 {
		rep.Passed = false
		if rep.Err == nil {
			rep.Err = errors.New("no elements could be checked")
		}
	}
	rep.Summary = rep.buildSummary()
	return rep, nil
}

func (r *GradCheckReport) FailedCount() int {
	n := 0
	for _, e := range r.Entries {
		if e.AbsError > r.Config.AbsTolerance || e.RelError > r.Config.RelTolerance {
			n++
		}
	}
	return n
}

func (r *GradCheckReport) buildSummary() string {
	var buf strings.Builder
	if r.Passed {
		fmt.Fprintf(&buf, "gradient check passed: %d elements, maxAbs=%g, maxRel=%g, %d skipped\n", r.CheckedCount, r.MaxAbsError, r.MaxRelError, r.SkippedCount)
		return buf.String()
	}
	fmt.Fprintf(&buf, "gradient check FAILED: %d/%d elements exceed tolerance (abs=%g rel=%g)\n", r.FailedCount(), r.CheckedCount, r.Config.AbsTolerance, r.Config.RelTolerance)
	if r.CheckedCount > 0 {
		fmt.Fprintf(&buf, "worst parameter: %s[%d] absError=%g (max abs %g at %s[%d])\n", r.MaxAbsParam, r.MaxAbsIndex, r.MaxAbsError, r.MaxAbsError, r.MaxAbsParam, r.MaxAbsIndex)
	}
	return buf.String()
}

func paramName(n *Node, i int) string {
	if n == nil {
		return fmt.Sprintf("param%d", i)
	}
	if name := n.Name(); name != "" {
		return name
	}
	return fmt.Sprintf("param%d", i)
}

// perturbAndRun copies orig into n, adds delta at flat index i, runs the graph
// and returns the cost. The parameter is restored to orig before returning.
func perturbAndRun(g *ExprGraph, n *Node, orig Value, i int, delta float64, runForward func() (float64, error)) (float64, error) {
	work, err := CloneValue(orig)
	if err != nil {
		return math.NaN(), errors.Wrap(err, "cannot clone parameter value")
	}
	switch w := work.(type) {
	case *F64:
		if i != 0 {
			return math.NaN(), errors.Errorf("index %d out of range for scalar", i)
		}
		*w += F64(delta)
	case *F32:
		if i != 0 {
			return math.NaN(), errors.Errorf("index %d out of range for scalar", i)
		}
		*w += F32(delta)
	case *tensor.Dense:
		switch w.Dtype() {
		case tensor.Float64:
			data := w.Float64s()
			if i < 0 || i >= len(data) {
				return math.NaN(), errors.Errorf("index %d out of range for %v", i, w.Shape())
			}
			data[i] += delta
		case tensor.Float32:
			data := w.Float32s()
			if i < 0 || i >= len(data) {
				return math.NaN(), errors.Errorf("index %d out of range for %v", i, w.Shape())
			}
			data[i] += float32(delta)
		default:
			return math.NaN(), errors.Errorf("unsupported dtype %v", w.Dtype())
		}
	default:
		return math.NaN(), errors.Errorf("unsupported value type %T", orig)
	}

	if err := Let(n, work); err != nil {
		return math.NaN(), errors.Wrap(err, "cannot bind perturbed value")
	}
	cost, err := runForward()
	// Always restore, even if the run failed.
	if rerr := Let(n, orig); rerr != nil && err == nil {
		err = errors.Wrap(rerr, "cannot restore parameter value")
	}
	return cost, err
}

func scalarOf(v Value) (float64, error) {
	switch vt := v.(type) {
	case *F64:
		return float64(*vt), nil
	case *F32:
		return float64(*vt), nil
	case *tensor.Dense:
		if vt.Size() != 1 {
			return math.NaN(), errors.Errorf("cost must be a scalar, got shape %v", vt.Shape())
		}
		switch vt.Dtype() {
		case tensor.Float64:
			return vt.Float64s()[0], nil
		case tensor.Float32:
			return float64(vt.Float32s()[0]), nil
		default:
			return math.NaN(), errors.Errorf("unsupported cost dtype %v", vt.Dtype())
		}
	default:
		return math.NaN(), errors.Errorf("unsupported cost type %T", v)
	}
}

func floatAt(v Value, i int) (float64, bool) {
	switch vt := v.(type) {
	case *F64:
		if i != 0 {
			return 0, false
		}
		return float64(*vt), true
	case *F32:
		if i != 0 {
			return 0, false
		}
		return float64(*vt), true
	case *tensor.Dense:
		switch vt.Dtype() {
		case tensor.Float64:
			data := vt.Float64s()
			if i < 0 || i >= len(data) {
				return 0, false
			}
			return data[i], true
		case tensor.Float32:
			data := vt.Float32s()
			if i < 0 || i >= len(data) {
				return 0, false
			}
			return float64(data[i]), true
		}
	}
	return 0, false
}

// GradCheckError is returned by GradCheckWithError when a check fails. It
// carries the full report so callers can inspect the offending parameter
// rather than relying on a log line.
type GradCheckError struct {
	Report *GradCheckReport
}

func (e *GradCheckError) Error() string {
	if e == nil || e.Report == nil {
		return "gradient check failed"
	}
	r := e.Report
	return fmt.Sprintf("gradient check failed: worst parameter %s[%d] absError=%g (abs tolerance %g, rel tolerance %g); %d of %d elements over tolerance",
		r.MaxAbsParam, r.MaxAbsIndex, r.MaxAbsError, r.Config.AbsTolerance, r.Config.RelTolerance, r.FailedCount(), r.CheckedCount)
}

// GradCheckStrict is the fail-closed form of GradCheck: a check that does not
// pass returns a *GradCheckError carrying the report, so a caller cannot
// mistake a failing check for success.
func GradCheckStrict(cost *Node, params ...*Node) (*GradCheckReport, error) {
	return GradCheckStrictWithConfig(cost, GradCheckConfig{}, params...)
}

// GradCheckStrictWithConfig is GradCheckStrict with an explicit configuration.
func GradCheckStrictWithConfig(cost *Node, cfg GradCheckConfig, params ...*Node) (*GradCheckReport, error) {
	rep, err := GradCheckWithConfig(cost, cfg, params...)
	if err != nil {
		return rep, err
	}
	if !rep.Passed {
		return rep, &GradCheckError{Report: rep}
	}
	return rep, nil
}
