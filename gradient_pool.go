package gorgonia

/*
This file implements a reusable, observable pool of gradient buffers.

The pool exists because training loops allocate a same-shaped float buffer on
every backward pass. Buffers are grouped by (shape, dtype), reused across
iterations, and zeroed at the point of take-out (Get), never at the point of
use: zeroing on return would be too early if the caller still needs the value,
and zeroing after accumulation would wipe the term that was just added.

Ownership is governed by the pool only. Each Get hands out a fresh *GradBuffer
handle. When a handle is Put back:
  - the handle is marked released permanently (generation counter),
  - any later method call on the old handle returns ErrBufferReleased,
  - the underlying storage may be handed to another caller through a new
    handle.

A buffer can be Retain()ed while an execution machine or a solver cache needs
it; Put refuses to reclaim a retained buffer. Pinning is reference counted.

All state transitions happen under the pool's lock. Counters are stored as
plain numbers, so the statistics never keep a buffer alive.
*/

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

var (
	// ErrBufferReleased is returned when a handle is used after it has been
	// returned to its pool.
	ErrBufferReleased = errors.New("gradient buffer has been returned to its pool")
	// ErrBufferForeign is returned when a handle is returned to a pool other
	// than the one that issued it.
	ErrBufferForeign = errors.New("gradient buffer does not belong to this pool")
	// ErrBufferDoubleReturn is returned when Put is called twice for one handle.
	ErrBufferDoubleReturn = errors.New("gradient buffer has already been returned")
	// ErrBufferPinned is returned when Put is attempted while the buffer is
	// still retained by a machine or a solver cache.
	ErrBufferPinned = errors.New("gradient buffer is still retained by a machine or solver cache")
	// ErrBufferNotFloat is returned when a non-floating dtype is requested.
	ErrBufferNotFloat = errors.New("gradient buffers require a floating point dtype")
	// ErrPoolBusy is returned by TryGet when the pool is busy and the pool was
	// configured to reject concurrent take-out instead of serialising it.
	ErrPoolBusy = errors.New("gradient pool is busy")
)

// GradientBufferKey identifies a group of interchangeable buffers.
type GradientBufferKey struct {
	Shape string // tensor.Shape.String(), kept as a value (no buffer reference)
	Dtype tensor.Dtype
}

func keyOf(shape tensor.Shape, dt tensor.Dtype) GradientBufferKey {
	return GradientBufferKey{Shape: fmt.Sprintf("%v", shape), Dtype: dt}
}

func (k GradientBufferKey) String() string { return fmt.Sprintf("%v/%v", k.Dtype, k.Shape) }

// gradSlot is the pool-owned storage. Handles point at it but hide it behind a
// generation number so that stale handles cannot touch a reissued slot.
type gradSlot struct {
	t     tensor.Tensor
	shape tensor.Shape
	dtype tensor.Dtype
	inUse bool
	pins  int
	gen   uint64
	bytes int64
}

// GradBuffer is a single-owner handle to pooled gradient storage.
//
// The handle is only valid between Get and Put. After Put, Tensor, Zero,
// Retain and Release all return ErrBufferReleased.
type GradBuffer struct {
	pool     *GradientPool
	slot     *gradSlot
	gen      uint64
	key      GradientBufferKey
	released bool
}

// Key returns the grouping key of the buffer.
func (b *GradBuffer) Key() GradientBufferKey { return b.key }

// Shape returns the buffer's shape.
func (b *GradBuffer) Shape() tensor.Shape { return b.slot.shape.Clone() }

// Dtype returns the buffer's dtype.
func (b *GradBuffer) Dtype() tensor.Dtype { return b.key.Dtype }

// Tensor returns the underlying tensor, which is already zeroed. It returns
// ErrBufferReleased once the handle has been returned to the pool.
func (b *GradBuffer) Tensor() (tensor.Tensor, error) {
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	if err := b.validLocked(); err != nil {
		return nil, err
	}
	return b.slot.t, nil
}

// Zero sets every element to zero.
func (b *GradBuffer) Zero() error {
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	if err := b.validLocked(); err != nil {
		return err
	}
	b.slot.t.Zero()
	b.pool.stats.ExplicitZeros++
	return nil
}

// Retain marks the buffer as held by a machine or solver cache. A retained
// buffer cannot be returned to the pool.
func (b *GradBuffer) Retain() error {
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	if err := b.validLocked(); err != nil {
		return err
	}
	b.slot.pins++
	return nil
}

// Release drops one Retain. Releasing more times than retaining is an error.
func (b *GradBuffer) Release() error {
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	if err := b.validLocked(); err != nil {
		return err
	}
	if b.slot.pins == 0 {
		return errors.Errorf("gradient buffer %v released without a matching Retain", b.key)
	}
	b.slot.pins--
	return nil
}

// Pins returns the current retain count.
func (b *GradBuffer) Pins() int {
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	if b.validLocked() != nil {
		return 0
	}
	return b.slot.pins
}

func (b *GradBuffer) validLocked() error {
	if b.released || b.slot.gen != b.gen || !b.slot.inUse {
		return ErrBufferReleased
	}
	return nil
}

// GradientPoolStats is a snapshot of the pool's counters. It contains no
// buffer references, so taking a snapshot never extends a buffer's lifetime.
type GradientPoolStats struct {
	// New is the number of buffers allocated from scratch.
	New int
	// Hits is the number of times a free buffer was reused.
	Hits int
	// ZerosOnBorrow is the number of times a buffer was zeroed at take-out.
	ZerosOnBorrow int
	// ExplicitZeros is the number of explicit Zero() calls on handles.
	ExplicitZeros int
	// Returns is the number of successful Put calls.
	Returns int
	// Outstanding is the number of buffers currently checked out.
	Outstanding int
	// Free is the number of buffers currently idle in the pool.
	Free int
	// Buffers is Outstanding + Free.
	Buffers int
	// Bytes is the total backing storage currently held by the pool (idle
	// and checked out).
	Bytes int64
	// PeakOutstanding is the highest Outstanding value ever observed.
	PeakOutstanding int
	// PeakBytes is the highest Bytes value ever observed.
	PeakBytes int64
	// Groups is the number of (shape, dtype) groups currently held.
	Groups int
}

// GradientPool is a concurrency-safe pool of gradient buffers.
//
// The zero value is not usable; use NewGradientPool.
type GradientPool struct {
	mu sync.Mutex

	free  map[GradientBufferKey][]*gradSlot
	all   []*gradSlot
	stats GradientPoolStats

	// session serialises take-out when nonBlocking is false, and rejects it
	// with ErrPoolBusy when nonBlocking is true. It is a channel rather than
	// a sync.Mutex so that non-blocking acquisition works on the Go versions
	// this module declares support for.
	session     chan struct{}
	nonBlocking bool
}

// GradientPoolOption configures a GradientPool.
type GradientPoolOption func(*GradientPool)

// WithNonBlockingTakeOut makes Get reject concurrent take-out with ErrPoolBusy
// in the same pool instead of serialising it. The default is to serialise.
func WithNonBlockingTakeOut() GradientPoolOption {
	return func(p *GradientPool) { p.nonBlocking = true }
}

// NewGradientPool creates a gradient pool.
func NewGradientPool(opts ...GradientPoolOption) *GradientPool {
	p := &GradientPool{
		free:    make(map[GradientBufferKey][]*gradSlot),
		session: make(chan struct{}, 1),
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Get borrows a zeroed buffer of the requested shape and dtype.
//
// Take-out is serialised per pool by default. A pooled buffer is zeroed before
// it is handed out, which is the only safe point: after this call the caller
// owns the buffer and may accumulate into it.
func (p *GradientPool) Get(shape tensor.Shape, dt tensor.Dtype) (*GradBuffer, error) {
	return p.get(shape, dt, false)
}

// TryGet is Get, but fails with ErrPoolBusy rather than waiting when another
// take-out is in progress and the pool was created WithNonBlockingTakeOut.
func (p *GradientPool) TryGet(shape tensor.Shape, dt tensor.Dtype) (*GradBuffer, error) {
	return p.get(shape, dt, true)
}

func (p *GradientPool) get(shape tensor.Shape, dt tensor.Dtype, try bool) (*GradBuffer, error) {
	switch dt {
	case tensor.Float64, tensor.Float32:
	default:
		return nil, errors.Wrapf(ErrBufferNotFloat, "requested %v", dt)
	}

	if !p.acquireSession(try) {
		return nil, ErrPoolBusy
	}
	defer p.releaseSession()

	key := keyOf(shape, dt)

	p.mu.Lock()
	defer p.mu.Unlock()

	var slot *gradSlot
	if free := p.free[key]; len(free) > 0 {
		slot = free[len(free)-1]
		p.free[key] = free[:len(free)-1]
		slot.t.Zero()
		p.stats.Hits++
		p.stats.ZerosOnBorrow++
	} else {
		t := tensor.New(tensor.WithShape(shape.Clone()...), tensor.Of(dt))
		slot = &gradSlot{
			t:     t,
			shape: shape.Clone(),
			dtype: dt,
			bytes: int64(t.MemSize()),
		}
		p.all = append(p.all, slot)
		p.stats.New++
		p.stats.ZerosOnBorrow++
		p.stats.Bytes += slot.bytes
		if p.stats.Bytes > p.stats.PeakBytes {
			p.stats.PeakBytes = p.stats.Bytes
		}
	}

	slot.gen++
	slot.inUse = true
	slot.pins = 0
	p.stats.Outstanding++
	if p.stats.Outstanding > p.stats.PeakOutstanding {
		p.stats.PeakOutstanding = p.stats.Outstanding
	}
	if p.stats.Bytes > p.stats.PeakBytes {
		p.stats.PeakBytes = p.stats.Bytes
	}

	return &GradBuffer{pool: p, slot: slot, gen: slot.gen, key: key}, nil
}

// Put returns a buffer to the pool. It is an error to return a foreign
// handle, the same handle twice, or a buffer that is still retained.
//
// The handle is invalidated: its Tensor() will return ErrBufferReleased, and
// the storage may be handed to another caller via a different handle.
func (p *GradientPool) Put(b *GradBuffer) error {
	if b == nil {
		return errors.New("cannot return a nil gradient buffer")
	}
	if b.pool != p {
		return ErrBufferForeign
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if b.released || b.slot.gen != b.gen || !b.slot.inUse {
		return ErrBufferDoubleReturn
	}
	if b.slot.pins > 0 {
		return errors.Wrapf(ErrBufferPinned, "%v still has %d retain(s)", b.key, b.slot.pins)
	}

	b.released = true
	b.slot.inUse = false
	b.slot.gen++ // invalidate every outstanding handle to this slot
	p.stats.Returns++
	p.stats.Outstanding--
	p.free[b.key] = append(p.free[b.key], b.slot)
	return nil
}

// Stats returns a snapshot of the pool counters.
func (p *GradientPool) Stats() GradientPoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.stats
	s.Free = 0
	for _, free := range p.free {
		s.Free += len(free)
	}
	keys := make(map[GradientBufferKey]struct{})
	for _, slot := range p.all {
		keys[keyOf(slot.shape, slot.dtype)] = struct{}{}
	}
	s.Buffers = s.Outstanding + s.Free
	s.Groups = len(keys)
	return s
}

// Release discards every idle buffer. Checked-out buffers are untouched; they
// become ordinary garbage once returned. Statistics keep their historical
// counters.
func (p *GradientPool) Release() {
	p.mu.Lock()
	defer p.mu.Unlock()
	kept := p.all[:0]
	for _, slot := range p.all {
		if slot.inUse {
			kept = append(kept, slot)
		} else {
			p.stats.Bytes -= slot.bytes
		}
	}
	p.all = kept
	p.free = make(map[GradientBufferKey][]*gradSlot)
}

func (p *GradientPool) acquireSession(try bool) bool {
	if try || p.nonBlocking {
		select {
		case p.session <- struct{}{}:
			return true
		default:
			return false
		}
	}
	p.session <- struct{}{}
	return true
}

func (p *GradientPool) releaseSession() { <-p.session }

// GradBufferPoolFor returns a process-wide pool suitable for a training
// session. Sessions that must not share buffers should create their own with
// NewGradientPool.
var defaultGradientPool = NewGradientPool()

// DefaultGradientPool returns the process-wide gradient pool.
func DefaultGradientPool() *GradientPool { return defaultGradientPool }

func init() {
	// Guard against accidental changes to the public stats contract: the
	// snapshot must remain a plain value type.
	if reflect.TypeOf(GradientPoolStats{}).Kind() != reflect.Struct {
		panic("GradientPoolStats must stay a value type")
	}
}
