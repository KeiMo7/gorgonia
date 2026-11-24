package gorgonia

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestGradientPool_ReuseAndZeroOnBorrow(t *testing.T) {
	p := NewGradientPool()
	shape := tensor.Shape{3, 2}

	first, err := p.Get(shape, tensor.Float64)
	require.NoError(t, err)
	tt, err := first.Tensor()
	require.NoError(t, err)
	data := tt.Data().([]float64)
	for i := range data {
		data[i] = float64(i + 1) // dirty the buffer
	}
	require.NoError(t, p.Put(first))

	stats := p.Stats()
	assert.Equal(t, 1, stats.New)
	assert.Equal(t, 0, stats.Hits)

	second, err := p.Get(shape, tensor.Float64)
	require.NoError(t, err)
	tt2, err := second.Tensor()
	require.NoError(t, err)
	for i, v := range tt2.Data().([]float64) {
		assert.Equal(t, 0.0, v, "element %d was not zeroed at take-out", i)
	}

	stats = p.Stats()
	assert.Equal(t, 1, stats.New, "buffer must be reused, not reallocated")
	assert.Equal(t, 1, stats.Hits)
	assert.Equal(t, 2, stats.ZerosOnBorrow, "each take-out zeroes")
}

func TestGradientPool_GroupedByShapeAndDtype(t *testing.T) {
	p := NewGradientPool()
	a, err := p.Get(tensor.Shape{2, 2}, tensor.Float64)
	require.NoError(t, err)
	b, err := p.Get(tensor.Shape{4}, tensor.Float64)
	require.NoError(t, err)
	c, err := p.Get(tensor.Shape{2, 2}, tensor.Float32)
	require.NoError(t, err)

	require.NoError(t, p.Put(a))
	require.NoError(t, p.Put(b))
	require.NoError(t, p.Put(c))

	stats := p.Stats()
	assert.Equal(t, 3, stats.New)
	assert.Equal(t, 3, stats.Groups, "shape and dtype must form different groups")

	// A {{2},{2}} f64 request may only reuse the {{2},{2}} f64 buffer.
	d, err := p.Get(tensor.Shape{2, 2}, tensor.Float64)
	require.NoError(t, err)
	require.NoError(t, p.Put(d))
	assert.Equal(t, 1, p.Stats().Hits)
}

func TestGradientPool_ReleasedHandleIsInvalid(t *testing.T) {
	p := NewGradientPool()
	b, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	require.NoError(t, p.Put(b))

	_, err = b.Tensor()
	assert.ErrorIs(t, err, ErrBufferReleased)
	assert.ErrorIs(t, b.Zero(), ErrBufferReleased)
	assert.ErrorIs(t, b.Retain(), ErrBufferReleased)

	// Return twice is an error, not a corruption.
	assert.ErrorIs(t, p.Put(b), ErrBufferDoubleReturn)
}

func TestGradientPool_StaleHandleAfterReissue(t *testing.T) {
	p := NewGradientPool()
	old, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	require.NoError(t, p.Put(old))

	// Get the same storage again; the old handle must stay dead.
	fresh, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	_, err = old.Tensor()
	assert.ErrorIs(t, err, ErrBufferReleased, "a reissued slot must not validate an old handle")
	require.NoError(t, p.Put(fresh))
}

func TestGradientPool_RetainBlocksReturn(t *testing.T) {
	p := NewGradientPool()
	b, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	require.NoError(t, b.Retain())
	assert.Equal(t, 1, b.Pins())
	assert.ErrorIs(t, p.Put(b), ErrBufferPinned)

	require.NoError(t, b.Release())
	require.NoError(t, p.Put(b))

	// Releasing after return is an error.
	assert.ErrorIs(t, b.Release(), ErrBufferReleased)
}

func TestGradientPool_ExclusiveOwnership(t *testing.T) {
	// Two live handles must never share storage.
	p := NewGradientPool()
	a, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	b, err := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	ta, _ := a.Tensor()
	tb, _ := b.Tensor()
	require.NotEqual(t, ta.Uintptr(), tb.Uintptr(), "two live buffers share memory")
	assert.Equal(t, 2, p.Stats().Outstanding)
}

func TestGradientPool_StatsSnapshotHoldsNoBuffers(t *testing.T) {
	p := NewGradientPool()
	b, err := p.Get(tensor.Shape{8}, tensor.Float64)
	require.NoError(t, err)
	stats := p.Stats()
	assert.Equal(t, 1, stats.Outstanding)
	assert.Equal(t, int64(64), stats.Bytes)
	require.NoError(t, p.Put(b))
	// Taking stats after release does not resurrect or touch the buffer.
	stats = p.Stats()
	assert.Equal(t, 0, stats.Outstanding)
	assert.Equal(t, 1, stats.Free)
	assert.Equal(t, 1, stats.PeakOutstanding)
}

func TestGradientPool_FloatOnly(t *testing.T) {
	p := NewGradientPool()
	_, err := p.Get(tensor.Shape{2}, tensor.Int)
	assert.ErrorIs(t, err, ErrBufferNotFloat)
	_, err = p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
}

func TestGradientPool_ConcurrentUse(t *testing.T) {
	p := NewGradientPool()
	const workers = 16
	const iterations = 200
	var wg sync.WaitGroup
	errs := make([]error, workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				b, err := p.Get(tensor.Shape{4, 4}, tensor.Float64)
				if err != nil {
					errs[w] = err
					return
				}
				tt, err := b.Tensor()
				if err != nil {
					errs[w] = err
					return
				}
				// Every borrowed buffer must be zero on arrival.
				for _, v := range tt.Data().([]float64) {
					if v != 0 {
						errs[w] = ErrBufferReleased // reuse a sentinel; checked below
						return
					}
				}
				tt.Data().([]float64)[0] = float64(w*iterations + i)
				if err := p.Put(b); err != nil {
					errs[w] = err
					return
				}
			}
		}(w)
	}
	wg.Wait()

	for w, err := range errs {
		require.NoError(t, err, "worker %d", w)
	}
	stats := p.Stats()
	assert.Equal(t, 0, stats.Outstanding)
	assert.Equal(t, workers*iterations, stats.Returns)
	assert.LessOrEqual(t, stats.New, workers, "buffers were not reused under concurrency")
	assert.Greater(t, stats.PeakOutstanding, 0)
}

func TestGradientPool_TryGetRejectsWhenBusy(t *testing.T) {
	p := NewGradientPool(WithNonBlockingTakeOut())
	// Hold the session slot by borrowing through a blocking-only path: use a
	// second pool sharing the same guard is not possible, so simulate by
	// acquiring the channel directly (internal test).
	p.session <- struct{}{}
	_, err := p.TryGet(tensor.Shape{2}, tensor.Float64)
	assert.ErrorIs(t, err, ErrPoolBusy)
	<-p.session

	b, err := p.TryGet(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, err)
	require.NoError(t, p.Put(b))
}

func TestGradientPool_Release(t *testing.T) {
	p := NewGradientPool()
	a, _ := p.Get(tensor.Shape{2}, tensor.Float64)
	require.NoError(t, p.Put(a))
	b, _ := p.Get(tensor.Shape{2}, tensor.Float64) // reused
	require.NoError(t, p.Put(b))

	p.Release()
	stats := p.Stats()
	assert.Equal(t, int64(0), stats.Bytes)
	assert.Equal(t, 0, stats.Free)
}
