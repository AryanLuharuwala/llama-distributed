package main

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestHFResolveCoalescesSameKey is the headline guarantee: N concurrent
// callers for the same (repo, revision, token) tuple must produce exactly
// one underlying fn() invocation, and every caller must receive the
// winner's result.
func TestHFResolveCoalescesSameKey(t *testing.T) {
	c := newHFResolveCoalescer()

	var calls atomic.Int64
	gate := make(chan struct{})
	expected := []hfFileInfo{{Path: "model.gguf", Size: 12345}}
	fn := func(ctx context.Context) ([]hfFileInfo, bool, error) {
		calls.Add(1)
		<-gate // hold the winner so racers pile in
		return expected, false, nil
	}

	const N = 50
	results := make([]struct {
		files []hfFileInfo
		conv  bool
		err   error
	}, N)
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		i := i
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			f, conv, err := c.resolve(ctx, "TheBloke/Llama-2-7B-GGUF", "main", "tok", fn)
			results[i].files = f
			results[i].conv = conv
			results[i].err = err
		}()
	}
	// Give all goroutines a moment to enter resolve() and queue on the singleflight.
	time.Sleep(50 * time.Millisecond)
	close(gate)
	wg.Wait()

	if got := calls.Load(); got != 1 {
		t.Errorf("expected exactly 1 underlying fn() invocation, got %d", got)
	}
	for i, r := range results {
		if r.err != nil {
			t.Errorf("caller %d got err: %v", i, r.err)
			continue
		}
		if len(r.files) != 1 || r.files[0].Path != "model.gguf" {
			t.Errorf("caller %d got unexpected files: %+v", i, r.files)
		}
		if r.conv {
			t.Errorf("caller %d got convertible=true, want false", i)
		}
	}
}

// TestHFResolveDistinctKeysParallelize confirms that different repos do
// NOT block each other — the dedup must be key-scoped.
func TestHFResolveDistinctKeysParallelize(t *testing.T) {
	c := newHFResolveCoalescer()
	var inflight, peak atomic.Int64
	start := make(chan struct{})

	fn := func(ctx context.Context) ([]hfFileInfo, bool, error) {
		now := inflight.Add(1)
		// Track peak concurrency.
		for {
			p := peak.Load()
			if now <= p || peak.CompareAndSwap(p, now) {
				break
			}
		}
		<-start
		inflight.Add(-1)
		return []hfFileInfo{{Path: "x.gguf", Size: 1}}, false, nil
	}

	const N = 5
	done := make(chan struct{}, N)
	for i := 0; i < N; i++ {
		repo := "repo" + string(rune('A'+i))
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, _, _ = c.resolve(ctx, repo, "main", "tok", fn)
			done <- struct{}{}
		}()
	}
	// Wait for all goroutines to enter fn().
	time.Sleep(50 * time.Millisecond)
	if got := peak.Load(); got < N {
		t.Errorf("distinct keys should run in parallel; peak inflight=%d, want >=%d", got, N)
	}
	close(start)
	for i := 0; i < N; i++ {
		<-done
	}
}

// TestHFResolveErrorNotCached confirms that an error from one resolve
// does not poison subsequent calls — each new fan-in should re-run fn().
func TestHFResolveErrorNotCached(t *testing.T) {
	c := newHFResolveCoalescer()
	var calls atomic.Int64
	bang := errors.New("transient HF flake")

	fn := func(ctx context.Context) ([]hfFileInfo, bool, error) {
		n := calls.Add(1)
		if n == 1 {
			return nil, false, bang
		}
		return []hfFileInfo{{Path: "ok.gguf", Size: 9}}, false, nil
	}
	ctx := context.Background()

	if _, _, err := c.resolve(ctx, "TheBloke/Foo", "main", "tok", fn); !errors.Is(err, bang) {
		t.Errorf("first call: want %v, got %v", bang, err)
	}
	files, _, err := c.resolve(ctx, "TheBloke/Foo", "main", "tok", fn)
	if err != nil {
		t.Errorf("second call should have succeeded, got: %v", err)
	}
	if len(files) != 1 || files[0].Path != "ok.gguf" {
		t.Errorf("second call: want one file 'ok.gguf', got %+v", files)
	}
}

// TestHFResolveLoserContextDoesNotCancelWinner ensures a caller that
// gives up does not tear down the work that another caller is still
// waiting for.
func TestHFResolveLoserContextDoesNotCancelWinner(t *testing.T) {
	c := newHFResolveCoalescer()
	hold := make(chan struct{})
	fn := func(ctx context.Context) ([]hfFileInfo, bool, error) {
		<-hold
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		return []hfFileInfo{{Path: "winner.gguf", Size: 1}}, false, nil
	}

	// Winner: long-lived context.
	winnerDone := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_, _, err := c.resolve(ctx, "repoZ", "main", "tok", fn)
		winnerDone <- err
	}()

	// Wait for the winner to enter the singleflight before the loser
	// piles on; otherwise the loser might enter first and become winner.
	time.Sleep(30 * time.Millisecond)

	// Loser: cancels almost immediately.
	loserCtx, loserCancel := context.WithCancel(context.Background())
	loserDone := make(chan error, 1)
	go func() {
		_, _, err := c.resolve(loserCtx, "repoZ", "main", "tok", fn)
		loserDone <- err
	}()
	time.Sleep(10 * time.Millisecond)
	loserCancel()

	if err := <-loserDone; err == nil {
		t.Errorf("loser should have got context.Canceled, got nil")
	}
	// Unblock the underlying fn.
	close(hold)
	if err := <-winnerDone; err != nil {
		t.Errorf("winner should have completed cleanly, got: %v", err)
	}
}
