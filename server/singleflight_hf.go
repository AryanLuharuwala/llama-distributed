package main

// HF resolve coalescing via golang.org/x/sync/singleflight.
//
// Problem: the dashboard's "import from HF" picker calls /api/hf/resolve on
// every keystroke completion.  Each call lists the repo's files via the HF
// API.  When several tabs or several keystrokes land within the same
// network round-trip, we hit HF N times for the same (repo, revision,
// token) tuple — wasted bandwidth, wasted HF quota, slower UX under their
// rate limits.
//
// singleflight.Group runs the underlying probe exactly once across all
// concurrent callers sharing a key; the rest block and receive the same
// result.  Distinct keys still parallelize.  Errors are NOT cached — a
// transient HF flake should not poison the next user's resolve.
//
// We deliberately do not cache successful resolves across requests:
//   - HF repos can be re-tagged/re-pushed; we want a fresh probe per click
//   - The signed download URLs we eventually mint are short-lived anyway
//   - rigCostCache solves the different problem of "cache + dedupe", and
//     for HF resolve we only want dedupe (each user's keystroke is fresh)
//
// Why a separate Group per concern (HF resolve here, model manifest in a
// later file): keys are namespaced by Group identity, so accidentally
// sharing a key across concerns is impossible.  Keeps the failure mode
// well-typed at the price of one extra pointer per Group.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"golang.org/x/sync/singleflight"
)

// hfResolveCoalescer coalesces concurrent HF list-files probes for the
// same (repo_id, revision, token) tuple.  The token is hashed into the
// key so we don't keep plaintext tokens on the heap as map keys.
type hfResolveCoalescer struct {
	g singleflight.Group
}

func newHFResolveCoalescer() *hfResolveCoalescer {
	return &hfResolveCoalescer{}
}

// resolveResult is the shape every coalesced caller receives.  Mirrors
// the success branch of handleHFResolve so callers don't need a second
// pass over the data.
type hfResolveResult struct {
	Files       []hfFileInfo
	Convertible bool
	Err         error
}

// resolve runs fn() exactly once across concurrent callers sharing the
// same (repoID, revision, tokenHash) tuple.  The context is honoured
// only for the *winning* caller — losers wait for the winner regardless
// of their own context.  If a loser's context fires we return early with
// ctx.Err(), but the winner keeps running so subsequent callers see a
// settled result instead of restarting.
//
// fn must be safe to invoke without a per-caller mutex; HF's HTTP client
// is goroutine-safe.
func (c *hfResolveCoalescer) resolve(
	ctx context.Context,
	repoID, revision, token string,
	fn func(ctx context.Context) ([]hfFileInfo, bool, error),
) ([]hfFileInfo, bool, error) {
	key := hfResolveKey(repoID, revision, token)
	type out struct {
		files       []hfFileInfo
		convertible bool
	}
	ch := c.g.DoChan(key, func() (interface{}, error) {
		// Run in a fresh background-derived context so a losing caller's
		// cancel doesn't tear down the work the winner is doing for
		// everyone else.  We still impose the resolve-timeout at the
		// outer handler so this can't run forever.
		bg, cancel := context.WithCancel(context.Background())
		defer cancel()
		files, conv, err := fn(bg)
		if err != nil {
			return nil, err
		}
		return out{files: files, convertible: conv}, nil
	})
	select {
	case <-ctx.Done():
		return nil, false, ctx.Err()
	case r := <-ch:
		if r.Err != nil {
			return nil, false, r.Err
		}
		v := r.Val.(out)
		return v.files, v.convertible, nil
	}
}

// hfResolveKey derives the dedup key.  Token gets sha256'd so we don't
// retain plaintext tokens in the map keys; collisions on this hash would
// already mean the user has bigger problems.
func hfResolveKey(repoID, revision, token string) string {
	if revision == "" {
		revision = "main"
	}
	sum := sha256.Sum256([]byte(token))
	return fmt.Sprintf("%s@%s#%s", repoID, revision, hex.EncodeToString(sum[:8]))
}

// hfResolveCoalescerSingleton is module-level so tests and handlers
// share the same Group across the process.  Reset() drops in-flight
// state for tests.
var (
	hfCoalescerOnce sync.Once
	hfCoalescer     *hfResolveCoalescer
)

func getHFCoalescer() *hfResolveCoalescer {
	hfCoalescerOnce.Do(func() {
		hfCoalescer = newHFResolveCoalescer()
	})
	return hfCoalescer
}
