package main

// Hybrid retrieval: dense (cosine) + BM25, fused with reciprocal rank
// fusion (RRF), then optionally reranked.
//
// Why RRF instead of weighted-sum:
//   - Scores from cosine and BM25 live on incomparable scales (the dense
//     side is bounded; BM25 is not).  Weighted sums force you to
//     normalise per-collection, which drifts as the index grows.
//   - RRF is parameterless beyond a single k (we use k=60, the value
//     from the original Cormack et al. 2009 paper).  It rewards items
//     that rank well in *either* list, which empirically matches what
//     users want from "find the document about X".
//
// The reranker is a thin interface — current default is a no-op (we
// just trust the fused order).  When the embed tier ships a real
// cross-encoder, slot it in via setReranker().

import (
	"context"
	"sort"
	"sync"
)

const rrfK = 60

// hybridConfig pins the parameters for retrieve().  Defaults match what
// /v1/chat/completions uses; the search HTTP endpoint can override.
type hybridConfig struct {
	K         int  // final number of hits returned
	PoolEach  int  // candidates pulled from each sub-retriever before fusion
	UseDense  bool
	UseBM25   bool
	Rerank    bool // run the reranker over the fused top-pool
}

func defaultHybridConfig() hybridConfig {
	return hybridConfig{K: 8, PoolEach: 32, UseDense: true, UseBM25: true, Rerank: false}
}

// hybridHit is the fused result.  Score is the RRF score (higher is
// better, but the absolute value is not comparable across queries).
// Per-source scores are kept for debugging/UI.
type hybridHit struct {
	Chunk      ragChunk
	Score      float32 // fused RRF score
	DenseScore float32
	BM25Score  float32
}

// retrieveHybrid runs both retrievers in parallel and fuses the results.
// Caller has already verified the user owns the collection; this
// function calls the ACL-aware vector store / BM25 path under the hood.
func (s *server) retrieveHybrid(ctx context.Context, uid, collectionID int64, query string, cfg hybridConfig) ([]hybridHit, error) {
	if cfg.K <= 0 {
		cfg.K = 8
	}
	if cfg.PoolEach <= 0 {
		cfg.PoolEach = 4 * cfg.K
	}

	var (
		wg          sync.WaitGroup
		denseHits   []ragHit
		bm25Hits    []bm25Hit
		denseErr    error
		bm25Err     error
	)

	if cfg.UseDense {
		wg.Add(1)
		go func() {
			defer wg.Done()
			emb := getEmbedder()
			vecs, err := emb.Embed(ctx, []string{query})
			if err != nil {
				denseErr = err
				return
			}
			store := newSQLiteVectorStore(s.db, s.dialect)
			denseHits, denseErr = store.searchByEmbedding(uid, collectionID, vecs[0], cfg.PoolEach)
		}()
	}
	if cfg.UseBM25 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bm25Hits, bm25Err = bm25Search(ctx, s.db, s.dialect, collectionID, query, cfg.PoolEach)
		}()
	}
	wg.Wait()
	if denseErr != nil {
		return nil, denseErr
	}
	if bm25Err != nil {
		return nil, bm25Err
	}

	fused := rrfFuse(denseHits, bm25Hits)
	sort.SliceStable(fused, func(i, j int) bool { return fused[i].Score > fused[j].Score })

	if cfg.Rerank {
		fused = applyReranker(ctx, query, fused)
	}
	if len(fused) > cfg.K {
		fused = fused[:cfg.K]
	}
	return fused, nil
}

// rrfFuse blends two ranked lists into one ordered by reciprocal-rank
// fusion.  RRF gives each item a score of Σ 1/(rrfK + rank_i) across
// the lists where it appears.  rrfK=60 is the canonical default.
func rrfFuse(dense []ragHit, bm25 []bm25Hit) []hybridHit {
	type bucket struct {
		chunk      ragChunk
		score      float32
		denseScore float32
		bm25Score  float32
		hasDense   bool
		hasBM25    bool
	}
	byID := make(map[int64]*bucket)

	for rank, h := range dense {
		b, ok := byID[h.Chunk.ID]
		if !ok {
			b = &bucket{chunk: h.Chunk}
			byID[h.Chunk.ID] = b
		}
		b.score += 1.0 / float32(rrfK+rank+1)
		b.denseScore = h.Score
		b.hasDense = true
	}
	for rank, h := range bm25 {
		b, ok := byID[h.Chunk.ID]
		if !ok {
			b = &bucket{chunk: h.Chunk}
			byID[h.Chunk.ID] = b
		}
		b.score += 1.0 / float32(rrfK+rank+1)
		b.bm25Score = h.Score
		b.hasBM25 = true
	}
	out := make([]hybridHit, 0, len(byID))
	for _, b := range byID {
		out = append(out, hybridHit{
			Chunk:      b.chunk,
			Score:      b.score,
			DenseScore: b.denseScore,
			BM25Score:  b.bm25Score,
		})
	}
	return out
}
