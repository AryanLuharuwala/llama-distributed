package main

// Cross-encoder reranker plug-in.
//
// A cross-encoder takes (query, candidate) pairs and outputs a
// relevance score — significantly higher quality than bi-encoder
// cosine because the model attends across both inputs.  We keep this
// behind an interface so the actual model can live on the embed tier
// (rag_embed.go's rig fleet) without ingesting any rerank code into
// the control-plane binary.
//
// The default reranker is a no-op: it returns hits in the order it
// got them.  setReranker() lets main() (or tests) plug in a real one
// once the model fleet advertises a "rerank" capability.

import (
	"context"
	"sync"
)

// reranker takes the (query, candidates) and returns the candidates
// re-ordered with updated scores.  Implementations must preserve the
// input slice contents but may shuffle order.
type reranker interface {
	Rerank(ctx context.Context, query string, hits []hybridHit) []hybridHit
}

// noopReranker is the default — just hands back the input.
type noopReranker struct{}

func (noopReranker) Rerank(_ context.Context, _ string, hits []hybridHit) []hybridHit {
	return hits
}

var (
	rerankerMu sync.RWMutex
	rerankerCurrent reranker = noopReranker{}
)

func getReranker() reranker {
	rerankerMu.RLock()
	defer rerankerMu.RUnlock()
	return rerankerCurrent
}

func setReranker(r reranker) {
	rerankerMu.Lock()
	if r == nil {
		rerankerCurrent = noopReranker{}
	} else {
		rerankerCurrent = r
	}
	rerankerMu.Unlock()
}

// applyReranker is a thin wrapper so retrieve.go doesn't have to know
// about the singleton.
func applyReranker(ctx context.Context, query string, hits []hybridHit) []hybridHit {
	return getReranker().Rerank(ctx, query, hits)
}
