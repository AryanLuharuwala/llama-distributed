package main

// Embedding provider for RAG.
//
// Two implementations:
//   • rigEmbedder dispatches a batch of strings to the rig fleet over the
//     existing inference WS path, asking for role="embed" rigs.  Production.
//   • hashEmbedder is a deterministic, dependency-free fallback that maps
//     text → fixed-dim vector via hashed n-grams.  Used in tests and on
//     dev boxes where no embed-capable rig is connected.
//
// The interface is intentionally tiny so we can swap in any model-server
// (Triton, Ray Serve, a local llama.cpp embedding pool) later without
// touching the ingestion or retrieval code.
//
// Why no caching here: the dedup-by-content-sha path in rag_ingest.go
// already prevents re-embedding the same document.  At the chunk level
// the same text rarely appears twice across distinct documents, so an
// in-memory cache would burn RAM for little hit-rate.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"unicode"
)

// embedder produces dense vectors for a batch of strings.  Embed must be
// safe for concurrent invocation; the chunk-level worker calls it from
// multiple goroutines.
type embedder interface {
	// Embed returns one vector per input string.  Order is preserved.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// Dim returns the output dimensionality.  Pinned to the collection
	// row at creation time so a mid-stream model swap surfaces loudly.
	Dim() int

	// ModelID identifies the embedding model (e.g. "BAAI/bge-large-en")
	// for the collection metadata.
	ModelID() string
}

// hashEmbedder produces deterministic, low-quality vectors via hashed
// character n-grams.  It exists so the test suite (and any single-box
// dev deploy without an embed rig) has a working RAG path end-to-end.
// Quality is significantly worse than a real model — do not use in
// production.  Output is L2-normalised so cosine similarity behaves.
type hashEmbedder struct {
	dim int
}

func newHashEmbedder(dim int) *hashEmbedder {
	if dim <= 0 {
		dim = 256
	}
	return &hashEmbedder{dim: dim}
}

func (h *hashEmbedder) Dim() int        { return h.dim }
func (h *hashEmbedder) ModelID() string { return fmt.Sprintf("hash-fnv-%d", h.dim) }

func (h *hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.one(t)
	}
	return out, nil
}

// one hashes character trigrams + word unigrams into the vector.  Two
// independent hash functions reduce collision pile-up.  Each token
// contributes a +1 to its first slot and a sign-randomised +1 to its
// second; this is the standard "feature hashing" trick from Weinberger
// et al. 2009.
func (h *hashEmbedder) one(s string) []float32 {
	v := make([]float32, h.dim)
	s = strings.ToLower(s)

	addToken := func(tok string) {
		if tok == "" {
			return
		}
		a := fnv.New32a()
		_, _ = a.Write([]byte(tok))
		ah := a.Sum32()
		i := int(ah) % h.dim
		if i < 0 {
			i += h.dim
		}
		// sign from the high bit so collisions don't always add
		sgn := float32(1)
		if ah&0x80000000 != 0 {
			sgn = -1
		}
		v[i] += sgn

		// second slot, decorrelated by sha256
		sh := sha256.Sum256([]byte(tok))
		j := int(binary.LittleEndian.Uint32(sh[:4])) % h.dim
		if j < 0 {
			j += h.dim
		}
		v[j] += sgn * 0.5
	}

	// word unigrams
	for _, w := range strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		addToken(w)
	}
	// character trigrams over a normalised string
	runes := []rune(strings.Join(strings.Fields(s), " "))
	for i := 0; i+3 <= len(runes); i++ {
		addToken(string(runes[i : i+3]))
	}

	// L2 normalise
	var sq float64
	for _, x := range v {
		sq += float64(x) * float64(x)
	}
	if sq == 0 {
		return v
	}
	inv := float32(1.0 / math.Sqrt(sq))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// embedderRegistry holds the process-wide selection.  Wired by main()
// once the rig fleet is up; tests construct their own via setHashEmbedder.
var (
	embedderMu      sync.RWMutex
	embedderCurrent embedder
)

// getEmbedder returns the current embedder, lazily falling back to a
// 256-dim hashEmbedder if main() never set one.  Lazy fallback keeps
// dev-mode RAG functional with no extra wiring.
func getEmbedder() embedder {
	embedderMu.RLock()
	e := embedderCurrent
	embedderMu.RUnlock()
	if e != nil {
		return e
	}
	embedderMu.Lock()
	defer embedderMu.Unlock()
	if embedderCurrent == nil {
		embedderCurrent = newHashEmbedder(256)
	}
	return embedderCurrent
}

func setEmbedder(e embedder) {
	embedderMu.Lock()
	embedderCurrent = e
	embedderMu.Unlock()
}
