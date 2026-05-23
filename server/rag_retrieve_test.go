package main

import (
	"context"
	"testing"
)

func TestBM25TokenizeStable(t *testing.T) {
	in := "Hello, World!  This is a TEST — punctuation, plus 42."
	toks := tokenizeBM25(in)
	want := []string{"hello", "world", "this", "is", "a", "test", "punctuation", "plus", "42"}
	if len(toks) != len(want) {
		t.Fatalf("len=%d want %d (%v)", len(toks), len(want), toks)
	}
	for i := range want {
		if toks[i] != want[i] {
			t.Errorf("tok[%d]=%q want %q", i, toks[i], want[i])
		}
	}
}

func TestBM25RanksByTermPresence(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(1)
	cid := newRAGCollection(t, s, uid, "bm25-test")

	// Two distinct documents in distinct topical worlds.
	docA := "rocket engines burn kerosene at high pressure. The combustion chamber sustains plasma."
	docB := "garden tomatoes ripen in late summer. Compost helps the soil retain moisture."
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:a", Body: []byte(docA),
	}); err != nil {
		t.Fatalf("ingest A: %v", err)
	}
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:b", Body: []byte(docB),
	}); err != nil {
		t.Fatalf("ingest B: %v", err)
	}

	hits, err := bm25Search(context.Background(), s.db, s.dialect, cid, "rocket engines", 5)
	if err != nil {
		t.Fatalf("bm25Search: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected hits for 'rocket engines'")
	}
	// The top hit must come from docA (about rockets), not docB (tomatoes).
	if !contains(hits[0].Chunk.Text, "rocket") && !contains(hits[0].Chunk.Text, "engines") {
		t.Errorf("top hit unexpectedly came from a non-rocket chunk: %q", hits[0].Chunk.Text)
	}

	// Garden query → should rank the garden doc first.
	hits2, err := bm25Search(context.Background(), s.db, s.dialect, cid, "tomatoes compost", 5)
	if err != nil {
		t.Fatalf("bm25 #2: %v", err)
	}
	if len(hits2) == 0 || (!contains(hits2[0].Chunk.Text, "tomato") && !contains(hits2[0].Chunk.Text, "compost")) {
		t.Errorf("expected garden chunk at top, got %q", func() string {
			if len(hits2) > 0 {
				return hits2[0].Chunk.Text
			}
			return "(none)"
		}())
	}

	// Query that matches nothing → empty result, no error.
	none, err := bm25Search(context.Background(), s.db, s.dialect, cid, "zzzzzzzz qqqqqqqq", 5)
	if err != nil {
		t.Fatalf("empty-match bm25: %v", err)
	}
	if len(none) != 0 {
		t.Errorf("expected 0 hits for nonsense, got %d", len(none))
	}
}

func TestHybridFusionPrefersBothSignals(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(9)
	cid := newRAGCollection(t, s, uid, "hybrid")

	// Three docs: one is the clear semantic match, one is a lexical
	// keyword match, one is a noise blob.
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:1", Body: []byte("the rocket carries fuel into orbit."),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:2", Body: []byte("propellant tanks pressurise during ascent."),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:3", Body: []byte("kittens like sunbeams in the afternoon."),
	}); err != nil {
		t.Fatal(err)
	}

	cfg := defaultHybridConfig()
	cfg.K = 3
	hits, err := s.retrieveHybrid(context.Background(), uid, cid, "rocket fuel orbit", cfg)
	if err != nil {
		t.Fatalf("retrieveHybrid: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected non-zero hits")
	}
	// Top hit should mention rocket/fuel/orbit — not kittens.
	top := hits[0].Chunk.Text
	if contains(top, "kitten") {
		t.Errorf("top hit was the noise doc: %q", top)
	}
	// Sanity: scores monotonically non-increasing.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Errorf("hits not sorted: %d:%v > %d:%v", i, hits[i].Score, i-1, hits[i-1].Score)
		}
	}
}

func TestRRFFuseMath(t *testing.T) {
	// Two ranked lists, one item appears in both.
	a := []ragHit{
		{Chunk: ragChunk{ID: 1}, Score: 0.9},
		{Chunk: ragChunk{ID: 2}, Score: 0.5},
	}
	b := []bm25Hit{
		{Chunk: ragChunk{ID: 2}, Score: 10},
		{Chunk: ragChunk{ID: 3}, Score: 5},
	}
	fused := rrfFuse(a, b)
	// ID=2 should outrank ID=1 and ID=3 because it appears in both lists.
	var s1, s2, s3 float32
	for _, h := range fused {
		switch h.Chunk.ID {
		case 1:
			s1 = h.Score
		case 2:
			s2 = h.Score
		case 3:
			s3 = h.Score
		}
	}
	if !(s2 > s1) || !(s2 > s3) {
		t.Errorf("RRF should reward dual-list membership: s1=%v s2=%v s3=%v", s1, s2, s3)
	}
}

func TestRerankerNoopByDefault(t *testing.T) {
	hits := []hybridHit{
		{Chunk: ragChunk{ID: 1}, Score: 0.5},
		{Chunk: ragChunk{ID: 2}, Score: 0.3},
	}
	out := applyReranker(context.Background(), "q", hits)
	if len(out) != 2 || out[0].Chunk.ID != 1 || out[1].Chunk.ID != 2 {
		t.Errorf("noop reranker should preserve order, got %+v", out)
	}
}

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
