package main

import (
	"math"
	"testing"
)

func TestEncodeDecodeEmbedding(t *testing.T) {
	cases := [][]float32{
		nil,
		{},
		{0},
		{1, -1, 0.5, -0.5},
		{float32(math.Pi), float32(math.E), -1e-30, 1e+30},
	}
	for _, in := range cases {
		buf := encodeEmbedding(in)
		out, err := decodeEmbedding(buf)
		if err != nil {
			t.Errorf("decodeEmbedding(%v) error: %v", in, err)
			continue
		}
		if len(out) != len(in) {
			t.Errorf("length mismatch: got %d want %d", len(out), len(in))
			continue
		}
		for i := range in {
			if out[i] != in[i] {
				t.Errorf("element %d: got %v want %v", i, out[i], in[i])
			}
		}
	}
}

func TestDecodeEmbeddingMalformed(t *testing.T) {
	// 3 bytes is not a multiple of 4.
	if _, err := decodeEmbedding([]byte{1, 2, 3}); err == nil {
		t.Error("expected error on non-multiple-of-4 input")
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	if got := cosineSimilarity(a, b); got < 0.999 || got > 1.001 {
		t.Errorf("identical vectors: cos=%v want 1", got)
	}
	c := []float32{0, 1, 0}
	if got := cosineSimilarity(a, c); got != 0 {
		t.Errorf("orthogonal: cos=%v want 0", got)
	}
	d := []float32{-1, 0, 0}
	if got := cosineSimilarity(a, d); got > -0.999 || got < -1.001 {
		t.Errorf("anti-parallel: cos=%v want -1", got)
	}
	// Length mismatch must not crash; returns -inf.
	if got := cosineSimilarity([]float32{1, 0}, []float32{1, 0, 0}); !math.IsInf(float64(got), -1) {
		t.Errorf("mismatched dims: got %v want -inf", got)
	}
	// Zero vector returns 0 (no NaN).
	z := []float32{0, 0, 0}
	if got := cosineSimilarity(a, z); got != 0 {
		t.Errorf("zero vector: cos=%v want 0", got)
	}
}

func TestRAGMigrateIdempotent(t *testing.T) {
	s, _ := openMCPTestDB(t)
	// migrateRAG ran during openMCPTestDB; running it again must succeed.
	if err := migrateRAG(s.db, s.dialect); err != nil {
		t.Fatalf("re-running migrateRAG failed: %v", err)
	}
	// Spot-check the three tables exist.
	for _, tbl := range []string{"rag_collections", "rag_documents", "rag_chunks"} {
		var name string
		if err := s.db.QueryRow(
			`SELECT name FROM sqlite_master WHERE type='table' AND name=?`, tbl,
		).Scan(&name); err != nil || name != tbl {
			t.Errorf("expected table %q present; got name=%q err=%v", tbl, name, err)
		}
	}
}
