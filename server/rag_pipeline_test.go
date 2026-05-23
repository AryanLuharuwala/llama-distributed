package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// helper: create a collection and return its ID.
func newRAGCollection(t *testing.T, s *server, uid int64, name string) int64 {
	t.Helper()
	emb := getEmbedder()
	now := time.Now().Unix()
	res, err := s.db.Exec(s.dialect.RewriteQuery(
		`INSERT INTO rag_collections (user_id, name, embedding_model, embedding_dim, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
	), uid, name, emb.ModelID(), emb.Dim(), now, now)
	if err != nil {
		t.Fatalf("insert collection: %v", err)
	}
	id, _ := res.LastInsertId()
	return id
}

func TestHashEmbedderShape(t *testing.T) {
	e := newHashEmbedder(128)
	if e.Dim() != 128 {
		t.Errorf("Dim=%d want 128", e.Dim())
	}
	v, err := e.Embed(context.Background(), []string{"hello world", "another text"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(v) != 2 || len(v[0]) != 128 || len(v[1]) != 128 {
		t.Fatalf("shape: %d × %d", len(v), len(v[0]))
	}
	// Deterministic: same input → same output.
	v2, _ := e.Embed(context.Background(), []string{"hello world"})
	for i := range v[0] {
		if v[0][i] != v2[0][i] {
			t.Fatalf("non-deterministic at idx %d", i)
		}
	}
	// Non-trivial: different inputs should not be identical.
	identical := true
	for i := range v[0] {
		if v[0][i] != v[1][i] {
			identical = false
			break
		}
	}
	if identical {
		t.Errorf("distinct inputs produced identical embeddings")
	}
}

func TestChunkerBudget(t *testing.T) {
	// Paragraph well under budget → single chunk.
	one := chunkText("Hello world.", defaultChunkerConfig())
	if len(one) != 1 || one[0] != "Hello world." {
		t.Errorf("short text: got %+v", one)
	}

	// Long text → multiple chunks, none exceeding 2× target.
	long := strings.Repeat("This is a sentence. ", 800)
	cfg := defaultChunkerConfig()
	out := chunkText(long, cfg)
	if len(out) < 2 {
		t.Fatalf("expected >=2 chunks for long input, got %d", len(out))
	}
	for i, c := range out {
		if approxTokens(c) > cfg.MaxTokens+cfg.OverlapToks {
			t.Errorf("chunk %d tokens=%d exceeds MaxTokens(%d)+overlap(%d)", i, approxTokens(c), cfg.MaxTokens, cfg.OverlapToks)
		}
	}

	// Empty input → no chunks.
	if got := chunkText("   \n\n   ", cfg); len(got) != 0 {
		t.Errorf("empty input: got %d chunks, want 0", len(got))
	}
}

func TestRAGIngestAndSearch(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(128))
	defer setEmbedder(nil) // reset for other tests

	const uid = int64(42)
	cid := newRAGCollection(t, s, uid, "test-coll")

	doc := "The rocket carries nine engines. Each engine burns kerosene. " +
		"\n\nThe second stage uses a single vacuum-optimised engine. " +
		"It restarts in space to circularise the orbit."

	res, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID:       uid,
		CollectionID: cid,
		URI:          "inline:rocket.txt",
		Body:         []byte(doc),
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if res.Duplicate {
		t.Errorf("first ingest should not be duplicate")
	}
	if res.ChunkCount < 1 {
		t.Fatalf("expected >=1 chunk, got %d", res.ChunkCount)
	}

	// Dedup path: same bytes ⇒ Duplicate=true, same doc id.
	res2, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:rocket.txt", Body: []byte(doc),
	})
	if err != nil {
		t.Fatalf("dedup ingest: %v", err)
	}
	if !res2.Duplicate || res2.DocumentID != res.DocumentID {
		t.Errorf("dedup: got %+v want Duplicate=true id=%d", res2, res.DocumentID)
	}

	// Search by embedding for a query semantically close to the first half.
	store := newSQLiteVectorStore(s.db, s.dialect)
	emb := getEmbedder()
	q, _ := emb.Embed(context.Background(), []string{"engines burning fuel"})
	hits, err := store.searchByEmbedding(uid, cid, q[0], 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least one hit")
	}
	// Hits should be sorted by score descending.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Errorf("hits not sorted: [%d].Score=%v > [%d].Score=%v", i, hits[i].Score, i-1, hits[i-1].Score)
		}
	}
}

func TestRAGACLBlocksCrossUser(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const userA, userB = int64(1), int64(2)
	cidA := newRAGCollection(t, s, userA, "alice")

	// User A ingests.
	_, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: userA, CollectionID: cidA, URI: "inline:a", Body: []byte("alpha bravo charlie"),
	})
	if err != nil {
		t.Fatalf("ingest A: %v", err)
	}

	// User B must not be able to search collection A.
	store := newSQLiteVectorStore(s.db, s.dialect)
	emb := getEmbedder()
	q, _ := emb.Embed(context.Background(), []string{"alpha"})
	_, err = store.searchByEmbedding(userB, cidA, q[0], 5)
	if err == nil {
		t.Errorf("expected ACL error for cross-user search, got nil")
	}

	// User B must not be able to ingest into collection A either.
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: userB, CollectionID: cidA, URI: "inline:b", Body: []byte("delta"),
	}); err == nil {
		t.Errorf("expected ACL error for cross-user ingest, got nil")
	}
}

func TestRAGRemoveDocument(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(7)
	cid := newRAGCollection(t, s, uid, "rem")
	r1, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:1", Body: []byte("foo bar baz"),
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	// Verify counters bumped.
	var docs, chunks int64
	_ = s.db.QueryRow(`SELECT documents_count, chunks_count FROM rag_collections WHERE id=?`, cid).
		Scan(&docs, &chunks)
	if docs != 1 || chunks != int64(r1.ChunkCount) {
		t.Errorf("after ingest: docs=%d chunks=%d want 1, %d", docs, chunks, r1.ChunkCount)
	}

	if err := s.removeDocument(uid, cid, r1.DocumentID); err != nil {
		t.Fatalf("removeDocument: %v", err)
	}

	// Counters back to zero, no chunks left.
	_ = s.db.QueryRow(`SELECT documents_count, chunks_count FROM rag_collections WHERE id=?`, cid).
		Scan(&docs, &chunks)
	if docs != 0 || chunks != 0 {
		t.Errorf("after delete: docs=%d chunks=%d want 0, 0", docs, chunks)
	}
	var leftover int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks WHERE document_id=?`, r1.DocumentID).Scan(&leftover)
	if leftover != 0 {
		t.Errorf("leftover chunks for deleted doc: %d", leftover)
	}
}

func TestRAGUpsertReplacesChunks(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(32))
	defer setEmbedder(nil)

	const uid = int64(11)
	cid := newRAGCollection(t, s, uid, "upserts")

	store := newSQLiteVectorStore(s.db, s.dialect)
	emb := getEmbedder()
	mkChunks := func(texts []string) []ragChunk {
		vecs, _ := emb.Embed(context.Background(), texts)
		out := make([]ragChunk, len(texts))
		for i, t := range texts {
			out[i] = ragChunk{OrdinalIdx: i, Text: t, TokenCount: approxTokens(t), Embedding: vecs[i]}
		}
		return out
	}

	// Insert a placeholder document row directly so we have a documentID.
	res, err := s.db.Exec(s.dialect.RewriteQuery(
		`INSERT INTO rag_documents (collection_id, user_id, uri, content_sha, mime_type, size_bytes, chunk_count, created_at)
		 VALUES (?, ?, ?, ?, '', 0, 0, ?)`,
	), cid, uid, "inline:up", "sha-x", time.Now().Unix())
	if err != nil {
		t.Fatalf("doc insert: %v", err)
	}
	did, _ := res.LastInsertId()

	if err := store.upsertChunks(uid, cid, did, mkChunks([]string{"one", "two", "three"})); err != nil {
		t.Fatalf("first upsert: %v", err)
	}
	var n int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks WHERE document_id=?`, did).Scan(&n)
	if n != 3 {
		t.Errorf("after first upsert: %d chunks, want 3", n)
	}

	// Second upsert with two chunks should replace, not append.
	if err := store.upsertChunks(uid, cid, did, mkChunks([]string{"alpha", "beta"})); err != nil {
		t.Fatalf("second upsert: %v", err)
	}
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks WHERE document_id=?`, did).Scan(&n)
	if n != 2 {
		t.Errorf("after second upsert: %d chunks, want 2", n)
	}
}

func TestValidateRAGCollectionName(t *testing.T) {
	good := []string{"alpha", "alpha-1", "a.b_c", "x"}
	for _, n := range good {
		if _, err := validateRAGCollectionName(n); err != nil {
			t.Errorf("validateRAGCollectionName(%q) = %v, want nil", n, err)
		}
	}
	bad := []string{"", "  ", "BadCaps", "with space", "weird?char", strings.Repeat("a", ragMaxNameLen+1)}
	for _, n := range bad {
		if _, err := validateRAGCollectionName(n); err == nil {
			t.Errorf("validateRAGCollectionName(%q) = nil, want error", n)
		}
	}
}

func TestRAGEmbedderDimMismatchRejected(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(3)
	cid := newRAGCollection(t, s, uid, "dimmatch")

	// Swap embedder to a different dim mid-stream.
	setEmbedder(newHashEmbedder(128))

	_, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:x", Body: []byte("hello"),
	})
	if err == nil || !strings.Contains(err.Error(), "embedder dim") {
		t.Errorf("expected dim-mismatch error, got: %v", err)
	}
}
