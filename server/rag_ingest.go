package main

// Ingestion: bytes → chunks → embeddings → store.
//
// The synchronous path is fine at small scale (a few-MB doc, <1s with
// hashEmbedder, ~10s with a real model on a single embed rig).  For
// larger uploads we serialise behind the document row state machine —
// status='pending' → 'embedding' → 'ready' / 'failed' — so a worker
// pool added later can pick up where the request left off.
//
// Deduplication: a document is keyed by (collection_id, content_sha).
// Uploading the same bytes twice is a no-op that returns the existing
// document_id.  Saves embedding budget on retries.

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"time"
)

// ingestRequest is what callers (rag_api.go) hand to ingestDocument.
// URI is informational — we don't fetch it server-side, the bytes are
// already in Body.  MIMEType is used for content-type metadata only.
type ingestRequest struct {
	UserID       int64
	CollectionID int64
	URI          string
	MIMEType     string
	Body         []byte
}

// ingestResult tells the caller whether they hit the dedup path and
// what the resulting document looks like.
type ingestResult struct {
	DocumentID int64
	ChunkCount int
	Duplicate  bool // true if (collection, content_sha) already existed
}

// ingestDocument is the synchronous end-to-end ingest path.  Caller
// already verified the user owns the collection.  Errors are wrapped
// with stage context so production logs surface where it broke.
func (s *server) ingestDocument(ctx context.Context, req ingestRequest) (*ingestResult, error) {
	if len(req.Body) == 0 {
		return nil, errors.New("rag: empty body")
	}
	sum := sha256.Sum256(req.Body)
	sha := hex.EncodeToString(sum[:])

	// Fast path: dedup hit.
	var existing int64
	var existingChunks int
	err := s.dbQueryRow(s.dialect.RewriteQuery(
		`SELECT id, chunk_count FROM rag_documents
		  WHERE collection_id=? AND content_sha=? AND user_id=?`,
	), req.CollectionID, sha, req.UserID).Scan(&existing, &existingChunks)
	if err == nil {
		return &ingestResult{DocumentID: existing, ChunkCount: existingChunks, Duplicate: true}, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("rag: dedup lookup: %w", err)
	}

	// Get embedding-dim from the collection, since we may have been called
	// against an existing collection whose dim differs from the current
	// process-wide embedder.  Mismatch is a hard error — the caller has to
	// either pick a matching embedder or create a new collection.
	var collDim int
	if err := s.dbQueryRow(s.dialect.RewriteQuery(
		`SELECT embedding_dim FROM rag_collections WHERE id=? AND user_id=?`,
	), req.CollectionID, req.UserID).Scan(&collDim); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("rag: collection %d not found", req.CollectionID)
		}
		return nil, fmt.Errorf("rag: load collection: %w", err)
	}
	emb := getEmbedder()
	if emb.Dim() != collDim {
		return nil, fmt.Errorf(
			"rag: embedder dim=%d, collection dim=%d — collection was created with a different model",
			emb.Dim(), collDim,
		)
	}

	// Chunk the bytes.  We treat all uploads as UTF-8 text for now; a
	// PDF/HTML extractor lives behind an interface in a follow-up patch.
	chunks := chunkText(string(req.Body), defaultChunkerConfig())
	if len(chunks) == 0 {
		return nil, errors.New("rag: chunking produced 0 chunks (empty / whitespace-only doc?)")
	}

	// Embed in one batch.  Future: split into sub-batches if a provider
	// caps request size.
	vecs, err := emb.Embed(ctx, chunks)
	if err != nil {
		return nil, fmt.Errorf("rag: embed: %w", err)
	}
	if len(vecs) != len(chunks) {
		return nil, fmt.Errorf("rag: embedder returned %d vectors for %d chunks", len(vecs), len(chunks))
	}

	now := time.Now().Unix()
	tx, err := s.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("rag: begin ingest: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	res, err := s.txExec(tx,
		`INSERT INTO rag_documents
			(collection_id, user_id, uri, content_sha, mime_type, size_bytes, chunk_count, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		req.CollectionID, req.UserID, req.URI, sha, req.MIMEType, int64(len(req.Body)), len(chunks), now)
	if err != nil {
		return nil, fmt.Errorf("rag: insert document: %w", err)
	}
	docID, err := res.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("rag: last insert id: %w", err)
	}

	ins, err := tx.Prepare(s.dialect.RewriteQuery(
		`INSERT INTO rag_chunks
			(collection_id, document_id, ordinal_idx, text, token_count, embedding)
			VALUES (?, ?, ?, ?, ?, ?)`,
	))
	if err != nil {
		return nil, fmt.Errorf("rag: prepare chunk insert: %w", err)
	}
	defer ins.Close()

	for i, c := range chunks {
		if _, err := ins.Exec(
			req.CollectionID, docID, i, c, approxTokens(c), encodeEmbedding(vecs[i]),
		); err != nil {
			return nil, fmt.Errorf("rag: insert chunk %d: %w", i, err)
		}
	}

	// Counter rollups on the collection — keeps the dashboard cheap.
	if _, err := s.txExec(tx,
		`UPDATE rag_collections
		    SET documents_count = documents_count + 1,
		        chunks_count    = chunks_count + ?,
		        updated_at      = ?
		  WHERE id=?`,
		len(chunks), now, req.CollectionID); err != nil {
		return nil, fmt.Errorf("rag: bump collection counters: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("rag: commit ingest: %w", err)
	}
	return &ingestResult{DocumentID: docID, ChunkCount: len(chunks)}, nil
}

// removeDocument deletes the chunks + document row + bumps the
// collection counters.  Used by DELETE /api/rag/documents/{id}.
func (s *server) removeDocument(uid, collectionID, documentID int64) error {
	// Load chunk count for counter rollback.
	var chunks int
	err := s.dbQueryRow(
		`SELECT chunk_count FROM rag_documents
		  WHERE id=? AND collection_id=? AND user_id=?`,
		documentID, collectionID, uid).Scan(&chunks)
	if errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("rag: document %d not found", documentID)
	}
	if err != nil {
		return fmt.Errorf("rag: load doc: %w", err)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("rag: begin remove: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := s.txExec(tx,
		`DELETE FROM rag_chunks WHERE document_id=? AND collection_id=?`,
		documentID, collectionID); err != nil {
		return fmt.Errorf("rag: delete chunks: %w", err)
	}
	if _, err := s.txExec(tx,
		`DELETE FROM rag_documents WHERE id=? AND collection_id=? AND user_id=?`,
		documentID, collectionID, uid); err != nil {
		return fmt.Errorf("rag: delete doc: %w", err)
	}
	if _, err := s.txExec(tx,
		`UPDATE rag_collections
		    SET documents_count = documents_count - 1,
		        chunks_count    = chunks_count - ?,
		        updated_at      = ?
		  WHERE id=?`,
		chunks, time.Now().Unix(), collectionID); err != nil {
		return fmt.Errorf("rag: roll back counters: %w", err)
	}
	return tx.Commit()
}
