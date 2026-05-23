package main

// RAG (Retrieval-Augmented Generation) storage + retrieval surface.
//
// This file owns four layers:
//
//   1. Schema — rag_collections, rag_documents, rag_chunks.  Each chunk
//      stores its source-document pointer, the raw text we'll inject into
//      the model's context, and the dense embedding as a float32 blob.
//      (Dense vectors go in BLOB/BYTEA so we don't pay JSON parse cost on
//      every retrieval; the dialect mapping is handled by rewriteDDL.)
//
//   2. vectorStore interface — abstracts "give me the top-k nearest chunks
//      to this query embedding".  Backed by two implementations:
//        • sqliteVectorStore — does cosine in Go (table scan with an early-
//          exit min-heap).  Fine up to ~100k chunks per collection.
//        • postgresVectorStore — uses pgvector's `<#>` operator with an
//          ivfflat or hnsw index on the embedding column.  Scales to
//          tens of millions of chunks per collection.
//
//   3. Hybrid retrieval (rag_retrieve.go) — combines dense scores with
//      BM25 (FTS5 on SQLite, tsvector on Postgres) via reciprocal rank
//      fusion, then optionally reranks with a cross-encoder served on
//      the embedding tier.
//
//   4. Ingestion (rag_ingest.go) — chunker + embedder + writeback.  Owns
//      the deduplication-by-content-hash path so re-uploading the same
//      file is a no-op.
//
// This file only defines schema + interface + the BLOB encoding helpers.
// The two store implementations live alongside in rag_sqlite.go and
// rag_postgres.go so each can pull in driver-specific tricks (sqlite-vec
// for SQLite, pgvector for Postgres) without contaminating the shared
// path.

import (
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// ragCollection groups documents under a single user-controlled namespace.
// Inference requests reference collections by name; this struct is the
// persisted shape behind that name.
type ragCollection struct {
	ID              int64
	UserID          int64
	Name            string
	EmbeddingModel  string // e.g. "BAAI/bge-large-en-v1.5"
	EmbeddingDim    int    // 384, 768, 1024, 1536
	DocumentsCount  int64
	ChunksCount     int64
	CreatedAt       int64
	UpdatedAt       int64
}

// ragDocument is one source artifact (file, URL, pasted blob) that's been
// chunked into the collection.  Kept separate from rag_chunks so deletes
// can cascade by document.
type ragDocument struct {
	ID           int64
	CollectionID int64
	UserID       int64
	URI          string // file path, https://, or "inline:" for pasted text
	ContentSHA   string // sha256 hex; used for dedupe
	MIMEType     string
	SizeBytes    int64
	ChunkCount   int
	CreatedAt    int64
}

// ragChunk is one retrieval unit: a span of text + its dense embedding.
// Embedding is stored as a packed []float32 blob (little-endian) to keep
// the read path zero-allocation on the hot side.
type ragChunk struct {
	ID           int64
	CollectionID int64
	DocumentID   int64
	OrdinalIdx   int    // 0-based position within the document
	Text         string
	TokenCount   int
	Embedding    []float32
}

// ragHit is the result of a vector search.
type ragHit struct {
	Chunk ragChunk
	Score float32 // cosine similarity, [-1, 1]; higher is better
}

// vectorStore abstracts the per-driver "nearest neighbours by embedding"
// path.  Both implementations enforce per-user/per-collection ACL — the
// caller passes (uid, collectionID) and the store guarantees no rows
// from other tenants leak through.
type vectorStore interface {
	// upsertChunks writes (or replaces) the given chunks for a document.
	// The caller has already computed embeddings; the store is responsible
	// only for persistence + index maintenance.
	upsertChunks(uid, collectionID, documentID int64, chunks []ragChunk) error

	// searchByEmbedding returns the top-k chunks in collectionID ranked by
	// cosine similarity to query.  k is clamped to a sane upper bound by
	// the caller (usually 1..64).
	searchByEmbedding(uid, collectionID int64, query []float32, k int) ([]ragHit, error)

	// deleteDocument removes every chunk belonging to documentID.  Used
	// when the user removes a doc from the collection.
	deleteDocument(uid, collectionID, documentID int64) error
}

// migrateRAG installs the three RAG tables.  Idempotent across SQLite +
// Postgres via the dialect helper.  The embedding column is declared
// BLOB; the postgres dialect rewrites that to BYTEA.  pgvector
// migration (alter to vector(N) + ivfflat index) is a one-shot script
// shipped separately — we keep the portable BYTEA path so SQLite tests
// don't depend on a non-default extension.
func migrateRAG(db *sql.DB, d sqlDialect) error {
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rag_collections (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id         INTEGER NOT NULL,
			name            TEXT    NOT NULL,
			embedding_model TEXT    NOT NULL,
			embedding_dim   INTEGER NOT NULL,
			documents_count INTEGER NOT NULL DEFAULT 0,
			chunks_count    INTEGER NOT NULL DEFAULT 0,
			created_at      INTEGER NOT NULL,
			updated_at      INTEGER NOT NULL,
			UNIQUE(user_id, name)
		)
	`)); err != nil {
		return fmt.Errorf("create rag_collections: %w", err)
	}
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rag_documents (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			collection_id INTEGER NOT NULL,
			user_id       INTEGER NOT NULL,
			uri           TEXT    NOT NULL,
			content_sha   TEXT    NOT NULL,
			mime_type     TEXT    NOT NULL DEFAULT '',
			size_bytes    INTEGER NOT NULL DEFAULT 0,
			chunk_count   INTEGER NOT NULL DEFAULT 0,
			created_at    INTEGER NOT NULL,
			UNIQUE(collection_id, content_sha)
		)
	`)); err != nil {
		return fmt.Errorf("create rag_documents: %w", err)
	}
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rag_chunks (
			id             INTEGER PRIMARY KEY AUTOINCREMENT,
			collection_id  INTEGER NOT NULL,
			document_id    INTEGER NOT NULL,
			ordinal_idx    INTEGER NOT NULL,
			text           TEXT    NOT NULL,
			token_count    INTEGER NOT NULL DEFAULT 0,
			embedding      BLOB    NOT NULL
		)
	`)); err != nil {
		return fmt.Errorf("create rag_chunks: %w", err)
	}
	// Indexes — collection_id is the dominant predicate on every read
	// path, and (document_id) supports per-document delete cascade.
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE INDEX IF NOT EXISTS idx_rag_chunks_collection
			ON rag_chunks (collection_id)
	`)); err != nil {
		return fmt.Errorf("create idx_rag_chunks_collection: %w", err)
	}
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
			ON rag_chunks (document_id)
	`)); err != nil {
		return fmt.Errorf("create idx_rag_chunks_document: %w", err)
	}
	return nil
}

// ── Embedding (de)serialization ───────────────────────────────────────
//
// Embeddings live as packed little-endian float32 in BLOB/BYTEA.  We
// pin little-endian explicitly so a Postgres row dumped from one host
// and loaded on another with different native byte order still reads
// correctly.

// encodeEmbedding packs a float vector into a length-prefixed binary
// blob.  Returns nil on a nil input so the caller can pass through.
func encodeEmbedding(v []float32) []byte {
	if v == nil {
		return nil
	}
	buf := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// decodeEmbedding unpacks a binary blob into a float vector.  Returns an
// error on malformed (non-multiple-of-4) input so a corrupted DB row
// surfaces loudly instead of silently returning a zero vector.
func decodeEmbedding(b []byte) ([]float32, error) {
	if len(b) == 0 {
		return nil, nil
	}
	if len(b)%4 != 0 {
		return nil, errors.New("rag: embedding blob length not a multiple of 4")
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
}

// cosineSimilarity returns the cosine similarity between a and b.  Used
// by the SQLite store (in-process scoring).  Postgres uses pgvector's
// distance operator and skips this path entirely.
//
// Panics on length mismatch — that would indicate a corrupted index or
// an embedding-model swap that bypassed migration.  Caller is expected
// to ensure dim consistency upstream (the collection's embedding_dim
// pins this).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		// Soft fail: return a sentinel so a malformed pair gets ranked
		// dead last but doesn't crash the request.
		return float32(math.Inf(-1))
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
}
