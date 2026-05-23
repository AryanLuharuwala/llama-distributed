package main

// SQLite implementation of vectorStore.
//
// Scoring strategy: cosine similarity computed in-process.  We pull every
// chunk in the target collection (gated by the (uid, collection_id) ACL
// upstream), decode its embedding, score against the query, and keep a
// size-k min-heap of best hits.  This costs one O(N·dim) sweep per
// query but avoids any DB-side vector extension — sqlite-vec is a great
// option but installing it requires a custom build, and we want the
// default sqlite driver to be enough for single-container deploys.
//
// At ~100k chunks × 768-dim float32 this is still <80ms on a modern CPU.
// Beyond that, the Postgres + pgvector path is the right answer; the
// dialect abstraction lets the rest of the codebase stay identical.

import (
	"container/heap"
	"context"
	"database/sql"
	"errors"
	"fmt"
)

type sqliteVectorStore struct {
	db *sql.DB
	d  sqlDialect
}

func newSQLiteVectorStore(db *sql.DB, d sqlDialect) *sqliteVectorStore {
	return &sqliteVectorStore{db: db, d: d}
}

// ownsCollection is the single ACL check every method runs before touching
// chunks.  Returns (collection.embedding_dim, nil) on success.  We return
// the dim so callers can validate query/chunk dimensionality without a
// second round-trip.
func (s *sqliteVectorStore) ownsCollection(uid, cid int64) (int, error) {
	var dim int
	err := s.db.QueryRow(s.d.RewriteQuery(
		`SELECT embedding_dim FROM rag_collections WHERE id=? AND user_id=?`,
	), cid, uid).Scan(&dim)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, fmt.Errorf("rag: collection %d not found for user %d", cid, uid)
	}
	if err != nil {
		return 0, fmt.Errorf("rag: ownsCollection: %w", err)
	}
	return dim, nil
}

// upsertChunks replaces every chunk for documentID with the given set.
// Done in a single transaction so a concurrent search never sees a torn
// half-write (zero chunks one moment, full set the next).
func (s *sqliteVectorStore) upsertChunks(uid, collectionID, documentID int64, chunks []ragChunk) error {
	dim, err := s.ownsCollection(uid, collectionID)
	if err != nil {
		return err
	}
	for i, c := range chunks {
		if len(c.Embedding) != dim {
			return fmt.Errorf("rag: chunk %d embedding dim=%d, collection dim=%d", i, len(c.Embedding), dim)
		}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("rag: begin upsert: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec(s.d.RewriteQuery(
		`DELETE FROM rag_chunks WHERE document_id=? AND collection_id=?`,
	), documentID, collectionID); err != nil {
		return fmt.Errorf("rag: clear chunks: %w", err)
	}

	ins, err := tx.Prepare(s.d.RewriteQuery(
		`INSERT INTO rag_chunks
			(collection_id, document_id, ordinal_idx, text, token_count, embedding)
			VALUES (?, ?, ?, ?, ?, ?)`,
	))
	if err != nil {
		return fmt.Errorf("rag: prepare insert: %w", err)
	}
	defer ins.Close()

	for _, c := range chunks {
		if _, err := ins.Exec(
			collectionID, documentID, c.OrdinalIdx, c.Text, c.TokenCount, encodeEmbedding(c.Embedding),
		); err != nil {
			return fmt.Errorf("rag: insert chunk %d: %w", c.OrdinalIdx, err)
		}
	}
	return tx.Commit()
}

// searchByEmbedding scans every chunk in the collection, scores against
// query via cosine similarity, and returns the top-k by score (highest
// first).  Returns an empty slice (not nil) on an empty collection.
func (s *sqliteVectorStore) searchByEmbedding(uid, collectionID int64, query []float32, k int) ([]ragHit, error) {
	dim, err := s.ownsCollection(uid, collectionID)
	if err != nil {
		return nil, err
	}
	if len(query) != dim {
		return nil, fmt.Errorf("rag: query dim=%d, collection dim=%d", len(query), dim)
	}
	if k <= 0 {
		return []ragHit{}, nil
	}

	rows, err := s.db.QueryContext(context.Background(), s.d.RewriteQuery(
		`SELECT id, document_id, ordinal_idx, text, token_count, embedding
		   FROM rag_chunks
		  WHERE collection_id=?`,
	), collectionID)
	if err != nil {
		return nil, fmt.Errorf("rag: query chunks: %w", err)
	}
	defer rows.Close()

	h := &hitMinHeap{}
	heap.Init(h)

	for rows.Next() {
		var c ragChunk
		c.CollectionID = collectionID
		var emb []byte
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.OrdinalIdx, &c.Text, &c.TokenCount, &emb); err != nil {
			return nil, fmt.Errorf("rag: scan chunk: %w", err)
		}
		v, err := decodeEmbedding(emb)
		if err != nil {
			return nil, fmt.Errorf("rag: decode chunk %d: %w", c.ID, err)
		}
		c.Embedding = v
		score := cosineSimilarity(query, v)
		if h.Len() < k {
			heap.Push(h, ragHit{Chunk: c, Score: score})
			continue
		}
		if score > (*h)[0].Score {
			(*h)[0] = ragHit{Chunk: c, Score: score}
			heap.Fix(h, 0)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rag: rows iter: %w", err)
	}

	out := make([]ragHit, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(ragHit)
	}
	return out, nil
}

// deleteDocument removes every chunk for documentID.  The rag_documents
// row itself is removed by the caller (rag_api.go) so document-level
// metadata stays consistent with the chunk delete.
func (s *sqliteVectorStore) deleteDocument(uid, collectionID, documentID int64) error {
	if _, err := s.ownsCollection(uid, collectionID); err != nil {
		return err
	}
	if _, err := s.db.Exec(s.d.RewriteQuery(
		`DELETE FROM rag_chunks WHERE document_id=? AND collection_id=?`,
	), documentID, collectionID); err != nil {
		return fmt.Errorf("rag: delete chunks: %w", err)
	}
	return nil
}

// hitMinHeap is a min-heap on Score; the smallest element sits at index 0
// so we can cheaply evict the worst candidate when a better one shows up.
type hitMinHeap []ragHit

func (h hitMinHeap) Len() int            { return len(h) }
func (h hitMinHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h hitMinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hitMinHeap) Push(x interface{}) { *h = append(*h, x.(ragHit)) }
func (h *hitMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
