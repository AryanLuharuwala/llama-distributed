package main

// HTTP surface for RAG: collections, documents, search.
//
// Auth: every handler resolves the user via s.userFromRequest and gates
// on UserID match.  The vector store layer enforces the same ACL on its
// own (defense in depth) so a future code path that wires through the
// store without going via these handlers can't accidentally leak.
//
// Wire shapes are intentionally JSON (not protobuf) — the dashboard
// consumes these endpoints directly.  The eventual chat-completions
// surface will wrap the same Go-level helpers, not the JSON layer.

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	ragMaxBodyBytes    = 8 << 20 // 8 MiB per upload — UI surface, not bulk import
	ragMaxCollections  = 200     // per user
	ragMaxNameLen      = 64
	ragSearchMaxK      = 64
	ragSearchDefaultK  = 8
)

type ragCollectionAPI struct {
	ID             int64  `json:"id"`
	Name           string `json:"name"`
	EmbeddingModel string `json:"embedding_model"`
	EmbeddingDim   int    `json:"embedding_dim"`
	DocumentsCount int64  `json:"documents_count"`
	ChunksCount    int64  `json:"chunks_count"`
	CreatedAt      int64  `json:"created_at"`
	UpdatedAt      int64  `json:"updated_at"`
}

type ragDocumentAPI struct {
	ID         int64  `json:"id"`
	URI        string `json:"uri"`
	ContentSHA string `json:"content_sha"`
	MIMEType   string `json:"mime_type"`
	SizeBytes  int64  `json:"size_bytes"`
	ChunkCount int    `json:"chunk_count"`
	CreatedAt  int64  `json:"created_at"`
}

type ragHitAPI struct {
	ChunkID    int64   `json:"chunk_id"`
	DocumentID int64   `json:"document_id"`
	Ordinal    int     `json:"ordinal"`
	Text       string  `json:"text"`
	TokenCount int     `json:"token_count"`
	Score      float32 `json:"score"`
}

// POST /api/rag/collections — body: {name, embedding_model?}
func (s *server) handleRAGCreateCollection(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		Name           string `json:"name"`
		EmbeddingModel string `json:"embedding_model"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}
	name, err := validateRAGCollectionName(body.Name)
	if err != nil {
		writeErr(w, 400, err.Error())
		return
	}

	var count int
	if err := s.dbQueryRow(s.dialect.RewriteQuery(
		`SELECT COUNT(*) FROM rag_collections WHERE user_id=?`,
	), u.ID).Scan(&count); err != nil {
		writeErr(w, 500, "count collections: "+err.Error())
		return
	}
	if count >= ragMaxCollections {
		writeErr(w, 429, fmt.Sprintf("per-user collection cap (%d) reached", ragMaxCollections))
		return
	}

	emb := getEmbedder()
	model := body.EmbeddingModel
	if model == "" {
		model = emb.ModelID()
	}
	now := time.Now().Unix()
	res, err := s.dbExec(s.dialect.RewriteQuery(
		`INSERT INTO rag_collections
			(user_id, name, embedding_model, embedding_dim, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?)`,
	), u.ID, name, model, emb.Dim(), now, now)
	if err != nil {
		if isUniqueConstraint(err) {
			writeErr(w, 409, "collection name already exists")
			return
		}
		writeErr(w, 500, "insert: "+err.Error())
		return
	}
	id, _ := res.LastInsertId()
	writeJSON(w, 200, ragCollectionAPI{
		ID:             id,
		Name:           name,
		EmbeddingModel: model,
		EmbeddingDim:   emb.Dim(),
		CreatedAt:      now,
		UpdatedAt:      now,
	})
}

// GET /api/rag/collections
func (s *server) handleRAGListCollections(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(s.dialect.RewriteQuery(
		`SELECT id, name, embedding_model, embedding_dim,
		        documents_count, chunks_count, created_at, updated_at
		   FROM rag_collections
		  WHERE user_id=?
		  ORDER BY updated_at DESC, id DESC`,
	), u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	out := []ragCollectionAPI{}
	for rows.Next() {
		var c ragCollectionAPI
		if err := rows.Scan(&c.ID, &c.Name, &c.EmbeddingModel, &c.EmbeddingDim,
			&c.DocumentsCount, &c.ChunksCount, &c.CreatedAt, &c.UpdatedAt); err != nil {
			continue
		}
		out = append(out, c)
	}
	writeJSON(w, 200, map[string]any{"collections": out})
}

// DELETE /api/rag/collections/{id}
func (s *server) handleRAGDeleteCollection(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid id")
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := s.txExec(tx,
		`DELETE FROM rag_chunks WHERE collection_id=?`, cid); err != nil {
		writeErr(w, 500, "delete chunks: "+err.Error())
		return
	}
	if _, err := s.txExec(tx,
		`DELETE FROM rag_documents WHERE collection_id=? AND user_id=?`, cid, u.ID); err != nil {
		writeErr(w, 500, "delete docs: "+err.Error())
		return
	}
	res, err := s.txExec(tx,
		`DELETE FROM rag_collections WHERE id=? AND user_id=?`, cid, u.ID)
	if err != nil {
		writeErr(w, 500, "delete collection: "+err.Error())
		return
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		writeErr(w, 404, "collection not found")
		return
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, 500, "commit: "+err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// POST /api/rag/collections/{id}/documents
//
// Body shape: {uri?, mime_type?, content_base64?, content?}
// Either content_base64 or content (UTF-8 text) is required.
func (s *server) handleRAGUploadDocument(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid collection id")
		return
	}
	var body struct {
		URI           string `json:"uri"`
		MIMEType      string `json:"mime_type"`
		Content       string `json:"content"`
		ContentBase64 string `json:"content_base64"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, ragMaxBodyBytes)).Decode(&body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}
	var payload []byte
	switch {
	case body.Content != "":
		payload = []byte(body.Content)
	case body.ContentBase64 != "":
		dec, err := decodeBase64Loose(body.ContentBase64)
		if err != nil {
			writeErr(w, 400, "content_base64 not valid base64")
			return
		}
		payload = dec
	default:
		writeErr(w, 400, "content or content_base64 required")
		return
	}
	if len(payload) == 0 {
		writeErr(w, 400, "empty content")
		return
	}
	if int64(len(payload)) > ragMaxBodyBytes {
		writeErr(w, 413, fmt.Sprintf("payload exceeds %d bytes", ragMaxBodyBytes))
		return
	}
	res, err := s.ingestDocument(r.Context(), ingestRequest{
		UserID:       u.ID,
		CollectionID: cid,
		URI:          body.URI,
		MIMEType:     body.MIMEType,
		Body:         payload,
	})
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"document_id": res.DocumentID,
		"chunk_count": res.ChunkCount,
		"duplicate":   res.Duplicate,
	})
}

// GET /api/rag/collections/{id}/documents
func (s *server) handleRAGListDocuments(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid id")
		return
	}
	rows, err := s.dbQuery(s.dialect.RewriteQuery(
		`SELECT id, uri, content_sha, mime_type, size_bytes, chunk_count, created_at
		   FROM rag_documents
		  WHERE collection_id=? AND user_id=?
		  ORDER BY created_at DESC, id DESC`,
	), cid, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	out := []ragDocumentAPI{}
	for rows.Next() {
		var d ragDocumentAPI
		if err := rows.Scan(&d.ID, &d.URI, &d.ContentSHA, &d.MIMEType,
			&d.SizeBytes, &d.ChunkCount, &d.CreatedAt); err != nil {
			continue
		}
		out = append(out, d)
	}
	writeJSON(w, 200, map[string]any{"documents": out})
}

// DELETE /api/rag/collections/{id}/documents/{doc_id}
func (s *server) handleRAGDeleteDocument(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid collection id")
		return
	}
	did, err := strconv.ParseInt(r.PathValue("doc_id"), 10, 64)
	if err != nil || did <= 0 {
		writeErr(w, 400, "invalid document id")
		return
	}
	if err := s.removeDocument(u.ID, cid, did); err != nil {
		if strings.Contains(err.Error(), "not found") {
			writeErr(w, 404, err.Error())
			return
		}
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// POST /api/rag/collections/{id}/search
// Body: {query: string, k?: int}
func (s *server) handleRAGSearch(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid id")
		return
	}
	var body struct {
		Query string `json:"query"`
		K     int    `json:"k"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}
	q := strings.TrimSpace(body.Query)
	if q == "" {
		writeErr(w, 400, "query required")
		return
	}
	k := body.K
	if k <= 0 {
		k = ragSearchDefaultK
	}
	if k > ragSearchMaxK {
		k = ragSearchMaxK
	}

	emb := getEmbedder()
	vecs, err := emb.Embed(r.Context(), []string{q})
	if err != nil {
		writeErr(w, 502, "embed: "+err.Error())
		return
	}
	if len(vecs) != 1 {
		writeErr(w, 502, "embedder returned no vector")
		return
	}
	store := newSQLiteVectorStore(s.db, s.dialect)
	hits, err := store.searchByEmbedding(u.ID, cid, vecs[0], k)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	out := make([]ragHitAPI, 0, len(hits))
	for _, h := range hits {
		out = append(out, ragHitAPI{
			ChunkID:    h.Chunk.ID,
			DocumentID: h.Chunk.DocumentID,
			Ordinal:    h.Chunk.OrdinalIdx,
			Text:       h.Chunk.Text,
			TokenCount: h.Chunk.TokenCount,
			Score:      h.Score,
		})
	}
	writeJSON(w, 200, map[string]any{"hits": out})
}

// isUniqueConstraint returns true when err looks like a UNIQUE-violation
// from either SQLite or Postgres.  Inline string matching is the
// portable answer; per-driver error types would require a build tag.
func isUniqueConstraint(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "UNIQUE constraint failed") ||
		strings.Contains(s, "duplicate key value") ||
		strings.Contains(s, "violates unique constraint")
}

// decodeBase64Loose accepts both standard and URL-safe base64, with or
// without padding.  Saves the frontend from second-guessing which
// encoding to send.
func decodeBase64Loose(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if b, err := base64.StdEncoding.DecodeString(s); err == nil {
		return b, nil
	}
	if b, err := base64.RawStdEncoding.DecodeString(s); err == nil {
		return b, nil
	}
	if b, err := base64.URLEncoding.DecodeString(s); err == nil {
		return b, nil
	}
	return base64.RawURLEncoding.DecodeString(s)
}

// POST /api/rag/collections/{id}/hybrid_search
// Body: {query, k?, mode?: "hybrid"|"dense"|"bm25", rerank?: bool}
func (s *server) handleRAGHybridSearch(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	cid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || cid <= 0 {
		writeErr(w, 400, "invalid id")
		return
	}
	var body struct {
		Query  string `json:"query"`
		K      int    `json:"k"`
		Mode   string `json:"mode"`
		Rerank bool   `json:"rerank"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}
	if strings.TrimSpace(body.Query) == "" {
		writeErr(w, 400, "query required")
		return
	}

	// Verify the user owns the collection up-front so we surface a clean
	// 404 instead of an empty result for cross-tenant probes.
	var owner int64
	if err := s.dbQueryRow(s.dialect.RewriteQuery(
		`SELECT user_id FROM rag_collections WHERE id=?`,
	), cid).Scan(&owner); err != nil {
		writeErr(w, 404, "collection not found")
		return
	}
	if owner != u.ID {
		writeErr(w, 404, "collection not found")
		return
	}

	cfg := defaultHybridConfig()
	switch body.Mode {
	case "", "hybrid":
		// keep defaults
	case "dense":
		cfg.UseBM25 = false
	case "bm25":
		cfg.UseDense = false
	default:
		writeErr(w, 400, "mode must be one of: hybrid, dense, bm25")
		return
	}
	if body.K > 0 {
		cfg.K = body.K
		if cfg.K > ragSearchMaxK {
			cfg.K = ragSearchMaxK
		}
	}
	cfg.Rerank = body.Rerank

	hits, err := s.retrieveHybrid(r.Context(), u.ID, cid, body.Query, cfg)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	type hybridHitAPI struct {
		ChunkID    int64   `json:"chunk_id"`
		DocumentID int64   `json:"document_id"`
		Ordinal    int     `json:"ordinal"`
		Text       string  `json:"text"`
		Score      float32 `json:"score"`
		DenseScore float32 `json:"dense_score"`
		BM25Score  float32 `json:"bm25_score"`
	}
	out := make([]hybridHitAPI, 0, len(hits))
	for _, h := range hits {
		out = append(out, hybridHitAPI{
			ChunkID:    h.Chunk.ID,
			DocumentID: h.Chunk.DocumentID,
			Ordinal:    h.Chunk.OrdinalIdx,
			Text:       h.Chunk.Text,
			Score:      h.Score,
			DenseScore: h.DenseScore,
			BM25Score:  h.BM25Score,
		})
	}
	writeJSON(w, 200, map[string]any{"hits": out})
}

// validateRAGCollectionName enforces the same naming discipline as MCP
// names: short, lowercase, no surprises in URL paths.
func validateRAGCollectionName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", errors.New("name required")
	}
	if len(name) > ragMaxNameLen {
		return "", fmt.Errorf("name too long (max %d)", ragMaxNameLen)
	}
	for _, r := range name {
		if !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && r != '-' && r != '_' && r != '.' {
			return "", errors.New("name: lowercase letters/digits/-_. only")
		}
	}
	return name, nil
}
