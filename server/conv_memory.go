package main

// Conversation memory: per-(user, conversation) summaries that survive
// across turns even after the underlying transcript scrolls out of the
// model's context window.
//
// Two retrieval modes:
//   • Recent — last N memory rows in chronological order.  Useful for
//     the "what did we just discuss" recall on every turn.
//   • Semantic — top-k rows by cosine similarity to a query embedding.
//     Useful when the user references something from much earlier in a
//     long conversation.  The system prompt blends both.
//
// Memory rows have an optional "fact" classification — a one-shot
// extractor (run on the agent or via the embed tier) can promote a
// derived fact ("user prefers TypeScript", "project deadline is Q3")
// out of the rolling summary so it's never compressed away.
//
// Storage shape mirrors rag_chunks: text + packed-float32 embedding
// blob.  We share encodeEmbedding/decodeEmbedding and cosineSimilarity
// from rag.go so the dim-handling code stays in one place.

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	memKindSummary = "summary"
	memKindFact    = "fact"
)

// migrateConvMemory installs the conv_memories table.  Idempotent across
// SQLite + Postgres via the dialect helper.
func migrateConvMemory(db *sql.DB, d sqlDialect) error {
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS conv_memories (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id         INTEGER NOT NULL,
			conversation_id TEXT    NOT NULL,
			kind            TEXT    NOT NULL,
			text            TEXT    NOT NULL,
			token_count     INTEGER NOT NULL DEFAULT 0,
			embedding       BLOB,
			created_at      INTEGER NOT NULL
		)
	`)); err != nil {
		return fmt.Errorf("create conv_memories: %w", err)
	}
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE INDEX IF NOT EXISTS idx_conv_memories_user_conv
			ON conv_memories (user_id, conversation_id, created_at)
	`)); err != nil {
		return fmt.Errorf("create idx_conv_memories_user_conv: %w", err)
	}
	return nil
}

type convMemory struct {
	ID             int64
	UserID         int64
	ConversationID string
	Kind           string
	Text           string
	TokenCount     int
	Embedding      []float32
	CreatedAt      int64
}

// appendMemory writes one row.  Embedding is computed by the caller
// (this keeps the embedder pluggable; tests can pass nil to skip).
func (s *server) appendMemory(ctx context.Context, m convMemory) (int64, error) {
	if m.UserID == 0 {
		return 0, errors.New("conv: user_id required")
	}
	m.ConversationID = strings.TrimSpace(m.ConversationID)
	if m.ConversationID == "" {
		return 0, errors.New("conv: conversation_id required")
	}
	if m.Kind == "" {
		m.Kind = memKindSummary
	}
	if m.Kind != memKindSummary && m.Kind != memKindFact {
		return 0, fmt.Errorf("conv: invalid kind %q", m.Kind)
	}
	if strings.TrimSpace(m.Text) == "" {
		return 0, errors.New("conv: text required")
	}
	if m.CreatedAt == 0 {
		m.CreatedAt = time.Now().Unix()
	}
	if m.TokenCount == 0 {
		m.TokenCount = approxTokens(m.Text)
	}
	res, err := s.db.ExecContext(ctx, s.dialect.RewriteQuery(
		`INSERT INTO conv_memories
			(user_id, conversation_id, kind, text, token_count, embedding, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?)`,
	), m.UserID, m.ConversationID, m.Kind, m.Text, m.TokenCount, encodeEmbedding(m.Embedding), m.CreatedAt)
	if err != nil {
		return 0, fmt.Errorf("conv: insert memory: %w", err)
	}
	id, _ := res.LastInsertId()
	return id, nil
}

// recentMemories returns up to n rows ordered newest-first.  Caller
// usually presents them in chronological order (reverse this slice)
// for the prompt builder.
func (s *server) recentMemories(ctx context.Context, uid int64, conv string, n int) ([]convMemory, error) {
	if n <= 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, s.dialect.RewriteQuery(
		`SELECT id, kind, text, token_count, embedding, created_at
		   FROM conv_memories
		  WHERE user_id=? AND conversation_id=?
		  ORDER BY created_at DESC, id DESC
		  LIMIT ?`,
	), uid, conv, n)
	if err != nil {
		return nil, fmt.Errorf("conv: query recent: %w", err)
	}
	defer rows.Close()
	out := []convMemory{}
	for rows.Next() {
		var m convMemory
		m.UserID = uid
		m.ConversationID = conv
		var emb []byte
		if err := rows.Scan(&m.ID, &m.Kind, &m.Text, &m.TokenCount, &emb, &m.CreatedAt); err != nil {
			return nil, fmt.Errorf("conv: scan: %w", err)
		}
		if v, err := decodeEmbedding(emb); err == nil {
			m.Embedding = v
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// recallSemantic returns the k memory rows whose embedding most
// resembles query (cosine).  Facts are upweighted by a small constant
// so a stable fact ("user prefers TypeScript") outranks a transient
// summary at the same cosine distance.
func (s *server) recallSemantic(ctx context.Context, uid int64, conv string, query []float32, k int) ([]convMemory, error) {
	if k <= 0 || len(query) == 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, s.dialect.RewriteQuery(
		`SELECT id, kind, text, token_count, embedding, created_at
		   FROM conv_memories
		  WHERE user_id=? AND conversation_id=? AND embedding IS NOT NULL`,
	), uid, conv)
	if err != nil {
		return nil, fmt.Errorf("conv: query semantic: %w", err)
	}
	defer rows.Close()

	type scored struct {
		m     convMemory
		score float32
	}
	var all []scored
	for rows.Next() {
		var m convMemory
		m.UserID = uid
		m.ConversationID = conv
		var emb []byte
		if err := rows.Scan(&m.ID, &m.Kind, &m.Text, &m.TokenCount, &emb, &m.CreatedAt); err != nil {
			return nil, fmt.Errorf("conv: scan semantic: %w", err)
		}
		v, err := decodeEmbedding(emb)
		if err != nil || len(v) == 0 {
			continue
		}
		m.Embedding = v
		score := cosineSimilarity(query, v)
		if m.Kind == memKindFact {
			score += 0.05
		}
		all = append(all, scored{m: m, score: score})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("conv: rows: %w", err)
	}
	// Partial sort top-k.
	if k > len(all) {
		k = len(all)
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(all); j++ {
			if all[j].score > all[best].score {
				best = j
			}
		}
		all[i], all[best] = all[best], all[i]
	}
	out := make([]convMemory, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].m
	}
	return out, nil
}

// forgetConversation removes every memory row for a conversation.  Used
// when the user explicitly clears chat history or deletes a thread.
func (s *server) forgetConversation(ctx context.Context, uid int64, conv string) (int64, error) {
	res, err := s.db.ExecContext(ctx, s.dialect.RewriteQuery(
		`DELETE FROM conv_memories WHERE user_id=? AND conversation_id=?`,
	), uid, conv)
	if err != nil {
		return 0, fmt.Errorf("conv: delete: %w", err)
	}
	n, _ := res.RowsAffected()
	return n, nil
}

// summarizeTurn is the writer-side helper: takes a freshly-completed
// turn (or a rolling window of turns), produces a compact summary, and
// stores it with an embedding so future turns can recall it.
//
// The actual summarisation is delegated to a summarizer interface so
// we can plug a real LLM call later.  For now the default extracts
// the user-side text verbatim (capped) — good enough to thread context
// in dev/test, and easy to swap.
type summarizer interface {
	Summarize(ctx context.Context, turn string) (string, error)
}

type passthroughSummarizer struct{ maxChars int }

func (p passthroughSummarizer) Summarize(_ context.Context, turn string) (string, error) {
	t := strings.TrimSpace(turn)
	if p.maxChars > 0 && len(t) > p.maxChars {
		t = t[:p.maxChars] + "…"
	}
	return t, nil
}

// summarizeAndStore is the convenience helper for the chat-completions
// surface: build a summary, embed it, append it.  Returns the row id.
func (s *server) summarizeAndStore(ctx context.Context, uid int64, conv string, kind, turn string, sum summarizer) (int64, error) {
	if sum == nil {
		sum = passthroughSummarizer{maxChars: 1024}
	}
	summary, err := sum.Summarize(ctx, turn)
	if err != nil {
		return 0, fmt.Errorf("conv: summarize: %w", err)
	}
	if summary == "" {
		return 0, errors.New("conv: empty summary")
	}
	vecs, err := getEmbedder().Embed(ctx, []string{summary})
	if err != nil {
		return 0, fmt.Errorf("conv: embed summary: %w", err)
	}
	return s.appendMemory(ctx, convMemory{
		UserID:         uid,
		ConversationID: conv,
		Kind:           kind,
		Text:           summary,
		Embedding:      vecs[0],
	})
}
