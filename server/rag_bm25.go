package main

// Okapi BM25 over the rag_chunks table, scored in-process.
//
// Why in-process and not FTS5/tsvector:
//   - FTS5 is a SQLite-specific virtual table.  tsvector is Postgres-only.
//     Maintaining both would mean a separate ingestion path per backend.
//   - Our collections are O(10k–100k chunks) per user; a single sweep
//     per query is ~milliseconds.  We pay one O(N·tokens) sweep on the
//     cold path; the dense path does the same.
//   - When the embedding tier graduates to a real model and per-tenant
//     scale exceeds 1M chunks, the right move is pgvector + tsvector on
//     Postgres, not a SQLite extension.  The dialect abstraction means
//     this swap happens behind the vectorStore / lexicalStore interface.
//
// BM25 implementation: Okapi BM25 with k1=1.2, b=0.75.  We tokenize as
// lower-cased word unigrams; the same normalisation the hashEmbedder
// uses, so the two scorers agree on what "a word" is.

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"strings"
	"unicode"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// bm25Hit mirrors ragHit but carries the BM25 score instead of cosine.
type bm25Hit struct {
	Chunk ragChunk
	Score float32
}

// bm25Search scans every chunk in collectionID, scoring against the
// tokenised query terms, and returns the top-k.  Empty query → empty
// result; we explicitly do not fall back to "return everything".
func bm25Search(ctx context.Context, db *sql.DB, d sqlDialect, collectionID int64, query string, k int) ([]bm25Hit, error) {
	terms := tokenizeBM25(query)
	if len(terms) == 0 || k <= 0 {
		return nil, nil
	}
	termSet := make(map[string]struct{}, len(terms))
	for _, t := range terms {
		termSet[t] = struct{}{}
	}

	// Pass 1: collect chunk_id → (text, tokens, doc-len) and per-term DF.
	type chunkRec struct {
		id       int64
		docID    int64
		ord      int
		text     string
		tokCount int
		tf       map[string]int
		docLen   int
	}
	rows, err := db.QueryContext(ctx, d.RewriteQuery(
		`SELECT id, document_id, ordinal_idx, text, token_count
		   FROM rag_chunks WHERE collection_id=?`,
	), collectionID)
	if err != nil {
		return nil, fmt.Errorf("bm25: query: %w", err)
	}
	defer rows.Close()

	var (
		all       []chunkRec
		df        = make(map[string]int)
		totalLen  int64
	)
	for rows.Next() {
		var c chunkRec
		if err := rows.Scan(&c.id, &c.docID, &c.ord, &c.text, &c.tokCount); err != nil {
			return nil, fmt.Errorf("bm25: scan: %w", err)
		}
		toks := tokenizeBM25(c.text)
		c.docLen = len(toks)
		c.tf = make(map[string]int, 8)
		seen := make(map[string]bool, 8)
		for _, t := range toks {
			c.tf[t]++
			if _, q := termSet[t]; q && !seen[t] {
				df[t]++
				seen[t] = true
			}
		}
		all = append(all, c)
		totalLen += int64(c.docLen)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("bm25: rows: %w", err)
	}
	if len(all) == 0 {
		return nil, nil
	}
	avgLen := float64(totalLen) / float64(len(all))
	N := float64(len(all))

	// Pass 2: score every chunk.  We only score chunks where at least
	// one query term hits — others get -inf and are filtered.
	type scored struct {
		rec   chunkRec
		score float64
	}
	out := make([]scored, 0, len(all))
	for _, c := range all {
		var s float64
		any := false
		for _, term := range terms {
			tf := float64(c.tf[term])
			if tf == 0 {
				continue
			}
			any = true
			d := float64(df[term])
			idf := math.Log(1 + (N-d+0.5)/(d+0.5))
			denom := tf + bm25K1*(1-bm25B+bm25B*(float64(c.docLen)/avgLen))
			s += idf * (tf * (bm25K1 + 1) / denom)
		}
		if !any {
			continue
		}
		out = append(out, scored{rec: c, score: s})
	}
	// Top-k via partial sort.
	if k > len(out) {
		k = len(out)
	}
	for i := 0; i < k; i++ {
		bestJ := i
		for j := i + 1; j < len(out); j++ {
			if out[j].score > out[bestJ].score {
				bestJ = j
			}
		}
		out[i], out[bestJ] = out[bestJ], out[i]
	}
	hits := make([]bm25Hit, k)
	for i := 0; i < k; i++ {
		c := out[i].rec
		hits[i] = bm25Hit{
			Chunk: ragChunk{
				ID:           c.id,
				CollectionID: collectionID,
				DocumentID:   c.docID,
				OrdinalIdx:   c.ord,
				Text:         c.text,
				TokenCount:   c.tokCount,
			},
			Score: float32(out[i].score),
		}
	}
	return hits, nil
}

// tokenizeBM25 lowercases + splits on non-alphanumeric runes.  Matches
// the unigram path inside hashEmbedder so the two scorers agree on what
// constitutes a "word" — useful when reasoning about why a query hit or
// missed.
func tokenizeBM25(s string) []string {
	if s == "" {
		return nil
	}
	s = strings.ToLower(s)
	return strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
