package main

// Chat-completion augmentation: stitch RAG retrieval + conversation
// memory into the prompt before the rig sees it.
//
// The OpenAI request shape is extended with two optional objects:
//
//   {
//     ...,
//     "rag": {
//       "collection_id":   123,           // OR
//       "collection_name": "my-docs",
//       "k": 6,                            // retrieval depth
//       "mode": "hybrid" | "dense" | "bm25"
//     },
//     "memory": {
//       "conversation_id": "thread-42",   // required to enable memory
//       "recall_k": 4,                    // semantic recall depth
//       "summarize_after": true           // store summary post-response
//     }
//   }
//
// When `rag` is set we run the configured retriever against the last
// user turn and prepend a system message with the retrieved chunks.
// When `memory.conversation_id` is set we pull recent + semantic
// memories and prepend them as another system message.  Both blocks
// are clearly delimited so the model can attribute its answer.
//
// Surfaced via the same handler the dashboard's chat panel uses.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

type chatRAGOpts struct {
	CollectionID   int64  `json:"collection_id"`
	CollectionName string `json:"collection_name"`
	K              int    `json:"k"`
	Mode           string `json:"mode"`
}

type chatMemoryOpts struct {
	ConversationID string `json:"conversation_id"`
	RecallK        int    `json:"recall_k"`
	SummarizeAfter bool   `json:"summarize_after"`
}

// augmentMessages prepends retrieval + memory context to the message
// thread.  Returns the augmented slice; the input is never mutated.
// Errors only when the user references a collection that doesn't
// exist / isn't owned by them — soft failures (no rig, empty hits)
// just degrade silently.
func (s *server) augmentMessages(
	ctx context.Context,
	uid int64,
	msgs []oaiMsg,
	rag *chatRAGOpts,
	mem *chatMemoryOpts,
) ([]oaiMsg, error) {
	if (rag == nil || rag.isEmpty()) && (mem == nil || mem.isEmpty()) {
		return msgs, nil
	}
	// Find the last user message to drive retrieval.  Falls back to
	// the concatenated thread if no user turn yet (rare; assistant-first
	// system prompt eval).
	query := lastUserContent(msgs)
	if query == "" {
		query = concatContent(msgs, 2048)
	}

	var prepend []oaiMsg

	if mem != nil && !mem.isEmpty() {
		block, err := s.buildMemoryBlock(ctx, uid, mem, query)
		if err != nil {
			return nil, err
		}
		if block != "" {
			prepend = append(prepend, oaiMsg{Role: "system", Content: block})
		}
	}

	if rag != nil && !rag.isEmpty() && query != "" {
		block, err := s.buildRAGBlock(ctx, uid, rag, query)
		if err != nil {
			return nil, err
		}
		if block != "" {
			prepend = append(prepend, oaiMsg{Role: "system", Content: block})
		}
	}

	if len(prepend) == 0 {
		return msgs, nil
	}
	out := make([]oaiMsg, 0, len(prepend)+len(msgs))
	out = append(out, prepend...)
	out = append(out, msgs...)
	return out, nil
}

func (o *chatRAGOpts) isEmpty() bool {
	return o == nil || (o.CollectionID == 0 && strings.TrimSpace(o.CollectionName) == "")
}

func (o *chatMemoryOpts) isEmpty() bool {
	return o == nil || strings.TrimSpace(o.ConversationID) == ""
}

func lastUserContent(msgs []oaiMsg) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return strings.TrimSpace(msgs[i].Content)
		}
	}
	return ""
}

func concatContent(msgs []oaiMsg, max int) string {
	var b strings.Builder
	for _, m := range msgs {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(m.Content)
		if b.Len() >= max {
			break
		}
	}
	out := b.String()
	if len(out) > max {
		out = out[:max]
	}
	return out
}

// resolveCollection finds the collection by id or name and verifies the
// caller owns it.  Returns the row's id+dim or an error.
func (s *server) resolveCollection(uid int64, opts *chatRAGOpts) (id int64, dim int, err error) {
	if opts.CollectionID > 0 {
		err = s.db.QueryRow(s.dialect.RewriteQuery(
			`SELECT id, embedding_dim FROM rag_collections WHERE id=? AND user_id=?`,
		), opts.CollectionID, uid).Scan(&id, &dim)
	} else {
		err = s.db.QueryRow(s.dialect.RewriteQuery(
			`SELECT id, embedding_dim FROM rag_collections WHERE name=? AND user_id=?`,
		), strings.TrimSpace(opts.CollectionName), uid).Scan(&id, &dim)
	}
	if err != nil {
		return 0, 0, fmt.Errorf("rag: collection not found: %w", err)
	}
	return id, dim, nil
}

func (s *server) buildRAGBlock(ctx context.Context, uid int64, opts *chatRAGOpts, query string) (string, error) {
	cid, _, err := s.resolveCollection(uid, opts)
	if err != nil {
		return "", err
	}
	cfg := defaultHybridConfig()
	if opts.K > 0 {
		cfg.K = opts.K
		if cfg.K > ragSearchMaxK {
			cfg.K = ragSearchMaxK
		}
	}
	switch opts.Mode {
	case "", "hybrid":
	case "dense":
		cfg.UseBM25 = false
	case "bm25":
		cfg.UseDense = false
	default:
		return "", errors.New("rag: mode must be hybrid|dense|bm25")
	}
	hits, err := s.retrieveHybrid(ctx, uid, cid, query, cfg)
	if err != nil {
		return "", err
	}
	if len(hits) == 0 {
		return "", nil
	}
	var b strings.Builder
	b.WriteString("[RAG context: top-")
	b.WriteString(augItoa(len(hits)))
	b.WriteString(" passages, score ranked]\n")
	for i, h := range hits {
		fmt.Fprintf(&b, "\n[#%d] (doc=%d ord=%d score=%.3f)\n",
			i+1, h.Chunk.DocumentID, h.Chunk.OrdinalIdx, h.Score)
		b.WriteString(h.Chunk.Text)
		b.WriteString("\n")
	}
	b.WriteString("\n[end RAG context]")
	return b.String(), nil
}

func (s *server) buildMemoryBlock(ctx context.Context, uid int64, opts *chatMemoryOpts, query string) (string, error) {
	conv := strings.TrimSpace(opts.ConversationID)
	if conv == "" {
		return "", nil
	}
	recent, err := s.recentMemories(ctx, uid, conv, 4)
	if err != nil {
		return "", err
	}

	var semantic []convMemory
	if opts.RecallK > 0 && query != "" {
		emb := getEmbedder()
		vecs, embErr := emb.Embed(ctx, []string{query})
		if embErr == nil && len(vecs) == 1 {
			semantic, _ = s.recallSemantic(ctx, uid, conv, vecs[0], opts.RecallK)
		}
	}

	// Dedup by ID — semantic + recent can overlap.
	seen := make(map[int64]bool)
	var b strings.Builder
	header := false
	add := func(m convMemory) {
		if seen[m.ID] {
			return
		}
		seen[m.ID] = true
		if !header {
			b.WriteString("[Conversation memory: prior context for this thread]\n")
			header = true
		}
		fmt.Fprintf(&b, "\n[%s] %s\n", strings.ToUpper(m.Kind), m.Text)
	}
	// Show recent in chronological order (oldest first).
	for i := len(recent) - 1; i >= 0; i-- {
		add(recent[i])
	}
	for _, m := range semantic {
		add(m)
	}
	if !header {
		return "", nil
	}
	b.WriteString("\n[end memory]")
	return b.String(), nil
}

// itoa is a tiny non-allocating int → string for hot prompt builders.
// strconv works fine; this just keeps the call site terse.
func augItoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// summarizeChatTurnSafely is the goroutine entry-point called after the
// handler returns the response.  It bounds the work with its own
// context + timeout so a slow summariser can't pile background
// goroutines forever, and recovers from any panic the embed path
// might raise (we're in a detached goroutine — a panic here takes
// down the whole process).
func (s *server) summarizeChatTurnSafely(uid int64, conv, userMsg, assistantMsg string) {
	defer func() {
		// Detached goroutine — never let a panic escape.
		_ = recover()
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = s.summarizeChatTurn(ctx, uid, conv, userMsg, assistantMsg)
}

// summarizeChatTurn writes the user→assistant turn into conv_memories.
// Called after the chat-completion handler has streamed the reply.  The
// summary is deliberately simple: "User: <q>\nAssistant: <a>" capped at
// 1 KiB.  A real LLM-based summariser plugs in via setSummarizer().
func (s *server) summarizeChatTurn(ctx context.Context, uid int64, conv, userMsg, assistantMsg string) error {
	if strings.TrimSpace(conv) == "" {
		return nil
	}
	if strings.TrimSpace(userMsg) == "" && strings.TrimSpace(assistantMsg) == "" {
		return nil
	}
	turn := fmt.Sprintf("User: %s\nAssistant: %s", userMsg, assistantMsg)
	_, err := s.summarizeAndStore(ctx, uid, conv, memKindSummary, turn, getSummarizer())
	return err
}

// summarizer plug — main() can swap in an LLM-backed implementation
// when the embed tier ships one.  Default is the passthrough that
// truncates at 1 KiB.
var summarizerCurrent summarizer = passthroughSummarizer{maxChars: 1024}

func getSummarizer() summarizer { return summarizerCurrent }
func setSummarizer(sm summarizer) {
	if sm == nil {
		summarizerCurrent = passthroughSummarizer{maxChars: 1024}
		return
	}
	summarizerCurrent = sm
}
