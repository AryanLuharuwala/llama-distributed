package main

import (
	"context"
	"strings"
	"testing"
)

func TestAugmentMessagesNoopsWhenNoOptions(t *testing.T) {
	s, _ := openMCPTestDB(t)
	in := []oaiMsg{{Role: "user", Content: "hi"}}
	out, err := s.augmentMessages(context.Background(), 1, in, nil, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 1 || out[0].Content != "hi" {
		t.Errorf("expected no augmentation, got %+v", out)
	}
}

func TestAugmentRAGPrependsContext(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(5)
	cid := newRAGCollection(t, s, uid, "ctx-test")
	if _, err := s.ingestDocument(context.Background(), ingestRequest{
		UserID: uid, CollectionID: cid, URI: "inline:a",
		Body: []byte("Frostpunk is a city-builder where steam-powered generators fight the cold."),
	}); err != nil {
		t.Fatal(err)
	}

	in := []oaiMsg{{Role: "user", Content: "Tell me about Frostpunk."}}
	out, err := s.augmentMessages(context.Background(), uid, in,
		&chatRAGOpts{CollectionID: cid, K: 3}, nil)
	if err != nil {
		t.Fatalf("augment: %v", err)
	}
	if len(out) < 2 {
		t.Fatalf("expected >=2 msgs after augmentation, got %d", len(out))
	}
	if out[0].Role != "system" || !strings.Contains(out[0].Content, "RAG context") {
		t.Errorf("expected RAG system block first, got: role=%q content=%q", out[0].Role, out[0].Content)
	}
	if !strings.Contains(out[0].Content, "Frostpunk") {
		t.Errorf("expected retrieved chunk text in block, got: %q", out[0].Content)
	}
}

func TestAugmentMemoryPrependsRecall(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	const uid = int64(3)
	_, _ = s.summarizeAndStore(context.Background(), uid, "conv1", memKindFact,
		"User's primary language is Spanish.", nil)
	_, _ = s.summarizeAndStore(context.Background(), uid, "conv1", memKindSummary,
		"Previously discussed React component patterns.", nil)

	in := []oaiMsg{{Role: "user", Content: "What language do I prefer?"}}
	out, err := s.augmentMessages(context.Background(), uid, in, nil,
		&chatMemoryOpts{ConversationID: "conv1", RecallK: 2})
	if err != nil {
		t.Fatalf("augment: %v", err)
	}
	if len(out) < 2 {
		t.Fatal("expected memory block to be prepended")
	}
	if out[0].Role != "system" || !strings.Contains(out[0].Content, "memory") {
		t.Errorf("expected memory system block, got: role=%q content=%q", out[0].Role, out[0].Content)
	}
}

func TestSummarizeChatTurnPersists(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(32))
	defer setEmbedder(nil)

	if err := s.summarizeChatTurn(context.Background(), 1, "thread-1",
		"What is 2+2?", "Four."); err != nil {
		t.Fatalf("summarize: %v", err)
	}
	rows, _ := s.recentMemories(context.Background(), 1, "thread-1", 5)
	if len(rows) != 1 {
		t.Fatalf("want 1 stored memory, got %d", len(rows))
	}
	if !strings.Contains(rows[0].Text, "2+2") || !strings.Contains(rows[0].Text, "Four") {
		t.Errorf("summary text missing pieces: %q", rows[0].Text)
	}
}

func TestResolveCollectionByName(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(32))
	defer setEmbedder(nil)

	const uid = int64(8)
	cid := newRAGCollection(t, s, uid, "mydocs")
	id, dim, err := s.resolveCollection(uid, &chatRAGOpts{CollectionName: "mydocs"})
	if err != nil || id != cid || dim != 32 {
		t.Errorf("resolveCollection: id=%d dim=%d err=%v want id=%d dim=32", id, dim, err, cid)
	}
	// Wrong user → not found.
	if _, _, err := s.resolveCollection(999, &chatRAGOpts{CollectionName: "mydocs"}); err == nil {
		t.Errorf("expected error for cross-user resolve")
	}
}

func TestLastUserContent(t *testing.T) {
	msgs := []oaiMsg{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "first question"},
		{Role: "assistant", Content: "first answer"},
		{Role: "user", Content: "second question"},
	}
	if got := lastUserContent(msgs); got != "second question" {
		t.Errorf("got %q want 'second question'", got)
	}
	if got := lastUserContent(nil); got != "" {
		t.Errorf("nil input should return empty, got %q", got)
	}
}
