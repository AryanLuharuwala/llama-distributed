package main

import (
	"context"
	"testing"
)

func TestConvMemoryAppendAndRecent(t *testing.T) {
	s, _ := openMCPTestDB(t)

	for _, txt := range []string{"first turn", "second turn", "third turn"} {
		if _, err := s.appendMemory(context.Background(), convMemory{
			UserID: 1, ConversationID: "c1", Kind: memKindSummary, Text: txt,
		}); err != nil {
			t.Fatalf("append %q: %v", txt, err)
		}
	}
	got, err := s.recentMemories(context.Background(), 1, "c1", 5)
	if err != nil {
		t.Fatalf("recent: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d rows, want 3", len(got))
	}
	if got[0].Text != "third turn" {
		t.Errorf("newest-first expected, got %q first", got[0].Text)
	}
}

func TestConvMemoryACL(t *testing.T) {
	s, _ := openMCPTestDB(t)
	_, err := s.appendMemory(context.Background(), convMemory{
		UserID: 1, ConversationID: "c1", Text: "alice note",
	})
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	rows, err := s.recentMemories(context.Background(), 2, "c1", 5)
	if err != nil {
		t.Fatalf("recent: %v", err)
	}
	if len(rows) != 0 {
		t.Errorf("user 2 should not see user 1's memory, got %d rows", len(rows))
	}
}

func TestConvMemorySemanticRecall(t *testing.T) {
	s, _ := openMCPTestDB(t)
	setEmbedder(newHashEmbedder(64))
	defer setEmbedder(nil)

	// Store three memories with embeddings.
	ctx := context.Background()
	if _, err := s.summarizeAndStore(ctx, 1, "c1", memKindSummary,
		"User asked about rocket engines and combustion chambers.", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.summarizeAndStore(ctx, 1, "c1", memKindFact,
		"User prefers Python over Rust for their day job.", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.summarizeAndStore(ctx, 1, "c1", memKindSummary,
		"User discussed gardening tomatoes and compost timing.", nil); err != nil {
		t.Fatal(err)
	}

	emb := getEmbedder()
	q, _ := emb.Embed(ctx, []string{"What programming language does the user use?"})
	hits, err := s.recallSemantic(ctx, 1, "c1", q[0], 2)
	if err != nil {
		t.Fatalf("recall: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one hit")
	}
	// Fact about Python/Rust should be at or near the top.
	foundFact := false
	for _, h := range hits {
		if h.Kind == memKindFact {
			foundFact = true
			break
		}
	}
	if !foundFact {
		t.Errorf("expected the fact row in recall results, got: %+v", hits)
	}
}

func TestConvMemoryForget(t *testing.T) {
	s, _ := openMCPTestDB(t)
	for i := 0; i < 5; i++ {
		_, _ = s.appendMemory(context.Background(), convMemory{
			UserID: 1, ConversationID: "c1", Text: "row",
		})
	}
	n, err := s.forgetConversation(context.Background(), 1, "c1")
	if err != nil {
		t.Fatalf("forget: %v", err)
	}
	if n != 5 {
		t.Errorf("forget removed %d, want 5", n)
	}
	rows, _ := s.recentMemories(context.Background(), 1, "c1", 10)
	if len(rows) != 0 {
		t.Errorf("after forget: %d rows remain", len(rows))
	}
}

func TestConvMemoryValidationRejectsBadInput(t *testing.T) {
	s, _ := openMCPTestDB(t)
	bad := []convMemory{
		{UserID: 0, ConversationID: "c1", Text: "x"},
		{UserID: 1, ConversationID: "", Text: "x"},
		{UserID: 1, ConversationID: "c1", Text: ""},
		{UserID: 1, ConversationID: "c1", Text: "x", Kind: "weird"},
	}
	for i, m := range bad {
		if _, err := s.appendMemory(context.Background(), m); err == nil {
			t.Errorf("case %d: expected error, got nil", i)
		}
	}
}
