package main

// Tests for the pluggable counter store.  The SQLite-backed default
// implements the legacy upsert-then-update path; the Bigtable stub
// proxies to the fallback and surfaces a call counter for ops to
// monitor before flipping the real SDK on.

import (
	"context"
	"testing"
)

func TestSQLiteCounterStore_RoundTrip(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "counter-user")
	cs := newSQLiteCounterStore(srv)

	ctx := context.Background()
	if err := cs.IncrBy(ctx, uid, "202605", 100, 200); err != nil {
		t.Fatalf("IncrBy: %v", err)
	}
	if err := cs.IncrBy(ctx, uid, "202605", 7, 13); err != nil {
		t.Fatalf("IncrBy: %v", err)
	}
	in, out, err := cs.Get(ctx, uid, "202605")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if in != 107 || out != 213 {
		t.Errorf("got (%d,%d), want (107,213)", in, out)
	}
}

func TestSQLiteCounterStore_MissingRow(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "no-counter-user")
	cs := newSQLiteCounterStore(srv)

	in, out, err := cs.Get(context.Background(), uid, "202605")
	if err != nil {
		t.Fatalf("Get on missing row should return zeros without error: %v", err)
	}
	if in != 0 || out != 0 {
		t.Errorf("got (%d,%d), want (0,0)", in, out)
	}
}

func TestBigtableCounterStore_FallsBackToSQLite(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "bt-user")
	fb := newSQLiteCounterStore(srv)
	bt, err := newBigtableCounterStore(bigtableCounterConfig{
		ProjectID: "p", InstanceID: "i", TableID: "t",
	}, fb)
	if err != nil {
		t.Fatalf("newBigtableCounterStore: %v", err)
	}
	ctx := context.Background()
	if err := bt.IncrBy(ctx, uid, "202605", 50, 75); err != nil {
		t.Fatalf("IncrBy: %v", err)
	}
	in, out, err := bt.Get(ctx, uid, "202605")
	if err != nil || in != 50 || out != 75 {
		t.Errorf("Get returned (%d,%d,%v), want (50,75,nil)", in, out, err)
	}
	if bt.calls.Load() != 1 {
		t.Errorf("calls=%d, want 1 (stub records every IncrBy)", bt.calls.Load())
	}
}

func TestBigtableCounterStore_RejectsBadConfig(t *testing.T) {
	srv := newTestServer(t)
	_ = srv
	fb := newSQLiteCounterStore(srv)
	if _, err := newBigtableCounterStore(bigtableCounterConfig{}, fb); err == nil {
		t.Error("expected error for missing project/instance/table")
	}
	if _, err := newBigtableCounterStore(bigtableCounterConfig{
		ProjectID: "p", InstanceID: "i", TableID: "t",
	}, nil); err == nil {
		t.Error("expected error for nil fallback")
	}
}

func TestRowKeyForCounter_OrderingIsLex(t *testing.T) {
	a := rowKeyForCounter(1, "202605")
	b := rowKeyForCounter(2, "202605")
	if a >= b {
		t.Errorf("row keys must sort by user_id within a period: %q vs %q", a, b)
	}
	c := rowKeyForCounter(1, "202604")
	if c >= a {
		t.Errorf("row keys must sort by period first: %q vs %q", c, a)
	}
}
