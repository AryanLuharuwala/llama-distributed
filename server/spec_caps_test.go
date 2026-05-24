package main

import "testing"

func TestUpsertSpecCaps_StoresFields(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "spec-user")
	srv.upsertSpecCaps(uid, "rig-medusa", map[string]any{
		"ok":               true,
		"method":           "medusa",
		"draft_tokens":     float64(4),
		"accept_rate_hint": float64(0.62),
	})
	var ok, k int
	var method string
	var hint float64
	if err := srv.dbQueryRow(
		`SELECT ok, method, draft_tokens, accept_rate_hint
		   FROM spec_caps WHERE user_id = ? AND agent_id = ?`,
		uid, "rig-medusa",
	).Scan(&ok, &method, &k, &hint); err != nil {
		t.Fatalf("select: %v", err)
	}
	if ok != 1 || method != "medusa" || k != 4 || hint < 0.61 || hint > 0.63 {
		t.Errorf("got (%d, %q, %d, %g), want (1, medusa, 4, ~0.62)", ok, method, k, hint)
	}
}

func TestUpsertSpecCaps_RejectsBadMethod(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "spec-bad")
	srv.upsertSpecCaps(uid, "rig-x", map[string]any{
		"ok":           true,
		"method":       "; DROP TABLE spec_caps; --",
		"draft_tokens": float64(8),
	})
	var method string
	if err := srv.dbQueryRow(
		`SELECT method FROM spec_caps WHERE user_id = ? AND agent_id = ?`,
		uid, "rig-x",
	).Scan(&method); err != nil {
		t.Fatalf("select: %v", err)
	}
	if method != "none" {
		t.Errorf("bogus method should normalize to 'none', got %q", method)
	}
}

func TestUpsertSpecCaps_ClampsDraftTokens(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "spec-clamp")
	srv.upsertSpecCaps(uid, "rig-y", map[string]any{
		"ok":           true,
		"method":       "eagle",
		"draft_tokens": float64(500), // way past the sanity cap
	})
	var k int
	if err := srv.dbQueryRow(
		`SELECT draft_tokens FROM spec_caps WHERE user_id = ? AND agent_id = ?`,
		uid, "rig-y",
	).Scan(&k); err != nil {
		t.Fatalf("select: %v", err)
	}
	if k != 32 {
		t.Errorf("draft_tokens=%d, want 32 (clamped)", k)
	}
}
