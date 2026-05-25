package main

// sdcpp_caps_test.go — CF12-W4/W6c: cap storage + role/block-split planner.
//
// These tests construct an isolated server (in-memory sqlite) and exercise
// the upsert + planSdcppRoleChain + planSdcppUnetSplit paths.  They do
// NOT spawn a worker — that's covered by the python smoke suites
// (test_sdcpp_daemon.py and test_sdcpp_role_chain.py).

import (
	"context"
	"testing"
)

func mustMigrateSdcpp(t *testing.T, s *server) {
	t.Helper()
	if err := migrateSdcppCaps(s.db, s.dialect); err != nil {
		t.Fatalf("migrateSdcppCaps: %v", err)
	}
}

func TestUpsertSdcppCaps_RolesArrayStored(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	s.upsertSdcppCaps(7, "rig-A", map[string]any{
		"ok":      true,
		"roles":   []any{"full", "te", "unet", "vae"},
		"worker":  "/usr/bin/dist-sdcpp-worker",
		"backend": "vulkan:1 cuda:0",
	})

	rows, err := s.dbQuery(
		`SELECT role FROM sdcpp_caps WHERE user_id = ? AND agent_id = ? ORDER BY role`,
		7, "rig-A")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	var got []string
	for rows.Next() {
		var r string
		if err := rows.Scan(&r); err != nil {
			t.Fatalf("scan: %v", err)
		}
		got = append(got, r)
	}
	want := []string{"full", "te", "unet", "vae"}
	if len(got) != len(want) {
		t.Fatalf("roles: got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("roles[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestUpsertSdcppCaps_DowngradeClearsOldRows(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	// Initial advert: four roles.
	s.upsertSdcppCaps(1, "rig", map[string]any{
		"ok":    true,
		"roles": []any{"full", "te", "unet", "vae"},
	})

	// Downgrade: just full (e.g. operator disabled per-role API).
	s.upsertSdcppCaps(1, "rig", map[string]any{
		"ok":    true,
		"roles": []any{"full"},
	})

	rows, err := s.dbQuery(
		`SELECT role FROM sdcpp_caps WHERE user_id = ? AND agent_id = ?`,
		1, "rig")
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	var n int
	for rows.Next() {
		var r string
		_ = rows.Scan(&r)
		if r != "full" {
			t.Errorf("unexpected role after downgrade: %q", r)
		}
		n++
	}
	if n != 1 {
		t.Errorf("row count after downgrade = %d, want 1", n)
	}
}

func TestUpsertSdcppCaps_NotOkClearsAll(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	s.upsertSdcppCaps(1, "rig", map[string]any{
		"ok":    true,
		"roles": []any{"full"},
	})
	s.upsertSdcppCaps(1, "rig", map[string]any{
		"ok":    false,
		"roles": []any{"full"},
	})

	row := s.dbQueryRow(
		`SELECT COUNT(*) FROM sdcpp_caps WHERE user_id = ? AND agent_id = ?`,
		1, "rig")
	var n int
	if err := row.Scan(&n); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if n != 0 {
		t.Errorf("rows after ok=false = %d, want 0", n)
	}
}

func TestPlanSdcppRoleChain_PicksOnePerRole(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	// Three rigs, each specialising on one role.
	s.upsertSdcppCaps(1, "rig-te",   map[string]any{"ok": true, "roles": []any{"te"}})
	s.upsertSdcppCaps(1, "rig-unet", map[string]any{"ok": true, "roles": []any{"unet"}})
	s.upsertSdcppCaps(1, "rig-vae",  map[string]any{"ok": true, "roles": []any{"vae"}})

	chain := s.planSdcppRoleChain(context.Background(), 1)
	if len(chain) != 3 {
		t.Fatalf("chain length = %d, want 3", len(chain))
	}
	if chain[0].Role != "te" || chain[0].AgentID != "rig-te" {
		t.Errorf("stage 0 = %+v, want te/rig-te", chain[0])
	}
	if chain[1].Role != "unet" || chain[1].AgentID != "rig-unet" {
		t.Errorf("stage 1 = %+v, want unet/rig-unet", chain[1])
	}
	if chain[2].Role != "vae" || chain[2].AgentID != "rig-vae" {
		t.Errorf("stage 2 = %+v, want vae/rig-vae", chain[2])
	}
}

func TestPlanSdcppRoleChain_NilWhenRoleMissing(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	// Missing VAE rig.
	s.upsertSdcppCaps(1, "rig-te",   map[string]any{"ok": true, "roles": []any{"te"}})
	s.upsertSdcppCaps(1, "rig-unet", map[string]any{"ok": true, "roles": []any{"unet"}})

	if chain := s.planSdcppRoleChain(context.Background(), 1); chain != nil {
		t.Errorf("expected nil chain when VAE missing; got %+v", chain)
	}
}

func TestPlanSdcppFull_PrefersOwnerRig(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	s.upsertSdcppCaps(9, "stranger", map[string]any{"ok": true, "roles": []any{"full"}})
	s.upsertSdcppCaps(1, "mine",     map[string]any{"ok": true, "roles": []any{"full"}})

	a, ok := s.planSdcppFull(context.Background(), 1)
	if !ok {
		t.Fatalf("expected a full rig")
	}
	if a.AgentID != "mine" {
		t.Errorf("got %s, want mine (owner pref)", a.AgentID)
	}
}

func TestPlanSdcppUnetSplit_EmptyWhenNoBlockRig(t *testing.T) {
	// W6c: planner gracefully degrades when no rig has block-split caps.
	// This is the current expected state until CF12-W6a upstream lands.
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	s.upsertSdcppCaps(1, "unet-only", map[string]any{"ok": true, "roles": []any{"unet"}})

	if ranges := s.planSdcppUnetSplit(context.Background(), 1, 7, 3); ranges != nil {
		t.Errorf("expected nil block-split plan; got %+v", ranges)
	}
}

func TestPlanSdcppUnetSplit_PartitionsBlocks(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	// Three rigs that *do* advertise unet_blocks.  These wouldn't exist
	// today (the C++ adapter returns ENOTIMPL for partial ranges) but
	// the planner is wire-ready for when CF12-W6a lands — exercising
	// the partitioning math here is cheap insurance.
	s.upsertSdcppCaps(1, "rig-a", map[string]any{"ok": true, "roles": []any{"unet_blocks"}})
	s.upsertSdcppCaps(1, "rig-b", map[string]any{"ok": true, "roles": []any{"unet_blocks"}})
	s.upsertSdcppCaps(1, "rig-c", map[string]any{"ok": true, "roles": []any{"unet_blocks"}})

	ranges := s.planSdcppUnetSplit(context.Background(), 1, 7, 3)
	if len(ranges) != 3 {
		t.Fatalf("stages = %d, want 3", len(ranges))
	}
	// Partition for total=7, stages=3: sizes 3,2,2 → [0,3),[3,5),[5,7).
	expects := []struct{ lo, hi int }{{0, 3}, {3, 5}, {5, 7}}
	for i, want := range expects {
		if ranges[i].BlockLo != want.lo || ranges[i].BlockHi != want.hi {
			t.Errorf("stage %d = [%d,%d), want [%d,%d)",
				i, ranges[i].BlockLo, ranges[i].BlockHi, want.lo, want.hi)
		}
		if ranges[i].BlockTotal != 7 {
			t.Errorf("stage %d total = %d, want 7", i, ranges[i].BlockTotal)
		}
	}
}

func TestPlanSdcppUnetSplit_NeedsEnoughRigs(t *testing.T) {
	s := newTestServer(t)
	mustMigrateSdcpp(t, s)

	s.upsertSdcppCaps(1, "rig-a", map[string]any{"ok": true, "roles": []any{"unet_blocks"}})

	if ranges := s.planSdcppUnetSplit(context.Background(), 1, 7, 3); ranges != nil {
		t.Errorf("expected nil when stages > available rigs; got %+v", ranges)
	}
}
