package main

import (
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestBytesToMaxTokens(t *testing.T) {
	cases := []struct {
		b    int
		want int
	}{
		{0, 0},
		{-1, 0},
		{1, 0},
		{3, 2},
		{15, 10},
		{1500, 1000},
	}
	for _, c := range cases {
		if got := bytesToMaxTokens(c.b); got != c.want {
			t.Errorf("bytesToMaxTokens(%d) = %d, want %d", c.b, got, c.want)
		}
	}
}

func TestClampReportedTokens(t *testing.T) {
	cases := []struct {
		name        string
		in, out     int
		promptChars int
		bytes       int
		wantIn      int
		wantOut     int
		wantClamped bool
	}{
		{"honest report passes through", 100, 50, 400, 90, 100, 50, false},
		{"out report inflated", 100, 1000, 400, 60, 100, 40, true}, // 60*2/3 = 40
		{"in report inflated", 9999, 50, 60, 90, 36, 50, true},     // 60/3 + 16 = 36
		{"negative gets normalized", -5, -1, 100, 100, 0, 0, true},
		{"both fields clamped", 9999, 9999, 60, 60, 36, 40, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotIn, gotOut, clamped := clampReportedTokens(c.in, c.out, c.promptChars, c.bytes)
			if gotIn != c.wantIn || gotOut != c.wantOut || clamped != c.wantClamped {
				t.Errorf("clampReportedTokens(%d, %d, %d, %d) = (%d, %d, %v); want (%d, %d, %v)",
					c.in, c.out, c.promptChars, c.bytes,
					gotIn, gotOut, clamped, c.wantIn, c.wantOut, c.wantClamped)
			}
		})
	}
}

func openDriftTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:?_journal=WAL&_busy_timeout=2000")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := migrateQuarantine(db, sqliteDialect{}); err != nil {
		t.Fatalf("migrateQuarantine: %v", err)
	}
	return db
}

func TestRigDriftQuarantineFires(t *testing.T) {
	db := openDriftTestDB(t)
	defer db.Close()
	tab := newRigDriftTable()

	// 100 settlements where the rig over-reports by 50%.
	for i := 0; i < driftSampleWindow-1; i++ {
		if got := tab.observe(db, "rig-evil", 150, 100, 1024); got {
			t.Fatalf("quarantine fired early at i=%d", i)
		}
	}
	if !tab.observe(db, "rig-evil", 150, 100, 1024) {
		t.Fatalf("expected quarantine on the %dth observation", driftSampleWindow)
	}
	if !tab.isQuarantined("rig-evil") {
		t.Errorf("isQuarantined should be true after threshold breach")
	}

	// Persistence check: open a fresh table and hydrate from db — the
	// rig should still be considered quarantined.
	fresh := newRigDriftTable()
	if err := fresh.hydrateQuarantine(db); err != nil {
		t.Fatalf("hydrate: %v", err)
	}
	if !fresh.isQuarantined("rig-evil") {
		t.Errorf("quarantine should survive a restart via rig_quarantine table")
	}
}

func TestRigDriftHonestRigStaysClean(t *testing.T) {
	db := openDriftTestDB(t)
	defer db.Close()
	tab := newRigDriftTable()

	for i := 0; i < 5*driftSampleWindow; i++ {
		// Honest rig reports exactly at the server ceiling.
		if got := tab.observe(db, "rig-honest", 100, 100, 1024); got {
			t.Fatalf("honest rig should never quarantine; fired at i=%d", i)
		}
	}
	if tab.isQuarantined("rig-honest") {
		t.Errorf("honest rig should not be quarantined")
	}
}

func TestRigDriftIgnoresTinyStreams(t *testing.T) {
	db := openDriftTestDB(t)
	defer db.Close()
	tab := newRigDriftTable()

	// A million tiny streams that lie outrageously must not trip the
	// quarantine — the sample threshold filters out noise.
	for i := 0; i < 5*driftSampleWindow; i++ {
		if got := tab.observe(db, "rig-tiny", 1_000_000, 1, driftMinBytesSample-1); got {
			t.Fatalf("tiny-stream observations must be ignored; fired at i=%d", i)
		}
	}
	if tab.isQuarantined("rig-tiny") {
		t.Errorf("tiny-stream observations must not cause quarantine")
	}
}

func TestSettleTokensClampsAndDoesNotPanic(t *testing.T) {
	db := openDriftTestDB(t)
	defer db.Close()
	s := &server{db: db, drift: newRigDriftTable()}

	gotIn, gotOut := s.settleTokens("rig-x", 999999, 999999, 60, 60)
	if gotIn != 36 || gotOut != 40 {
		t.Errorf("settleTokens clamp wrong: got (%d,%d); want (36,40)", gotIn, gotOut)
	}
}
