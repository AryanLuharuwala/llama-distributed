package main

// Server-side tokenisation defence for billing.
//
// The inference protocol carries rig-reported tokIn/tokOut counts in the
// chunk header (see inference.go encodeChunk).  Those counts feed
// recordTokens() which writes to usage_counters — and ultimately the
// invoice.  Trusting a counter that originates on a peer's machine is a
// classic billing exploit: a malicious rig owner inflates a victim's
// monthly token usage to evict them from their own quota or run up their
// bill on a paid plan.
//
// We can't run the model's real BPE tokenizer here without pulling in
// per-family vocabs (llama, qwen, mistral, …); that pushes binary size
// and is brittle to silent vocab swaps.  Instead we treat the rig's
// reported output as upper-bounded by the byte stream we actually
// observed: tokens are at least ~1.5 characters on average for the
// English-dominant text these models emit, and even adversarial UTF-8
// can't squeeze more than one token per byte in any tokenizer we ship.
// So a rig that reports 1 000 output tokens when only 100 bytes of
// payload crossed the WS is lying, and we clamp.
//
// Beyond the per-request clamp we keep a small in-memory running drift
// estimator per rig: every settlement records (reported, server-cap)
// and, once we have a hundred samples for that rig, if the cumulative
// drift exceeds 5 % we record a row in rig_quarantine and stop crediting
// reputation for it.  Quarantine is persisted so a restart doesn't wipe
// the evidence, but the in-memory counter resets — that's fine, a
// reformed rig will recover after a clean window of 100 requests.

import (
	"database/sql"
	"log"
	"sync"
)

// bytesToMaxTokens returns the conservative upper bound on the number
// of tokens a stream of `b` bytes could plausibly represent.  Floor
// of b / 1.5 — equivalent to (2*b)/3.  We round down so the clamp is
// strict on the rig (any honest reporting will be well under the cap;
// only inflation hits it).
func bytesToMaxTokens(b int) int {
	if b <= 0 {
		return 0
	}
	return (2 * b) / 3
}

// clampReportedTokens enforces serverside upper bounds on rig-reported
// token counts.  For prompt tokens we trust the server's own
// estimateTokens() of the original prompt as a generous ceiling
// (estimateTokens already over-counts by using 4 chars/token).  For
// completion tokens we use bytesToMaxTokens against the observed wire
// payload.
//
// Returns the possibly-clamped (inTok, outTok) plus a boolean indicating
// whether either side was clamped (i.e. suspicious activity worth
// logging or feeding into the drift estimator).
func clampReportedTokens(
	reportedIn, reportedOut int,
	promptChars, completionBytes int,
) (inTok, outTok int, clamped bool) {
	inTok = reportedIn
	outTok = reportedOut

	// Lower bound on rig outputs: agents must report at least zero.  A
	// negative is a bug or a wraparound; treat as zero.
	if inTok < 0 {
		inTok, clamped = 0, true
	}
	if outTok < 0 {
		outTok, clamped = 0, true
	}

	// Input ceiling: server-side estimate of the prompt.  Add 25 % slack
	// to absorb BPE variance — we don't want to penalise an honest rig
	// whose real BPE happens to produce a few more tokens than our
	// 4-chars-per-token heuristic.
	inCeil := promptChars/3 + 16 // ~1.33 * estimateTokens
	if inCeil > 0 && inTok > inCeil {
		inTok, clamped = inCeil, true
	}

	// Output ceiling: bytes that actually crossed the wire.
	outCeil := bytesToMaxTokens(completionBytes)
	if outTok > outCeil {
		outTok, clamped = outCeil, true
	}
	return
}

// ─── per-rig drift estimator ──────────────────────────────────────────

const (
	driftSampleWindow   = 100   // requests
	driftThresholdPct   = 5     // % over which we quarantine
	driftMinBytesSample = 64    // skip tiny streams from the running calc
)

type rigDriftCounter struct {
	n             int
	sumReported   int64
	sumServerCeil int64
	quarantined   bool
}

// rigDriftTable tracks per-rig running drift between reported tokens
// and server-side ceilings.  Lives in memory; persists to
// rig_quarantine on threshold breach so future processes know to refuse
// reputation credit even if the in-memory counter has reset.
type rigDriftTable struct {
	mu sync.Mutex
	m  map[string]*rigDriftCounter
}

func newRigDriftTable() *rigDriftTable {
	return &rigDriftTable{m: make(map[string]*rigDriftCounter)}
}

// observe records one settled inference event against the running
// counter for agentID.  reportedOut is whatever the rig sent; serverCeil
// is the server-derived upper bound from clampReportedTokens.  When the
// window fills, drift is evaluated and quarantine triggers if the rig
// has been consistently over-reporting beyond the threshold.
//
// Returns true if the rig got newly quarantined on this observation so
// the caller can take audit action (skip reputation credit, log, etc.).
func (t *rigDriftTable) observe(
	db *sql.DB, agentID string,
	reportedOut, serverCeil, completionBytes int,
) bool {
	if agentID == "" {
		return false
	}
	if completionBytes < driftMinBytesSample {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	c, ok := t.m[agentID]
	if !ok {
		c = &rigDriftCounter{}
		t.m[agentID] = c
	}
	if c.quarantined {
		return false
	}
	c.n++
	c.sumReported += int64(reportedOut)
	c.sumServerCeil += int64(serverCeil)
	if c.n < driftSampleWindow {
		return false
	}
	// Drift evaluated as (sumReported - sumServerCeil) / sumServerCeil.
	// Negative drift (reporting less than the ceiling) is not punished —
	// that's an honest rig under-reporting if anything.  Only over-
	// reporting above the threshold is suspicious.
	if c.sumServerCeil <= 0 {
		// Reset and wait for another window.
		c.n, c.sumReported, c.sumServerCeil = 0, 0, 0
		return false
	}
	overPct := float64(c.sumReported-c.sumServerCeil) * 100.0 / float64(c.sumServerCeil)
	// Reset window before any returns so we don't repeatedly fire on
	// every following observation.
	c.n, c.sumReported, c.sumServerCeil = 0, 0, 0
	if overPct <= float64(driftThresholdPct) {
		return false
	}
	c.quarantined = true
	persistQuarantine(db, agentID, overPct)
	log.Printf("rig %s quarantined: token-drift %.1f%% over %d-request window",
		agentID, overPct, driftSampleWindow)
	return true
}

// isQuarantined reports whether agentID has been flagged in this
// process's in-memory table.  Persistent state in rig_quarantine is
// loaded into this table at startup via hydrateQuarantine.
func (t *rigDriftTable) isQuarantined(agentID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	c := t.m[agentID]
	return c != nil && c.quarantined
}

// hydrateQuarantine reads existing rig_quarantine rows into the in-memory
// drift table so a restart doesn't temporarily un-quarantine known bad
// rigs.  Safe to call once during boot before WS connections are
// accepted.
func (t *rigDriftTable) hydrateQuarantine(db *sql.DB) error {
	if db == nil {
		return nil
	}
	rows, err := db.Query(`SELECT agent_id FROM rig_quarantine`)
	if err != nil {
		return err
	}
	defer rows.Close()
	t.mu.Lock()
	defer t.mu.Unlock()
	for rows.Next() {
		var aid string
		if err := rows.Scan(&aid); err != nil {
			continue
		}
		t.m[aid] = &rigDriftCounter{quarantined: true}
	}
	return rows.Err()
}

func migrateQuarantine(db *sql.DB, d sqlDialect) error {
	_, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rig_quarantine (
			agent_id      TEXT PRIMARY KEY,
			reason        TEXT NOT NULL,
			drift_pct     REAL NOT NULL,
			quarantined_at INTEGER NOT NULL
		)
	`))
	return err
}

func persistQuarantine(db *sql.DB, agentID string, driftPct float64) {
	if db == nil {
		return
	}
	_, err := db.Exec(
		`INSERT INTO rig_quarantine (agent_id, reason, drift_pct, quarantined_at)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT (agent_id) DO UPDATE SET
		     drift_pct      = excluded.drift_pct,
		     quarantined_at = excluded.quarantined_at`,
		agentID, "token_drift", driftPct, nowUnix(),
	)
	if err != nil {
		log.Printf("persistQuarantine: %v", err)
	}
}
