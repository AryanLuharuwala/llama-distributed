package main

// Per-user rate limiting.
//
// Two dimensions:
//   req_per_min       — sliding-minute cap on /api/infer calls.
//   tokens_per_month  — soft monthly budget (input + output tokens).
//
// Defaults live in defaultRateLimit() and are applied when a user has no row
// in the rate_limits table.  We don't touch the DB on the fast path unless a
// limit is exceeded; counters are flushed asynchronously after each request.

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

type rateLimitPolicy struct {
	ReqPerMin      int `json:"req_per_min"`
	TokensPerMonth int `json:"tokens_per_month"`
}

func defaultRateLimit() rateLimitPolicy {
	return rateLimitPolicy{ReqPerMin: 30, TokensPerMonth: 1_000_000}
}

type usageSnapshot struct {
	ReqThisMinute  int `json:"req_this_minute"`
	TokensThisMo   int `json:"tokens_this_month"`
	InputThisMo    int `json:"input_tokens_this_month"`
	OutputThisMo   int `json:"output_tokens_this_month"`
	Period         string `json:"period"`
}

func currentMonthKey(t time.Time) string {
	return t.UTC().Format("2006-01")
}

func currentMinuteKey(t time.Time) string {
	return fmt.Sprintf("minute:%d", t.UTC().Unix()/60)
}

func (s *server) loadRateLimit(userID int64) rateLimitPolicy {
	var p rateLimitPolicy
	err := s.db.QueryRow(
		`SELECT req_per_min, tokens_per_month FROM rate_limits WHERE user_id = ?`,
		userID,
	).Scan(&p.ReqPerMin, &p.TokensPerMonth)
	if err != nil {
		return defaultRateLimit()
	}
	return p
}

func (s *server) usageSnapshot(userID int64) usageSnapshot {
	var u usageSnapshot
	u.Period = currentMonthKey(time.Now())

	minuteKey := currentMinuteKey(time.Now())
	_ = s.db.QueryRow(
		`SELECT requests FROM usage_counters WHERE user_id = ? AND period = ?`,
		userID, minuteKey,
	).Scan(&u.ReqThisMinute)

	_ = s.db.QueryRow(
		`SELECT input_tokens, output_tokens FROM usage_counters WHERE user_id = ? AND period = ?`,
		userID, u.Period,
	).Scan(&u.InputThisMo, &u.OutputThisMo)
	u.TokensThisMo = u.InputThisMo + u.OutputThisMo
	return u
}

// rlMu serialises concurrent counter updates for a single user so increments
// don't get lost under concurrent infer requests.  A real production system
// would use SQLite UPSERT + atomic increments, or Redis — but for now this
// keeps the semantics clean.
var rlMu sync.Mutex

// refundRequestSlot undoes one increment from the rolling minute counter.
// Used when a reservation was successfully taken but the request never made
// it to an agent (no online rig, agent busy, send-buffer full).  Without
// this a flapping pool can churn the user's per-minute budget to zero
// without serving a single token.
func (s *server) refundRequestSlot(userID int64) {
	minuteKey := currentMinuteKey(time.Now())
	rlMu.Lock()
	defer rlMu.Unlock()
	_, _ = s.db.Exec(
		`UPDATE usage_counters SET requests = MAX(0, requests - 1)
		 WHERE user_id = ? AND period = ?`,
		userID, minuteKey,
	)
}

// reserveRequestSlot atomically checks and consumes one request from the
// rolling minute.  Returns ok=false if the cap would be exceeded.
func (s *server) reserveRequestSlot(userID int64) (bool, rateLimitPolicy, usageSnapshot) {
	rlMu.Lock()
	defer rlMu.Unlock()

	policy := s.loadRateLimit(userID)
	snap := s.usageSnapshot(userID)
	if snap.ReqThisMinute >= policy.ReqPerMin {
		return false, policy, snap
	}
	if snap.TokensThisMo >= policy.TokensPerMonth {
		return false, policy, snap
	}
	// Bump the minute counter.  We do this before the request so a burst of
	// concurrent calls can't all pass the check.
	minuteKey := currentMinuteKey(time.Now())
	_, _ = s.db.Exec(
		`INSERT INTO usage_counters (user_id, period, requests, input_tokens, output_tokens)
		 VALUES (?, ?, 1, 0, 0)
		 ON CONFLICT (user_id, period) DO UPDATE SET requests = requests + 1`,
		userID, minuteKey,
	)
	snap.ReqThisMinute++
	return true, policy, snap
}

// recordTokens is called after the stream completes to attribute token usage
// to the calling user.  Errors are logged but not surfaced to the client.
//
// Agent-reported counts are untrusted (a malicious public-pool rig owner
// could inflate a victim's usage to evict them from their own quota).  Cap
// at sensible per-request maxima before persisting.  The caps are generous
// (1M input, 256k output) — they only kick in for clearly-broken or hostile
// agents, not honest reporting.
func (s *server) recordTokens(userID int64, inTok, outTok int) {
	const maxInPerReq = 1_000_000
	const maxOutPerReq = 256_000
	if inTok < 0 {
		inTok = 0
	}
	if outTok < 0 {
		outTok = 0
	}
	if inTok > maxInPerReq {
		inTok = maxInPerReq
	}
	if outTok > maxOutPerReq {
		outTok = maxOutPerReq
	}
	periodKey := currentMonthKey(time.Now())
	rlMu.Lock()
	defer rlMu.Unlock()
	_, _ = s.db.Exec(
		`INSERT INTO usage_counters (user_id, period, requests, input_tokens, output_tokens)
		 VALUES (?, ?, 0, ?, ?)
		 ON CONFLICT (user_id, period) DO UPDATE SET
		   input_tokens  = input_tokens  + excluded.input_tokens,
		   output_tokens = output_tokens + excluded.output_tokens`,
		userID, periodKey, inTok, outTok,
	)
}

// ─── REST: GET /api/usage ──────────────────────────────────────────────────

func (s *server) handleUsage(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	policy := s.loadRateLimit(u.ID)
	snap := s.usageSnapshot(u.ID)
	writeJSON(w, 200, map[string]any{
		"policy": policy,
		"usage":  snap,
	})
}
