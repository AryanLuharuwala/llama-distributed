package main

// Per-rig reputation tracking.
//
// Every time we ask a rig to relay a session (p2p_relay_assign), we record
// the start; when the session ends cleanly we mark success and credit the
// bytes the rig reported forwarding; when the rig disconnects mid-session
// or the cleanup fires after an error, we mark failure.  The score this
// produces is fed back into findRelayAgent so the planner stops picking
// flaky rigs without operator intervention.
//
// Compute-side reputation (rigs that drop ACTV frames or take too long to
// produce tokens) is tracked on the same table — pp_route bumps the
// compute_* counters in its cleanup path.
//
// Scoring uses Bayesian smoothing so a brand-new rig isn't ranked above an
// established one purely because it has zero failures:
//
//   success_rate = (success + 1) / (total + 2)
//   confidence   = min(1, log(total+1) / log(20))     // 20 sessions ≈ trusted
//   recency      = 1 if last_failure_at > now-5min, 0.3 if very recent
//   score        = success_rate * confidence * recency_factor
//
// score is in [0, 1].  Newcomers sit at ~0.5*confidence so they get tried
// gradually as confidence grows.

import (
	"database/sql"
	"math"
	mrand "math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// rigReputation is the persisted shape from the rig_reputation table.
type rigReputation struct {
	AgentID               string
	RelaySessionsTotal    int64
	RelaySessionsSuccess  int64
	RelaySessionsFailed   int64
	RelayBytesForwarded   int64
	ComputeSessionsTotal  int64
	ComputeSessionsFailed int64
	LastSuccessAt         int64
	LastFailureAt         int64
}

// relayAssignment is the in-memory record of an active relay assignment.
// Used to (a) attribute bytes to the right rig when release arrives, and
// (b) detect mid-session disconnects (the relay rig went offline before
// release was sent).
type relayAssignment struct {
	SessionID  string
	AgentID    string // relay rig
	LeftPeer   string
	RightPeer  string
	StartedAt  int64
	BytesL2R   int64 // updated by release frame
	BytesR2L   int64
}

// activeRelays tracks all in-flight relay assignments keyed by sessionID +
// relayAgentID.  Concurrent map operations are common (assign on one
// goroutine, release on another, unregisterAgent on a third) so we hold a
// single mutex around the map — relay assignments are rare events vs the
// hot WS path so contention is a non-issue.
type activeRelays struct {
	mu      sync.Mutex
	byKey   map[relayKey]*relayAssignment
	byAgent map[string]map[string]*relayAssignment // agentID → sessionID → assignment
}

type relayKey struct {
	sessionID string
	agentID   string
}

func newActiveRelays() *activeRelays {
	return &activeRelays{
		byKey:   make(map[relayKey]*relayAssignment),
		byAgent: make(map[string]map[string]*relayAssignment),
	}
}

func (ar *activeRelays) add(a *relayAssignment) {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	ar.byKey[relayKey{a.SessionID, a.AgentID}] = a
	per, ok := ar.byAgent[a.AgentID]
	if !ok {
		per = make(map[string]*relayAssignment)
		ar.byAgent[a.AgentID] = per
	}
	per[a.SessionID] = a
}

func (ar *activeRelays) remove(sessionID, agentID string) *relayAssignment {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	k := relayKey{sessionID, agentID}
	a, ok := ar.byKey[k]
	if !ok {
		return nil
	}
	delete(ar.byKey, k)
	if per := ar.byAgent[agentID]; per != nil {
		delete(per, sessionID)
		if len(per) == 0 {
			delete(ar.byAgent, agentID)
		}
	}
	return a
}

// drainForAgent returns and removes every assignment owned by agentID.
// Called from unregisterAgent so we can mark them all failed in one shot.
func (ar *activeRelays) drainForAgent(agentID string) []*relayAssignment {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	per := ar.byAgent[agentID]
	if len(per) == 0 {
		return nil
	}
	out := make([]*relayAssignment, 0, len(per))
	for sid, a := range per {
		out = append(out, a)
		delete(ar.byKey, relayKey{sid, agentID})
	}
	delete(ar.byAgent, agentID)
	return out
}

// migrateReputation adds the rig_reputation table.  Idempotent.
func migrateReputation(db *sql.DB, d sqlDialect) error {
	_, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rig_reputation (
			agent_id                 TEXT PRIMARY KEY,
			relay_sessions_total     INTEGER NOT NULL DEFAULT 0,
			relay_sessions_success   INTEGER NOT NULL DEFAULT 0,
			relay_sessions_failed    INTEGER NOT NULL DEFAULT 0,
			relay_bytes_forwarded    INTEGER NOT NULL DEFAULT 0,
			compute_sessions_total   INTEGER NOT NULL DEFAULT 0,
			compute_sessions_failed  INTEGER NOT NULL DEFAULT 0,
			last_success_at          INTEGER NOT NULL DEFAULT 0,
			last_failure_at          INTEGER NOT NULL DEFAULT 0,
			updated_at               INTEGER NOT NULL DEFAULT 0
		)
	`))
	return err
}

// loadReputation reads the current row for agentID.  Returns a zero-valued
// struct (not an error) if the rig has never been scored — newcomers are
// treated as "unknown but worth trying".
func (s *server) loadReputation(agentID string) rigReputation {
	var r rigReputation
	r.AgentID = agentID
	if s == nil || s.db == nil || agentID == "" {
		return r
	}
	_ = s.db.QueryRow(`
		SELECT relay_sessions_total, relay_sessions_success, relay_sessions_failed,
		       relay_bytes_forwarded,
		       compute_sessions_total, compute_sessions_failed,
		       last_success_at, last_failure_at
		FROM rig_reputation WHERE agent_id = ?`, agentID).
		Scan(&r.RelaySessionsTotal, &r.RelaySessionsSuccess, &r.RelaySessionsFailed,
			&r.RelayBytesForwarded,
			&r.ComputeSessionsTotal, &r.ComputeSessionsFailed,
			&r.LastSuccessAt, &r.LastFailureAt)
	return r
}

// maxAggregateRelayBytes caps a single rig's lifetime relay-byte counter.
// 1 PiB is hilariously beyond any honest rig — at 10 Gbps line rate it
// would take ~9.7 days of continuous saturation to reach.  The cap is
// purely defense-in-depth on top of A4's per-session clamp: it prevents
// a malicious rig from overflowing INTEGER (or crowding out an honest
// rig in reputation-sorted views) by repeatedly closing sessions just
// under the per-session cap.
const maxAggregateRelayBytes int64 = 1 << 50

// recordRelaySuccess credits a successful relay session.  bytesForwarded is
// summed across both directions (l→r + r→l) as reported by the rig.
// The persisted aggregate is hard-capped at maxAggregateRelayBytes.
func (s *server) recordRelaySuccess(agentID string, bytesForwarded int64) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	if bytesForwarded < 0 {
		bytesForwarded = 0
	}
	if bytesForwarded > maxAggregateRelayBytes {
		bytesForwarded = maxAggregateRelayBytes
	}
	now := nowUnix()
	if _, err := s.db.Exec(`
		INSERT INTO rig_reputation (agent_id, relay_sessions_total,
		    relay_sessions_success, relay_bytes_forwarded,
		    last_success_at, updated_at)
		VALUES (?, 1, 1, ?, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
		    relay_sessions_total    = relay_sessions_total + 1,
		    relay_sessions_success  = relay_sessions_success + 1,
		    relay_bytes_forwarded   = MIN(?, relay_bytes_forwarded + excluded.relay_bytes_forwarded),
		    last_success_at         = excluded.last_success_at,
		    updated_at              = excluded.updated_at
	`, agentID, bytesForwarded, now, now, maxAggregateRelayBytes); err != nil {
		// Reputation is best-effort — log but don't propagate.
	}
}

// recordRelayFailure marks a relay session as failed.  Called on mid-session
// disconnect and on cleanup-with-error paths.
func (s *server) recordRelayFailure(agentID string) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	now := nowUnix()
	_, _ = s.db.Exec(`
		INSERT INTO rig_reputation (agent_id, relay_sessions_total,
		    relay_sessions_failed, last_failure_at, updated_at)
		VALUES (?, 1, 1, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
		    relay_sessions_total   = relay_sessions_total + 1,
		    relay_sessions_failed  = relay_sessions_failed + 1,
		    last_failure_at        = excluded.last_failure_at,
		    updated_at             = excluded.updated_at
	`, agentID, now, now)
}

// recordComputeFailure increments compute-side failure counters.  Reserved
// for pp_route cleanup paths that detect a stage dropped frames or timed
// out before producing tokens.  Right now nothing calls this — the hook is
// here so the planner can use it without another migration.
func (s *server) recordComputeFailure(agentID string) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	now := nowUnix()
	_, _ = s.db.Exec(`
		INSERT INTO rig_reputation (agent_id, compute_sessions_total,
		    compute_sessions_failed, last_failure_at, updated_at)
		VALUES (?, 1, 1, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
		    compute_sessions_total  = compute_sessions_total + 1,
		    compute_sessions_failed = compute_sessions_failed + 1,
		    last_failure_at         = excluded.last_failure_at,
		    updated_at              = excluded.updated_at
	`, agentID, now, now)
}

// recordComputeSuccess increments compute-side success counters.  Wired up
// to the pp_route happy path so the planner can prefer rigs that actually
// produce tokens for the model they advertise.
func (s *server) recordComputeSuccess(agentID string) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	now := nowUnix()
	_, _ = s.db.Exec(`
		INSERT INTO rig_reputation (agent_id, compute_sessions_total,
		    last_success_at, updated_at)
		VALUES (?, 1, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
		    compute_sessions_total = compute_sessions_total + 1,
		    last_success_at        = excluded.last_success_at,
		    updated_at             = excluded.updated_at
	`, agentID, now, now)
}

// relayRngCounter feeds an atomic counter into each rng seed, so two
// findRelayAgent calls happening in the same nanosecond still get
// distinct seeds (and therefore independent draws).
var relayRngCounter atomic.Int64

// mathRandFresh returns a thread-private *math/rand.Rand seeded from
// the wall clock combined with a process-monotonic counter.  Avoids
// the global math/rand mutex that all callers of rand.Float64()
// share — critical when many requests run findRelayAgent in parallel.
func mathRandFresh() *mrand.Rand {
	seed := time.Now().UnixNano() ^ relayRngCounter.Add(1)
	return mrand.New(mrand.NewSource(seed))
}

// relayScoreSampled is the Thompson-sampled variant of relayScore.
//
// Theory: relay-rig selection is a multi-armed bandit. Each rig is an
// arm; "success rate" is the unknown Bernoulli parameter we want to
// learn. Maintaining a Beta(alpha=successes+1, beta=failures+1)
// posterior over that parameter is exact under a uniform prior.
// Thompson sampling — draw one number from each arm's posterior, pick
// the arm with the highest draw — is provably regret-optimal for
// Bernoulli bandits (Russo, van Roy, Kazerouni, Osband, Wen 2018,
// "A Tutorial on Thompson Sampling").
//
// The recency penalty is preserved as a *multiplicative weight* on the
// sample, so a freshly-failed rig still gets demoted regardless of how
// optimistic the posterior is.  This trades a tiny amount of explore
// for a hard guardrail against immediately re-picking a sick rig.
//
// rng is the caller-supplied source so the choice is reproducible in
// tests.  In production, callers pass a per-request rand.New from a
// crypto-seeded global; this keeps each relay pick independent and
// avoids the global math/rand lock.
func relayScoreSampled(r rigReputation, nowUnixSec int64, rng *mrand.Rand) float64 {
	total := r.RelaySessionsTotal
	success := r.RelaySessionsSuccess
	failure := total - success
	if failure < 0 {
		failure = 0
	}

	// Beta(α, β) with α = successes+1, β = failures+1.  +1 priors
	// give Beta(1,1) ≡ Uniform[0,1] for unseen rigs — they get a
	// random draw on the unit interval, ensuring they're tried.
	alpha := float64(success + 1)
	beta := float64(failure + 1)
	draw := betaSample(rng, alpha, beta)

	// Recency penalty — same multiplicative shape as relayScore so the
	// "just-failed rig gets demoted hard" property is preserved.
	recency := relayRecencyPenalty(r, nowUnixSec)

	return draw * recency
}

// relayRecencyPenalty returns the [0,1] recency multiplier shared
// between relayScore and relayScoreSampled.  Extracted so a change to
// the recency model only needs one edit.
func relayRecencyPenalty(r rigReputation, nowUnixSec int64) float64 {
	if r.LastFailureAt <= 0 {
		return 1.0
	}
	age := nowUnixSec - r.LastFailureAt
	if age < 0 {
		age = 0
	}
	if age < 60 {
		return 0.1
	}
	if age < 300 {
		return 0.3 + 0.7*(float64(age-60)/240.0)
	}
	return 1.0
}

// betaSample draws from Beta(alpha, beta) via the ratio of two Gamma
// samples.  Both shape parameters must be ≥ 1 (callers guarantee this
// via the +1 prior).
func betaSample(rng *mrand.Rand, alpha, beta float64) float64 {
	x := gammaSample(rng, alpha)
	y := gammaSample(rng, beta)
	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

// gammaSample draws from Gamma(shape=k, scale=1) using the Marsaglia-Tsang
// (2000) acceptance-rejection method.  Requires k ≥ 1.  O(1) expected
// draws per sample (rejection rate < 5%).
func gammaSample(rng *mrand.Rand, k float64) float64 {
	if k < 1.0 {
		k = 1.0
	}
	d := k - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		x := rng.NormFloat64()
		v := 1.0 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		x2 := x * x
		if u < 1.0-0.0331*x2*x2 {
			return d * v
		}
		if math.Log(u) < 0.5*x2+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// relayScore returns a [0,1] score for a rig's relay reliability.
//
// The score blends three factors:
//   1. Bayesian-smoothed success rate (so a single failure doesn't tank a
//      rig with 100 prior successes, and a rig with one success isn't
//      ranked above one with a thousand).
//   2. Confidence based on session count — rigs with few sessions get a
//      penalty so newcomers earn trust gradually.
//   3. Recency — a failure in the last 5 minutes is a strong negative
//      signal even if the rig's lifetime rate is fine.
//
// Pure function so it's trivially testable.
func relayScore(r rigReputation, nowUnixSec int64) float64 {
	total := r.RelaySessionsTotal
	success := r.RelaySessionsSuccess

	// Bayesian smoothing — beta(1,1) prior pulls everyone toward 0.5.
	successRate := float64(success+1) / float64(total+2)

	// Confidence — logarithmic ramp; reaches 1.0 at 20 sessions.
	confidence := math.Log(float64(total+1)) / math.Log(20.0)
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}

	// Recency — strongly punish very recent failures (the rig might still
	// be sick).  Penalty fades out over 5 minutes.
	recency := 1.0
	if r.LastFailureAt > 0 {
		age := nowUnixSec - r.LastFailureAt
		if age < 0 {
			age = 0
		}
		if age < 60 {
			recency = 0.1
		} else if age < 300 {
			// Linear ramp from 0.3 (just past 1 min) back up to 1.0 at 5 min.
			recency = 0.3 + 0.7*(float64(age-60)/240.0)
		}
	}

	// Final score.  We hold a floor at 0.05 for not-yet-rated rigs so the
	// planner still rotates traffic to them; without this we'd never give
	// a brand-new rig its first session and confidence would stay zero
	// forever.
	score := successRate * confidence * recency
	if total == 0 {
		// Brand-new rig — give it 0.4 (under the typical established score
		// of 0.5+ but well above proven-bad).
		score = 0.4 * recency
	}
	return score
}

// allReputations bulk-loads reputation for a slice of agent IDs in one
// query.  Avoids the N+1 problem when scoring relay candidates.
func (s *server) allReputations(agentIDs []string) map[string]rigReputation {
	out := make(map[string]rigReputation, len(agentIDs))
	if s == nil || s.db == nil || len(agentIDs) == 0 {
		return out
	}
	// Bound the IN-clause size; SQLite caps host params at 999 by default
	// but we only ever pass a handful (online relay candidates).  Build
	// the placeholder string manually to keep the query parameterised.
	q := `SELECT agent_id, relay_sessions_total, relay_sessions_success,
	             relay_sessions_failed, relay_bytes_forwarded,
	             last_success_at, last_failure_at
	      FROM rig_reputation WHERE agent_id IN (`
	args := make([]any, 0, len(agentIDs))
	for i, id := range agentIDs {
		if i > 0 {
			q += ","
		}
		q += "?"
		args = append(args, id)
	}
	q += ")"
	rows, err := s.db.Query(q, args...)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var r rigReputation
		if err := rows.Scan(&r.AgentID,
			&r.RelaySessionsTotal, &r.RelaySessionsSuccess,
			&r.RelaySessionsFailed, &r.RelayBytesForwarded,
			&r.LastSuccessAt, &r.LastFailureAt); err == nil {
			out[r.AgentID] = r
		}
	}
	return out
}

// secondsSince is a tiny helper so tests can fake the clock easily.
func secondsSince(t time.Time) int64 { return time.Since(t).Nanoseconds() / 1e9 }

// Stale-assignment timeout.  An inference session that doesn't release its
// relay within an hour is almost certainly leaked — either the client
// vanished without a clean release frame, or the agent's release didn't
// reach us.  Reap so the activeRelays map can't grow unbounded.
const maxRelayAssignmentAge = int64(3600)

// reapStale walks the byKey map and removes assignments older than
// maxAgeSec.  Returns the reaped list so callers can attribute failure
// reputation.  Cheap O(n) scan — n is bounded by concurrent sessions.
func (ar *activeRelays) reapStale(nowUnixSec, maxAgeSec int64) []*relayAssignment {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	var out []*relayAssignment
	for k, a := range ar.byKey {
		if a.StartedAt > 0 && nowUnixSec-a.StartedAt > maxAgeSec {
			out = append(out, a)
			delete(ar.byKey, k)
			if per := ar.byAgent[a.AgentID]; per != nil {
				delete(per, a.SessionID)
				if len(per) == 0 {
					delete(ar.byAgent, a.AgentID)
				}
			}
		}
	}
	return out
}

// pruneIdleReputation deletes rig_reputation rows that haven't been
// touched in maxIdleSec.  Defends against unbounded growth from agents
// that probe once and never come back; we can always rebuild from
// scratch since reputation is empirical.
func (s *server) pruneIdleReputation(maxIdleSec int64) (int64, error) {
	if s == nil || s.db == nil {
		return 0, nil
	}
	cutoff := nowUnix() - maxIdleSec
	res, err := s.db.Exec(`DELETE FROM rig_reputation WHERE updated_at < ? AND updated_at > 0`, cutoff)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return n, nil
}
