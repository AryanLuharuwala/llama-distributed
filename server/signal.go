package main

// P2P signalling.
//
// The browser runs a WebRTC data channel to bypass the server relay whenever
// possible.  We don't terminate SDP ourselves — we just ferry offers/answers
// between the browser and the selected agent over the agent's control-plane
// WebSocket.
//
// POST /api/signal { pool_id, offer:{...} }
//   → { answer:{...}, ice_servers:[...] }       (2xx with SDP answer)
//   → { error:"...", ice_servers:[...] }        (4xx when peer unreachable —
//                                                 the client should fall back
//                                                 to the /ws/client relay)
//
// Agent JSON protocol additions (text frames):
//   server → agent:  {"kind":"signal", "req_id":N, "offer":..., "ice_servers":[...]}
//   agent  → server: {"kind":"signal_answer", "req_id":N, "answer":...}
//                    {"kind":"signal_error",  "req_id":N, "message":"..."}

import (
	"context"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// rigTURNSecret derives a per-rig HMAC secret from the operator's master
// secret.  Each rig only ever sees its own derived secret (via the welcome
// frame), so a compromised rig cannot mint credentials valid against any
// other rig's sidecar.  The server derives the same value on demand when
// minting credentials targeted at a specific relay rig.
//
// Returned as lowercase hex of HMAC-SHA256(master, agent_id).  We hex-encode
// rather than base64 so the secret can sit in a URL/CLI arg without quoting
// concerns when operators copy it for debugging.
func (s *server) rigTURNSecret(agentID string) string {
	if s.cfg.turnSecret == "" || agentID == "" {
		return ""
	}
	mac := hmac.New(sha256.New, []byte(s.cfg.turnSecret))
	mac.Write([]byte("dist-turn-rig|"))
	mac.Write([]byte(agentID))
	return hex.EncodeToString(mac.Sum(nil))
}

// Default STUN servers advertised to both ends.  TURN is appended by
// iceServersFor when configured.
func defaultSTUNServers() []map[string]any {
	return []map[string]any{
		{"urls": "stun:stun.l.google.com:19302"},
		{"urls": "stun:stun1.l.google.com:19302"},
	}
}

// iceServersFor builds the ICE candidate list to advertise to a given
// pair-peer.  It always includes the public STUN servers and, if the
// operator has configured one, appends a TURN entry with credentials
// scoped to `audience` (an agent_id or session id used as part of the
// ephemeral username).  When neither static nor ephemeral TURN is
// configured we return STUN-only and the WS path remains the only
// fallback for symmetric-NAT pairs.
func (s *server) iceServersFor(audience string) []map[string]any {
	ice := defaultSTUNServers()

	// Operator-configured TURN.  Wins precedence over peer relays —
	// dedicated coturn is always more reliable than borrowing a user's
	// rig, when one is available.
	if s.cfg.turnURL != "" {
		if user, pass := s.mintTURNCreds(audience); user != "" {
			ice = append(ice, map[string]any{
				"urls":       s.cfg.turnURL,
				"username":   user,
				"credential": pass,
			})
		}
	}

	// Peer-relay candidates: well-connected rigs (cone NAT or open) that
	// volunteered as forwarders.  The C++ adapter recognises the
	// `peer:<agent_id>` URL scheme and opens a libdatachannel forwarding
	// session through that rig instead of a real TURN allocation.  We
	// cap the list so a single audience doesn't enumerate the whole swarm.
	for _, e := range s.pickPeerRelays(audience, 3) {
		ice = append(ice, e)
	}
	return ice
}

// findRelayAgent returns a live agentConn that has volunteered as a peer
// relay (friendly NAT, relay_capable=true) and is not in the `exclude`
// list.  Returns nil if no candidate is available.  Used by the pipeline
// planner to insert a relay between two adjacent stages when either of
// them is behind a symmetric NAT.
//
// Selection is Thompson-sampled (see reputation.go::relayScoreSampled):
// each call draws once from each rig's Beta posterior and picks the
// max. Provably regret-optimal for the Bernoulli-bandit framing of
// "which relay is best?" — converges to the true optimum while still
// exploring untrusted newcomers.
//
// rng is private to this call so concurrent picks don't contend on a
// shared math/rand mutex.  Seeding from time.Now().UnixNano() ^ a
// per-call pointer is enough entropy here — the goal is decorrelation
// across picks, not crypto-strong unpredictability.
func (s *server) findRelayAgent(exclude map[string]struct{}) *agentConn {
	if s == nil || s.hub == nil {
		return nil
	}
	type candidate struct {
		ac    *agentConn
		score float64
	}
	var cands []candidate
	var ids []string
	for _, a := range s.hub.snapshotAgents() {
		if _, skip := exclude[a.agentID]; skip {
			continue
		}
		st := a.snapshotStatus()
		if !st.RelayCapable {
			continue
		}
		switch st.NATType {
		case "open", "cone":
			cands = append(cands, candidate{ac: a})
			ids = append(ids, a.agentID)
		}
	}
	if len(cands) == 0 {
		return nil
	}
	reps := s.allReputations(ids)
	now := nowUnix()
	rng := mathRandFresh()
	best := -1.0
	var pick *agentConn
	for i := range cands {
		r := reps[cands[i].ac.agentID]
		r.AgentID = cands[i].ac.agentID
		cands[i].score = relayScoreSampled(r, now, rng)
		if cands[i].score > best {
			best = cands[i].score
			pick = cands[i].ac
		}
	}
	return pick
}

// pickPeerRelays returns up to `limit` ICE entries pointing at relay-capable
// peer rigs that have advertised a friendly NAT type in their last heartbeat.
// Entries use the `peer:<agent_id>` URL scheme — the dist-node adapter knows
// to route through that rig over an existing or fresh WebRTC channel.
//
// audience is opaque (typically the agent_id of the consumer); we don't
// currently filter on it but the parameter is reserved for future per-pair
// allowlisting.
func (s *server) pickPeerRelays(audience string, limit int) []map[string]any {
	if s == nil || s.hub == nil || limit <= 0 {
		return nil
	}
	agents := s.hub.snapshotAgents()
	out := make([]map[string]any, 0, limit)
	for _, a := range agents {
		st := a.snapshotStatus()
		if !st.RelayCapable {
			continue
		}
		// Skip the audience itself — a rig can't relay to itself.
		if a.agentID == audience {
			continue
		}
		// Only friendly NAT types make useful relays.  "unknown" is
		// excluded so we don't burn handshake budget on dead-ends.
		switch st.NATType {
		case "open", "cone":
			// good
		default:
			continue
		}
		// Prefer the rig's bundled TURN sidecar when it's running —
		// that gives the consumer a real coturn-compatible relay that
		// forwards UDP without terminating DTLS, so neither this server
		// nor the relay rig sees plaintext ACTV bytes.  Falls back to
		// the legacy `peer:` sentinel when no sidecar is up; that path
		// works but lets the relay rig see plaintext.
		if st.CoturnPort > 0 && s.cfg.turnSecret != "" {
			// Prefer the STUN-discovered IP the rig reported (the actual
			// UDP-reachable address) over the WS source IP.  Fall back to
			// remoteIP when the rig couldn't gather an srflx candidate.
			host := st.PublicIP
			if host == "" {
				host = a.remoteIP
			}
			if host == "" {
				continue
			}
			url := "turn:" + host + ":" + strconv.Itoa(st.CoturnPort)
			// Use the relay rig's *derived* secret so credentials only
			// validate against that one rig's sidecar.  A compromised rig
			// cannot impersonate any other rig's TURN.
			user, pass := s.mintRigTURNCreds(a.agentID, audience)
			if user != "" {
				out = append(out, map[string]any{
					"urls":       url,
					"username":   user,
					"credential": pass,
				})
				if len(out) >= limit {
					break
				}
				continue
			}
		}
		out = append(out, map[string]any{
			"urls": "peer:" + a.agentID,
		})
		if len(out) >= limit {
			break
		}
	}
	return out
}

// mintRigTURNCreds returns credentials valid against a specific rig's
// bundled dist-turn sidecar (per-rig secret), independent of whether the
// operator has configured a global turnURL.  Returns empty strings when no
// master secret is configured.
func (s *server) mintRigTURNCreds(relayAgentID, audience string) (string, string) {
	rigSecret := s.rigTURNSecret(relayAgentID)
	if rigSecret == "" {
		return "", ""
	}
	ttl := s.cfg.turnTTL
	if ttl <= 0 {
		ttl = time.Hour
	}
	exp := time.Now().Add(ttl).Unix()
	if audience == "" {
		audience = "anon"
	}
	user := strconv.FormatInt(exp, 10) + ":" + audience
	mac := hmac.New(sha1.New, []byte(rigSecret))
	mac.Write([]byte(user))
	cred := base64.StdEncoding.EncodeToString(mac.Sum(nil))
	return user, cred
}

// mintTURNCreds returns a (username, credential) pair appropriate for the
// configured TURN server.  Ephemeral (coturn `use-auth-secret`) wins over
// static creds when both are set.  Returns empty strings if TURN is
// disabled.
//
// Ephemeral scheme (coturn-compatible REST credentials):
//
//	username = "<unix-expiry>:<audience>"
//	credential = base64(HMAC-SHA1(<turnSecret>, username))
//
// coturn validates by recomputing the HMAC and checking the expiry has
// not passed.  No static password ever leaves the server in this mode.
func (s *server) mintTURNCreds(audience string) (string, string) {
	if s.cfg.turnURL == "" {
		return "", ""
	}
	if s.cfg.turnSecret != "" {
		ttl := s.cfg.turnTTL
		if ttl <= 0 {
			ttl = time.Hour
		}
		exp := time.Now().Add(ttl).Unix()
		if audience == "" {
			audience = "anon"
		}
		user := strconv.FormatInt(exp, 10) + ":" + audience
		mac := hmac.New(sha1.New, []byte(s.cfg.turnSecret))
		mac.Write([]byte(user))
		cred := base64.StdEncoding.EncodeToString(mac.Sum(nil))
		return user, cred
	}
	if s.cfg.turnStaticUser != "" || s.cfg.turnStaticPass != "" {
		return s.cfg.turnStaticUser, s.cfg.turnStaticPass
	}
	return "", ""
}

// Outstanding signal requests keyed by req_id.
type signalWaiter struct {
	answer chan json.RawMessage
	errMsg chan string
	once   sync.Once
}

func (w *signalWaiter) close() {
	w.once.Do(func() { close(w.answer); close(w.errMsg) })
}

var (
	signalMu      sync.Mutex
	signalWaiters = map[uint32]*signalWaiter{}
	signalReqCtr  uint32
)

func registerSignalWaiter() (uint32, *signalWaiter) {
	id := atomic.AddUint32(&signalReqCtr, 1)
	w := &signalWaiter{
		answer: make(chan json.RawMessage, 1),
		errMsg: make(chan string, 1),
	}
	signalMu.Lock()
	signalWaiters[id] = w
	signalMu.Unlock()
	return id, w
}

func takeSignalWaiter(id uint32) (*signalWaiter, bool) {
	signalMu.Lock()
	defer signalMu.Unlock()
	w, ok := signalWaiters[id]
	if ok {
		delete(signalWaiters, id)
	}
	return w, ok
}

// deliverSignalAnswer is called from the agent reader when a signal_answer or
// signal_error text frame arrives.  Returns true if the frame was consumed.
func deliverSignalAnswer(msg map[string]any) bool {
	kind, _ := msg["kind"].(string)
	if kind != "signal_answer" && kind != "signal_error" {
		return false
	}
	idf, ok := msg["req_id"].(float64)
	if !ok {
		return false
	}
	w, ok := takeSignalWaiter(uint32(idf))
	if !ok {
		return false
	}
	if kind == "signal_answer" {
		raw, _ := json.Marshal(msg["answer"])
		select {
		case w.answer <- raw:
		default:
		}
	} else {
		m, _ := msg["message"].(string)
		select {
		case w.errMsg <- m:
		default:
		}
	}
	return true
}

// ─── POST /api/signal ──────────────────────────────────────────────────────

type signalReq struct {
	PoolID int64           `json:"pool_id"`
	Offer  json.RawMessage `json:"offer"`
}

func (s *server) handleSignal(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body signalReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 256<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	if body.PoolID == 0 || len(body.Offer) == 0 {
		writeErr(w, 400, "pool_id and offer required")
		return
	}
	if _, isMember := s.userIsMember(body.PoolID, u.ID); !isMember {
		writeErr(w, 403, "not a pool member")
		return
	}

	ice := s.iceServersFor("user-" + strconv.FormatInt(u.ID, 10))

	ac, ok := s.pickOnlineRigInPool(body.PoolID)
	if !ok {
		writeJSON(w, 503, map[string]any{
			"error":       "no online rig in pool",
			"ice_servers": ice,
		})
		return
	}

	id, waiter := registerSignalWaiter()
	defer func() {
		if _, stillThere := takeSignalWaiter(id); stillThere {
			waiter.close()
		}
	}()

	ac.send(map[string]any{
		"kind":        "signal",
		"req_id":      id,
		"offer":       body.Offer,
		"ice_servers": ice,
	})

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	select {
	case ans := <-waiter.answer:
		writeJSON(w, 200, map[string]any{
			"answer":      json.RawMessage(ans),
			"ice_servers": ice,
		})
	case msg := <-waiter.errMsg:
		writeJSON(w, 502, map[string]any{
			"error":       "agent: " + msg,
			"ice_servers": ice,
		})
	case <-ctx.Done():
		writeJSON(w, 504, map[string]any{
			"error":       "agent signalling timed out",
			"ice_servers": ice,
		})
	}
}
