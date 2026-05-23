package main

// Agent-to-agent WebRTC signaling relay.
//
// When the planner picks two rigs for adjacent stages of a pipeline, those
// rigs can talk to each other directly over a WebRTC data channel instead of
// relaying every ACTV through this server.  The server's only job is to
// shuttle SDP offers/answers and ICE candidates between them — it never sees
// or parses the actual data plane.
//
// Wire shapes (text frames on /ws/agent):
//   agent → server: {"kind":"p2p_offer",  "to":"<peer_agent_id>", "session_id":"...", "sdp":"..."}
//   agent → server: {"kind":"p2p_answer", "to":"<peer_agent_id>", "session_id":"...", "sdp":"..."}
//   agent → server: {"kind":"p2p_ice",    "to":"<peer_agent_id>", "session_id":"...", "candidate":"...", "mid":"...", "index":N}
//   server → agent: same shapes but with "from" instead of "to", plus
//                   "from_user" when the peer is in a different user's
//                   account (cross-user pool membership).
//
// To prevent rigs from using the server as a generic NAT-traversal proxy,
// signaling is only forwarded between agents that have an outstanding,
// server-issued permission (added by the pipeline planner via allowP2PPair).
// Permissions are TTL-scoped (default 5 min) and are cleaned up lazily on
// the next read.

import (
	"log"
	"sync"
	"time"
)

const p2pSignalTTL = 5 * time.Minute

// fmtSessionID builds an opaque-ish per-request session id rigs can use to
// correlate signaling messages with the request they belong to.  Format
// is informational only; the server never parses it.
func fmtSessionID(prefix string, reqID uint16) string {
	return prefix + "-" + itoa16(reqID) + "-" + itoa64(time.Now().UnixNano())
}

func itoa16(v uint16) string {
	const hex = "0123456789abcdef"
	out := []byte{hex[(v>>12)&0xF], hex[(v>>8)&0xF], hex[(v>>4)&0xF], hex[v&0xF]}
	return string(out)
}

func itoa64(v int64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

type p2pPairKey struct {
	// from is the originator's agentID; to is the destination agentID.
	from string
	to   string
}

type p2pPermission struct {
	// expires is the absolute deadline.  After this time the pair is no
	// longer authorised to signal each other.
	expires time.Time
	// sessionID is the opaque per-request id the planner stamped.  Used
	// so we can scope log messages to a request and so an old session's
	// late-arriving offers don't accidentally race with a new one.
	sessionID string
}

var (
	p2pMu          sync.Mutex
	p2pPermissions = map[p2pPairKey]p2pPermission{}
)

// allowP2PPair authorises one-way signaling from -> to for ttl.  Call once
// for each direction (the helper does both).
func allowP2PPair(from, to, sessionID string, ttl time.Duration) {
	if ttl <= 0 {
		ttl = p2pSignalTTL
	}
	exp := time.Now().Add(ttl)
	p2pMu.Lock()
	defer p2pMu.Unlock()
	p2pPermissions[p2pPairKey{from, to}] = p2pPermission{exp, sessionID}
	p2pPermissions[p2pPairKey{to, from}] = p2pPermission{exp, sessionID}
}

// revokeP2PPair clears any outstanding permission.  Called when the planner
// tears down a request — best-effort cleanup; the TTL is the real safety net.
func revokeP2PPair(a, b string) {
	p2pMu.Lock()
	defer p2pMu.Unlock()
	delete(p2pPermissions, p2pPairKey{a, b})
	delete(p2pPermissions, p2pPairKey{b, a})
}

func p2pPermissionFor(from, to string) (p2pPermission, bool) {
	p2pMu.Lock()
	defer p2pMu.Unlock()
	p, ok := p2pPermissions[p2pPairKey{from, to}]
	if !ok {
		return p2pPermission{}, false
	}
	if time.Now().After(p.expires) {
		// Lazy GC — we'll see the rest of the map's expired entries on
		// future calls or via the periodic sweep below.
		delete(p2pPermissions, p2pPairKey{from, to})
		return p2pPermission{}, false
	}
	return p, true
}

// p2pSignalSweep runs once per minute to garbage-collect expired permissions
// the lazy path didn't reach.  Cheap; the map is small.
func p2pSignalSweep() {
	now := time.Now()
	p2pMu.Lock()
	defer p2pMu.Unlock()
	for k, v := range p2pPermissions {
		if now.After(v.expires) {
			delete(p2pPermissions, k)
		}
	}
}

// p2pSignalFrame is the subset of fields the relay cares about.  We don't
// touch the SDP/candidate payloads themselves.
type p2pSignalFrame struct {
	Kind      string `json:"kind"`
	To        string `json:"to,omitempty"`
	From      string `json:"from,omitempty"`
	FromUser  int64  `json:"from_user,omitempty"`
	SessionID string `json:"session_id,omitempty"`
}

// deliverP2PSignal is called from the agent reader when a p2p_offer /
// p2p_answer / p2p_ice frame arrives.  Returns true if the frame was
// consumed (regardless of whether it was forwarded or dropped).
//
// fromUID/fromAgentID is the sending rig; msg is the parsed JSON.  The
// hub lookup is done here so we can keep the routing logic in one place.
func (s *server) deliverP2PSignal(fromUID int64, fromAgentID string, msg map[string]any) bool {
	kind, _ := msg["kind"].(string)
	if kind != "p2p_offer" && kind != "p2p_answer" && kind != "p2p_ice" {
		return false
	}
	toAgentID, _ := msg["to"].(string)
	if toAgentID == "" {
		return true // consumed (malformed); don't fall through
	}
	if toAgentID == fromAgentID {
		// Self-signaling is meaningless; protect against a misbehaving rig
		// trying to loop frames through the server.
		return true
	}

	perm, ok := p2pPermissionFor(fromAgentID, toAgentID)
	if !ok {
		// Drop silently — a misbehaving rig shouldn't get diagnostic
		// info about what's allowed.  Log only.
		log.Printf("p2p_signal: dropping %s from=%s to=%s (no permission)", kind, fromAgentID, toAgentID)
		return true
	}
	_ = perm

	// Find the destination.  We allow cross-user signaling because pools
	// can have multiple-user membership; permission is the gate, not user
	// identity.  Look in every user that has this agentID online.
	target := s.hub.findAgentByID(toAgentID)
	if target == nil {
		log.Printf("p2p_signal: dropping %s from=%s to=%s (peer offline)", kind, fromAgentID, toAgentID)
		return true
	}

	// Rewrite the frame: strip "to", add "from" / "from_user".
	out := make(map[string]any, len(msg)+1)
	for k, v := range msg {
		if k == "to" {
			continue
		}
		out[k] = v
	}
	out["from"] = fromAgentID
	out["from_user"] = fromUID

	target.send(out)
	return true
}
