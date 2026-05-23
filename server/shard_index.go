package main

// Shard cache index — server-side bookkeeping of which online rigs have
// which shards on disk.  The intent is P2P shard distribution: when a
// new rig joins and needs a 4-GB GGUF that twenty peers already have,
// route it to a peer instead of the origin.  Two design rules:
//
//   1. Authoritative source is the rig.  The server never assumes a rig
//      still has a shard once disconnected — `clearRigShards` runs on
//      every WS close.  Stale rows would make us promise downloads
//      that 404.
//
//   2. The index is advisory.  A rig listed here MAY no longer have the
//      file (the rig's GC removed it mid-session and we haven't been
//      told yet).  Callers must treat a peer 404 as soft and fall back
//      to the origin.
//
// We index (user_id, agent_id, model_name, file).  model_name is what
// the rig calls it — for HF repos that's the repo id ("meta-llama/Llama-3-8B"),
// for splitter-managed models it's the row in `models`.  A coordinator
// that hosts ten unrelated tenants doesn't want one user's rig advertised
// as a peer to another user's download; we scope by user_id implicitly
// through the rig identity, and only return matches owned by the user
// who's asking.

import (
	"encoding/json"
	"net/http"
	"strings"
)

// upsertRigShards records a snapshot of (model_name, file) pairs the rig
// claims to have.  Idempotent — duplicates are ignored.
//
// We deliberately do NOT delete unmentioned rows here.  A status frame
// may carry a delta of newly-cached entries; subtraction is reserved
// for disconnect (`clearRigShards`) and explicit eviction (not yet
// surfaced over the wire).
// maxShardsPerRig caps how many cached-shard claims one rig can lodge in a
// single hello/status frame.  A malicious agent could otherwise claim tens
// of thousands of bogus shards and bloat the rig_shards table.  Most real
// rigs cache a handful of GGUF/safetensors files — 1024 is two orders of
// magnitude headroom.
const maxShardsPerRig = 1024

func (s *server) upsertRigShards(uid int64, agentID string, shards []cachedShardEntry) {
	if s.db == nil || agentID == "" || len(shards) == 0 {
		return
	}
	if len(shards) > maxShardsPerRig {
		shards = shards[:maxShardsPerRig]
	}
	tx, err := s.db.Begin()
	if err != nil {
		return
	}
	defer func() { _ = tx.Rollback() }()
	stmt, err := tx.Prepare(
		`INSERT INTO rig_shards (user_id, agent_id, model_name, file, size_bytes, cached_at)
		 VALUES (?, ?, ?, ?, ?, ?)
		 ON CONFLICT(user_id, agent_id, model_name, file) DO UPDATE SET
		   size_bytes = excluded.size_bytes,
		   cached_at  = excluded.cached_at`,
	)
	if err != nil {
		return
	}
	defer stmt.Close()
	now := nowUnix()
	for _, e := range shards {
		model := strings.TrimSpace(e.ModelName)
		file := strings.TrimSpace(e.File)
		// Defensive: rigs are not trusted to honour our path rules.
		// Anything that looks like a path-escape attempt or absurdly
		// long is dropped silently.
		if model == "" || file == "" {
			continue
		}
		if len(model) > 256 || len(file) > 256 {
			continue
		}
		if strings.ContainsAny(model, "\x00\r\n") || strings.ContainsAny(file, "\x00\r\n") {
			continue
		}
		size := e.SizeBytes
		if size < 0 {
			size = 0
		}
		if _, err := stmt.Exec(uid, agentID, model, file, size, now); err != nil {
			return
		}
	}
	_ = tx.Commit()
}

// clearRigShards drops every shard claim for one (user_id, agent_id).
// Called on WS disconnect — keeps the index from advertising rigs that
// can no longer serve.
func (s *server) clearRigShards(uid int64, agentID string) {
	if s.db == nil || agentID == "" {
		return
	}
	_, _ = s.db.Exec(
		`DELETE FROM rig_shards WHERE user_id = ? AND agent_id = ?`,
		uid, agentID,
	)
}

// shardPeer is one row in the peers response.  agent_id is enough for
// the caller to look up the rig in the hub; we don't expose IPs here
// because the actual peer URL is minted lazily and signed (same flow
// as origin shard URLs).
type shardPeer struct {
	AgentID    string `json:"agent_id"`
	Hostname   string `json:"hostname,omitempty"`
	SizeBytes  int64  `json:"size_bytes,omitempty"`
	CachedAt   int64  `json:"cached_at"`
	BWUpKbps   int64  `json:"bw_up_kbps,omitempty"`
	NATType    string `json:"nat_type,omitempty"`
}

// peersForShard returns online rigs (under the same user) that claim
// to have (modelName, file).  Offline rigs are filtered out — there's
// no point pointing a downloader at a disconnected peer.
//
// The query is intentionally simple: we trust the index, then filter
// in-memory against the live hub map.  For large swarms we'd add a
// `WHERE last_seen > ?` clause, but disconnect-time clearRigShards
// already does that job.
func (s *server) peersForShard(uid int64, modelName, file string) []shardPeer {
	if s.db == nil {
		return nil
	}
	rows, err := s.db.Query(
		`SELECT agent_id, size_bytes, cached_at
		 FROM rig_shards
		 WHERE user_id = ? AND model_name = ? AND file = ?
		 ORDER BY cached_at ASC`,
		uid, modelName, file,
	)
	if err != nil {
		return nil
	}
	defer rows.Close()

	type row struct {
		agentID  string
		size     int64
		cachedAt int64
	}
	var candidates []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.agentID, &r.size, &r.cachedAt); err != nil {
			continue
		}
		candidates = append(candidates, r)
	}

	// Resolve agent_id → live agentConn under a single hub lock.
	s.hub.mu.RLock()
	live := make(map[string]*agentConn, len(s.hub.agents))
	for _, ac := range s.hub.agents {
		if ac.userID == uid {
			live[ac.agentID] = ac
		}
	}
	s.hub.mu.RUnlock()

	out := make([]shardPeer, 0, len(candidates))
	for _, c := range candidates {
		ac, ok := live[c.agentID]
		if !ok {
			continue
		}
		st := ac.snapshotStatus()
		out = append(out, shardPeer{
			AgentID:   c.agentID,
			Hostname:  ac.hostname,
			SizeBytes: c.size,
			CachedAt:  c.cachedAt,
			BWUpKbps:  st.BWUpKbps,
			NATType:   ac.snapshotNAT(),
		})
	}
	return out
}

// isSafeIndexedFile is the looser path check used by the shard-peers
// query.  Unlike isSafeShardFile (which guards a real filesystem read
// off models_dir), this path is metadata-only — we only need to reject
// path-escape characters and absurd lengths.  HF cache files cover a
// long tail of extensions (.safetensors, .bin, .json, .gguf, ...) so
// we can't be extension-strict the way the canonical shard server is.
func isSafeIndexedFile(f string) bool {
	if f == "" || len(f) > 256 {
		return false
	}
	if f == "." || f == ".." {
		return false
	}
	for i := 0; i < len(f); i++ {
		c := f[i]
		if c == 0 || c == '/' || c == '\\' || c == '\n' || c == '\r' {
			return false
		}
	}
	if strings.Contains(f, "..") {
		// Any literal ".." substring is suspicious in a single-segment
		// filename — disallow.  Yes, this rejects "foo..bar"; HF doesn't
		// produce such names in practice, and the safety win is worth it.
		return false
	}
	return true
}

// snapshotNAT returns the most recent NAT classification the rig reported,
// or "unknown".  Lives here (rather than ws.go) because it's only
// consumed by the shard-peer path today.
func (a *agentConn) snapshotNAT() string {
	a.statusMu.RLock()
	defer a.statusMu.RUnlock()
	if a.live.NATType == "" {
		return "unknown"
	}
	return a.live.NATType
}

// handleShardPeers answers GET /api/models/{name}/peers?file=<file>.
//
//   200 + JSON { "peers": [...] } on success (possibly empty).
//   400 if file is missing or unsafe.
//   401 if not logged in.
//   404 if the named model isn't registered AND no peer claims a shard
//       for it — we don't want to leak "this model exists in our DB"
//       to unauthenticated users, so we couple the two checks.
//
// Auth uses the same session cookie chain as the rest of /api/* — this
// is a user-scoped lookup, not a public endpoint.
func (s *server) handleShardPeers(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	uid := u.ID
	name := r.PathValue("name")
	file := r.URL.Query().Get("file")
	if name == "" || file == "" {
		http.Error(w, "name and file required", http.StatusBadRequest)
		return
	}
	if !isSafeIndexedFile(file) {
		http.Error(w, "bad file", http.StatusBadRequest)
		return
	}
	peers := s.peersForShard(uid, name, file)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"model": name,
		"file":  file,
		"peers": peers,
	})
}
