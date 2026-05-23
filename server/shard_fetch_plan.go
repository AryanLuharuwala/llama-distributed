package main

// Multi-source shard fetch planner.
//
// Background: a rig joining a swarm that already has 20 peers caching
// the same 4-GB GGUF should pull chunks from those peers in parallel
// rather than hammer the origin.  Two pieces wire this up:
//
//   1. `rig_shards` (see shard_index.go) tells us which peers claim
//      to have which file — populated from hello/status frames.
//
//   2. This endpoint translates a (model, file) request into a fetch
//      *plan*: an ordered list of source URLs (peers first, origin
//      last) plus a recommended chunk size.  The rig-side downloader
//      walks the byte range in chunks and assigns each chunk to the
//      next source round-robin, falling back to the next source on
//      404 / Range-not-honored / stalled.
//
// Why server-side instead of pure peer-to-peer discovery?  Two reasons:
//   - The server already has the auth + signed-URL machinery, so it
//     can mint per-source URLs the requester can use without a session.
//   - The server can refuse to advertise peers that are NAT'd in a way
//     that makes direct HTTP impossible.  Today every peer URL is
//     routed via the origin (which holds the canonical shard); once
//     dist-relay grows a peer-shard transport, this same planner gets
//     a richer source list.
//
// What this endpoint does NOT do today:
//   - Actually proxy from peers.  Peer URLs in the response point at
//     the origin (with a peer-routed query tag for metrics).  Adding
//     a real peer-relay transport is downstream work — but the rig
//     fetcher doesn't need to know that until then.
//   - Verify peer claims.  rig_shards is advisory; a peer that lied
//     surfaces as a 404 → the rig falls back to the next source.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// shardFetchSource is one entry in the fetch plan.  The rig downloader
// fetches `url` with a Range header set by chunk index.
type shardFetchSource struct {
	URL     string `json:"url"`
	Kind    string `json:"kind"`               // "origin" | "peer"
	AgentID string `json:"agent_id,omitempty"` // for kind=peer
	// HintBWUpKbps is the peer's reported upload bandwidth — the rig
	// can weight chunk assignment so fast peers get more of the file.
	// Zero on origin and on peers that never reported BW.
	HintBWUpKbps int64 `json:"hint_bw_up_kbps,omitempty"`
}

// shardFetchPlan is the response payload.  `chunk_size` is the
// recommended granularity — the rig is free to ignore it (Range-GET
// is byte-precise), but using it lines up cleanly with the 16 MiB
// LFS chunk grain HF itself uses.
type shardFetchPlan struct {
	Model     string             `json:"model"`
	File      string             `json:"file"`
	SizeBytes int64              `json:"size_bytes,omitempty"`
	ChunkSize int64              `json:"chunk_size"`
	Sources   []shardFetchSource `json:"sources"`
	// TTL is when every signed URL in `sources` stops being honored.
	// The rig should re-plan before then.
	TTLUnix int64 `json:"ttl_unix"`
}

// handleShardFetchPlan answers GET /api/models/{name}/fetch-plan?file=<file>.
// Auth: session cookie (same as /peers).
func (s *server) handleShardFetchPlan(w http.ResponseWriter, r *http.Request) {
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

	// Resolve the canonical model id (so we can mint a signed origin URL).
	// If the name doesn't match a registered model, we can still serve
	// a peer-only plan — useful for HF-cached repos that haven't been
	// imported on this coordinator yet.
	var (
		modelID     int64
		shardsDir   string
		originURL   string
		sizeBytes   int64
	)
	_ = s.db.QueryRow(
		`SELECT id, shards_dir FROM models WHERE name = ?`, name,
	).Scan(&modelID, &shardsDir)

	const ttl = 30 * time.Minute
	exp := time.Now().Add(ttl).Unix()
	if modelID != 0 && isSafeShardFile(file) {
		// Origin URL: same shape as mintShardURL, but we keep modelID + file
		// so the existing handleShardDownload validates the signature.
		sig := s.signShardURL(modelID, file, exp)
		originURL = fmt.Sprintf("%s/models/%d/shards/%s?exp=%d&sig=%s",
			strings.TrimRight(s.cfg.publicURL, "/"), modelID, file, exp, sig)
		// Inspect the on-disk shard so the plan can include the byte count.
		// Falling back to zero (unknown) is harmless — the rig will issue
		// a HEAD first if it needs the size.
		if shardsDir != "" {
			if fi, err := os.Stat(filepath.Join(shardsDir, file)); err == nil {
				sizeBytes = fi.Size()
			}
		}
	}

	peers := s.peersForShard(uid, name, file)
	sources := make([]shardFetchSource, 0, len(peers)+1)
	// Peers first so the rig prefers them; sort by claimed upload BW
	// descending (zero BW falls to the end, but still ahead of origin).
	for _, p := range peers {
		// Peer URL today routes through origin with a metrics tag —
		// see file-header comment.  Once a real peer transport lands,
		// this is the one line that changes.
		if modelID == 0 {
			// Without a registered model the origin URL is unavailable,
			// so we drop the peer source too — there's nothing to point
			// at.  An HF-only peer plan is something we'll wire when the
			// HF cache path becomes a download target.
			continue
		}
		sig := s.signShardURL(modelID, file, exp)
		// AgentID is escaped because nothing pre-validates the charset on
		// agent_ids — an id with '&' or '=' would otherwise smash the
		// query string and silently break the rig downloader.
		purl := fmt.Sprintf("%s/models/%d/shards/%s?exp=%d&sig=%s&via=%s",
			strings.TrimRight(s.cfg.publicURL, "/"), modelID, file, exp, sig,
			url.QueryEscape(p.AgentID))
		sources = append(sources, shardFetchSource{
			URL:          purl,
			Kind:         "peer",
			AgentID:      p.AgentID,
			HintBWUpKbps: p.BWUpKbps,
		})
	}
	if originURL != "" {
		sources = append(sources, shardFetchSource{
			URL:  originURL,
			Kind: "origin",
		})
	}
	if len(sources) == 0 {
		http.Error(w, "no sources", http.StatusNotFound)
		return
	}

	resp := shardFetchPlan{
		Model:     name,
		File:      file,
		SizeBytes: sizeBytes,
		ChunkSize: 16 * 1024 * 1024, // 16 MiB
		Sources:   sources,
		TTLUnix:   exp,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
