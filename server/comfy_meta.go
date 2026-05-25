package main

// CF11: proxy ComfyUI's native HTTP surface through the control plane.
//
// ComfyUI exposes a useful metadata API the SDK community already
// builds against — /system_stats, /object_info, /embeddings, /queue,
// /history, /interrupt, /free.  Without a coordinator-side proxy each
// rig would have to be reached directly, which (a) means clients need
// per-rig credentials, (b) defeats the whole control-plane model, and
// (c) makes object_info caching impossible (each client redownloads
// the ~MB JSON every time).  We tunnel these as `comfy_meta` request
// frames over the existing WS, with a strict path allowlist on both
// ends so a rogue rig (or a future bug) can't be used to pivot to
// arbitrary URLs on the agent host.
//
// The wire shape is tiny by design:
//   server → rig: {"kind":"comfy_meta","corr_id":"…","method":"GET","path":"/system_stats","body":""}
//   rig → server: {"kind":"comfy_meta_result","corr_id":"…","status":200,"body_b64":"…"}
//
// body_b64 is base64-encoded so binary responses (none today, but
// future endpoints like /view return PNGs) survive intact.  Older rigs
// without b64 support inline a UTF-8 "body" field — ws.go handles both.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// comfyMetaTimeout bounds a single proxy round-trip.  Most metadata
// calls return in <100ms, but /object_info on a fresh ComfyUI can
// scan custom_nodes and take a few seconds.  10s is the cushion.
const comfyMetaTimeout = 10 * time.Second

// comfyMetaMaxBody caps the upstream response we forward to the
// client.  /object_info on a maxed-out install is ~2-5 MiB; we cap at
// 16 MiB to allow headroom without letting a buggy rig OOM the
// coordinator if the agent ever decides to send /view image bytes.
const comfyMetaMaxBody = 16 * 1024 * 1024

// comfyMetaAllowedPaths is the closed set of upstream paths a rig is
// permitted to relay.  Wildcards expand inside the agent-side dispatch
// (e.g. /object_info/{class}).  Anything not in this list is rejected
// before the WS frame is sent, so the agent can apply a matching
// allowlist as defence-in-depth without having to invent its own.
//
//   GET-side endpoints are pure metadata (introspection).
//   POST-side endpoints are control: /interrupt and /free.  We
//   deliberately do NOT proxy /prompt, /history mutations, /upload, or
//   /queue mutations — those go through our own /api/comfy/* surface
//   which has user/pool authorisation and accounting attached.
type comfyMetaRoute struct {
	method     string
	prefix     string // exact match unless allowSuffix is true
	allowExtra bool   // permits /object_info/{class}, /models/{folder}, /history/{id}
}

var comfyMetaAllowed = []comfyMetaRoute{
	{"GET", "/system_stats", false},
	{"GET", "/features", false},
	{"GET", "/embeddings", false},
	{"GET", "/models", false},
	{"GET", "/models/", true},
	{"GET", "/object_info", false},
	{"GET", "/object_info/", true},
	{"GET", "/prompt", false}, // upstream's "get queue info" — read-only despite the name
	{"GET", "/queue", false},
	{"GET", "/history", false},
	{"GET", "/history/", true},
	{"POST", "/interrupt", false},
	{"POST", "/free", false},
}

func comfyMetaPathAllowed(method, path string) bool {
	for _, r := range comfyMetaAllowed {
		if r.method != method {
			continue
		}
		if r.allowExtra && strings.HasPrefix(path, r.prefix) {
			return true
		}
		if !r.allowExtra && path == r.prefix {
			return true
		}
	}
	return false
}

// newComfyMetaCorrID returns 16 hex chars — plenty for the small
// number of in-flight meta calls per rig at any time, while staying
// short enough that the JSON envelope is cheap.
func newComfyMetaCorrID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fall back to a time-mix so we never return an empty id.
		return fmt.Sprintf("t%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// queryComfyMeta dispatches a single GET or POST to a rig's local
// ComfyUI and returns the raw response body + upstream status.
// Caller is responsible for picking the rig and for serialising the
// returned body to its HTTP client.
func (s *server) queryComfyMeta(
	ctx context.Context,
	a *agentConn,
	method, path string,
	body string,
) (status int, respBody []byte, err error) {
	if a == nil {
		return 0, nil, errors.New("nil agent")
	}
	if !comfyMetaPathAllowed(method, path) {
		return 0, nil, fmt.Errorf("comfy_meta: path %s %s not allowed", method, path)
	}
	corrID := newComfyMetaCorrID()
	ch := a.subscribeComfyMeta(corrID)
	defer a.unsubscribeComfyMeta(corrID)

	req := map[string]any{
		"kind":    "comfy_meta",
		"corr_id": corrID,
		"method":  method,
		"path":    path,
	}
	if body != "" {
		req["body"] = body
	}
	if !a.trySend(req) {
		return 0, nil, errors.New("comfy_meta: rig outbound queue full or closed")
	}

	deadline := time.NewTimer(comfyMetaTimeout)
	defer deadline.Stop()
	select {
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	case <-deadline.C:
		return 0, nil, fmt.Errorf("comfy_meta: rig timed out after %s", comfyMetaTimeout)
	case msg, ok := <-ch:
		if !ok {
			return 0, nil, errors.New("comfy_meta: rig disconnected")
		}
		if msg.Err != nil {
			return msg.HTTPStatus, msg.Body, msg.Err
		}
		if len(msg.Body) > comfyMetaMaxBody {
			return msg.HTTPStatus, nil, fmt.Errorf("comfy_meta: response too large (%d > %d)",
				len(msg.Body), comfyMetaMaxBody)
		}
		return msg.HTTPStatus, msg.Body, nil
	}
}

// pickOneComfyRig selects a single comfy-capable agent owned by uid
// (any pool).  Used by metadata endpoints where one rig's snapshot is
// representative.  Returns nil if none are online.
func (s *server) pickOneComfyRig(uid int64) *agentConn {
	picks := s.pickComfyRigs(uid, 0, 1)
	if len(picks) == 0 {
		return nil
	}
	return picks[0]
}

// pickAllComfyRigs is the multi-rig variant — used by /system_stats
// and /queue where the caller wants a per-rig fan-out.  Capped at 16
// because the calling endpoint blocks on each rig's reply.
func (s *server) pickAllComfyRigs(uid int64) []*agentConn {
	const max = 16
	return s.pickComfyRigs(uid, 0, max)
}

// ─── HTTP handlers ─────────────────────────────────────────────────────────

// proxyComfyGetOne serves a metadata GET by querying one comfy rig and
// passing through the body verbatim.  Used for /object_info,
// /embeddings, /models — calls that return rig-local snapshots.
func (s *server) proxyComfyGetOne(w http.ResponseWriter, r *http.Request, path string) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	if !comfyMetaPathAllowed("GET", path) {
		writeErr(w, 400, "path not permitted")
		return
	}
	a := s.pickOneComfyRig(u.ID)
	if a == nil {
		writeErr(w, 503, "no comfy-capable rigs online")
		return
	}
	status, body, err := s.queryComfyMeta(r.Context(), a, "GET", path, "")
	if err != nil {
		writeErr(w, 502, "rig query failed: "+err.Error())
		return
	}
	// Pass the upstream content-type through where we can guess — all
	// allowed endpoints return JSON today.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(translateUpstreamStatus(status))
	_, _ = w.Write(body)
}

// handleComfyMetaSystemStats fans out /system_stats to all online
// comfy rigs and returns a per-rig array.  Unlike object_info this is
// genuinely useful to aggregate — the operator wants to see fleet
// memory at a glance.  Each entry includes the rig's agent_id so the
// client can correlate with /api/swarm.
func (s *server) handleComfyMetaSystemStats(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rigs := s.pickAllComfyRigs(u.ID)
	if len(rigs) == 0 {
		writeErr(w, 503, "no comfy-capable rigs online")
		return
	}
	type entry struct {
		AgentID string          `json:"agent_id"`
		Status  int             `json:"status"`
		Stats   json.RawMessage `json:"stats,omitempty"`
		Error   string          `json:"error,omitempty"`
	}
	out := make([]entry, len(rigs))
	var wg sync.WaitGroup
	for i, a := range rigs {
		wg.Add(1)
		go func(i int, a *agentConn) {
			defer wg.Done()
			st, body, err := s.queryComfyMeta(r.Context(), a, "GET", "/system_stats", "")
			e := entry{AgentID: a.agentID, Status: st}
			if err != nil {
				e.Error = err.Error()
			} else if len(body) > 0 && json.Valid(body) {
				e.Stats = body
			} else if len(body) > 0 {
				e.Error = "rig returned non-JSON body"
			}
			out[i] = e
		}(i, a)
	}
	wg.Wait()
	writeJSON(w, 200, map[string]any{"rigs": out})
}

// handleComfyMetaObjectInfo returns the merged node catalogue from a
// representative rig.  We do NOT aggregate across rigs by default —
// node sets can differ when one rig has custom_nodes another doesn't,
// and merging would silently mask that.  Caller can query
// /api/comfy/object_info?agent=<id> for per-rig variants in a future
// iteration; today we just hand back the first rig's view, which is
// the same behaviour as a typical single-rig ComfyUI deployment.
func (s *server) handleComfyMetaObjectInfo(w http.ResponseWriter, r *http.Request) {
	s.proxyComfyGetOne(w, r, "/object_info")
}

func (s *server) handleComfyMetaEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.proxyComfyGetOne(w, r, "/embeddings")
}

func (s *server) handleComfyMetaModels(w http.ResponseWriter, r *http.Request) {
	s.proxyComfyGetOne(w, r, "/models")
}

func (s *server) handleComfyMetaFeatures(w http.ResponseWriter, r *http.Request) {
	s.proxyComfyGetOne(w, r, "/features")
}

// handleComfyMetaQueue aggregates queue snapshots across all comfy
// rigs.  Useful when a pool admin wants to see every rig's current
// load.  Each entry is the raw /queue body, untouched.
func (s *server) handleComfyMetaQueue(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rigs := s.pickAllComfyRigs(u.ID)
	if len(rigs) == 0 {
		writeErr(w, 503, "no comfy-capable rigs online")
		return
	}
	type entry struct {
		AgentID string          `json:"agent_id"`
		Status  int             `json:"status"`
		Queue   json.RawMessage `json:"queue,omitempty"`
		Error   string          `json:"error,omitempty"`
	}
	out := make([]entry, len(rigs))
	var wg sync.WaitGroup
	for i, a := range rigs {
		wg.Add(1)
		go func(i int, a *agentConn) {
			defer wg.Done()
			st, body, err := s.queryComfyMeta(r.Context(), a, "GET", "/queue", "")
			e := entry{AgentID: a.agentID, Status: st}
			if err != nil {
				e.Error = err.Error()
			} else if len(body) > 0 && json.Valid(body) {
				e.Queue = body
			}
			out[i] = e
		}(i, a)
	}
	wg.Wait()
	writeJSON(w, 200, map[string]any{"rigs": out})
}

// handleComfyMetaInterrupt sends /interrupt to the rig running the
// specified job.  Unlike /cancel (which deletes the job row + tells
// the rig to abort), /interrupt is the soft variant — ComfyUI stops
// the current step but the queue continues with the next item.
// Useful when a user wants to skip a slow long-prompt without losing
// the rest of a batch.
func (s *server) handleComfyMetaInterrupt(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	idStr := r.PathValue("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, 400, "bad job id")
		return
	}
	// Lookup the rig assigned to this job, verifying ownership in the
	// same query so callers can't probe other users' jobs.
	var aUID int64
	var aID string
	var owner int64
	if err := s.dbQueryRow(
		`SELECT user_id, COALESCE(agent_user_id, 0), COALESCE(agent_id, '')
		   FROM comfy_jobs WHERE id = ?`, id).Scan(&owner, &aUID, &aID); err != nil {
		writeErr(w, 404, "job not found")
		return
	}
	if owner != u.ID {
		// Mirror /cancel's 404 to avoid an existence oracle.
		writeErr(w, 404, "job not found")
		return
	}
	if aID == "" {
		writeErr(w, 409, "job not yet bound to a rig")
		return
	}
	a, ok := s.hub.findAgent(aUID, aID)
	if !ok {
		writeErr(w, 503, "rig offline")
		return
	}
	// Soft interrupt — no prompt_id supplied means ComfyUI does a
	// global interrupt of the rig's current step.  We don't pass a
	// prompt_id because the rig owns that mapping and may have
	// reassigned its workflow id by the time the frame lands.
	st, body, err := s.queryComfyMeta(r.Context(), a, "POST", "/interrupt", "{}")
	if err != nil {
		writeErr(w, 502, "rig interrupt failed: "+err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"job_id":         id,
		"upstream":       translateUpstreamStatus(st),
		"upstream_body":  string(body),
		"upstream_bytes": len(body),
	})
}

// handleComfyMetaFree fans out the /free endpoint, asking comfy rigs
// to unload models / free GPU memory.  This is a self-service garbage
// collector that's safe to call (ComfyUI will lazy-reload on the next
// prompt) but it does evict cached weights, so we cap it to the
// caller's own rigs only.  When the admin role lands (P20-D1) we'll
// add a fleet-wide variant gated on it.
func (s *server) handleComfyMetaFree(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		UnloadModels bool `json:"unload_models"`
		FreeMemory   bool `json:"free_memory"`
	}
	// Decode is best-effort — empty body means "both flags off",
	// matching ComfyUI's own /free behaviour.
	_ = json.NewDecoder(r.Body).Decode(&body)
	upstream, _ := json.Marshal(body)

	rigs := s.pickAllComfyRigs(u.ID)
	if len(rigs) == 0 {
		writeErr(w, 503, "no comfy-capable rigs online")
		return
	}
	type entry struct {
		AgentID string `json:"agent_id"`
		Status  int    `json:"status"`
		Error   string `json:"error,omitempty"`
	}
	out := make([]entry, len(rigs))
	var wg sync.WaitGroup
	for i, a := range rigs {
		wg.Add(1)
		go func(i int, a *agentConn) {
			defer wg.Done()
			st, _, err := s.queryComfyMeta(r.Context(), a, "POST", "/free", string(upstream))
			e := entry{AgentID: a.agentID, Status: st}
			if err != nil {
				e.Error = err.Error()
			}
			out[i] = e
		}(i, a)
	}
	wg.Wait()
	writeJSON(w, 200, map[string]any{"rigs": out})
}

// translateUpstreamStatus maps the upstream HTTP status into the
// proxy response code.  ComfyUI's 5xx becomes our 502 (so callers can
// distinguish coordinator vs rig faults); 4xx passes through; 0
// (unreachable) becomes 503.
func translateUpstreamStatus(st int) int {
	switch {
	case st == 0:
		return 503
	case st >= 500:
		return 502
	case st >= 400:
		return st
	case st >= 200 && st < 300:
		return st
	default:
		return 502
	}
}

