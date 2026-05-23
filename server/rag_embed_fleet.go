package main

// Server-side scaffolding for the rig-fleet embedding tier.
//
// Wire contract (in the control plane direction):
//   →  {"type":"embed_request", "req_id":"<uuid>", "model":"<model-id>",
//       "texts":["...", ...]}
//   ←  {"type":"embed_reply", "req_id":"<uuid>",
//       "vectors":[[...], [...]], "model":"<model-id>", "dim":<int>,
//       "error":"" }
//
// An agent advertises participation by including "embed" in its status
// RolesHeld slice and (optionally) the served model in ModelsHeld.
//
// This file ships:
//   • rigEmbedFleet — picks an online rig that advertises role="embed",
//     pinned to the desired model + dim if specified.
//   • rigEmbedder — embedder implementation that uses the fleet, with
//     a process-wide timeout and a hashEmbedder fallback for dev /
//     no-rig deployments.
//
// The actual WS request/response is *not* wired yet — agent-side
// support is on the prod-branch roadmap (task R6 in the sweep tracker,
// agent path explicitly deferred per scope).  Until then Embed()
// returns the hash-fallback result, but selection telemetry is recorded
// so observability is in place ahead of the agent ship.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

type rigEmbedFleet struct {
	s       *server
	model   string // empty = accept any
	dim     int    // 0 = accept any
	fallback embedder

	// Telemetry: number of times we fell back because no rig was usable.
	// Surfaced via dashboards once the agent path lands.
	fallbacks atomic.Int64
	requests  atomic.Int64
}

// newRigEmbedFleet constructs a fleet view bound to (model, dim).  Pass
// model="" / dim=0 to accept any embed-capable rig.
func newRigEmbedFleet(s *server, model string, dim int, fallback embedder) *rigEmbedFleet {
	if fallback == nil {
		// Always have something to fall back to so a misconfigured fleet
		// doesn't break dev/test envs.
		d := dim
		if d <= 0 {
			d = 256
		}
		fallback = newHashEmbedder(d)
	}
	return &rigEmbedFleet{s: s, model: model, dim: dim, fallback: fallback}
}

// pickRig walks the live agent set and returns one that advertises the
// "embed" role and matches the (model, dim) filter.  Returns ("", false)
// when nothing matches.
func (f *rigEmbedFleet) pickRig() (agentID string, ok bool) {
	if f.s == nil || f.s.hub == nil {
		return "", false
	}
	cands := f.s.hub.snapshotAgents()
	var best *agentConn
	var bestUpdated int64
	for _, a := range cands {
		st := a.snapshotStatus()
		if !roleHeld(st.RolesHeld, "embed") {
			continue
		}
		if f.model != "" && !modelHeld(st.ModelsHeld, f.model) {
			continue
		}
		// Prefer the rig with the most recent status frame — proxy for
		// freshness / least-likely-to-be-stuck.
		if best == nil || st.UpdatedAt > bestUpdated {
			best = a
			bestUpdated = st.UpdatedAt
		}
	}
	if best == nil {
		return "", false
	}
	return best.agentID, true
}

func (f *rigEmbedFleet) Dim() int {
	if f.dim > 0 {
		return f.dim
	}
	return f.fallback.Dim()
}

func (f *rigEmbedFleet) ModelID() string {
	if f.model != "" {
		return f.model
	}
	return f.fallback.ModelID()
}

// Embed runs the configured model on the rig fleet.  Until the agent-side
// wire lands, this routes through the fallback embedder — but we still
// pick a rig and record a request so observability is in place ahead
// of time.  The timeout below is sized for a real model RTT; it does
// not affect the fallback path.
func (f *rigEmbedFleet) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	f.requests.Add(1)
	if _, ok := f.pickRig(); !ok {
		f.fallbacks.Add(1)
		return f.fallback.Embed(ctx, texts)
	}
	// TODO(R6 agent-side): replace this with a real WS request/response.
	//   reqID := newReqID()
	//   reply, err := f.s.dispatchEmbedRequest(ctx, rigID, embedRequest{...})
	//   return reply.Vectors, err
	// For now we still return the fallback result so end-to-end RAG works,
	// but the rig selection is exercised and counted.
	subCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	_ = subCtx
	return f.fallback.Embed(ctx, texts)
}

// roleHeld is a small helper — kept here rather than in ws.go so the
// dependency direction stays clean.
func roleHeld(roles []string, want string) bool {
	for _, r := range roles {
		if strings.EqualFold(r, want) {
			return true
		}
	}
	return false
}

func modelHeld(models []string, want string) bool {
	want = strings.ToLower(want)
	for _, m := range models {
		if strings.ToLower(m) == want {
			return true
		}
	}
	return false
}

// FleetStats is the telemetry surface for the embed fleet.  Exposed via
// /api/console/embed_fleet once a dashboard panel exists.
type FleetStats struct {
	Requests       int64 `json:"requests"`
	Fallbacks      int64 `json:"fallbacks"`
	RigsAdvertised int   `json:"rigs_advertised"`
}

func (f *rigEmbedFleet) Stats() FleetStats {
	advertised := 0
	if f.s != nil && f.s.hub != nil {
		for _, a := range f.s.hub.snapshotAgents() {
			st := a.snapshotStatus()
			if roleHeld(st.RolesHeld, "embed") {
				advertised++
			}
		}
	}
	return FleetStats{
		Requests:       f.requests.Load(),
		Fallbacks:      f.fallbacks.Load(),
		RigsAdvertised: advertised,
	}
}

// errNoEmbedRig is returned when caller-policy demands a real rig and
// none is available.  Currently unused (we fall back silently) but
// surfaced for the agent-side ship.
var errNoEmbedRig = errors.New("rag: no embed-capable rig connected")

// guardrails: callers may want a strict-mode fleet that refuses to
// silently fall back.  Wire later via cfg.
var _ = fmt.Sprintf // keep fmt import warm for the future WS payloads
