package main

// sd.cpp result dispatcher — fans `sdcpp_done` / `sdcpp_error` /
// `sdcpp_role_done` / `sdcpp_progress` frames from rigs into the
// request goroutine that owns the matching req_id.
//
// Why server-level (not per-agent like comfyResults): the role chain
// crosses three different rigs (TE → UNet → VAE), so the subscriber
// needs to receive frames from any agent that participates in the
// req_id, not just one. The dispatcher in sdcpp_comfyjob owns the
// req_id lifecycle: it subscribes before fanning out role frames and
// unsubscribes when it has the final PNG (or hits an error).
//
// The frame shapes are documented at the top of src/sdcpp_worker.cpp:
//
//   {"kind":"sdcpp_progress",  "req_id":N,"step":i,"steps":S,"t":secs}
//   {"kind":"sdcpp_done",      "req_id":N,"png_b64":"…"}
//   {"kind":"sdcpp_error",     "req_id":N,"error":"…"}
//   {"kind":"sdcpp_role_done", "req_id":N,"role":"te|unet|vae","frame_b64":"…"}

import (
	"sync"
)

// sdcppResultKind tags the variant the receiver should switch on.
type sdcppResultKind int

const (
	sdcppResultProgress sdcppResultKind = iota
	sdcppResultRoleDone
	sdcppResultDone
	sdcppResultError
)

// sdcppResultMsg is the normalised event delivered to the per-req_id
// subscriber. Only the fields relevant to Kind are populated.
type sdcppResultMsg struct {
	Kind    sdcppResultKind
	Role    string // role_done: which role completed
	FrameB64 string // role_done: TE cond / UNet latent payload
	PNGB64  string // done: final image
	ErrMsg  string // error
	Step    int    // progress
	Steps   int    // progress
	AgentID string // provenance — useful for error reporting
}

// sdcppResultDispatcher routes inbound sd.cpp frames into per-req_id
// channels. The job goroutine subscribes before issuing the first
// route and unsubscribes when it terminates (success, error, or
// timeout).
type sdcppResultDispatcher struct {
	mu   sync.Mutex
	subs map[uint16]chan sdcppResultMsg
}

func newSdcppResultDispatcher() *sdcppResultDispatcher {
	return &sdcppResultDispatcher{
		subs: map[uint16]chan sdcppResultMsg{},
	}
}

// Subscribe registers interest in frames for req_id. Returns a buffered
// channel (cap 8 — enough for one role_done per role plus a few progress
// ticks in flight). Caller MUST call Unsubscribe.
func (d *sdcppResultDispatcher) Subscribe(reqID uint16) chan sdcppResultMsg {
	ch := make(chan sdcppResultMsg, 8)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.subs[reqID] = ch
	return ch
}

func (d *sdcppResultDispatcher) Unsubscribe(reqID uint16) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if ch, ok := d.subs[reqID]; ok {
		delete(d.subs, reqID)
		close(ch)
	}
}

// deliver routes msg to the registered channel. Returns false when no
// subscriber exists (stale frame after timeout/cancel). Non-blocking
// send — a slow subscriber drops progress events but a Kind=Done frame
// is critical; we still attempt non-blocking and rely on the channel
// buffer being large enough for the role-chain's three role_done frames
// plus one done.
func (d *sdcppResultDispatcher) deliver(reqID uint16, msg sdcppResultMsg) bool {
	d.mu.Lock()
	ch, ok := d.subs[reqID]
	d.mu.Unlock()
	if !ok {
		return false
	}
	select {
	case ch <- msg:
	default:
	}
	return true
}

// ingestSdcppFrame parses one WS frame from a rig and routes it. Called
// from the WS reader for kinds: sdcpp_done, sdcpp_error, sdcpp_progress,
// sdcpp_role_done.
func (s *server) ingestSdcppFrame(agentID string, kind string, msg map[string]any) bool {
	if s.sdcppResults == nil {
		return false
	}
	rf, _ := msg["req_id"].(float64)
	reqID := uint16(rf)
	out := sdcppResultMsg{AgentID: agentID}
	switch kind {
	case "sdcpp_progress":
		out.Kind = sdcppResultProgress
		if v, ok := msg["step"].(float64); ok {
			out.Step = int(v)
		}
		if v, ok := msg["steps"].(float64); ok {
			out.Steps = int(v)
		}
	case "sdcpp_role_done":
		out.Kind = sdcppResultRoleDone
		out.Role, _ = msg["role"].(string)
		out.FrameB64, _ = msg["frame_b64"].(string)
	case "sdcpp_done":
		out.Kind = sdcppResultDone
		out.PNGB64, _ = msg["png_b64"].(string)
	case "sdcpp_error":
		out.Kind = sdcppResultError
		if v, ok := msg["error"].(string); ok && v != "" {
			out.ErrMsg = v
		} else if v, ok := msg["message"].(string); ok {
			out.ErrMsg = v
		}
	default:
		return false
	}
	return s.sdcppResults.deliver(reqID, out)
}
