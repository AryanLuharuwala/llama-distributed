package main

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/coder/websocket"
	ctrlpb "github.com/llama-distributed/server/ctrlpb"
	"google.golang.org/protobuf/proto"
)

// wireCodec is the per-connection serialization for /ws/agent control
// frames.  Two implementations:
//
//   - jsonCodec — text frames with json-encoded map[string]any.  This
//     is the default for legacy rigs and remains byte-for-byte
//     identical to the pre-P4 wire.
//   - protoCodec — binary frames with proto-encoded ClientFrame /
//     ServerFrame envelopes (distpool.ctrl.v1).  Selected when the rig
//     advertises Sec-WebSocket-Protocol: distpool.proto.v1 on the
//     /ws/agent handshake.
//
// The codec surface is intentionally map-shaped: the reader loop and
// the outbound goroutine in handleAgentWS already operate on
// map[string]any keyed by "kind", and forking that for the proto path
// would double the dispatch logic.  The codec is responsible for
// (de)serializing the map to/from its wire encoding, leaving the
// existing dispatch unchanged.
//
// Binary relay frames (ACTV passthrough on the inference data plane)
// are NOT routed through the codec — they remain raw bytes on
// ac.binCh and through dispatchBinaryFromAgent.  The codec's decode
// method returns isRelay=true on a binary frame the protoCodec
// couldn't parse, so the existing relay path absorbs anything that
// isn't a recognized ClientFrame.
type wireCodec interface {
	name() string

	// encodeServer turns a server-emitted message (map[string]any
	// keyed by "kind", today's format) into bytes ready for
	// conn.Write, along with the WebSocket message type.
	encodeServer(msg any) (websocket.MessageType, []byte, error)

	// decodeClient turns an incoming WS frame into either:
	//   - a parsed control frame as map[string]any (isRelay=false)
	//   - or a signal that the bytes are a relay passthrough
	//     (isRelay=true, body=nil)
	// The map shape is identical to what json.Unmarshal would
	// produce on the legacy JSON wire, so the dispatch loop in
	// handleAgentWS does not need to know which codec is active.
	decodeClient(mt websocket.MessageType, data []byte) (body map[string]any, isRelay bool, err error)
}

// codecForSubprotocol resolves the negotiated subprotocol header
// (returned by websocket.Conn.Subprotocol()) to a codec.  The empty
// string means JSON — same as legacy rigs.
func codecForSubprotocol(sub string) wireCodec {
	switch sub {
	case protoSubprotocolV1:
		return protoCodec{}
	default:
		return jsonCodec{}
	}
}

// protoSubprotocolV1 is the Sec-WebSocket-Protocol token the rig
// sends to switch the /ws/agent wire to length-delimited proto.
const protoSubprotocolV1 = "distpool.proto.v1"

// ────────────────────────────────────────────────────────────────────
// JSON codec — preserves the pre-P4 wire byte-for-byte.

type jsonCodec struct{}

func (jsonCodec) name() string { return "json" }

func (jsonCodec) encodeServer(msg any) (websocket.MessageType, []byte, error) {
	b, err := json.Marshal(msg)
	if err != nil {
		return 0, nil, err
	}
	return websocket.MessageText, b, nil
}

func (jsonCodec) decodeClient(mt websocket.MessageType, data []byte) (map[string]any, bool, error) {
	if mt == websocket.MessageBinary {
		// Today's invariant: binary frames are ACTV relay bytes, not
		// control.  Hand them back to the caller as relay.
		return nil, true, nil
	}
	var msg map[string]any
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, false, err
	}
	return msg, false, nil
}

// ────────────────────────────────────────────────────────────────────
// Proto codec — distpool.ctrl.v1 over WebSocket BINARY frames.
//
// The decode path projects a ClientFrame oneof into the same
// map[string]any shape the JSON path produces, so the reader loop in
// handleAgentWS works without modification.  Unknown / unparseable
// binary frames are reported as isRelay=true so the relay passthrough
// keeps working.

type protoCodec struct{}

func (protoCodec) name() string { return "proto.v1" }

func (protoCodec) encodeServer(msg any) (websocket.MessageType, []byte, error) {
	m, ok := msg.(map[string]any)
	if !ok {
		// Server callers occasionally pass a typed struct (e.g.
		// agentHello echoed back).  Round-trip through JSON to
		// normalize into the map shape we know how to encode.
		raw, err := json.Marshal(msg)
		if err != nil {
			return 0, nil, err
		}
		m = map[string]any{}
		if err := json.Unmarshal(raw, &m); err != nil {
			return 0, nil, err
		}
	}
	frame, err := serverFrameFromMap(m)
	if err != nil {
		return 0, nil, err
	}
	b, err := proto.Marshal(frame)
	if err != nil {
		return 0, nil, err
	}
	return websocket.MessageBinary, b, nil
}

func (protoCodec) decodeClient(mt websocket.MessageType, data []byte) (map[string]any, bool, error) {
	if mt == websocket.MessageText {
		// A rig negotiated proto.v1 but is sending a TEXT frame —
		// most likely a legacy code path on the rig side.  Decode
		// it as JSON so the connection still works.
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, false, err
		}
		return msg, false, nil
	}
	var frame ctrlpb.ClientFrame
	if err := proto.Unmarshal(data, &frame); err != nil {
		// Not a control frame — pass to the relay path.  This is
		// the routing rule: in proto mode the relay plane still
		// uses raw binary frames; control frames are
		// distinguishable by the fact that they parse as a
		// ClientFrame, relay frames are not.
		return nil, true, nil
	}
	body := clientFrameToMap(&frame)
	if body == nil {
		// Empty oneof — treat as relay passthrough.
		return nil, true, nil
	}
	return body, false, nil
}

// ────────────────────────────────────────────────────────────────────
// proto <-> map[string]any projections.

func clientFrameToMap(f *ctrlpb.ClientFrame) map[string]any {
	if f == nil {
		return nil
	}
	switch b := f.Body.(type) {
	case *ctrlpb.ClientFrame_Hello:
		return helloToMap(b.Hello)
	case *ctrlpb.ClientFrame_Status:
		return statusToMap(b.Status)
	case *ctrlpb.ClientFrame_RelayStats:
		return map[string]any{
			"kind":       "relay_stats",
			"session_id": b.RelayStats.GetSessionId(),
			"bytes_l2r":  float64(b.RelayStats.GetBytesL2R()),
			"bytes_r2l":  float64(b.RelayStats.GetBytesR2L()),
		}
	case *ctrlpb.ClientFrame_ComfyResult:
		return jsonBytesToMap("comfy_result", b.ComfyResult.GetJson())
	case *ctrlpb.ClientFrame_ComfyCaps:
		return jsonBytesToMap("comfy_caps", b.ComfyCaps.GetJson())
	case *ctrlpb.ClientFrame_P2PSignal:
		return jsonBytesToMap("p2p_signal", b.P2PSignal.GetJson())
	default:
		return nil
	}
}

func jsonBytesToMap(kind string, raw []byte) map[string]any {
	m := map[string]any{}
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &m)
	}
	m["kind"] = kind
	return m
}

func helloToMap(h *ctrlpb.AgentHello) map[string]any {
	m := map[string]any{
		"kind":             h.GetKind(),
		"token":            h.GetToken(),
		"agent_key":        h.GetAgentKey(),
		"agent_id":         h.GetAgentId(),
		"hostname":         h.GetHostname(),
		"n_gpus":           float64(h.GetNGpus()),
		"vram_bytes":       float64(h.GetVramBytes()),
		"pubkey":           h.GetPubkeyB64(),
		"protocol_version": float64(h.GetProtocolVersion()),
		"client_build":     h.GetClientBuild(),
	}
	if shards := h.GetCachedShards(); len(shards) > 0 {
		out := make([]any, 0, len(shards))
		for _, s := range shards {
			out = append(out, map[string]any{
				"model":      s.GetModel(),
				"file":       s.GetFile(),
				"size_bytes": float64(s.GetSizeBytes()),
			})
		}
		m["cached_shards"] = out
	}
	return m
}

func statusToMap(s *ctrlpb.AgentStatus) map[string]any {
	m := map[string]any{
		"kind":           "status",
		"tokens_sec":     s.GetTokensSec(),
		"n_gpus":         float64(s.GetNGpus()),
		"gpu_model":      s.GetGpuModel(),
		"vram_total":     float64(s.GetVramTotal()),
		"vram_free":      float64(s.GetVramFree()),
		"uptime_sec":     float64(s.GetUptimeSec()),
		"inflight":       float64(s.GetInflight()),
		"bw_up_kbps":     float64(s.GetBwUpKbps()),
		"bw_dn_kbps":     float64(s.GetBwDnKbps()),
		"last_error":     s.GetLastError(),
		"nat_type":       s.GetNatType(),
		"relay_capable":  s.GetRelayCapable(),
		"coturn_port":    float64(s.GetCoturnPort()),
		"public_ip":      s.GetPublicIp(),
		"max_concurrent": float64(s.GetMaxConcurrent()),
	}
	if g := s.GetGpuUtil(); len(g) > 0 {
		anys := make([]any, len(g))
		for i, v := range g {
			anys[i] = v
		}
		m["gpu_util"] = anys
	}
	if r := s.GetRoles(); len(r) > 0 {
		anys := make([]any, len(r))
		for i, v := range r {
			anys[i] = v
		}
		m["roles"] = anys
	}
	if mm := s.GetModels(); len(mm) > 0 {
		anys := make([]any, len(mm))
		for i, v := range mm {
			anys[i] = v
		}
		m["models"] = anys
	}
	if shards := s.GetCachedShards(); len(shards) > 0 {
		out := make([]any, 0, len(shards))
		for _, sh := range shards {
			out = append(out, map[string]any{
				"model":      sh.GetModel(),
				"file":       sh.GetFile(),
				"size_bytes": float64(sh.GetSizeBytes()),
			})
		}
		m["cached_shards"] = out
	}
	return m
}

// serverFrameFromMap is the inverse of clientFrameToMap for the
// server-emitted side.  We recognize the well-known control frames
// (welcome, error, challenge, command); anything else falls into the
// OpaqueJSON pocket so the caller doesn't need to enumerate every
// possible message at the codec layer.
func serverFrameFromMap(m map[string]any) (*ctrlpb.ServerFrame, error) {
	kind, _ := m["kind"].(string)
	switch kind {
	case "welcome":
		w := &ctrlpb.Welcome{
			AgentKey:           getString(m, "agent_key"),
			UserId:             getInt64(m, "user_id"),
			DisplayName:        getString(m, "display_name"),
			NegotiatedProtocol: getInt32(m, "negotiated_protocol"),
		}
		return &ctrlpb.ServerFrame{Body: &ctrlpb.ServerFrame_Welcome{Welcome: w}}, nil
	case "error":
		e := &ctrlpb.ErrorFrame{
			Message:           getString(m, "message"),
			ServerProtocolMin: getInt32(m, "server_protocol_min"),
			ServerProtocolMax: getInt32(m, "server_protocol_max"),
		}
		return &ctrlpb.ServerFrame{Body: &ctrlpb.ServerFrame_Error{Error: e}}, nil
	case "challenge":
		c := &ctrlpb.Challenge{Nonce: getString(m, "nonce")}
		return &ctrlpb.ServerFrame{Body: &ctrlpb.ServerFrame_Challenge{Challenge: c}}, nil
	case "command":
		cmd := &ctrlpb.Command{
			Cmd: getString(m, "cmd"),
		}
		// Payload may be passed as either a string or arbitrary
		// JSON-able object.  Encode whatever's there to bytes so
		// the rig can re-decode on its end.
		if p, ok := m["payload"]; ok && p != nil {
			if s, isString := p.(string); isString {
				cmd.Payload = []byte(s)
			} else {
				raw, err := json.Marshal(p)
				if err != nil {
					return nil, fmt.Errorf("encode command payload: %w", err)
				}
				cmd.Payload = raw
			}
		}
		return &ctrlpb.ServerFrame{Body: &ctrlpb.ServerFrame_Command{Command: cmd}}, nil
	default:
		// Unknown server-to-rig frame — wrap as opaque JSON so the
		// connection stays usable even when new server frames ship
		// before the codec is updated.
		raw, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		return &ctrlpb.ServerFrame{Body: &ctrlpb.ServerFrame_Other{Other: &ctrlpb.OpaqueJSON{Json: raw}}}, nil
	}
}

// ────────────────────────────────────────────────────────────────────
// Small map[string]any accessors.  Server-emitted maps are built by
// hand and types are uneven (int vs int64 vs float64) — these
// normalize without panicking on missing keys.

func getString(m map[string]any, k string) string {
	if v, ok := m[k].(string); ok {
		return v
	}
	return ""
}

func getInt64(m map[string]any, k string) int64 {
	switch v := m[k].(type) {
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case int64:
		return v
	case float64:
		return int64(v)
	case float32:
		return int64(v)
	}
	return 0
}

func getInt32(m map[string]any, k string) int32 {
	return int32(getInt64(m, k))
}

// errCodecNoSubprotocol is returned by negotiateCodec when the rig
// requested a subprotocol the server doesn't speak.  Connection
// is rejected by the caller before any auth work.
var errCodecNoSubprotocol = errors.New("requested subprotocol not supported")
