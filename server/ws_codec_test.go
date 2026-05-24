package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	ctrlpb "github.com/llama-distributed/server/ctrlpb"
	"google.golang.org/protobuf/proto"
)

// JSON codec must be byte-for-byte indistinguishable from a direct
// json.Marshal — this is what keeps the legacy wire intact.
func TestJSONCodec_EncodeServer_IsPlainJSON(t *testing.T) {
	c := jsonCodec{}
	msg := map[string]any{"kind": "welcome", "user_id": int64(42), "display_name": "n"}
	mt, b, err := c.encodeServer(msg)
	if err != nil {
		t.Fatalf("encodeServer: %v", err)
	}
	if mt != websocket.MessageText {
		t.Errorf("JSON codec must use text frames, got %v", mt)
	}
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("not json: %v (raw=%s)", err, b)
	}
	if back["kind"] != "welcome" || back["display_name"] != "n" {
		t.Errorf("wrong roundtrip: %+v", back)
	}
}

func TestJSONCodec_DecodeClient_BinaryIsRelay(t *testing.T) {
	c := jsonCodec{}
	body, isRelay, err := c.decodeClient(websocket.MessageBinary, []byte{0x00, 0x01, 0x02})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !isRelay {
		t.Errorf("binary frame on JSON wire must be flagged as relay")
	}
	if body != nil {
		t.Errorf("relay frames have no decoded body, got %+v", body)
	}
}

func TestJSONCodec_DecodeClient_TextHello(t *testing.T) {
	c := jsonCodec{}
	raw := []byte(`{"kind":"hello","agent_id":"rig-7","token":"abc","n_gpus":2}`)
	body, isRelay, err := c.decodeClient(websocket.MessageText, raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if isRelay {
		t.Errorf("text frame must never be relay")
	}
	if body["kind"] != "hello" || body["agent_id"] != "rig-7" {
		t.Errorf("wrong decode: %+v", body)
	}
}

// Proto codec round-trip: encode a ClientFrame.Hello on the wire,
// decode through the codec, verify the map shape matches what the
// JSON wire would produce.
func TestProtoCodec_DecodeClient_HelloRoundTrip(t *testing.T) {
	frame := &ctrlpb.ClientFrame{
		Body: &ctrlpb.ClientFrame_Hello{
			Hello: &ctrlpb.AgentHello{
				Kind:            "hello",
				Token:           "pair-token",
				AgentId:         "rig-1",
				Hostname:        "lab-1",
				NGpus:           2,
				VramBytes:       8 << 30,
				ProtocolVersion: 1,
				CachedShards: []*ctrlpb.CachedShardEntry{
					{Model: "m", File: "f.safetensors", SizeBytes: 1234},
				},
			},
		},
	}
	wire, err := proto.Marshal(frame)
	if err != nil {
		t.Fatalf("proto marshal: %v", err)
	}
	body, isRelay, err := protoCodec{}.decodeClient(websocket.MessageBinary, wire)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if isRelay {
		t.Fatalf("hello must not be flagged as relay")
	}
	if body["kind"] != "hello" || body["agent_id"] != "rig-1" {
		t.Errorf("hello map shape: %+v", body)
	}
	if body["token"] != "pair-token" {
		t.Errorf("token missing: %+v", body)
	}
	if body["n_gpus"].(float64) != 2 {
		t.Errorf("n_gpus: %+v", body["n_gpus"])
	}
	shards, ok := body["cached_shards"].([]any)
	if !ok || len(shards) != 1 {
		t.Fatalf("cached_shards: %+v", body["cached_shards"])
	}
	sh := shards[0].(map[string]any)
	if sh["model"] != "m" {
		t.Errorf("shard model: %+v", sh)
	}
}

func TestProtoCodec_DecodeClient_StatusRoundTrip(t *testing.T) {
	st := &ctrlpb.AgentStatus{
		TokensSec:     7.5,
		NGpus:         1,
		GpuUtil:       []float64{0.8},
		Roles:         []string{"text_encoder"},
		Models:        []string{"meta/llama"},
		MaxConcurrent: 4,
		NatType:       "cone",
		RelayCapable:  true,
	}
	wire, err := proto.Marshal(&ctrlpb.ClientFrame{Body: &ctrlpb.ClientFrame_Status{Status: st}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	body, isRelay, err := protoCodec{}.decodeClient(websocket.MessageBinary, wire)
	if err != nil || isRelay {
		t.Fatalf("decode: relay=%v err=%v", isRelay, err)
	}
	if body["kind"] != "status" {
		t.Errorf("wrong kind: %+v", body["kind"])
	}
	if body["tokens_sec"].(float64) != 7.5 {
		t.Errorf("tokens_sec: %+v", body["tokens_sec"])
	}
	if body["relay_capable"].(bool) != true {
		t.Errorf("relay_capable: %+v", body["relay_capable"])
	}
	if body["max_concurrent"].(float64) != 4 {
		t.Errorf("max_concurrent: %+v", body["max_concurrent"])
	}
	roles := body["roles"].([]any)
	if len(roles) != 1 || roles[0].(string) != "text_encoder" {
		t.Errorf("roles: %+v", roles)
	}
}

func TestProtoCodec_DecodeClient_RelayStatsRoundTrip(t *testing.T) {
	rs := &ctrlpb.RelayStats{SessionId: "sess-9", BytesL2R: 1_000_000, BytesR2L: 2_000_000}
	wire, _ := proto.Marshal(&ctrlpb.ClientFrame{Body: &ctrlpb.ClientFrame_RelayStats{RelayStats: rs}})
	body, isRelay, err := protoCodec{}.decodeClient(websocket.MessageBinary, wire)
	if err != nil || isRelay {
		t.Fatalf("decode: relay=%v err=%v", isRelay, err)
	}
	if body["session_id"] != "sess-9" {
		t.Errorf("session_id: %+v", body["session_id"])
	}
	if body["bytes_l2r"].(float64) != 1_000_000 {
		t.Errorf("bytes_l2r: %+v", body["bytes_l2r"])
	}
}

// Server-side encode: welcome must produce a parseable ServerFrame.
func TestProtoCodec_EncodeServer_Welcome(t *testing.T) {
	c := protoCodec{}
	mt, data, err := c.encodeServer(map[string]any{
		"kind":         "welcome",
		"agent_key":    "k",
		"user_id":      int64(7),
		"display_name": "n",
	})
	if err != nil {
		t.Fatalf("encodeServer: %v", err)
	}
	if mt != websocket.MessageBinary {
		t.Errorf("proto codec must use binary frames, got %v", mt)
	}
	var frame ctrlpb.ServerFrame
	if err := proto.Unmarshal(data, &frame); err != nil {
		t.Fatalf("not a server frame: %v", err)
	}
	w := frame.GetWelcome()
	if w == nil {
		t.Fatalf("welcome body missing: %+v", frame.Body)
	}
	if w.GetAgentKey() != "k" || w.GetUserId() != 7 || w.GetDisplayName() != "n" {
		t.Errorf("welcome fields: %+v", w)
	}
}

// Unknown server-side kind falls back to OpaqueJSON so the wire stays
// usable when new frames ship before the codec gains a typed message.
func TestProtoCodec_EncodeServer_UnknownKindBecomesOpaqueJSON(t *testing.T) {
	c := protoCodec{}
	_, data, err := c.encodeServer(map[string]any{"kind": "agent_message", "blob": "anything"})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var frame ctrlpb.ServerFrame
	if err := proto.Unmarshal(data, &frame); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	other := frame.GetOther()
	if other == nil {
		t.Fatalf("expected OpaqueJSON, got %+v", frame.Body)
	}
	var back map[string]any
	if err := json.Unmarshal(other.GetJson(), &back); err != nil {
		t.Fatalf("opaque body not json: %v", err)
	}
	if back["blob"] != "anything" {
		t.Errorf("opaque payload lost fields: %+v", back)
	}
}

// Non-proto binary bytes on the proto wire must fall through to the
// relay path, not crash the connection.
func TestProtoCodec_DecodeClient_UnknownBinaryIsRelay(t *testing.T) {
	// Random bytes that don't decode as a ClientFrame.
	junk := []byte{0xff, 0xff, 0xff, 0xff, 0xff}
	body, isRelay, err := protoCodec{}.decodeClient(websocket.MessageBinary, junk)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !isRelay {
		t.Errorf("unparseable binary must be relay, got body=%+v", body)
	}
}

// Subprotocol negotiation: dialing /ws/agent with the
// distpool.proto.v1 token must have the server echo it back so the
// rig knows the binary wire is active.
func TestSubprotocolNegotiation(t *testing.T) {
	// Stand up a tiny ws.Accept that mirrors what handleAgentWS does.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
			OriginPatterns:     []string{"*"},
			Subprotocols:       []string{protoSubprotocolV1},
		})
		if err != nil {
			return
		}
		_ = conn.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()

	url := strings.Replace(srv.URL, "http://", "ws://", 1)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// Client opts in:
	conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		Subprotocols: []string{protoSubprotocolV1},
	})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	got := conn.Subprotocol()
	_ = conn.Close(websocket.StatusNormalClosure, "")
	if got != protoSubprotocolV1 {
		t.Errorf("expected subprotocol %q, got %q", protoSubprotocolV1, got)
	}
	if c := codecForSubprotocol(got); c.name() != "proto.v1" {
		t.Errorf("codecForSubprotocol: got %s, want proto.v1", c.name())
	}

	// Legacy client (no subprotocol) gets JSON.
	ctx2, cancel2 := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel2()
	conn2, _, err := websocket.Dial(ctx2, url, nil)
	if err != nil {
		t.Fatalf("legacy dial: %v", err)
	}
	got2 := conn2.Subprotocol()
	_ = conn2.Close(websocket.StatusNormalClosure, "")
	if got2 != "" {
		t.Errorf("legacy client should get empty subprotocol, got %q", got2)
	}
	if c := codecForSubprotocol(got2); c.name() != "json" {
		t.Errorf("legacy codec: got %s, want json", c.name())
	}
}
