#pragma once

// ACTV-over-WebRTC peer-to-peer transport.
//
// When DIST_HAVE_P2P is defined at build time (controlled by the
// DIST_USE_P2P CMake option), this module wraps libdatachannel to give
// dist-node a direct binary channel to its pipeline neighbour, bypassing
// the dist-server relay for the ACTV data plane.  When the macro is not
// defined, every entry point compiles to a no-op stub that reports
// "unavailable" — the surrounding code falls back to the legacy WS relay.
//
// Lifecycle for a single pipeline-stage neighbour:
//
//   open_actv_peer(session_id, peer_agent_id, is_offerer, signal_send_fn,
//                  on_open_fn, on_actv_frame_fn)
//       Returns a session handle.  Spawns a libdatachannel
//       PeerConnection and (if is_offerer) immediately creates an SDP
//       offer; otherwise waits for the remote offer to arrive via
//       handle_signal().
//
//   handle_signal(handle, kind, payload_json)
//       Feed inbound `p2p_offer` / `p2p_answer` / `p2p_ice` JSON
//       frames (as they arrive from the dist-server WS) into the
//       state machine.
//
//   send_actv(handle, bytes, n)
//       Send an ACTV frame over the data channel.  Returns false if
//       the channel isn't open yet (caller should fall back to the
//       WS relay for this frame).
//
//   close_actv_peer(handle)
//       Tear down the peer connection.
//
// signal_send_fn is the callback the module uses to push outbound
// signaling messages (offer/answer/ICE) up to the WS reader so they
// reach the dist-server, which relays them to the peer.  on_open_fn
// fires once both ends have established the data channel.  Inbound
// ACTV frames are dispatched via on_actv_frame_fn.
//
// The handle is an opaque pointer; the caller owns nothing — the
// module manages all lifetimes.

#include <cstddef>
#include <functional>
#include <string>
#include <memory>
#include <vector>

namespace dist {

struct ActvPeer; // opaque (full def lives in src/actv_p2p.cpp)

// Custom deleter — needed because the default unique_ptr deleter would
// require the full ActvPeer type at every call site that instantiates
// the destructor, defeating the opaque/PIMPL contract.
struct ActvPeerDeleter {
    void operator()(ActvPeer* p) const noexcept;
};
using ActvPeerPtr = std::unique_ptr<ActvPeer, ActvPeerDeleter>;

// SignalSendFn ships a fully-formed JSON text frame on the dist-node→server
// WebSocket.  actv_p2p assembles the entire payload (including "kind", "to",
// "session_id", and the SDP/ICE body) — the callback's only job is the
// transport.
using SignalSendFn   = std::function<void(const std::string& wire_json)>;
using ActvOnOpenFn   = std::function<void()>;
using ActvOnFrameFn  = std::function<void(const std::byte* data, std::size_t n)>;

struct IceServer {
    std::string url;        // e.g. "stun:stun.l.google.com:19302"
    std::string username;   // optional (TURN)
    std::string credential; // optional (TURN)
};

// Returns nullptr if the build was compiled without P2P support.
//
// peer_agent_id is informational (used in log messages); the remote end
// is identified at the server by the signaling routes set up by the
// planner, not by any field in this struct.
ActvPeerPtr open_actv_peer(const std::string& session_id,
                           const std::string& peer_agent_id,
                           bool is_offerer,
                           const std::vector<IceServer>& ice,
                           SignalSendFn   signal_send,
                           ActvOnOpenFn   on_open,
                           ActvOnFrameFn  on_frame);

// Feed an inbound signaling frame into the session.  kind must be one of
// "p2p_offer", "p2p_answer", "p2p_ice".  Safe to call even on a stub build —
// it'll be ignored.
void handle_signal(ActvPeer& peer, const std::string& kind,
                   const std::string& payload_json);

// Returns false when the data channel isn't open yet (or P2P is disabled
// at build time).  Callers fall back to the WS relay path on false.
bool send_actv(ActvPeer& peer, const std::byte* data, std::size_t n);

// Cheap test: is the channel actually open?  Used by the send-path to
// decide between P2P and relay without a full TX attempt.
bool is_open(const ActvPeer& peer);

void close_actv_peer(ActvPeerPtr peer);

// NAT-type probe result.  See probe_nat_type().
struct NatProbe {
    // One of: "open", "cone", "symmetric", "blocked", "unknown".
    std::string type;
    // True iff this rig is a useful relay candidate for other peers —
    // currently aliased to (type == "open" || type == "cone").
    bool relay_capable = false;
    // The first server-reflexive (or public host) IPv4 we saw during
    // gathering.  Empty when no srflx/public candidate was discovered.
    // Used by dist-node to auto-fill --external-ip for the bundled TURN
    // sidecar so peers behind 1:1 NAT (EC2, GCP) advertise correctly.
    std::string public_ip;
};

// Probe the local NAT environment by gathering ICE candidates against the
// supplied STUN servers.  Blocks for up to `timeout_ms` (or until gathering
// completes).  Cheap to call once at startup; do not call on a hot path.
// In stub builds (DIST_HAVE_P2P undefined) returns {"unknown", false}
// instantly.
NatProbe probe_nat_type(const std::vector<IceServer>& stun_servers,
                        int timeout_ms = 4000);

// Compile-time flag the rest of the codebase can branch on.
constexpr bool kP2PCompiled =
#ifdef DIST_HAVE_P2P
    true;
#else
    false;
#endif

} // namespace dist
