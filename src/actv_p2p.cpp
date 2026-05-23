// ACTV-over-WebRTC implementation.
//
// Two compile flavours share this file: a real one backed by libdatachannel
// (when DIST_HAVE_P2P is defined) and a thin stub that compiles to no-ops.
// The stub exists so the rest of the codebase can call into this module
// unconditionally and let the link decide whether to engage P2P.

#include "actv_p2p.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <iostream>

#ifdef DIST_HAVE_P2P
#  include <rtc/rtc.hpp>
#endif

namespace dist {

#ifdef DIST_HAVE_P2P
// ─── Tiny JSON helpers ─────────────────────────────────────────────────────
//
// We don't drag in a real JSON library here — the signaling messages are
// tightly scoped (kind, sdp, candidate, mid, index) so a couple of
// hand-rolled extractors are easier to audit than another dependency.
// Compiled only in the P2P build; the stub doesn't touch JSON.

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c & 0xFF);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
    return out;
}

// Pull a string-valued field out of a flat JSON object.  Returns "" if
// the field isn't present or isn't a string.
static std::string json_string_field(const std::string& js, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto p = js.find(needle);
    if (p == std::string::npos) return {};
    p = js.find(':', p);
    if (p == std::string::npos) return {};
    ++p;
    while (p < js.size() && (js[p] == ' ' || js[p] == '\t')) ++p;
    if (p >= js.size() || js[p] != '"') return {};
    ++p;
    std::string out;
    while (p < js.size()) {
        char c = js[p++];
        if (c == '\\' && p < js.size()) {
            char e = js[p++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                default:   out.push_back(e);    break;
            }
        } else if (c == '"') {
            return out;
        } else {
            out.push_back(c);
        }
    }
    return {};
}

#endif // DIST_HAVE_P2P (json helpers)

// ─── Stub mode (DIST_HAVE_P2P not defined) ─────────────────────────────────

#ifndef DIST_HAVE_P2P

struct ActvPeer {
    // Empty.  Existing only so the unique_ptr<ActvPeer> public API has a
    // concrete type to point at.
};

void ActvPeerDeleter::operator()(ActvPeer* p) const noexcept { delete p; }

ActvPeerPtr open_actv_peer(const std::string&, const std::string&, bool,
                           const std::vector<IceServer>&,
                           SignalSendFn, ActvOnOpenFn, ActvOnFrameFn) {
    return ActvPeerPtr{}; // signals "P2P unavailable" — caller uses relay
}

void handle_signal(ActvPeer&, const std::string&, const std::string&) {}
bool send_actv(ActvPeer&, const std::byte*, std::size_t)    { return false; }
bool is_open(const ActvPeer&)                                { return false; }
void close_actv_peer(ActvPeerPtr)                            {}

NatProbe probe_nat_type(const std::vector<IceServer>&, int) {
    NatProbe p; p.type = "unknown"; p.relay_capable = false; return p;
}

#else // DIST_HAVE_P2P

// ─── Real implementation (libdatachannel) ──────────────────────────────────

struct ActvPeer {
    std::string sessionID;
    std::string peerAgentID;
    bool        offerer = false;

    SignalSendFn  signalSend;
    ActvOnOpenFn  onOpen;
    ActvOnFrameFn onFrame;

    std::shared_ptr<rtc::PeerConnection> pc;
    std::shared_ptr<rtc::DataChannel>    dc;

    std::mutex  mu;
    bool        opened = false;
};

// Assemble a complete signaling text frame ready to ship over the dist-node WS.
// Shape: {"kind":"p2p_*","to":"<peer>","session_id":"...","<extra-fields>"}
static std::string build_signal_frame(const std::string& kind,
                                      const std::string& peerAgentID,
                                      const std::string& sessionID,
                                      const std::string& extraFields) {
    std::ostringstream os;
    os << "{"
       << "\"kind\":"       << json_escape(kind)        << ","
       << "\"to\":"         << json_escape(peerAgentID) << ","
       << "\"session_id\":" << json_escape(sessionID);
    if (!extraFields.empty()) {
        os << "," << extraFields;
    }
    os << "}";
    return os.str();
}

static void wire_data_channel(ActvPeer* self, std::shared_ptr<rtc::DataChannel> dc) {
    self->dc = dc;
    dc->onOpen([self]() {
        {
            std::lock_guard<std::mutex> lk(self->mu);
            self->opened = true;
        }
        if (self->onOpen) self->onOpen();
    });
    dc->onClosed([self]() {
        std::lock_guard<std::mutex> lk(self->mu);
        self->opened = false;
    });
    dc->onMessage([self](auto data) {
        // libdatachannel hands us either std::string or rtc::binary
        // (std::vector<std::byte>).  ACTV frames are binary; ignore text.
        if (auto* bin = std::get_if<rtc::binary>(&data)) {
            if (self->onFrame) {
                self->onFrame(bin->data(), bin->size());
            }
        }
    });
}

void ActvPeerDeleter::operator()(ActvPeer* p) const noexcept { delete p; }

ActvPeerPtr open_actv_peer(const std::string& session_id,
                           const std::string& peer_agent_id,
                           bool is_offerer,
                           const std::vector<IceServer>& ice,
                           SignalSendFn   signal_send,
                           ActvOnOpenFn   on_open,
                           ActvOnFrameFn  on_frame) {
    ActvPeerPtr self(new ActvPeer);
    self->sessionID   = session_id;
    self->peerAgentID = peer_agent_id;
    self->offerer     = is_offerer;
    self->signalSend  = std::move(signal_send);
    self->onOpen      = std::move(on_open);
    self->onFrame     = std::move(on_frame);

    rtc::Configuration cfg;
    for (const auto& s : ice) {
        try {
            // libdatachannel parses "stun:host:port" / "turn:host:port" URL
            // form directly.  TURN credentials are set as struct fields
            // after construction.
            rtc::IceServer parsed(s.url);
            if (!s.username.empty()) {
                parsed.username = s.username;
                parsed.password = s.credential;
            }
            cfg.iceServers.emplace_back(std::move(parsed));
        } catch (const std::exception& e) {
            std::cerr << "[actv_p2p] bad ICE server " << s.url << ": " << e.what() << "\n";
        }
    }

    self->pc = std::make_shared<rtc::PeerConnection>(cfg);

    ActvPeer* raw = self.get();
    self->pc->onLocalDescription([raw](rtc::Description desc) {
        std::string type = desc.typeString();
        std::string sdp  = std::string(desc);
        std::string kind = (type == "offer") ? "p2p_offer" : "p2p_answer";
        std::string extra =
            "\"sdp_type\":" + json_escape(type) + "," +
            "\"sdp\":"      + json_escape(sdp);
        if (raw->signalSend) {
            raw->signalSend(build_signal_frame(kind, raw->peerAgentID,
                                               raw->sessionID, extra));
        }
    });

    self->pc->onLocalCandidate([raw](rtc::Candidate cand) {
        std::string extra =
            "\"candidate\":" + json_escape(std::string(cand)) + "," +
            "\"mid\":"       + json_escape(cand.mid());
        if (raw->signalSend) {
            raw->signalSend(build_signal_frame("p2p_ice", raw->peerAgentID,
                                               raw->sessionID, extra));
        }
    });

    self->pc->onDataChannel([raw](std::shared_ptr<rtc::DataChannel> incoming) {
        wire_data_channel(raw, incoming);
    });

    if (is_offerer) {
        // Pre-negotiated channel; data-channel creation drives SDP offer.
        // Reliable+ordered (default) is what ACTV expects.
        auto dc = self->pc->createDataChannel("actv");
        wire_data_channel(raw, dc);
    }

    return self;
}

void handle_signal(ActvPeer& peer, const std::string& kind,
                   const std::string& payload_json) {
    if (!peer.pc) return;
    try {
        if (kind == "p2p_offer" || kind == "p2p_answer") {
            std::string sdp  = json_string_field(payload_json, "sdp");
            std::string type = json_string_field(payload_json, "sdp_type");
            if (sdp.empty() || type.empty()) {
                std::cerr << "[actv_p2p] " << kind
                          << " parse failed: sdp.empty=" << sdp.empty()
                          << " type.empty=" << type.empty() << "\n";
                return;
            }
            peer.pc->setRemoteDescription(rtc::Description(sdp, type));
            std::cout << "[actv_p2p] applied " << kind
                      << " (type=" << type << ", sdp_len=" << sdp.size() << ")\n";
        } else if (kind == "p2p_ice") {
            std::string cand = json_string_field(payload_json, "candidate");
            std::string mid  = json_string_field(payload_json, "mid");
            if (cand.empty()) return;
            peer.pc->addRemoteCandidate(rtc::Candidate(cand, mid));
        }
    } catch (const std::exception& e) {
        std::cerr << "[actv_p2p] signal " << kind << " failed: " << e.what() << "\n";
    }
}

bool send_actv(ActvPeer& peer, const std::byte* data, std::size_t n) {
    {
        std::lock_guard<std::mutex> lk(peer.mu);
        if (!peer.opened || !peer.dc) return false;
    }
    try {
        // rtc::binary is a vector<std::byte>; we copy because the channel
        // owns the queue once we hand the buffer over.
        rtc::binary buf(data, data + n);
        return peer.dc->send(std::move(buf));
    } catch (const std::exception& e) {
        std::cerr << "[actv_p2p] send_actv failed: " << e.what() << "\n";
        return false;
    }
}

bool is_open(const ActvPeer& peer) {
    // Cast away const just for the lock; the read itself is benign.
    auto& mu = const_cast<std::mutex&>(peer.mu);
    std::lock_guard<std::mutex> lk(mu);
    return peer.opened;
}

void close_actv_peer(ActvPeerPtr peer) {
    if (!peer) return;
    try {
        if (peer->dc) peer->dc->close();
        if (peer->pc) peer->pc->close();
    } catch (...) {}
}

// ─── NAT-type probe ────────────────────────────────────────────────────────
//
// We classify the local network by spinning a throwaway PeerConnection with
// the supplied STUN servers, letting it gather candidates for up to
// `timeout_ms`, and inspecting what came back:
//
//   * any non-private (public) host candidate            -> "open"
//   * any srflx (server-reflexive) candidate             -> "cone"
//   * only host candidates inside RFC1918 + no srflx     -> "symmetric"
//                                                          (best guess —
//                                                           we can't fully
//                                                           differentiate
//                                                           symmetric vs
//                                                           blocked here)
//   * nothing useful gathered before the timeout         -> "blocked"
//
// "relay_capable" mirrors (type == "open" || type == "cone").  This is
// advisory — the planner uses it as a soft hint for picking peer relays.
NatProbe probe_nat_type(const std::vector<IceServer>& stun_servers,
                        int timeout_ms) {
    NatProbe out;
    out.type = "unknown";
    out.relay_capable = false;

    rtc::Configuration cfg;
    for (const auto& s : stun_servers) {
        if (s.url.empty()) continue;
        // Only feed real STUN/TURN URLs into the probe; "peer:..." entries
        // are sentinels for the routing layer and would break IceServer.
        if (s.url.rfind("stun:", 0) != 0 && s.url.rfind("turn:", 0) != 0 &&
            s.url.rfind("turns:", 0) != 0) continue;
        try {
            cfg.iceServers.emplace_back(s.url);
        } catch (...) { /* ignore — bad URL */ }
    }
    if (cfg.iceServers.empty()) {
        cfg.iceServers.emplace_back("stun:stun.l.google.com:19302");
    }

    std::mutex      mu;
    std::condition_variable cv;
    bool            done = false;
    bool            got_public_host = false;
    bool            got_srflx       = false;
    bool            got_private_host = false;

    auto is_private_ipv4 = [](const std::string& ip) -> bool {
        // crude RFC1918 / link-local / loopback / CGNAT check
        if (ip.rfind("10.", 0) == 0) return true;
        if (ip.rfind("127.", 0) == 0) return true;
        if (ip.rfind("192.168.", 0) == 0) return true;
        if (ip.rfind("169.254.", 0) == 0) return true;
        if (ip.rfind("100.", 0) == 0) {
            // 100.64.0.0/10 CGNAT
            int sec = 0;
            size_t p = 4;
            while (p < ip.size() && ip[p] >= '0' && ip[p] <= '9') {
                sec = sec * 10 + (ip[p] - '0');
                ++p;
            }
            if (sec >= 64 && sec <= 127) return true;
        }
        if (ip.rfind("172.", 0) == 0) {
            int sec = 0;
            size_t p = 4;
            while (p < ip.size() && ip[p] >= '0' && ip[p] <= '9') {
                sec = sec * 10 + (ip[p] - '0');
                ++p;
            }
            if (sec >= 16 && sec <= 31) return true;
        }
        return false;
    };

    try {
        auto pc = std::make_shared<rtc::PeerConnection>(cfg);

        pc->onLocalCandidate([&](rtc::Candidate cand) {
            std::string s = std::string(cand);
            // candidate strings look like:
            //   "candidate:1 1 UDP 2130706431 192.168.1.10 51544 typ host"
            //   "candidate:2 1 UDP 1694498815 203.0.113.5 51544 typ srflx ..."
            auto p = s.find(" typ ");
            if (p == std::string::npos) return;
            std::string typ = s.substr(p + 5);
            auto sp = typ.find(' ');
            if (sp != std::string::npos) typ = typ.substr(0, sp);

            // Extract the IP — the address sits two tokens before "typ".
            std::string addr;
            {
                // Walk back from p, find the address token.  Tokens are
                // space-separated; address is 4 tokens to the left of typ.
                size_t end = p;
                int back = 0;
                while (end > 0 && back < 4) {
                    if (s[end - 1] == ' ') ++back;
                    if (back >= 4) break;
                    --end;
                }
                size_t start = end;
                while (start > 0 && s[start - 1] != ' ') --start;
                addr = s.substr(start, end - start);
            }

            if (typ == "host") {
                if (is_private_ipv4(addr)) {
                    got_private_host = true;
                } else {
                    got_public_host  = true;
                    // First public host wins; otherwise srflx fills it.
                    std::lock_guard<std::mutex> lk(mu);
                    if (out.public_ip.empty()) out.public_ip = addr;
                }
            } else if (typ == "srflx" || typ == "prflx") {
                got_srflx = true;
                std::lock_guard<std::mutex> lk(mu);
                if (out.public_ip.empty()) out.public_ip = addr;
            }
        });

        pc->onGatheringStateChange([&](rtc::PeerConnection::GatheringState gs) {
            if (gs == rtc::PeerConnection::GatheringState::Complete) {
                std::lock_guard<std::mutex> lk(mu);
                done = true;
                cv.notify_all();
            }
        });

        // Need at least one data channel to actually gather.
        auto dc = pc->createDataChannel("probe");
        (void) dc;
        pc->setLocalDescription();

        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                    [&]{ return done; });

        try { pc->close(); } catch (...) {}
    } catch (const std::exception& e) {
        std::cerr << "[actv_p2p] nat probe failed: " << e.what() << "\n";
        return out;
    }

    if (got_public_host)       out.type = "open";
    else if (got_srflx)        out.type = "cone";
    else if (got_private_host) out.type = "symmetric";
    else                       out.type = "blocked";

    out.relay_capable = (out.type == "open" || out.type == "cone");
    return out;
}

#endif // DIST_HAVE_P2P

} // namespace dist
