// ACTV-over-WebRTC implementation.
//
// Two compile flavours share this file: a real one backed by libdatachannel
// (when DIST_HAVE_P2P is defined) and a thin stub that compiles to no-ops.
// The stub exists so the rest of the codebase can call into this module
// unconditionally and let the link decide whether to engage P2P.

#include "actv_p2p.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <iostream>

#ifdef __unix__
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define DIST_HAVE_BSD_SOCKETS 1
#endif

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

// ─── UDP reachability test ────────────────────────────────────────────────
//
// Ground-truth check for whether a peer can actually deliver packets to our
// public mapping.  STUN-based classification is a heuristic — a port-
// preserving symmetric NAT will look like cone when you compare reflexive
// ports across STUN servers, but inbound packets from a fresh source still
// get dropped.
//
// We bind socket A, STUN-discover its external (ip:port), then send a probe
// from socket B (a fresh source port) to that external endpoint and see if
// A receives it.  Returns true iff at least one packet round-trips.
#ifdef DIST_HAVE_BSD_SOCKETS
namespace {

constexpr uint32_t kStunMagicCookie = 0x2112A442u;

bool stun_query(int sock, const sockaddr_in& server,
                std::string& out_ip, uint16_t& out_port) {
    uint8_t req[20];
    std::memset(req, 0, sizeof(req));
    req[0] = 0x00; req[1] = 0x01;
    req[2] = 0x00; req[3] = 0x00;
    uint32_t magic_be = htonl(kStunMagicCookie);
    std::memcpy(req + 4, &magic_be, 4);
    std::mt19937 rng{std::random_device{}()};
    for (int i = 8; i < 20; ++i) req[i] = static_cast<uint8_t>(rng() & 0xff);

    if (::sendto(sock, req, sizeof(req), 0,
                 reinterpret_cast<const sockaddr*>(&server),
                 sizeof(server)) < 0) return false;

    pollfd pfd{sock, POLLIN, 0};
    if (::poll(&pfd, 1, 2500) <= 0) return false;

    uint8_t buf[1024];
    sockaddr_in from{};
    socklen_t fl = sizeof(from);
    ssize_t n = ::recvfrom(sock, buf, sizeof(buf), 0,
                           reinterpret_cast<sockaddr*>(&from), &fl);
    if (n < 20 || buf[0] != 0x01 || buf[1] != 0x01) return false;

    size_t off = 20;
    while (off + 4 <= static_cast<size_t>(n)) {
        uint16_t type = (uint16_t(buf[off]) << 8) | buf[off + 1];
        uint16_t alen = (uint16_t(buf[off + 2]) << 8) | buf[off + 3];
        off += 4;
        if (off + alen > static_cast<size_t>(n)) break;
        if ((type == 0x0020 || type == 0x0001) && alen >= 8 && buf[off + 1] == 0x01) {
            uint16_t p = (uint16_t(buf[off + 2]) << 8) | buf[off + 3];
            uint32_t a;
            std::memcpy(&a, buf + off + 4, 4);
            if (type == 0x0020) {
                p ^= static_cast<uint16_t>(kStunMagicCookie >> 16);
                a ^= htonl(kStunMagicCookie);
            }
            char ip[INET_ADDRSTRLEN];
            ::inet_ntop(AF_INET, &a, ip, sizeof(ip));
            out_ip = ip;
            out_port = p;
            return true;
        }
        off += (alen + 3) & ~3u;
    }
    return false;
}

bool resolve_udp_v4(const std::string& host, uint16_t port, sockaddr_in& out) {
    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    char pb[8];
    std::snprintf(pb, sizeof(pb), "%u", port);
    if (::getaddrinfo(host.c_str(), pb, &hints, &res) != 0 || !res) return false;
    std::memcpy(&out, res->ai_addr, sizeof(sockaddr_in));
    ::freeaddrinfo(res);
    return true;
}

// Returns true iff a packet sent from a fresh source port to our STUN-
// discovered external mapping is delivered inbound.  On false, the NAT is
// effectively symmetric / inbound-blocked from this rig's perspective — it
// cannot be a TURN relay regardless of what STUN candidate geometry says.
//
// Also writes the discovered external (ip:port) into `ext_ip` so the caller
// can use it as a fallback for public_ip.
bool udp_reachability_check(std::string& ext_ip, uint16_t& ext_port) {
    ext_ip.clear();
    ext_port = 0;

    int a = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (a < 0) return false;
    int b = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (b < 0) { ::close(a); return false; }

    sockaddr_in any{};
    any.sin_family = AF_INET;
    any.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(a, reinterpret_cast<sockaddr*>(&any), sizeof(any)) < 0 ||
        ::bind(b, reinterpret_cast<sockaddr*>(&any), sizeof(any)) < 0) {
        ::close(a); ::close(b); return false;
    }

    sockaddr_in stun{};
    if (!resolve_udp_v4("stun.l.google.com", 19302, stun)) {
        ::close(a); ::close(b); return false;
    }
    if (!stun_query(a, stun, ext_ip, ext_port)) {
        ::close(a); ::close(b); return false;
    }

    // Fire a probe from socket B at our own external mapping.
    sockaddr_in tgt{};
    tgt.sin_family = AF_INET;
    tgt.sin_port = htons(ext_port);
    ::inet_pton(AF_INET, ext_ip.c_str(), &tgt.sin_addr);

    const char* payload = "dist-reachability";
    bool reached = false;
    // Send a few times in case of single-packet loss.
    for (int i = 0; i < 3 && !reached; ++i) {
        ::sendto(b, payload, std::strlen(payload), 0,
                 reinterpret_cast<sockaddr*>(&tgt), sizeof(tgt));
        pollfd pfd{a, POLLIN, 0};
        if (::poll(&pfd, 1, 600) > 0) {
            uint8_t buf[256];
            sockaddr_in src{};
            socklen_t sl = sizeof(src);
            ssize_t n = ::recvfrom(a, buf, sizeof(buf), 0,
                                   reinterpret_cast<sockaddr*>(&src), &sl);
            if (n > 0) reached = true;
        }
    }

    ::close(a);
    ::close(b);
    return reached;
}

} // namespace
#endif // DIST_HAVE_BSD_SOCKETS

// ─── NAT-type probe ────────────────────────────────────────────────────────
//
// We classify the local network by spinning a throwaway PeerConnection with
// the supplied STUN servers, letting it gather candidates for up to
// `timeout_ms`, and inspecting what came back:
//
//   * any non-private (public) host candidate            -> "open"
//   * srflx with SAME (ip:port) across ≥2 STUN servers   -> "cone"
//   * srflx with DIFFERENT ports across STUN servers     -> "symmetric"
//                                                          (port mapping is
//                                                           destination-
//                                                           dependent — no
//                                                           direct hole-punch)
//   * srflx from a single STUN only                      -> "cone" (best
//                                                          guess; cannot
//                                                          distinguish
//                                                          without a second
//                                                          STUN reachable)
//   * only host candidates inside RFC1918 + no srflx     -> "symmetric"
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
    int stun_count = 0;
    for (const auto& s : stun_servers) {
        if (s.url.empty()) continue;
        // Only feed real STUN/TURN URLs into the probe; "peer:..." entries
        // are sentinels for the routing layer and would break IceServer.
        if (s.url.rfind("stun:", 0) != 0 && s.url.rfind("turn:", 0) != 0 &&
            s.url.rfind("turns:", 0) != 0) continue;
        try {
            cfg.iceServers.emplace_back(s.url);
            ++stun_count;
        } catch (...) { /* ignore — bad URL */ }
    }
    // Symmetric-NAT discrimination requires ≥2 STUN destinations on different
    // hosts.  Always append a second well-known STUN so we can compare
    // reflexive port mappings.
    if (stun_count == 0) {
        cfg.iceServers.emplace_back("stun:stun.l.google.com:19302");
        cfg.iceServers.emplace_back("stun:stun.cloudflare.com:3478");
        stun_count = 2;
    } else if (stun_count == 1) {
        try {
            cfg.iceServers.emplace_back("stun:stun.cloudflare.com:3478");
            ++stun_count;
        } catch (...) {}
    }

    std::mutex      mu;
    std::condition_variable cv;
    bool            done = false;
    bool            got_public_host = false;
    bool            got_srflx       = false;
    bool            got_private_host = false;
    // Unique reflexive mappings.  For symmetric NAT detection we look at
    // whether two different STUN servers see the same external port; if
    // they don't, the NAT is symmetric (port-restricted per destination).
    std::set<std::string> srflx_ip_port;
    std::set<std::string> srflx_ips;
    std::set<int>         srflx_ports;

    auto is_private_or_local = [](const std::string& ip) -> bool {
        if (ip.empty()) return true;
        // IPv4 RFC1918 / link-local / loopback / CGNAT.
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
        // IPv6: a global unicast address on a NIC does NOT prove inbound
        // reachability — campus / corp firewalls routinely hand out 2000::/3
        // addresses but block inbound v6.  We treat ALL IPv6 host candidates
        // as non-evidence of "open" mode; only an srflx via STUN can promote
        // us to "cone".  Callers can still see the v6 address via srflx if
        // STUN can echo it back.
        if (ip.find(':') != std::string::npos) return true;
        return false;
    };

    try {
        auto pc = std::make_shared<rtc::PeerConnection>(cfg);

        pc->onLocalCandidate([&](rtc::Candidate cand) {
            std::string s = std::string(cand);
            // SDP candidate grammar (RFC 8445 §5.1):
            //   candidate:<foundation> <component-id> <transport> <priority>
            //             <connection-address> <port> typ <cand-type> ...
            // Tokenise on spaces — addr is tokens[4], type is the token
            // immediately after "typ".
            std::vector<std::string> tok;
            {
                std::string cur;
                for (char c : s) {
                    if (c == ' ') {
                        if (!cur.empty()) tok.push_back(std::move(cur));
                        cur.clear();
                    } else {
                        cur += c;
                    }
                }
                if (!cur.empty()) tok.push_back(std::move(cur));
            }
            if (tok.size() < 8) return;
            const std::string& addr = tok[4];
            std::string typ;
            for (size_t i = 5; i + 1 < tok.size(); ++i) {
                if (tok[i] == "typ") { typ = tok[i + 1]; break; }
            }
            if (typ.empty()) return;

            if (typ == "host") {
                if (is_private_or_local(addr)) {
                    got_private_host = true;
                } else {
                    got_public_host  = true;
                    // First public host wins; otherwise srflx fills it.
                    std::lock_guard<std::mutex> lk(mu);
                    if (out.public_ip.empty()) out.public_ip = addr;
                }
            } else if (typ == "srflx" || typ == "prflx") {
                got_srflx = true;
                // Port is tok[5] per RFC 8445 grammar.
                int port = 0;
                if (tok.size() > 5) {
                    for (char c : tok[5]) {
                        if (c < '0' || c > '9') { port = 0; break; }
                        port = port * 10 + (c - '0');
                    }
                }
                std::lock_guard<std::mutex> lk(mu);
                if (out.public_ip.empty()) out.public_ip = addr;
                srflx_ips.insert(addr);
                if (port > 0) {
                    srflx_ports.insert(port);
                    srflx_ip_port.insert(addr + ":" + std::to_string(port));
                }
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

    if (got_public_host) {
        out.type = "open";
    } else if (got_srflx) {
        // If we probed ≥2 STUN servers and saw ≥2 distinct external
        // (ip:port) mappings, the NAT is allocating a fresh port per
        // destination — symmetric, no direct hole-punch.  A single
        // unique mapping across multiple STUN servers means cone/EIM.
        bool multi_probe = stun_count >= 2;
        bool divergent   = srflx_ip_port.size() >= 2 ||
                           srflx_ports.size()   >= 2;
        if (multi_probe && divergent) {
            out.type = "symmetric";
        } else {
            out.type = "cone";
        }
    } else if (got_private_host) {
        out.type = "symmetric";
    } else {
        out.type = "blocked";
    }

    // Symmetric NAT needs a TURN relay; direct WebRTC hole-punching to
    // peers behind other NATs will not work.
    out.relay_capable = (out.type == "open" || out.type == "cone");

#ifdef DIST_HAVE_BSD_SOCKETS
    // Ground-truth override: a port-preserving symmetric NAT can fool the
    // STUN-only classifier into reporting "cone" (both STUN destinations see
    // the same reflexive port, so the divergence heuristic doesn't fire).
    // The reachability check sends a packet from a fresh source port to our
    // discovered external mapping; if it never lands, the NAT is filtering
    // inbound from anyone other than the STUN server that opened the hole,
    // and we cannot serve as a relay regardless of candidate geometry.
    //
    // Only run when the heuristic claimed relay-capability, since a "blocked"
    // or already-symmetric verdict needs no further demotion.
    if (out.relay_capable && out.type != "open") {
        std::string ext_ip;
        uint16_t    ext_port = 0;
        bool reached = udp_reachability_check(ext_ip, ext_port);
        if (!ext_ip.empty() && out.public_ip.empty()) out.public_ip = ext_ip;
        if (!reached) {
            out.type          = "symmetric";
            out.relay_capable = false;
        }
    }
#endif

    return out;
}

#endif // DIST_HAVE_P2P

} // namespace dist
