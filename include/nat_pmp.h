#pragma once
//
// P17: NAT-PMP (RFC 6886) and PCP (RFC 6887) port-mapping probe.
//
// Before falling back to a TURN relay for the ACTV data plane, ask
// the upstream router to map an external port directly.  When the
// router speaks PCP or NAT-PMP this gives us a server-reflexive
// candidate at roughly the cost of two UDP round-trips, with the
// router itself doing the forwarding (no relay byte-counting, no
// extra latency hop).
//
// Surface:
//   - default_gateway_v4()  — best-effort discovery of the LAN gateway
//   - try_map_udp(internal_port, suggested_ext_port, lifetime_s)
//                           — attempt a UDP port mapping; returns the
//                             external (ip:port, lifetime) on success
//   - PortMapper            — background renewer that keeps the
//                             mapping alive for as long as it lives
//
// The wire functions are scalar, sync, and use only POSIX sockets.
// On Linux the gateway is read from /proc/net/route.  Other platforms
// currently return an empty string and the mapper falls back to
// the user-supplied gateway (DIST_PORTMAP_GATEWAY env var).

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>

namespace dist {

struct MappedPort {
    std::string external_ip;   // IPv4 dotted (or v6 string from PCP)
    uint16_t    external_port;
    uint16_t    internal_port;
    uint32_t    lifetime_s;    // server-granted lifetime
    std::string method;        // "pcp" or "nat-pmp"
};

// Best-effort discovery of the v4 default gateway.  Empty string if
// the platform can't be inspected.
std::string default_gateway_v4();

// Attempt to map a UDP port.  Sends PCP MAP first (version 2); on
// "UNSUPP_VERSION" or no reply, retries with NAT-PMP (version 0).
// `gateway` is the router IP; if empty, we call default_gateway_v4().
// `timeout_ms` bounds the whole probe (PCP + NAT-PMP combined).
//
// On success returns the granted mapping.  On failure returns nullopt;
// callers should fall back to STUN / TURN.
std::optional<MappedPort> try_map_udp(uint16_t internal_port,
                                      uint16_t suggested_ext_port,
                                      uint32_t lifetime_s,
                                      const std::string& gateway = "",
                                      int timeout_ms = 1500);

// Background renewer.  Constructs the initial mapping and refreshes
// halfway through the granted lifetime.  Stops on destruction or
// when stop() is called.
class PortMapper {
public:
    PortMapper(uint16_t internal_port,
               uint16_t suggested_ext_port = 0,
               uint32_t lifetime_s = 3600,
               std::string gateway = "");
    ~PortMapper();

    // Latest known mapping, or nullopt if the probe has never
    // succeeded.  Cheap; safe to call from any thread.
    std::optional<MappedPort> current() const;

    // Whether the mapper has ever succeeded.
    bool has_mapping() const;

    // Stop the renewal thread (does not deactivate the mapping with
    // the router — that just expires naturally).
    void stop();

private:
    void renew_loop();

    uint16_t internal_;
    uint16_t suggested_;
    uint32_t lifetime_;
    std::string gateway_;
    mutable std::atomic<bool> running_;
    mutable std::atomic<bool> have_;
    mutable MappedPort latest_;
    std::thread th_;
    mutable std::atomic<uint32_t> latest_seq_{0};
};

} // namespace dist
