// nat_pmp.cpp — NAT-PMP (RFC 6886) and PCP (RFC 6887) implementation.
//
// Both protocols target UDP/5351 on the gateway.  PCP is a strict
// superset of NAT-PMP semantically but uses a different wire format,
// so we try PCP first, and on UNSUPP_VERSION (1) or silence we
// retry with NAT-PMP.

#include "nat_pmp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socklen_t_local = int;
  #define DIST_CLOSESOCK closesocket
#else
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/types.h>
  #include <unistd.h>
  using socklen_t_local = socklen_t;
  #define DIST_CLOSESOCK ::close
#endif

namespace dist {

namespace {

constexpr uint16_t PMP_PCP_PORT = 5351;

// ─── platform: default gateway ────────────────────────────────────────

#if defined(__linux__)
std::string read_proc_default_gateway_v4() {
    // /proc/net/route: tab-separated table, gateway hex in column 2,
    // flags in column 3.  Default route has destination 00000000.
    std::ifstream f("/proc/net/route");
    if (!f) return {};
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string iface, dest, gw, flags_s;
        if (!(is >> iface >> dest >> gw >> flags_s)) continue;
        if (dest != "00000000") continue;
        // gw is hex little-endian: e.g. "0100A8C0" → 192.168.0.1
        if (gw.size() != 8) continue;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned int byte = 0;
            std::sscanf(gw.c_str() + i*2, "%02x", &byte);
            v |= (byte << (i*8));
        }
        char buf[INET_ADDRSTRLEN];
        in_addr a; a.s_addr = v; // already in network byte order
        if (!inet_ntop(AF_INET, &a, buf, sizeof(buf))) return {};
        return std::string(buf);
    }
    return {};
}
#endif

std::string env_override_gateway() {
    const char* g = std::getenv("DIST_PORTMAP_GATEWAY");
    return (g && *g) ? std::string(g) : std::string();
}

// ─── sockets ──────────────────────────────────────────────────────────

struct ScopedSocket {
    int fd = -1;
    ~ScopedSocket() { if (fd >= 0) DIST_CLOSESOCK(fd); }
};

bool set_recv_timeout(int fd, int ms) {
#if defined(_WIN32)
    DWORD t = (DWORD)ms;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                      (const char*)&t, sizeof(t)) == 0;
#else
    timeval tv{};
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

bool send_to_gateway(int fd, const std::string& gateway, uint16_t port,
                     const uint8_t* buf, size_t n) {
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(port);
    if (inet_pton(AF_INET, gateway.c_str(), &sa.sin_addr) != 1) return false;
    ssize_t k = sendto(fd, (const char*)buf, (int)n, 0,
                       (sockaddr*)&sa, sizeof(sa));
    return k == (ssize_t)n;
}

ssize_t recv_from(int fd, uint8_t* buf, size_t cap) {
    sockaddr_in sa{};
    socklen_t_local sl = sizeof(sa);
    return recvfrom(fd, (char*)buf, (int)cap, 0, (sockaddr*)&sa, &sl);
}

// ─── PCP MAP (RFC 6887) ───────────────────────────────────────────────
//
// Common header (24 bytes):
//   version (1)   = 2
//   R+opcode (1)  = 1 (MAP request, R=0)
//   reserved (2)  = 0
//   lifetime (4)  = seconds
//   client IP (16): IPv4-mapped IPv6
//
// MAP opcode body (36 bytes):
//   nonce (12)
//   protocol (1)  = 17 (UDP)
//   reserved (3)
//   internal port (2)
//   suggested external port (2)
//   suggested external IP (16)
//
// Response: same header (opcode high-bit set), then MAP body.

std::optional<MappedPort> try_pcp_map(int fd, const std::string& gateway,
                                      uint16_t internal_port,
                                      uint16_t suggested_ext_port,
                                      uint32_t lifetime_s) {
    uint8_t pkt[60] = {0};
    pkt[0] = 2;                                  // version
    pkt[1] = 1;                                  // opcode = MAP, R=0
    pkt[2] = pkt[3] = 0;                         // reserved
    pkt[4] = (uint8_t)(lifetime_s >> 24);
    pkt[5] = (uint8_t)(lifetime_s >> 16);
    pkt[6] = (uint8_t)(lifetime_s >> 8);
    pkt[7] = (uint8_t)(lifetime_s);
    // Client IP: IPv4-mapped IPv6.  Best-effort — bind a UDP socket
    // and read its local v4 to populate; otherwise leave ::ffff:0.0.0.0
    // which most routers accept anyway.
    pkt[18] = 0xff; pkt[19] = 0xff;              // ::ffff:0.0.0.0

    // MAP body at offset 24
    std::mt19937 rng((uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());
    for (int i = 24; i < 24 + 12; ++i) pkt[i] = (uint8_t)(rng() & 0xff);
    pkt[36] = 17;                                // UDP
    // 37..39 reserved
    pkt[40] = (uint8_t)(internal_port >> 8);
    pkt[41] = (uint8_t)(internal_port);
    pkt[42] = (uint8_t)(suggested_ext_port >> 8);
    pkt[43] = (uint8_t)(suggested_ext_port);
    // 44..59 suggested external IP — leave zero, ask router to pick.

    if (!send_to_gateway(fd, gateway, PMP_PCP_PORT, pkt, sizeof(pkt))) {
        return std::nullopt;
    }

    uint8_t resp[1100];
    ssize_t k = recv_from(fd, resp, sizeof(resp));
    if (k < 60) return std::nullopt;
    if (resp[0] != 2) return std::nullopt;       // not PCP
    if ((resp[1] & 0x7F) != 1) return std::nullopt; // not MAP opcode echo
    uint8_t result = resp[3];
    if (result != 0) {                           // 0 = SUCCESS
        return std::nullopt;
    }
    uint32_t granted = ((uint32_t)resp[8] << 24) | ((uint32_t)resp[9] << 16) |
                       ((uint32_t)resp[10] << 8) |  (uint32_t)resp[11];

    // MAP body at offset 24 in response too.
    uint16_t ext_port = ((uint16_t)resp[42] << 8) | resp[43];
    // External IP at resp[44..59].  If it's a v4-mapped v6, render dotted.
    bool v4mapped = true;
    for (int i = 0; i < 10; ++i) if (resp[44+i] != 0) { v4mapped = false; break; }
    if (resp[54] != 0xff || resp[55] != 0xff) v4mapped = false;
    std::string ext_ip;
    if (v4mapped) {
        char buf[INET_ADDRSTRLEN];
        in_addr a;
        std::memcpy(&a.s_addr, &resp[56], 4);
        if (inet_ntop(AF_INET, &a, buf, sizeof(buf))) ext_ip = buf;
    } else {
        char buf[INET6_ADDRSTRLEN];
        in6_addr a6;
        std::memcpy(&a6, &resp[44], 16);
        if (inet_ntop(AF_INET6, &a6, buf, sizeof(buf))) ext_ip = buf;
    }

    MappedPort m;
    m.external_ip   = ext_ip;
    m.external_port = ext_port;
    m.internal_port = internal_port;
    m.lifetime_s    = granted;
    m.method        = "pcp";
    return m;
}

// ─── NAT-PMP MAP (RFC 6886) ───────────────────────────────────────────
//
// MAP request (12 bytes):
//   version (1) = 0
//   opcode (1)  = 1 (UDP) / 2 (TCP)
//   reserved (2)
//   internal port (2)
//   suggested external port (2)
//   lifetime (4)
//
// Response (16 bytes):
//   version (1) = 0
//   opcode (1)  = 128 + req_opcode
//   result (2)
//   secs since epoch (4)
//   internal port (2)
//   external port (2)
//   lifetime (4)

std::optional<MappedPort> try_natpmp_map(int fd, const std::string& gateway,
                                         uint16_t internal_port,
                                         uint16_t suggested_ext_port,
                                         uint32_t lifetime_s) {
    uint8_t pkt[12] = {0};
    pkt[0] = 0;                                  // version
    pkt[1] = 1;                                  // UDP map
    pkt[4] = (uint8_t)(internal_port >> 8);
    pkt[5] = (uint8_t)(internal_port);
    pkt[6] = (uint8_t)(suggested_ext_port >> 8);
    pkt[7] = (uint8_t)(suggested_ext_port);
    pkt[8]  = (uint8_t)(lifetime_s >> 24);
    pkt[9]  = (uint8_t)(lifetime_s >> 16);
    pkt[10] = (uint8_t)(lifetime_s >> 8);
    pkt[11] = (uint8_t)(lifetime_s);

    if (!send_to_gateway(fd, gateway, PMP_PCP_PORT, pkt, sizeof(pkt))) {
        return std::nullopt;
    }

    uint8_t resp[64];
    ssize_t k = recv_from(fd, resp, sizeof(resp));
    if (k < 16) return std::nullopt;
    if (resp[0] != 0) return std::nullopt;
    if (resp[1] != (128 + 1)) return std::nullopt;
    uint16_t result = ((uint16_t)resp[2] << 8) | resp[3];
    if (result != 0) return std::nullopt;
    uint16_t int_port = ((uint16_t)resp[8]  << 8) | resp[9];
    uint16_t ext_port = ((uint16_t)resp[10] << 8) | resp[11];
    uint32_t granted  = ((uint32_t)resp[12] << 24) | ((uint32_t)resp[13] << 16) |
                        ((uint32_t)resp[14] << 8)  |  (uint32_t)resp[15];
    if (int_port != internal_port) {
        // Some routers echo the suggested external in this field;
        // keep going regardless.
    }
    MappedPort m;
    // NAT-PMP doesn't return the external IP in the MAP response —
    // a separate opcode (0) does.  Fire it to learn the address.
    {
        uint8_t addr_req[2] = {0, 0};
        uint8_t addr_resp[16];
        if (send_to_gateway(fd, gateway, PMP_PCP_PORT, addr_req, sizeof(addr_req))) {
            ssize_t kk = recv_from(fd, addr_resp, sizeof(addr_resp));
            if (kk >= 12 && addr_resp[0] == 0 && addr_resp[1] == 128) {
                char buf[INET_ADDRSTRLEN];
                in_addr a;
                std::memcpy(&a.s_addr, &addr_resp[8], 4);
                if (inet_ntop(AF_INET, &a, buf, sizeof(buf))) m.external_ip = buf;
            }
        }
    }
    m.external_port = ext_port;
    m.internal_port = internal_port;
    m.lifetime_s    = granted;
    m.method        = "nat-pmp";
    return m;
}

} // namespace

// ─── public surface ───────────────────────────────────────────────────

std::string default_gateway_v4() {
    auto e = env_override_gateway();
    if (!e.empty()) return e;
#if defined(__linux__)
    return read_proc_default_gateway_v4();
#else
    return std::string();
#endif
}

std::optional<MappedPort> try_map_udp(uint16_t internal_port,
                                      uint16_t suggested_ext_port,
                                      uint32_t lifetime_s,
                                      const std::string& gateway_in,
                                      int timeout_ms) {
    std::string gateway = gateway_in.empty() ? default_gateway_v4() : gateway_in;
    if (gateway.empty()) return std::nullopt;

    int fd = (int)::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return std::nullopt;
    ScopedSocket _g{fd};
    // Split the budget so both protocols get a fair shake.
    set_recv_timeout(fd, std::max(200, timeout_ms / 2));

    if (auto m = try_pcp_map(fd, gateway, internal_port,
                             suggested_ext_port, lifetime_s)) {
        return m;
    }
    // Re-arm for the NAT-PMP retry.
    set_recv_timeout(fd, std::max(200, timeout_ms / 2));
    return try_natpmp_map(fd, gateway, internal_port,
                          suggested_ext_port, lifetime_s);
}

// ─── PortMapper ───────────────────────────────────────────────────────

PortMapper::PortMapper(uint16_t internal_port, uint16_t suggested_ext_port,
                       uint32_t lifetime_s, std::string gateway)
    : internal_(internal_port),
      suggested_(suggested_ext_port),
      lifetime_(lifetime_s ? lifetime_s : 3600),
      gateway_(std::move(gateway)),
      running_(true),
      have_(false) {
    th_ = std::thread([this]{ renew_loop(); });
}

PortMapper::~PortMapper() { stop(); }

void PortMapper::stop() {
    if (running_.exchange(false)) {
        if (th_.joinable()) th_.join();
    }
}

bool PortMapper::has_mapping() const { return have_.load(); }

std::optional<MappedPort> PortMapper::current() const {
    if (!have_.load()) return std::nullopt;
    // Read with a sequence guard so we never tear a write.
    for (int i = 0; i < 4; ++i) {
        uint32_t s1 = latest_seq_.load(std::memory_order_acquire);
        if (s1 & 1u) continue;
        MappedPort copy = latest_;
        uint32_t s2 = latest_seq_.load(std::memory_order_acquire);
        if (s1 == s2) return copy;
    }
    return std::nullopt;
}

void PortMapper::renew_loop() {
    while (running_.load()) {
        auto m = try_map_udp(internal_, suggested_, lifetime_, gateway_);
        if (m) {
            latest_seq_.fetch_add(1, std::memory_order_release);
            latest_ = *m;
            latest_seq_.fetch_add(1, std::memory_order_release);
            have_.store(true);
            uint32_t granted = m->lifetime_s ? m->lifetime_s : lifetime_;
            // Refresh halfway through the granted lifetime, with a
            // sensible floor.
            uint32_t sleep_s = std::max<uint32_t>(30, granted / 2);
            for (uint32_t i = 0; i < sleep_s && running_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        } else {
            // Backoff after a failed probe; retry every 30 s.
            for (uint32_t i = 0; i < 30 && running_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }
}

} // namespace dist
