// nat-pingpong — standalone UDP reachability tester.
//
// Two modes:
//   nat-pingpong listen [--stun host:port] [--port N]
//     Binds a UDP socket, sends a STUN binding request to discover the
//     external (srflx) mapping, prints it, then echoes every received
//     datagram back to the sender for up to 10 minutes.
//
//   nat-pingpong probe <ip:port> [--count N] [--timeout-ms MS]
//     Sends N "ping" datagrams to <ip:port> and waits for echoes.  Prints
//     per-packet RTT and an aggregate loss summary.  Run this from a
//     different network than the listener (e.g. rtxserver) — same-host
//     pings round-trip through the NAT's hairpin and don't prove inbound
//     reachability.
//
// Purpose: confirm whether a node behind NAT is actually reachable from a
// fresh remote sender.  A node behind a port-randomising (symmetric) NAT
// will print one external port via STUN, but the mapping closes / rotates
// before the remote prober's packet arrives — so the probe sees 100% loss
// while the listener sits silent.  A cone / EIM NAT will accept the probe
// and we see successful echoes.

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t STUN_MAGIC_COOKIE = 0x2112A442u;

void usage() {
    std::fprintf(stderr,
        "nat-pingpong — UDP reachability tester\n"
        "\n"
        "  nat-pingpong listen [--stun host:port] [--port N]\n"
        "      Bind UDP, discover external mapping via STUN, echo datagrams.\n"
        "      Default STUN: stun.l.google.com:19302\n"
        "\n"
        "  nat-pingpong probe <ip:port> [--count N] [--timeout-ms MS]\n"
        "      Send N pings to ip:port and wait for echoes.\n"
        "      Default: --count 10  --timeout-ms 3000\n");
}

bool resolve_udp(const std::string& host, uint16_t port, sockaddr_in& out) {
    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    char portbuf[8];
    std::snprintf(portbuf, sizeof(portbuf), "%u", port);
    if (getaddrinfo(host.c_str(), portbuf, &hints, &res) != 0 || !res) return false;
    std::memcpy(&out, res->ai_addr, sizeof(sockaddr_in));
    freeaddrinfo(res);
    return true;
}

bool stun_binding(int sock, const sockaddr_in& server,
                  std::string& ext_ip, uint16_t& ext_port) {
    uint8_t req[20];
    std::memset(req, 0, sizeof(req));
    req[0] = 0x00; req[1] = 0x01;            // Binding Request
    req[2] = 0x00; req[3] = 0x00;            // length = 0
    uint32_t magic_be = htonl(STUN_MAGIC_COOKIE);
    std::memcpy(req + 4, &magic_be, 4);
    std::mt19937 rng{std::random_device{}()};
    for (int i = 8; i < 20; ++i) req[i] = static_cast<uint8_t>(rng() & 0xff);

    if (sendto(sock, req, sizeof(req), 0,
               reinterpret_cast<const sockaddr*>(&server), sizeof(server)) < 0) {
        std::perror("sendto(stun)");
        return false;
    }

    pollfd pfd{sock, POLLIN, 0};
    if (poll(&pfd, 1, 3000) <= 0) {
        std::fprintf(stderr, "stun: no response within 3s\n");
        return false;
    }

    uint8_t buf[1024];
    sockaddr_in from{};
    socklen_t fromlen = sizeof(from);
    ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                        reinterpret_cast<sockaddr*>(&from), &fromlen);
    if (n < 20) { std::fprintf(stderr, "stun: short reply (%zd)\n", n); return false; }
    if (!(buf[0] == 0x01 && buf[1] == 0x01)) {
        std::fprintf(stderr, "stun: not a binding success\n");
        return false;
    }
    uint16_t attrs_len = (uint16_t(buf[2]) << 8) | buf[3];
    if (20 + attrs_len > n) {
        std::fprintf(stderr, "stun: truncated attrs\n");
        return false;
    }

    size_t off = 20;
    while (off + 4 <= static_cast<size_t>(n)) {
        uint16_t type = (uint16_t(buf[off]) << 8) | buf[off + 1];
        uint16_t alen = (uint16_t(buf[off + 2]) << 8) | buf[off + 3];
        off += 4;
        if (off + alen > static_cast<size_t>(n)) break;
        // XOR-MAPPED-ADDRESS (0x0020) preferred, MAPPED-ADDRESS (0x0001) fallback
        if ((type == 0x0020 || type == 0x0001) && alen >= 8 && buf[off + 1] == 0x01) {
            uint16_t port_raw = (uint16_t(buf[off + 2]) << 8) | buf[off + 3];
            uint32_t addr_raw;
            std::memcpy(&addr_raw, buf + off + 4, 4);
            if (type == 0x0020) {
                port_raw ^= static_cast<uint16_t>(STUN_MAGIC_COOKIE >> 16);
                addr_raw ^= htonl(STUN_MAGIC_COOKIE);
            }
            char ipbuf[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr_raw, ipbuf, sizeof(ipbuf));
            ext_ip = ipbuf;
            ext_port = port_raw;
            return true;
        }
        // 32-bit alignment
        off += (alen + 3) & ~3u;
    }
    std::fprintf(stderr, "stun: no mapped-address attribute\n");
    return false;
}

int cmd_listen(int argc, char** argv) {
    std::string stun_host = "stun.l.google.com";
    uint16_t    stun_port = 19302;
    uint16_t    bind_port = 0;

    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--stun" && i + 1 < argc) {
            std::string s = argv[++i];
            auto colon = s.find(':');
            if (colon == std::string::npos) { std::fprintf(stderr, "bad --stun\n"); return 1; }
            stun_host = s.substr(0, colon);
            stun_port = static_cast<uint16_t>(std::stoi(s.substr(colon + 1)));
        } else if (a == "--port" && i + 1 < argc) {
            bind_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (a == "-h" || a == "--help") {
            usage(); return 0;
        }
    }

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    local.sin_port = htons(bind_port);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
        std::perror("bind"); return 1;
    }
    socklen_t llen = sizeof(local);
    if (::getsockname(sock, reinterpret_cast<sockaddr*>(&local), &llen) < 0) {
        std::perror("getsockname"); return 1;
    }
    std::printf("[listen] bound 0.0.0.0:%u\n", ntohs(local.sin_port));

    sockaddr_in stun_addr{};
    if (!resolve_udp(stun_host, stun_port, stun_addr)) {
        std::fprintf(stderr, "could not resolve %s:%u\n", stun_host.c_str(), stun_port);
        return 1;
    }

    std::string ext_ip;
    uint16_t    ext_port = 0;
    if (!stun_binding(sock, stun_addr, ext_ip, ext_port)) {
        std::fprintf(stderr, "[listen] STUN failed — external mapping unknown\n");
    } else {
        std::printf("[listen] external mapping (via %s): %s:%u\n",
                    stun_host.c_str(), ext_ip.c_str(), ext_port);
        std::printf("[listen] tell the prober:\n");
        std::printf("           nat-pingpong probe %s:%u\n", ext_ip.c_str(), ext_port);
    }
    std::printf("[listen] echoing for up to 10 minutes — Ctrl-C to stop\n");
    std::fflush(stdout);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    uint64_t rx = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        pollfd pfd{sock, POLLIN, 0};
        int pr = poll(&pfd, 1, 1000);
        if (pr <= 0) continue;
        uint8_t buf[2048];
        sockaddr_in from{};
        socklen_t flen = sizeof(from);
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                            reinterpret_cast<sockaddr*>(&from), &flen);
        if (n <= 0) continue;
        // Drop STUN-formed packets (avoid loops if probe is misconfigured).
        if (n >= 20) {
            uint32_t magic;
            std::memcpy(&magic, buf + 4, 4);
            if (magic == htonl(STUN_MAGIC_COOKIE)) continue;
        }
        char fip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &from.sin_addr, fip, sizeof(fip));
        ++rx;
        std::printf("[listen] rx #%llu from %s:%u (%zd bytes) — echoing\n",
                    static_cast<unsigned long long>(rx), fip,
                    ntohs(from.sin_port), n);
        std::fflush(stdout);
        sendto(sock, buf, n, 0, reinterpret_cast<sockaddr*>(&from), flen);
    }
    std::printf("[listen] done — total received: %llu\n",
                static_cast<unsigned long long>(rx));
    ::close(sock);
    return 0;
}

int cmd_probe(int argc, char** argv) {
    if (argc < 1) { usage(); return 1; }
    std::string target = argv[0];
    int count = 10;
    int timeout_ms = 3000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--count" && i + 1 < argc) count = std::stoi(argv[++i]);
        else if (a == "--timeout-ms" && i + 1 < argc) timeout_ms = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(); return 0; }
    }
    auto colon = target.rfind(':');
    if (colon == std::string::npos) { std::fprintf(stderr, "bad target — expect ip:port\n"); return 1; }
    std::string host = target.substr(0, colon);
    uint16_t    port = static_cast<uint16_t>(std::stoi(target.substr(colon + 1)));

    sockaddr_in dst{};
    if (!resolve_udp(host, port, dst)) {
        std::fprintf(stderr, "could not resolve %s:%u\n", host.c_str(), port);
        return 1;
    }

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }
    sockaddr_in any{}; any.sin_family = AF_INET; any.sin_addr.s_addr = htonl(INADDR_ANY);
    ::bind(sock, reinterpret_cast<sockaddr*>(&any), sizeof(any));

    std::printf("[probe] target %s:%u  count=%d  timeout=%dms\n",
                host.c_str(), port, count, timeout_ms);

    int rx = 0;
    std::vector<double> rtts;
    for (int i = 1; i <= count; ++i) {
        char payload[64];
        std::snprintf(payload, sizeof(payload), "ping #%d", i);
        size_t plen = std::strlen(payload);

        auto t0 = std::chrono::steady_clock::now();
        if (sendto(sock, payload, plen, 0,
                   reinterpret_cast<sockaddr*>(&dst), sizeof(dst)) < 0) {
            std::perror("sendto"); continue;
        }
        std::printf("[probe] sent #%d ... ", i);
        std::fflush(stdout);

        pollfd pfd{sock, POLLIN, 0};
        int pr = poll(&pfd, 1, timeout_ms);
        if (pr <= 0) {
            std::printf("(no echo, timeout)\n");
            continue;
        }
        uint8_t buf[2048];
        sockaddr_in from{};
        socklen_t flen = sizeof(from);
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                            reinterpret_cast<sockaddr*>(&from), &flen);
        auto t1 = std::chrono::steady_clock::now();
        if (n <= 0) { std::printf("(recv error)\n"); continue; }
        double rtt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        rtts.push_back(rtt);
        ++rx;
        char fip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &from.sin_addr, fip, sizeof(fip));
        bool from_target = (from.sin_addr.s_addr == dst.sin_addr.s_addr &&
                            from.sin_port == dst.sin_port);
        std::printf("echo from %s:%u %s  rtt=%.2fms  payload=\"%.*s\"\n",
                    fip, ntohs(from.sin_port),
                    from_target ? "(matches target)" : "(MISMATCH)",
                    rtt, static_cast<int>(n), buf);
    }

    std::printf("\n[probe] summary: %d/%d received  (%.0f%% loss)\n",
                rx, count, 100.0 * (count - rx) / std::max(count, 1));
    if (!rtts.empty()) {
        double sum = 0, mn = rtts[0], mx = rtts[0];
        for (double r : rtts) { sum += r; mn = std::min(mn, r); mx = std::max(mx, r); }
        std::printf("[probe] rtt: min=%.2fms  avg=%.2fms  max=%.2fms\n",
                    mn, sum / rtts.size(), mx);
    } else {
        std::printf("[probe] no echoes — target is unreachable from this network.\n");
        std::printf("        likely cause: symmetric NAT on the listener side, or\n");
        std::printf("        the listener's STUN-discovered port has already rotated.\n");
    }
    ::close(sock);
    return rx > 0 ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) { usage(); return 1; }
    std::string sub = argv[1];
    if (sub == "listen") return cmd_listen(argc - 2, argv + 2);
    if (sub == "probe")  return cmd_probe(argc - 2, argv + 2);
    if (sub == "-h" || sub == "--help") { usage(); return 0; }
    std::fprintf(stderr, "unknown subcommand: %s\n\n", sub.c_str());
    usage();
    return 1;
}
