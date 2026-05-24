// nat-portmap — operator tool to probe NAT-PMP / PCP on the upstream
// router.  Prints the granted (external_ip, external_port, lifetime,
// method) on success, exits non-zero on failure.
//
// Usage:
//   nat-portmap [--port <internal>] [--ext <suggested>] [--lifetime <s>]
//               [--gateway <ip>] [--timeout <ms>] [--keep]
//
// With --keep, runs in the foreground and renews the mapping until
// SIGINT.  Useful to verify the router holds the binding under
// sustained renewal.

#include "nat_pmp.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop.store(true); }

void print_help() {
    std::cout <<
        "nat-portmap — probe NAT-PMP / PCP on the upstream router.\n"
        "\n"
        "Options:\n"
        "  --port <n>      Internal UDP port to map (default: 51820)\n"
        "  --ext <n>       Suggested external port (0 = router decides)\n"
        "  --lifetime <s>  Requested lifetime in seconds (default: 3600)\n"
        "  --gateway <ip>  Override default gateway (also DIST_PORTMAP_GATEWAY)\n"
        "  --timeout <ms>  Total probe timeout (default: 1500)\n"
        "  --keep          Renew the mapping until SIGINT (foreground)\n"
        "  --help          Show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    uint16_t internal_port = 51820;
    uint16_t suggested = 0;
    uint32_t lifetime = 3600;
    std::string gateway;
    int timeout_ms = 1500;
    bool keep = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { print_help(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--port")          internal_port = (uint16_t)std::atoi(next());
        else if (a == "--ext")      suggested     = (uint16_t)std::atoi(next());
        else if (a == "--lifetime") lifetime      = (uint32_t)std::atoi(next());
        else if (a == "--gateway")  gateway       = next();
        else if (a == "--timeout")  timeout_ms    = std::atoi(next());
        else if (a == "--keep")     keep          = true;
        else if (a == "--help" || a == "-h") { print_help(); return 0; }
        else {
            std::cerr << "unknown arg: " << a << "\n";
            print_help();
            return 2;
        }
    }

    if (gateway.empty()) gateway = dist::default_gateway_v4();
    std::cout << "[nat-portmap] gateway = "
              << (gateway.empty() ? "(unknown)" : gateway) << "\n";
    if (gateway.empty()) {
        std::cerr << "no default gateway discovered. Set DIST_PORTMAP_GATEWAY=<ip> "
                     "or pass --gateway.\n";
        return 1;
    }

    auto m = dist::try_map_udp(internal_port, suggested, lifetime,
                               gateway, timeout_ms);
    if (!m) {
        std::cerr << "[nat-portmap] no PCP or NAT-PMP response; "
                     "router may not support port mapping (or it is disabled).\n";
        return 1;
    }
    std::cout << "[nat-portmap] mapped:\n"
              << "  method:        " << m->method << "\n"
              << "  external_ip:   " << m->external_ip << "\n"
              << "  external_port: " << m->external_port << "\n"
              << "  internal_port: " << m->internal_port << "\n"
              << "  lifetime_s:    " << m->lifetime_s << "\n";

    if (!keep) return 0;

    std::signal(SIGINT, on_sigint);
    std::cout << "[nat-portmap] holding mapping; Ctrl-C to release.\n";
    dist::PortMapper mapper(internal_port, m->external_port, lifetime, gateway);
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    mapper.stop();
    std::cout << "[nat-portmap] stopped.\n";
    return 0;
}
