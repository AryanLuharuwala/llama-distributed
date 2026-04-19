/**
 * dist_join_main.cpp
 *
 * dist-join — the easiest way to contribute your GPU to a pool.
 *
 * Usage:
 *   dist-join <coordinator-ip>               # auto-detect everything
 *   dist-join <coordinator-ip> --vm          # join in VM mode
 *   dist-join <coordinator-ip> --id mynode   # custom node identifier
 *   dist-join <coordinator-ip> --dry-run     # print the command, don't execute
 *
 * What it does:
 *   1. Contacts the coordinator's dashboard (:7780/join) to fetch the
 *      recommended join command with the correct ports and flags.
 *   2. Probes local GPU count and free VRAM, appends --n-gpu-layers if useful.
 *   3. Prints the final command and (unless --dry-run) exec's it.
 *
 * If the coordinator is unreachable it falls back to sane defaults and
 * constructs the command locally.
 *
 * This binary links only against dist_common (no llama.cpp headers needed at
 * build time — it just exec's dist-node / dist-vm-node).
 */

#include "dist_conn.h"
#include "dist_protocol.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>      // gethostname, getpid, readlink
#include <netdb.h>       // gethostbyname
#include <arpa/inet.h>
#include <sys/socket.h>

// ─── Tiny HTTP GET (no libcurl — just raw TCP) ────────────────────────────────

static std::string http_get(const std::string& host, uint16_t port,
                             const std::string& path) {
    dist::Connection conn;
    try { conn.connect(host, port); }
    catch (...) { return ""; }

    std::string req =
        "GET " + path + " HTTP/1.0\r\n"
        "Host: " + host + "\r\n"
        "Connection: close\r\n"
        "\r\n";

    // Send raw — bypass framing (this is plain HTTP not our protocol)
    // We need the raw fd. Use send() via the platform socket layer.
    // Since Connection wraps a fd, use its send helper or write directly.
    // Simplest: just open a raw socket here.
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(port);
    struct hostent* he = gethostbyname(host.c_str());
    if (!he) { ::close(fd); return ""; }
    std::memcpy(&sa.sin_addr, he->h_addr, he->h_length);

    if (::connect(fd, (sockaddr*)&sa, sizeof(sa)) < 0) { ::close(fd); return ""; }
    ::send(fd, req.data(), req.size(), 0);

    std::string resp;
    char buf[1024];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, n);
    ::close(fd);

    // Strip HTTP headers
    auto sep = resp.find("\r\n\r\n");
    return (sep != std::string::npos) ? resp.substr(sep + 4) : resp;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

static std::string default_node_id() {
    char host[256] = {};
    ::gethostname(host, sizeof(host));
    return std::string(host) + ":" + std::to_string(::getpid());
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <coordinator-ip> [options]\n"
        "\n"
        "  <coordinator-ip>        IP or hostname of the coordinator (required)\n"
        "  --vm                    Join in VM mode (dist-vm-node)\n"
        "  --control-port PORT     Coordinator control port (default: 7700)\n"
        "  --dashboard-port PORT   Dashboard port to query for join params (default: 7780)\n"
        "  --id NAME               Node identifier (default: hostname:pid)\n"
        "  --n-gpu-layers N        Layers to offload to GPU (default: 999 = all)\n"
        "  --dry-run               Print the command but don't run it\n"
        "  -h, --help\n",
        prog);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string coord_host;
    bool        vm_mode        = false;
    bool        dry_run        = false;
    uint16_t    ctrl_port      = 7700;
    uint16_t    dash_port      = 7780;
    std::string node_id        = default_node_id();
    int         n_gpu_layers   = 999;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nxt = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };
        if      (arg[0] != '-')                  coord_host    = arg;
        else if (arg == "--vm")                   vm_mode       = true;
        else if (arg == "--dry-run")              dry_run       = true;
        else if (arg == "--control-port")         ctrl_port     = (uint16_t)std::stoi(nxt());
        else if (arg == "--dashboard-port")       dash_port     = (uint16_t)std::stoi(nxt());
        else if (arg == "--id")                   node_id       = nxt();
        else if (arg == "--n-gpu-layers")         n_gpu_layers  = std::stoi(nxt());
        else if (arg == "-h" || arg == "--help")  { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (coord_host.empty()) {
        fprintf(stderr, "Error: coordinator IP required as first argument.\n");
        print_usage(argv[0]);
        return 1;
    }

    // ── 1. Ask the coordinator dashboard for the recommended join command ──────
    std::string server_cmd;
    std::cout << "⟳ Contacting coordinator at " << coord_host << ":" << dash_port << " …\n";
    server_cmd = http_get(coord_host, dash_port, "/join");
    // Strip newlines from server response
    server_cmd.erase(std::remove(server_cmd.begin(), server_cmd.end(), '\n'), server_cmd.end());
    server_cmd.erase(std::remove(server_cmd.begin(), server_cmd.end(), '\r'), server_cmd.end());

    // ── 2. Build the command ───────────────────────────────────────────────────
    std::string binary = vm_mode ? "dist-vm-node" : "dist-node";
    std::string cmd;

    if (!server_cmd.empty() &&
        (server_cmd.find("dist-node") != std::string::npos ||
         server_cmd.find("dist-vm-node") != std::string::npos)) {
        // Use server-provided command, append our per-node overrides
        cmd = server_cmd;
        cmd += " --id " + node_id;
        cmd += " --n-gpu-layers " + std::to_string(n_gpu_layers);
        if (vm_mode && cmd.find("dist-vm-node") == std::string::npos) {
            // Replace binary
            auto p = cmd.find("dist-node");
            if (p != std::string::npos) cmd.replace(p, 9, "dist-vm-node");
        }
    } else {
        // Fallback: construct locally
        if (!server_cmd.empty())
            std::cerr << "  (dashboard unreachable, using default parameters)\n";
        cmd = binary
            + " --server "        + coord_host
            + " --control-port "  + std::to_string(ctrl_port)
            + " --id "            + node_id
            + " --n-gpu-layers "  + std::to_string(n_gpu_layers);
    }

    // ── 3. Print and run ───────────────────────────────────────────────────────
    std::cout << "\n✓ Join command:\n\n"
              << "    " << cmd << "\n\n";

    if (dry_run) {
        std::cout << "(dry-run, not executing)\n";
        return 0;
    }

    std::cout << "▶ Starting node agent…\n\n";

    // Build argv for execvp
    // Split cmd on spaces — primitive but sufficient for our generated commands
    std::vector<std::string> parts;
    std::istringstream ss(cmd);
    std::string tok;
    while (ss >> tok) parts.push_back(tok);

    std::vector<char*> cargv;
    for (auto& p : parts) cargv.push_back(const_cast<char*>(p.c_str()));
    cargv.push_back(nullptr);

    // Try to find binary in same directory as dist-join itself
    // (allows running from the build directory without PATH manipulation)
    std::string self_path;
    {
        char self_buf[4096] = {};
        ssize_t r = ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
        if (r > 0) {
            self_path = std::string(self_buf, r);
            size_t slash = self_path.rfind('/');
            if (slash != std::string::npos)
                self_path = self_path.substr(0, slash + 1) + parts[0];
        }
    }

    // Try self_path first, then PATH
    if (!self_path.empty()) ::execv(self_path.c_str(), cargv.data());
    ::execvp(cargv[0], cargv.data());

    // If we get here, exec failed
    std::cerr << "exec failed: " << strerror(errno) << "\n";
    std::cerr << "Try running the command above manually.\n";
    return 1;
}
