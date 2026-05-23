/**
 * dist_node_main.cpp
 *
 * Entry point for a Node Agent — one per GPU server.
 *
 * Usage:
 *   dist-node -s COORDINATOR_HOST [options]
 *
 *   -s, --server HOST           Coordinator host (required)
 *   -p, --control-port PORT     Coordinator control port (default: 7700)
 *   -d, --data-port PORT        This node's data port (default: 7701)
 *   -g, --n-gpu-layers N        GPU layers (default: 999 = all)
 *   -c, --context N             Context window (default: 4096)
 *   -b, --batch N               Batch size (default: 512)
 *   --id NAME                   Node ID override (default: hostname:pid)
 *   -h, --help
 */

#include "node_agent.h"
#include "dist_ws_client.h"
#include "pp_engine.h"
#include "shard_download.h"
#include "gpu_lock.h"
#include "comfy_adapter.h"
#include "diffusion_pp_adapter.h"
#include "agent_identity.h"
#include "actv_p2p.h"

#include "llama.h"

#include "platform_compat.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <signal.h>
#include <unistd.h>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  login [--server URL]       Browser device-code login (recommended)\n"
        "  connect                    Run the agent using the saved login\n"
        "  disconnect                 Stop the background agent started by connect\n"
        "  logout                     Forget the saved login\n"
        "  url [--openai|--anthropic|--env|--json]\n"
        "                             Print API endpoint + bearer for this account\n"
        "\n"
        "Legacy / advanced flags (no subcommand):\n"
        "  -s, --server HOST         Coordinator host (direct-join mode)\n"
        "  -p, --control-port PORT   Coordinator control port (default: 7700)\n"
        "  -d, --data-port PORT      This node's data listen port (default: 7701)\n"
        "  -g, --n-gpu-layers N      GPU layers to offload (default: 999=all)\n"
        "  -c, --context N           Context window size (default: 4096)\n"
        "  -b, --batch N             Batch size (default: 512)\n"
        "      --id NAME             Override node ID\n"
        "      --token-id NAME       Auth token id (optional)\n"
        "      --token-secret HEX    Auth token secret, 64 hex chars (optional)\n"
        "      --pair URL            Deep-link pairing URL (distpool://…)\n"
        "  -h, --help\n",
        prog);
}

// ─── Pair mode helpers ──────────────────────────────────────────────────────

// Parse a distpool://pair?token=T&server=WS URL.
// Returns true if the URL was well-formed.
static bool parse_pair_url(const std::string& url,
                            std::string& out_token,
                            std::string& out_server) {
    const std::string prefix = "distpool://pair?";
    if (url.rfind(prefix, 0) != 0) return false;
    std::string qs = url.substr(prefix.size());

    auto pull = [](const std::string& s, const std::string& key) -> std::string {
        size_t p = s.find(key + "=");
        if (p == std::string::npos) return "";
        size_t start = p + key.size() + 1;
        size_t end   = s.find('&', start);
        return s.substr(start, end == std::string::npos ? std::string::npos : end - start);
    };

    out_token  = pull(qs, "token");
    out_server = pull(qs, "server");
    return !out_token.empty() && !out_server.empty();
}

// Minimal JSON string escaper (ASCII, no control chars expected).
static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

static std::string default_agent_id() {
    char host[256] = {};
    ::gethostname(host, sizeof(host));
    return std::string(host) + ":" + std::to_string(::getpid());
}

// GpuInfo — coarse single-GPU telemetry pulled from `nvidia-smi`.  Used by
// the heartbeat to populate the swarm dashboard (gpu_model, vram_total,
// vram_free).  Soft-fails on machines without nvidia-smi — the dashboard
// just renders "?" for those rigs.
struct GpuInfo {
    std::string name;        // e.g. "RTX 3050 Laptop GPU"
    int64_t     vram_total = 0; // bytes
    int64_t     vram_free  = 0; // bytes
    bool        ok = false;
};

static GpuInfo query_gpu_info() {
    GpuInfo gi;
#ifndef _WIN32
    // CSV: name, memory.total, memory.free — all in MiB.
    FILE* fp = ::popen(
        "nvidia-smi --query-gpu=name,memory.total,memory.free "
        "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!fp) return gi;
    char buf[512];
    if (::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        // Strip trailing newline.
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        // Split on ", " — exactly two commas in well-formed output.
        size_t c1 = line.find(',');
        size_t c2 = (c1 != std::string::npos) ? line.find(',', c1 + 1)
                                              : std::string::npos;
        if (c1 != std::string::npos && c2 != std::string::npos) {
            gi.name = line.substr(0, c1);
            // Trim whitespace on name.
            while (!gi.name.empty() && gi.name.back() == ' ') gi.name.pop_back();
            try {
                std::string tot = line.substr(c1 + 1, c2 - c1 - 1);
                std::string fre = line.substr(c2 + 1);
                gi.vram_total = (int64_t) std::stoll(tot) * 1024 * 1024;
                gi.vram_free  = (int64_t) std::stoll(fre) * 1024 * 1024;
                gi.ok = true;
            } catch (...) {}
        }
    }
    ::pclose(fp);
#endif
    return gi;
}

// State directory — where we persist agent_key across restarts so the rig
// can resume without burning a new pair token.  Order:
//   $DIST_STATE_DIR       (explicit override)
//   $XDG_STATE_HOME/llama-distributed       (linux)
//   $HOME/Library/Application Support/llama-distributed (mac)
//   %LOCALAPPDATA%\llama-distributed        (windows)
//   $HOME/.local/state/llama-distributed    (linux fallback)
static std::string state_dir() {
    if (const char* s = std::getenv("DIST_STATE_DIR"); s && *s) return s;
#ifdef _WIN32
    if (const char* s = std::getenv("LOCALAPPDATA"); s && *s)
        return std::string(s) + "\\llama-distributed";
#endif
    if (const char* s = std::getenv("XDG_STATE_HOME"); s && *s)
        return std::string(s) + "/llama-distributed";
    if (const char* h = std::getenv("HOME"); h && *h) {
#ifdef __APPLE__
        return std::string(h) + "/Library/Application Support/llama-distributed";
#else
        return std::string(h) + "/.local/state/llama-distributed";
#endif
    }
    return "./llama-distributed-state";
}

static std::string state_path(const std::string& name) {
    return state_dir() + "/" + name;
}

// Load a small UTF-8 value from disk; returns empty on any error.  Used for
// agent.id and agent.key.  Trailing newline (if present) is stripped.
static std::string state_read(const std::string& name) {
    std::ifstream f(state_path(name), std::ios::binary);
    if (!f.good()) return "";
    std::ostringstream buf;
    buf << f.rdbuf();
    std::string s = buf.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// Persist a small value atomically (write to .tmp, rename).  Returns true on
// success — caller should log but tolerate failures so a pair still completes
// even if the state dir is read-only.
static bool state_write(const std::string& name, const std::string& value) {
    std::error_code ec;
    std::filesystem::create_directories(state_dir(), ec);
    (void)ec;
    const std::string dest = state_path(name);
    const std::string tmp  = dest + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.good()) return false;
        f << value;
    }
    std::filesystem::rename(tmp, dest, ec);
    return !ec;
}

// Binary variants — used to persist the ed25519 private key, which is 32
// random bytes that may contain '\n' or '\0'.  state_read/state_write
// strip trailing whitespace and would corrupt the key.
static std::vector<uint8_t> state_read_bytes(const std::string& name) {
    std::ifstream f(state_path(name), std::ios::binary);
    if (!f.good()) return {};
    std::ostringstream buf;
    buf << f.rdbuf();
    const std::string& s = buf.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

static bool state_write_bytes(const std::string& name,
                              const std::vector<uint8_t>& bytes) {
    std::error_code ec;
    std::filesystem::create_directories(state_dir(), ec);
    (void)ec;
    const std::string dest = state_path(name);
    const std::string tmp  = dest + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.good()) return false;
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        if (!f.good()) return false;
    }
    std::filesystem::rename(tmp, dest, ec);
    if (ec) return false;
    // chmod 0600 so other users on multi-tenant boxes can't read the
    // private key.  Best-effort — silently ignored on Windows.
#ifndef _WIN32
    ::chmod(dest.c_str(), 0600);
#endif
    return true;
}

// ─── Tiny HTTPS client (OpenSSL) ────────────────────────────────────────────
// Used by the `dist-node login` / `dist-node url` subcommands to talk to the
// control-plane HTTP API.  Returns true and fills (status, body) on success.
// Supports http:// and https://.  Path includes the query string.
struct HttpResp {
    int         status = 0;
    std::string body;
};

namespace {
    bool g_dn_ssl_inited = false;
    void dn_ssl_init() {
        if (g_dn_ssl_inited) return;
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
        g_dn_ssl_inited = true;
    }
}

static bool parse_http_url_dn(const std::string& url,
                              bool& tls, std::string& host,
                              uint16_t& port, std::string& path) {
    std::string u = url;
    if      (u.rfind("https://", 0) == 0) { tls = true;  u = u.substr(8); port = 443; }
    else if (u.rfind("http://",  0) == 0) { tls = false; u = u.substr(7); port = 80;  }
    else return false;
    size_t slash = u.find('/');
    std::string hp = (slash == std::string::npos) ? u : u.substr(0, slash);
    path = (slash == std::string::npos) ? "/" : u.substr(slash);
    size_t colon = hp.find(':');
    if (colon == std::string::npos) host = hp;
    else { host = hp.substr(0, colon); port = (uint16_t)std::stoi(hp.substr(colon + 1)); }
    return !host.empty();
}

static bool http_request(const std::string& base_url, const std::string& path,
                         const std::string& method, const std::string& body,
                         const std::vector<std::string>& extra_headers,
                         HttpResp& out, std::string& err) {
    bool tls; std::string host; uint16_t port; std::string root_path;
    if (!parse_http_url_dn(base_url, tls, host, port, root_path)) {
        err = "bad base URL: " + base_url;
        return false;
    }
    (void)root_path;
    // Resolve.
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16]; std::snprintf(ps, sizeof(ps), "%u", (unsigned)port);
    if (getaddrinfo(host.c_str(), ps, &hints, &res) != 0 || !res) {
        err = "dns: " + host; return false;
    }
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { freeaddrinfo(res); err = "socket"; return false; }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res); dist::close_sock(fd);
        err = "connect " + host; return false;
    }
    freeaddrinfo(res);

    SSL_CTX* ctx = nullptr;
    SSL*     ssl = nullptr;
    if (tls) {
        dn_ssl_init();
        ctx = SSL_CTX_new(TLS_client_method());
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        SSL_CTX_set_default_verify_paths(ctx);
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host.c_str());
        SSL_set1_host(ssl, host.c_str());
        SSL_set_fd(ssl, fd);
        if (SSL_connect(ssl) != 1) {
            unsigned long e = ERR_get_error(); char buf[256] = {0};
            ERR_error_string_n(e, buf, sizeof(buf));
            err = std::string("TLS: ") + buf;
            SSL_free(ssl); SSL_CTX_free(ctx); dist::close_sock(fd);
            return false;
        }
    }

    const bool default_port = (tls && port == 443) || (!tls && port == 80);
    std::string host_hdr = default_port ? host : host + ":" + std::to_string(port);

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n"
        << "Host: " << host_hdr << "\r\n"
        << "User-Agent: dist-node/0.1\r\n"
        << "Accept: application/json\r\n"
        << "Connection: close\r\n";
    if (!body.empty()) {
        req << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    }
    for (const auto& h : extra_headers) req << h << "\r\n";
    req << "\r\n" << body;
    std::string r = req.str();

    auto sendall = [&](const char* p, size_t n) -> bool {
        while (n > 0) {
            int k = tls ? SSL_write(ssl, p, (int)n) : (int)::send(fd, p, n, 0);
            if (k <= 0) return false;
            p += k; n -= (size_t)k;
        }
        return true;
    };
    if (!sendall(r.data(), r.size())) { err = "send"; goto fail; }

    {
        std::string resp;
        char rb[4096];
        while (true) {
            int k = tls ? SSL_read(ssl, rb, sizeof(rb))
                        : (int)::recv(fd, rb, sizeof(rb), 0);
            if (k <= 0) break;
            resp.append(rb, (size_t)k);
        }
        // Status line
        size_t sp1 = resp.find(' ');
        size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : resp.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) { err = "bad status"; goto fail; }
        out.status = std::atoi(resp.substr(sp1 + 1, sp2 - sp1 - 1).c_str());
        size_t hdr_end = resp.find("\r\n\r\n");
        std::string body_raw = (hdr_end == std::string::npos) ? "" : resp.substr(hdr_end + 4);

        // Crude chunked decoder — strip "<hex>\r\n" length prefixes if present.
        std::string te_key = "Transfer-Encoding:";
        size_t te = resp.find(te_key);
        bool chunked = (te != std::string::npos && te < (hdr_end == std::string::npos ? resp.size() : hdr_end)
                        && resp.substr(te, hdr_end - te).find("chunked") != std::string::npos);
        if (chunked) {
            std::string decoded;
            size_t i = 0;
            while (i < body_raw.size()) {
                size_t eol = body_raw.find("\r\n", i);
                if (eol == std::string::npos) break;
                size_t len = std::strtoul(body_raw.substr(i, eol - i).c_str(), nullptr, 16);
                i = eol + 2;
                if (len == 0) break;
                if (i + len > body_raw.size()) break;
                decoded.append(body_raw, i, len);
                i += len + 2; // skip trailing \r\n
            }
            out.body = decoded;
        } else {
            out.body = body_raw;
        }
    }

    if (tls) { SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); }
    ::shutdown(fd, SHUT_RDWR);
    dist::close_sock(fd);
    return true;

fail:
    if (tls) { if (ssl) SSL_free(ssl); if (ctx) SSL_CTX_free(ctx); }
    dist::close_sock(fd);
    return false;
}

// Extract a JSON string value, very permissively.  Returns "" on not-found.
// Only works for flat string fields; fine for welcome {"agent_key":"…"}.
static std::string json_peek_string(const std::string& msg, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    size_t p = msg.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    std::string out;
    while (p < msg.size() && msg[p] != '"') {
        if (msg[p] == '\\' && p + 1 < msg.size()) { out += msg[p + 1]; p += 2; continue; }
        out += msg[p++];
    }
    return out;
}

// Peek a JSON number field as a string of its digits.  Returns "" if the
// key isn't a bare numeric value (e.g. it was wrapped in quotes).  Used
// to read fields like "ts":1779515581 out of the challenge frame.
static std::string json_peek_int(const std::string& msg, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t p = msg.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    // Skip whitespace.
    while (p < msg.size() && (msg[p] == ' ' || msg[p] == '\t')) ++p;
    if (p >= msg.size() || msg[p] == '"') return "";
    std::string out;
    if (msg[p] == '-') { out += '-'; ++p; }
    while (p < msg.size() && msg[p] >= '0' && msg[p] <= '9') {
        out += msg[p++];
    }
    return out;
}

// Resolve the absolute path of the running dist-node binary so we can find
// our `dist-turn` sibling next to it.  Falls back to "dist-turn" on PATH.
static std::string find_dist_turn_binary() {
    if (const char* env = std::getenv("DIST_TURN_BIN"); env && *env) {
        return env;
    }
#ifdef __linux__
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = 0;
        std::filesystem::path p(buf);
        auto sibling = p.parent_path() / "dist-turn";
        if (std::filesystem::exists(sibling)) return sibling.string();
    }
#endif
    return "dist-turn"; // assume PATH
}

// ── dist-turn sidecar manager ─────────────────────────────────────────────
//
// All state lives in file-scope variables guarded by g_turn_mu so the
// heartbeat loop and signal/atexit hooks can interact safely.  The model is:
//
//   * spawn_dist_turn() is idempotent on (secret, realm, port, ext_ip).  If a
//     live child already matches those args it's a no-op; if it's running
//     with stale args we kill and re-launch.
//   * The child runs in its own process group with PR_SET_PDEATHSIG, so an
//     uncatchable parent death (SIGKILL/segfault) still tears down the
//     sidecar — atexit alone is not enough.
//   * The child's stderr is wired to a pipe drained by a daemon thread that
//     prefixes each line with [turn-child] so operators see startup errors
//     in dist-node's own log.
//   * turn_sidecar_alive() does a non-blocking waitpid; if the child died on
//     its own it returns false and clears state so the next ensure_*() call
//     respawns.

static std::mutex                   g_turn_mu;
static std::atomic<pid_t>           g_turn_sidecar_pid{0};
static std::string                  g_turn_secret_cur;
static std::string                  g_turn_realm_cur;
static int                          g_turn_port_cur = 0;
static std::string                  g_turn_ext_ip_cur;
static std::thread                  g_turn_log_thread;
static std::atomic<bool>            g_turn_log_stop{false};
static int                          g_turn_log_fd = -1;

static void kill_turn_sidecar() {
#ifdef _WIN32
    pid_t p = g_turn_sidecar_pid.exchange(0);
    if (p > 0 && g_turn_job_handle) {
        // Closing the job handle with KILL_ON_JOB_CLOSE set terminates
        // every process inside.  Then wait briefly to reap.
        ::CloseHandle(g_turn_job_handle);
        g_turn_job_handle = nullptr;
        if (g_turn_proc_handle) {
            ::WaitForSingleObject(g_turn_proc_handle, 1000);
            ::CloseHandle(g_turn_proc_handle);
            g_turn_proc_handle = nullptr;
        }
    }
#else
    pid_t p = g_turn_sidecar_pid.exchange(0);
    if (p > 0) {
        // SIGTERM the whole process group — pion + any goroutines.
        ::kill(-p, SIGTERM);
        for (int i = 0; i < 20; ++i) {
            if (::waitpid(p, nullptr, WNOHANG) == p) goto reaped;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ::kill(-p, SIGKILL);
        ::waitpid(p, nullptr, 0);
    reaped:;
    }
    g_turn_log_stop.store(true);
    if (g_turn_log_fd >= 0) { ::close(g_turn_log_fd); g_turn_log_fd = -1; }
    if (g_turn_log_thread.joinable()) {
        // Detach: we can't safely join from atexit if we're already in
        // shutdown and the thread may be blocked on read.  The pipe close
        // above unsticks it; detachment is fine for an exiting process.
        g_turn_log_thread.detach();
    }
#endif
}

// True iff the sidecar PID is still a running child of ours.  If the child
// died on its own we reap it and clear state so the next ensure_*() call
// will respawn.
static bool turn_sidecar_alive() {
    pid_t p = g_turn_sidecar_pid.load();
    if (p <= 0) return false;
#ifdef _WIN32
    if (!g_turn_proc_handle) return false;
    DWORD r = ::WaitForSingleObject(g_turn_proc_handle, 0);
    if (r == WAIT_TIMEOUT) return true;
    // Exited.
    ::CloseHandle(g_turn_proc_handle);
    g_turn_proc_handle = nullptr;
    if (g_turn_job_handle) {
        ::CloseHandle(g_turn_job_handle);
        g_turn_job_handle = nullptr;
    }
    g_turn_sidecar_pid.store(0);
    {
        std::lock_guard<std::mutex> lk(g_turn_mu);
        g_turn_secret_cur.clear();
        g_turn_realm_cur.clear();
        g_turn_port_cur = 0;
        g_turn_ext_ip_cur.clear();
    }
    return false;
#else
    int status = 0;
    pid_t r = ::waitpid(p, &status, WNOHANG);
    if (r == 0) return true;  // still running
    // r == p (exited) or r == -1 (ECHILD); either way it's gone.
    g_turn_sidecar_pid.store(0);
    std::lock_guard<std::mutex> lk(g_turn_mu);
    g_turn_secret_cur.clear();
    g_turn_realm_cur.clear();
    g_turn_port_cur = 0;
    g_turn_ext_ip_cur.clear();
    return false;
#endif
}

#ifdef _WIN32
// Windows job-object handle holding the sidecar.  Anything launched into the
// job dies when the last open handle closes — perfect for tying the sidecar
// to the parent's lifetime (no atexit hook can survive SIGKILL on POSIX, and
// on Windows JobObjects + KILL_ON_JOB_CLOSE replace prctl(PR_SET_PDEATHSIG)).
static HANDLE g_turn_job_handle  = nullptr;
static HANDLE g_turn_proc_handle = nullptr;
#endif

// Internal: actually fork + exec the sidecar.  Caller must hold g_turn_mu and
// must have already killed any previous instance.  Returns child pid on
// success, 0 on failure (binary missing, fork failed, bound port taken).
static pid_t do_spawn_turn(int port,
                           const std::string& secret,
                           const std::string& realm,
                           const std::string& ext_ip) {
#ifdef _WIN32
    if (secret.empty()) return 0;
    std::string bin = find_dist_turn_binary();
    // Build command line: a single quoted string with all args.  CreateProcess
    // doesn't take argv, so we have to do our own quoting.  None of our args
    // contain unescaped quotes or backslashes meaningful to Windows, so simple
    // " wrapping is fine.
    auto quote = [](const std::string& s) {
        return std::string("\"") + s + "\"";
    };
    std::string cmd = quote(bin) +
        " --listen=0.0.0.0:" + std::to_string(port) +
        " --realm=" + quote(realm) +
        " --auth-secret=" + quote(secret);
    if (!ext_ip.empty()) {
        cmd += " --external-ip=" + ext_ip;
    } else if (const char* envip = std::getenv("DIST_TURN_EXTERNAL_IP");
               envip && *envip) {
        cmd += std::string(" --external-ip=") + envip;
    }

    // Job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE — the kernel kills
    // every process in the job when the last handle to it closes.  We hold
    // g_turn_job_handle until kill_turn_sidecar releases it.
    if (g_turn_job_handle == nullptr) {
        g_turn_job_handle = ::CreateJobObjectA(nullptr, nullptr);
        if (g_turn_job_handle == nullptr) {
            std::cerr << "[turn] CreateJobObject failed (err=" <<
                ::GetLastError() << ")\n";
            return 0;
        }
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        ::SetInformationJobObject(g_turn_job_handle,
            JobObjectExtendedLimitInformation, &info, sizeof(info));
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<char> mut(cmd.begin(), cmd.end()); mut.push_back(0);
    if (!::CreateProcessA(nullptr, mut.data(), nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP,
            nullptr, nullptr, &si, &pi)) {
        std::cerr << "[turn] CreateProcess(" << bin << ") failed (err="
                  << ::GetLastError() << ")\n";
        return 0;
    }
    if (!::AssignProcessToJobObject(g_turn_job_handle, pi.hProcess)) {
        std::cerr << "[turn] AssignProcessToJobObject failed (err="
                  << ::GetLastError() << "); killing child\n";
        ::TerminateProcess(pi.hProcess, 1);
        ::CloseHandle(pi.hThread); ::CloseHandle(pi.hProcess);
        return 0;
    }
    ::ResumeThread(pi.hThread);
    ::CloseHandle(pi.hThread);

    // Wait up to 1.5s; if the child exits sooner the launch failed.
    DWORD r = ::WaitForSingleObject(pi.hProcess, 1500);
    if (r == WAIT_OBJECT_0) {
        DWORD ec = 0;
        ::GetExitCodeProcess(pi.hProcess, &ec);
        std::cerr << "[turn] sidecar exited during startup (code="
                  << ec << ")\n";
        ::CloseHandle(pi.hProcess);
        return 0;
    }
    // Still running → success.
    g_turn_proc_handle = pi.hProcess;
    return (pid_t) pi.dwProcessId;
#else
    if (secret.empty()) return 0;
    std::string bin = find_dist_turn_binary();

    int err_pipe[2] = {-1, -1};
    if (::pipe(err_pipe) != 0) {
        std::cerr << "[turn] pipe failed: " << std::strerror(errno) << "\n";
        return 0;
    }

    pid_t pid = fork();
    if (pid < 0) {
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        std::cerr << "[turn] fork failed: " << std::strerror(errno) << "\n";
        return 0;
    }
    if (pid == 0) {
        // ── Child ──────────────────────────────────────────────────────
        // New process group so kill(-pid, SIG) hits any goroutines pion
        // might fork off; tied to parent death on Linux so an unclean
        // parent exit (SIGKILL / segfault) still tears us down.
        ::setpgid(0, 0);
#ifdef __linux__
        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        // Re-check: if parent already died between fork() and prctl(),
        // PR_SET_PDEATHSIG won't fire — exit voluntarily.
        if (::getppid() == 1) _exit(0);
#endif
        // Wire stderr → pipe; stdout is inherited so operators running
        // dist-node interactively still see anything pion prints there.
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);

        std::string listen = "0.0.0.0:" + std::to_string(port);
        std::vector<std::string> args = {
            bin,
            "--listen=" + listen,
            "--realm=" + realm,
            "--auth-secret=" + secret,
        };
        if (!ext_ip.empty()) {
            args.emplace_back(std::string("--external-ip=") + ext_ip);
        } else if (const char* envip = std::getenv("DIST_TURN_EXTERNAL_IP");
                   envip && *envip) {
            // Operator override takes precedence over autodetected IP.
            args.emplace_back(std::string("--external-ip=") + envip);
        }
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        ::execvp(bin.c_str(), argv.data());
        // execvp returned → it failed.  Write the error to the pipe so
        // the parent logger picks it up before _exit.
        std::string msg = std::string("execvp(") + bin + ") failed: " +
                          std::strerror(errno) + "\n";
        (void) ::write(STDERR_FILENO, msg.data(), msg.size());
        _exit(127);
    }

    // ── Parent ─────────────────────────────────────────────────────────
    ::close(err_pipe[1]); // we only read from the pipe

    // Tear down any prior log thread before starting a new one.
    g_turn_log_stop.store(true);
    if (g_turn_log_fd >= 0) { ::close(g_turn_log_fd); g_turn_log_fd = -1; }
    if (g_turn_log_thread.joinable()) g_turn_log_thread.detach();
    g_turn_log_stop.store(false);
    g_turn_log_fd = err_pipe[0];
    g_turn_log_thread = std::thread([fd = err_pipe[0]]() {
        std::string buf;
        char        chunk[512];
        while (!g_turn_log_stop.load()) {
            ssize_t n = ::read(fd, chunk, sizeof(chunk));
            if (n <= 0) break; // EOF or error
            buf.append(chunk, chunk + n);
            for (;;) {
                auto nl = buf.find('\n');
                if (nl == std::string::npos) break;
                std::cerr << "[turn-child] " << buf.substr(0, nl) << "\n";
                buf.erase(0, nl + 1);
            }
        }
        if (!buf.empty()) std::cerr << "[turn-child] " << buf << "\n";
    });

    // Give pion ~1.5s to bind and start listening, then probe.  pion logs
    // "dist-turn: listening on ..." once it's actually serving requests;
    // we don't need to parse that — waitpid + a localhost UDP smoke test
    // is enough.  The longer window forgives slow/loaded VMs.
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(1500);
    while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            std::cerr << "[turn] sidecar exited during startup (status="
                      << status << ")\n";
            return 0;
        }
        // Cheap aliveness check: open a UDP socket, try to send a zero-byte
        // packet to 127.0.0.1:port.  If the port is unbound the kernel
        // returns ECONNREFUSED via ICMP on the next recv; sendto itself
        // will succeed regardless.  So we use SOCK_DGRAM connect() which
        // *does* surface ECONNREFUSED synchronously on Linux.
        int sk = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sk >= 0) {
            sockaddr_in sa{};
            sa.sin_family = AF_INET;
            sa.sin_port   = htons(port);
            ::inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
            if (::connect(sk, (sockaddr*)&sa, sizeof(sa)) == 0) {
                // Connected datagram socket — a real send tells us if the
                // port is bound.  An unbound UDP port returns ECONNREFUSED
                // on the *next* IO op, so send a 1-byte probe and check.
                char one = 0;
                ::send(sk, &one, 1, MSG_DONTWAIT);
                char rb;
                ssize_t rr = ::recv(sk, &rb, 1, MSG_DONTWAIT);
                if (rr < 0 && errno != ECONNREFUSED && errno != EAGAIN &&
                    errno != EWOULDBLOCK) {
                    // Strange error; fall through to delay/retry.
                } else if (errno != ECONNREFUSED) {
                    // Port is bound (we either got EAGAIN/EWOULDBLOCK
                    // because pion didn't echo, or rr >= 0).
                    ::close(sk);
                    return pid;
                }
            }
            ::close(sk);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Final waitpid — if pion is still up after 1.5s, accept it even
    // without a successful probe (probe may fail on hosts where loopback
    // is firewalled but external is fine).
    int status = 0;
    pid_t r = ::waitpid(pid, &status, WNOHANG);
    if (r == pid) {
        std::cerr << "[turn] sidecar exited during startup (status="
                  << status << ")\n";
        return 0;
    }
    return pid;
#endif
}

// Idempotent ensure-up.  If a sidecar is already running with matching args
// this is a cheap no-op.  Returns the running pid (or 0 on failure).
// Setting want=false unconditionally tears the sidecar down — used when the
// server stops shipping a turn_secret or we lose relay_capable status.
static pid_t ensure_turn_sidecar(bool want,
                                 const std::string& secret,
                                 const std::string& realm,
                                 int port,
                                 const std::string& ext_ip) {
    std::lock_guard<std::mutex> lk(g_turn_mu);
    if (!want || secret.empty()) {
        if (g_turn_sidecar_pid.load() > 0) kill_turn_sidecar();
        return 0;
    }
    // Already running with matching args?  Liveness was already validated
    // by the caller (or will be re-validated by turn_sidecar_alive on the
    // next heartbeat) — we just compare args here.
    pid_t cur = g_turn_sidecar_pid.load();
    if (cur > 0 &&
        secret == g_turn_secret_cur &&
        realm  == g_turn_realm_cur  &&
        port   == g_turn_port_cur   &&
        ext_ip == g_turn_ext_ip_cur) {
        return cur;
    }
    // Stale args or no sidecar — kill any survivor and respawn.
    if (cur > 0) {
        std::cerr << "[turn] respawning sidecar (args changed)\n";
        // kill_turn_sidecar already manipulates g_turn_*; we're holding
        // g_turn_mu but kill_turn_sidecar doesn't re-lock, so this is fine.
        kill_turn_sidecar();
    }
    pid_t pid = do_spawn_turn(port, secret, realm, ext_ip);
    if (pid > 0) {
        g_turn_sidecar_pid.store(pid);
        g_turn_secret_cur = secret;
        g_turn_realm_cur  = realm;
        g_turn_port_cur   = port;
        g_turn_ext_ip_cur = ext_ip;
        static std::once_flag s_atexit;
        std::call_once(s_atexit, []{ std::atexit(kill_turn_sidecar); });
    }
    return pid;
}

// Enter pair mode: connect to the server WebSocket, send hello, and loop on
// status updates.  Returns process exit code.
//
// Reconnection model:
//   - On first run we have a pair token; send {kind:"hello"}, parse the
//     returned agent_key out of the welcome frame, and persist it to the
//     state dir alongside agent_id.
//   - On every subsequent run (the service restart case) we read both files
//     and send {kind:"resume"} — no pair token, no round-trip to the dash.
//   - If the server rejects the resume (bad agent_key → StatusPolicyViolation)
//     we delete the local state and fall back to the pair token so the rig
//     repairs itself on the next try.
static int run_pair_mode(const std::string& token, const std::string& server,
                         const std::string& agent_id_override,
                         int n_gpu_layers) {
    dist::WsClient ws;
    if (!ws.connect(server)) {
        std::cerr << "[pair] connect failed: " << ws.last_error() << "\n";
        return 1;
    }
    std::cout << "[pair] connected to " << server << "\n";

    // Agent identity: prefer explicit override; otherwise prefer the id we
    // persisted on a prior successful pair; fall back to hostname:pid only on
    // a truly first boot so the id stays stable across restarts.
    std::string saved_agent_id  = state_read("agent.id");
    std::string saved_agent_key = state_read("agent.key");
    std::string agent_id        = !agent_id_override.empty() ? agent_id_override
                                   : !saved_agent_id.empty()  ? saved_agent_id
                                   : default_agent_id();

    char hostname[256] = {};
    ::gethostname(hostname, sizeof(hostname));

    const bool have_resume = !saved_agent_key.empty() && !saved_agent_id.empty()
                             && agent_id_override.empty();

    // ── Signed-identity bootstrap ────────────────────────────────────────
    // Load or generate the ed25519 keypair we use to prove possession of
    // this agent_id.  Stored under <state>/agent.priv as 32 raw bytes,
    // chmod 0600.  On first boot we generate a fresh key; on every later
    // launch we load the saved private key and re-derive the public key.
    std::vector<uint8_t> agent_priv = state_read_bytes("agent.priv");
    std::vector<uint8_t> agent_pub;
    bool have_keypair = false;
    if (agent_priv.size() == dist::ED25519_PRIV_SIZE) {
        if (dist::ed25519_pub_from_priv(agent_priv, agent_pub)) {
            have_keypair = true;
        } else {
            std::cerr << "[pair] WARN: saved priv key failed to load; regenerating\n";
            agent_priv.clear();
        }
    }
    if (!have_keypair) {
        if (dist::ed25519_generate(agent_priv, agent_pub)) {
            if (!state_write_bytes("agent.priv", agent_priv)) {
                std::cerr << "[pair] WARN: could not persist agent.priv to "
                          << state_dir() << " — restarts will re-pair\n";
            }
            have_keypair = true;
            std::cout << "[pair] generated fresh ed25519 keypair\n";
        } else {
            std::cerr << "[pair] WARN: ed25519 keygen failed; resume sigs unavailable\n";
        }
    }
    std::string agent_pub_b64 = have_keypair
        ? dist::b64url_encode(agent_pub)
        : std::string();

    // Wire-protocol version we speak.  Server clamps to its own
    // supported range and returns the negotiated value in the welcome
    // frame; we save it for branching later (no consumer today, but
    // future server changes can fan out behind this).
    constexpr int kClientProtocolVersion = 1;

    std::ostringstream hello;
    if (have_resume) {
        std::cout << "[pair] resuming as agent_id=" << agent_id
                  << " (using saved agent_key)\n";
        hello << "{\"kind\":\"resume\","
              << "\"agent_key\":\"" << json_escape(saved_agent_key) << "\","
              << "\"agent_id\":\""  << json_escape(agent_id)        << "\","
              << "\"hostname\":\""  << json_escape(hostname)        << "\","
              << "\"protocol_version\":" << kClientProtocolVersion  << ","
              << "\"n_gpus\":0,\"vram_bytes\":0}";
    } else {
        hello << "{\"kind\":\"hello\","
              << "\"token\":\""    << json_escape(token)    << "\","
              << "\"agent_id\":\"" << json_escape(agent_id) << "\","
              << "\"hostname\":\"" << json_escape(hostname) << "\","
              << "\"protocol_version\":" << kClientProtocolVersion << ","
              << "\"n_gpus\":0,\"vram_bytes\":0";
        if (!agent_pub_b64.empty()) {
            hello << ",\"pubkey\":\"" << agent_pub_b64 << "\"";
        }
        hello << "}";
    }

    if (!ws.send_text(hello.str())) {
        std::cerr << "[pair] send hello failed\n";
        return 1;
    }

    std::string msg;
    if (!ws.recv_text(msg)) {
        std::cerr << "[pair] no welcome received\n";
        return 1;
    }
    // Resume challenge: if the server asks us to prove key ownership,
    // sign the (agent_id|nonce|ts) blob and reply with a "sig" frame.
    if (msg.find("\"kind\":\"challenge\"") != std::string::npos) {
        std::string nonce = json_peek_string(msg, "nonce");
        std::string ts    = json_peek_int(msg, "ts");
        if (nonce.empty() || ts.empty() || !have_keypair) {
            std::cerr << "[pair] challenge received but cannot answer (have_keypair="
                      << have_keypair << ", nonce_len=" << nonce.size() << ")\n";
            return 1;
        }
        const std::string sig_input =
            "dist-agent-v1|" + agent_id + "|" + nonce + "|" + ts;
        std::vector<uint8_t> sig_bytes;
        if (!dist::ed25519_sign(agent_priv, sig_input, sig_bytes)) {
            std::cerr << "[pair] ed25519 sign failed\n";
            return 1;
        }
        const std::string sig_b64 = dist::b64url_encode(sig_bytes);
        std::ostringstream sig_frame;
        sig_frame << "{\"kind\":\"sig\",\"sig\":\"" << sig_b64 << "\"}";
        if (!ws.send_text(sig_frame.str())) {
            std::cerr << "[pair] send sig failed\n";
            return 1;
        }
        // Now wait for the real welcome.
        msg.clear();
        if (!ws.recv_text(msg)) {
            std::cerr << "[pair] no welcome after sig\n";
            return 1;
        }
    }
    std::cout << "[pair] server reply: " << msg << "\n";
    if (msg.find("\"kind\":\"welcome\"") == std::string::npos) {
        // Resume failed (most likely the server lost the rig row — user
        // probably reset the DB or the key is stale).  Scrub local state so
        // the next launch does a fresh pair instead of looping on bad key.
        if (have_resume) {
            std::cerr << "[pair] resume rejected; clearing saved agent_key\n";
            std::error_code ec;
            std::filesystem::remove(state_path("agent.key"),    ec);
            std::filesystem::remove(state_path("agent.id"),     ec);
            std::filesystem::remove(state_path("agent.server"), ec);
        }
        // Specific hint for the common first-pair failure mode: a stale
        // one-liner (the user waited past the 30-min token TTL before
        // pasting, or already consumed it from another machine).  The
        // operator-facing remedy is always the same — regenerate from the
        // dashboard — so surface it prominently.
        if (msg.find("bad pair token") != std::string::npos) {
            std::cerr << "[pair] pair token rejected — it may be stale or already used.\n"
                      << "[pair] open the dashboard and click 'Generate one-liner' again,\n"
                      << "[pair] then re-run this installer with the new command.\n";
        }
        std::cerr << "[pair] not welcomed — exiting\n";
        // Sleep briefly so systemd's restart backoff picks up the log line
        // instead of spamming on instant exit.
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return 1;
    }

    // TURN-sidecar coordination: the server ships its shared secret + realm
    // in the welcome frame when it has DIST_TURN_SECRET configured.  We
    // capture them here even when DIST_WITH_TURN isn't set at launch time —
    // they're cheap to keep around and let a future SIGHUP-style reload
    // engage the sidecar without re-pairing.
    std::string turn_secret_from_srv = json_peek_string(msg, "turn_secret");
    std::string turn_realm_from_srv  = json_peek_string(msg, "turn_realm");
    if (turn_realm_from_srv.empty()) turn_realm_from_srv = "dist";

    // First-pair path: persist the agent_key that came back in welcome so the
    // next launch can resume without a pair token.  Also persist the server
    // URL we dialed so the service can relaunch with no arguments at all.
    // On resume welcomes the server does not re-emit agent_key; the files
    // stay as-is.
    if (!have_resume) {
        std::string fresh_key = json_peek_string(msg, "agent_key");
        if (!fresh_key.empty()) {
            bool ok = state_write("agent.key",    fresh_key) &&
                      state_write("agent.id",     agent_id)  &&
                      state_write("agent.server", server);
            if (!ok) {
                std::cerr << "[pair] WARN: could not persist agent_key to "
                          << state_dir() << " — restarts will re-pair\n";
            } else {
                std::cout << "[pair] persisted agent_key to " << state_dir() << "\n";
            }
        }
    }

    std::cout << "[pair] paired successfully; entering heartbeat loop\n";

    // ── ComfyUI capability advertisement ─────────────────────────────────
    //
    // Probe the local ComfyUI (DIST_COMFY_URL or default 127.0.0.1:8188).
    // If reachable, advertise comfy_caps so the control plane will route
    // image/video jobs here.  We never block the heartbeat loop on this —
    // a 1.5s timeout is fine; if ComfyUI is starting up we'll re-probe on
    // the first comfy_run.
    dist::ComfyClient comfy = dist::make_default_comfy_client();
    {
        const bool force = dist::comfy_force_enabled();
        dist::ComfyProbe probe = comfy.probe(force ? 3000 : 1500);
        if (probe.ok || force) {
            std::ostringstream caps;
            caps << "{\"kind\":\"comfy_caps\",\"ok\":true,"
                 << "\"version\":\"" << json_escape(probe.version) << "\","
                 << "\"models\":[";
            for (size_t i = 0; i < probe.models.size(); ++i) {
                if (i) caps << ",";
                caps << "\"" << json_escape(probe.models[i]) << "\"";
            }
            caps << "]}";
            ws.send_text(caps.str());
            std::cout << "[pair] comfy_caps ok="
                      << (probe.ok ? "1" : "force")
                      << " version=" << probe.version
                      << " models=" << probe.models.size() << "\n";
        } else {
            std::cout << "[pair] comfy probe failed: " << probe.error
                      << " (set DIST_WITH_COMFYUI=1 to advertise anyway)\n";
        }
    }

    // ── Diffusion-PP capability advertisement ────────────────────────────
    //
    // Probe the local Python runtime (the dpp_runtime package).  If the
    // module is importable we advertise dpp_caps so the control plane can
    // assign us a diffusion role (text_encoder / unet / vae) in a sharded
    // image-gen job.  Cheap probe — just `python -c "import
    // dpp_runtime.wire"`.  The full model load happens later, on first
    // dpp_route.
    dist::DppAdapter dpp;
    const std::string dpp_python_bin =
        std::getenv("DIST_PYTHON") ? std::getenv("DIST_PYTHON") : "python3";
    const std::string dpp_module_path = []() -> std::string {
        if (const char* env = std::getenv("DIST_DPP_PYTHONPATH")) return env;
        // Default: assume the package was installed alongside the binary
        // under <prefix>/python.  Fall back to a relative path useful in
        // dev checkouts.
        return "../python";
    }();
    {
        std::string dpp_err;
        bool dpp_ok = dist::DppAdapter::probe_local_caps(
            dpp_python_bin, dpp_module_path, dpp_err);
        std::ostringstream caps;
        caps << "{\"kind\":\"dpp_caps\","
             << "\"ok\":" << (dpp_ok ? "true" : "false") << ","
             << "\"roles\":[";
        if (dpp_ok) {
            caps << "\"text_encoder\",\"unet\",\"vae\"";
        }
        caps << "],"
             << "\"python\":\"" << json_escape(dpp_python_bin) << "\","
             << "\"module_path\":\"" << json_escape(dpp_module_path) << "\","
             << "\"error\":\"" << json_escape(dpp_err) << "\"}";
        ws.send_text(caps.str());
        std::cout << "[pair] dpp_caps ok=" << (dpp_ok ? "1" : "0")
                  << " err=" << dpp_err << "\n";
    }

    // Base64 (standard alphabet, no line breaks) — used to package binary
    // ComfyUI outputs into the JSON `comfy_result` frames the control plane
    // expects.  Keeps the WS protocol all-text for symmetry with the rest
    // of the agent-side messages.
    auto b64_encode = [](const std::vector<uint8_t>& in) -> std::string {
        static const char alpha[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((in.size() + 2) / 3) * 4);
        size_t i = 0;
        while (i + 3 <= in.size()) {
            uint32_t v = ((uint32_t)in[i] << 16) |
                         ((uint32_t)in[i+1] << 8) |
                          (uint32_t)in[i+2];
            out.push_back(alpha[(v >> 18) & 0x3F]);
            out.push_back(alpha[(v >> 12) & 0x3F]);
            out.push_back(alpha[(v >>  6) & 0x3F]);
            out.push_back(alpha[ v        & 0x3F]);
            i += 3;
        }
        if (i < in.size()) {
            uint32_t v = (uint32_t)in[i] << 16;
            if (i + 1 < in.size()) v |= (uint32_t)in[i+1] << 8;
            out.push_back(alpha[(v >> 18) & 0x3F]);
            out.push_back(alpha[(v >> 12) & 0x3F]);
            out.push_back(i + 1 < in.size() ? alpha[(v >> 6) & 0x3F] : '=');
            out.push_back('=');
        }
        return out;
    };

    // ── Cross-thread outbox ──────────────────────────────────────────────
    //
    // dist::WsClient::send_text is not internally serialized, so anything
    // not on the main heartbeat thread (today: just the ComfyUI worker)
    // pushes its frames into `comfy_outbox` and the main loop drains it.
    // Simple unbounded vector + mutex; the comfy thread is the only
    // producer and frames are small JSON envelopes.
    std::mutex comfy_outbox_mu;
    std::vector<std::string> comfy_outbox;
    auto drain_comfy_outbox = [&]() {
        std::vector<std::string> local;
        {
            std::lock_guard<std::mutex> lk(comfy_outbox_mu);
            if (comfy_outbox.empty()) return;
            local.swap(comfy_outbox);
        }
        for (auto& s : local) {
            if (!ws.is_open()) break;
            ws.send_text(s);
        }
    };

    // One pp_req per in-flight request assigned to this node.  Keyed on
    // req_id; populated on pp_route, mutated by ACTV frames, dropped when the
    // request finishes (terminal emits `done`, or an error short-circuits).
    struct pp_req {
        uint16_t req_id = 0;
        uint16_t stage_idx = 0;
        uint16_t stage_count = 0;
        bool is_first = false;
        bool is_last  = false;
        int32_t layer_lo = 0;
        int32_t layer_hi = 0;
        std::string shard_url;
        std::string shard_file;
        uint32_t max_tokens  = 128;
        int32_t  pos = 0;
        uint32_t out_tokens = 0;
        // Per-request seq id in the shared llama KV.  Assigned on first
        // pp_route; released on completion.
        int      seq_id = 0;
        // P2P / WebRTC ACTV transport (libdatachannel).  next_peer is set
        // when this stage has a successor — we initiate the WebRTC offer
        // toward stage+1.  prev_peer is set when this stage has a
        // predecessor — we wait for stage-1's offer.  Either may be null
        // (terminal stage / first stage / build without DIST_USE_P2P).
        // Forward ACTV frames go P2P-first, falling back to WS relay
        // whenever the channel isn't open yet.
        std::string session_id;
        std::string next_peer_id;
        std::string prev_peer_id;
        dist::ActvPeerPtr next_peer;
        dist::ActvPeerPtr prev_peer;
        // Counters for observability — how many forward ACTV frames we
        // emitted via P2P vs WS-relay since the request opened.
        uint64_t actv_sent_p2p   = 0;
        uint64_t actv_sent_relay = 0;
    };
    std::map<uint16_t, pp_req> reqs;

    // Peer-relay sessions for which this rig is acting as the relay.
    //
    // The planner picks a relay rig (cone/open NAT) between two stages
    // whose endpoints are behind symmetric NATs.  This rig opens two
    // PeerConnections — one to each end — and forwards every frame from
    // one channel onto the other.  We don't peek into the ACTV bytes
    // here; the relay is a dumb forwarder.
    struct relay_session {
        std::string session_id;
        std::string left_peer_id;
        std::string right_peer_id;
        dist::ActvPeerPtr left_peer;
        dist::ActvPeerPtr right_peer;
        // Counters for observability.
        uint64_t bytes_l2r = 0;
        uint64_t bytes_r2l = 0;
    };
    std::map<std::string, relay_session> relays;

    // Pool of seq_ids.  llama_batch_init(n_seq_max=1) from load() means we
    // can only use seq_id 0 today; the batched-decode path walks this pool.
    // For a real multi-tenant deployment bump n_seq_max at load().
    int next_seq_id = 0;

    // One engine cached across requests that share the same shard content.
    // Signed URLs change every request, so we key on (file, layer_lo, layer_hi)
    // and reuse the downloaded file + loaded model between requests.
    std::unique_ptr<dist::PpEngine> engine;
    std::string engine_key;   // "<file>#<lo>-<hi>"
    std::string engine_path;  // local path of the downloaded shard

    // Separate engine for single-rig INFR.  Its llama_context is configured
    // with embeddings=false / emit_logits=true so we can sample directly from
    // logits at the last position.  Coexists with the pipeline engine above.
    std::unique_ptr<dist::PpEngine> infer_engine;
    std::string infer_engine_key;
    std::string infer_engine_path;

    // Cross-process lock over device 0.  Any two dist-node processes sharing
    // one CUDA device serialise their decodes through this flock; a single
    // ggml CUDA backend cannot host two independent llama_contexts doing
    // concurrent decodes without corrupting each other's state.
    dist::GpuLock gpu_lock;
    if (n_gpu_layers > 0) {
        if (!gpu_lock.open(0)) {
            std::cerr << "[pair] gpu lock open failed: "
                      << gpu_lock.last_error() << "\n";
        }
    }

    // ACTV framing (reused by both the recv path and the batched flush).
    const uint32_t MAGIC_INFR = 0x494E4652u;
    const uint32_t MAGIC_ACTV = 0x41435456u;
    auto be32 = [](const uint8_t* p) -> uint32_t {
        return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
    };
    auto be16 = [](const uint8_t* p) -> uint16_t {
        return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
    };
    auto put32 = [](uint8_t* p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24);
        p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);
        p[3] = (uint8_t)v;
    };
    auto put16 = [](uint8_t* p, uint16_t v) {
        p[0] = (uint8_t)(v >> 8);
        p[1] = (uint8_t)v;
    };
    auto send_actv = [&](uint8_t type, uint16_t req_id, uint16_t stage,
                         uint32_t tok_seq, uint8_t dtype, uint8_t flags,
                         const std::vector<uint32_t>& dims,
                         const std::string& payload) {
        const size_t fixed = 20;
        size_t dims_bytes = 4 * dims.size();
        size_t total_hdr = fixed + dims_bytes + 4;
        std::vector<uint8_t> out(total_hdr + payload.size());
        put32(out.data() + 0, MAGIC_ACTV);
        out[4] = 0x01;
        out[5] = type;
        put16(out.data() + 6, req_id);
        put16(out.data() + 8, stage);
        put32(out.data() + 10, tok_seq);
        out[14] = dtype;
        out[15] = (uint8_t)dims.size();
        out[16] = flags;
        out[17] = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
            put32(out.data() + fixed + 4 * i, dims[i]);
        }
        put32(out.data() + fixed + dims_bytes, (uint32_t)payload.size());
        std::memcpy(out.data() + total_hdr, payload.data(), payload.size());

        // Forward ACTV (type 0x01) is the only frame kind that goes
        // stage→stage; prefer the WebRTC data channel when it's already
        // open, fall back to the WS relay otherwise.  Tokens (0x02),
        // done (0x03) and error (0x04) all flow back to the coordinator,
        // which is reachable only over the WS.
        bool sent_p2p = false;
        if (type == 0x01) {
            auto it = reqs.find(req_id);
            if (it != reqs.end() && it->second.next_peer &&
                dist::is_open(*it->second.next_peer)) {
                sent_p2p = dist::send_actv(
                    *it->second.next_peer,
                    reinterpret_cast<const std::byte*>(out.data()),
                    out.size());
                if (sent_p2p) {
                    if (it->second.actv_sent_p2p == 0) {
                        std::cout << "[p2p] req=" << req_id
                                  << " first ACTV sent over P2P\n";
                    }
                    ++it->second.actv_sent_p2p;
                }
            }
        }
        if (!sent_p2p) {
            ws.send_binary(out.data(), out.size());
            if (type == 0x01) {
                auto it = reqs.find(req_id);
                if (it != reqs.end()) ++it->second.actv_sent_relay;
            }
        }
    };

    // Rolling throughput — updated by flush_pending below, consumed by the
    // heartbeat.  Declared up here so the flush lambda can capture them by
    // reference.
    uint64_t dec_tokens = 0;
    auto     dec_window_start = std::chrono::steady_clock::now();

    // Pending ACTVs accumulate here between recv→drain cycles, then flush().
    struct PendingActv {
        uint8_t  type = 0;
        uint16_t req_id = 0;
        uint16_t stage_in = 0;
        uint32_t tok_seq = 0;
        uint8_t  dtype = 0;
        uint8_t  flags = 0;
        std::vector<uint32_t> dims;
        std::string payload;
    };
    std::vector<PendingActv> pending_actvs;

    // accept_actv_bytes parses a single ACTV binary frame (whether arrived
    // on the dist-server WS or on a peer-to-peer data channel) and queues
    // it for the next flush.  Returns true if the frame was consumed (or
    // intentionally dropped — e.g. DPP frames re-routed to the diffusion
    // adapter); false on malformed input.
    auto accept_actv_bytes = [&](const uint8_t* data, size_t n) -> bool {
        if (n < 20 || be32(data) != MAGIC_ACTV || data[4] != 0x01) return false;
        // DPP-flagged frames (latent / final / image) belong to the
        // diffusion adapter — hand them off and skip the LLM-PP path.
        constexpr uint8_t DPP_FLAGS = 0x08 | 0x10 | 0x20;
        if ((data[16] & DPP_FLAGS) != 0) {
            if (!dpp.dispatch_actv(data, n)) {
                std::cerr << "[dpp] no runtime for req=" << be16(data + 6)
                          << " (missing dpp_route?)\n";
            }
            return true;
        }
        PendingActv pa;
        pa.type     = data[5];
        pa.req_id   = be16(data + 6);
        pa.stage_in = be16(data + 8);
        pa.tok_seq  = be32(data + 10);
        pa.dtype    = data[14];
        uint8_t rank = data[15];
        pa.flags    = data[16];
        size_t hdr  = 20 + 4 * (size_t)rank;
        if (n < hdr + 4) return false;
        pa.dims.resize(rank);
        for (uint8_t i = 0; i < rank; ++i) {
            pa.dims[i] = be32(data + 20 + 4*i);
        }
        uint32_t payload_len = be32(data + hdr);
        if (n < hdr + 4 + payload_len) return false;
        pa.payload.assign((const char*)(data + hdr + 4), payload_len);
        if (pa.type == 0x01) {
            pending_actvs.push_back(std::move(pa));
        }
        return true;
    };

    // One shot: group pending ACTVs by (stage-kind), run one batched decode
    // per group, emit output frames.  Splits by stage because stage 0 uses
    // the token path while later stages use the embedding path.
    auto flush_pending = [&]() {
        if (pending_actvs.empty()) return;
        if (!engine || !engine->ready()) {
            for (auto & pa : pending_actvs) {
                send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, "engine not ready");
            }
            pending_actvs.clear();
            return;
        }

        // Bucket entries into three groups.  `idx` tracks the seqs[] slot
        // we hand to the engine so we can locate its output back here.
        struct Stage0Entry { size_t qi; std::vector<int32_t> toks; };
        struct StageNEntry { size_t qi; };
        std::vector<Stage0Entry> s0_entries;
        std::vector<StageNEntry> mid_entries;
        std::vector<StageNEntry> term_entries;
        std::vector<dist::PpEngine::TokenSeq>  s0_seqs;
        std::vector<dist::PpEngine::EmbedSeq>  mid_seqs;
        std::vector<dist::PpEngine::EmbedSeq>  term_seqs;

        for (size_t qi = 0; qi < pending_actvs.size(); ++qi) {
            auto & pa = pending_actvs[qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) {
                std::cout << "[pair] flush: no route for req=" << pa.req_id << " — drop\n";
                continue;
            }
            auto & r = it->second;
            const bool is_prompt = (pa.flags & 0x01) != 0;
            // Reset per-request KV on new prompt.
            if (is_prompt) {
                engine->reset_seq(r.seq_id);
                r.pos = 0;
                if (r.is_last) r.out_tokens = 0;
            }

            if (r.is_first) {
                std::vector<int32_t> toks = engine->tokenize(pa.payload, is_prompt);
                if (toks.empty()) {
                    send_actv(0x04, pa.req_id, r.stage_idx, 0, 4, 0, {},
                              "tokenize failed");
                    continue;
                }
                Stage0Entry e; e.qi = qi; e.toks = std::move(toks);
                s0_entries.push_back(std::move(e));
                dist::PpEngine::TokenSeq s;
                s.seq_id = r.seq_id;
                s.pos    = r.pos;
                s.tokens = s0_entries.back().toks.data();
                s.n      = (int32_t) s0_entries.back().toks.size();
                s0_seqs.push_back(std::move(s));
            } else {
                if (pa.dtype != 0) {
                    send_actv(0x04, pa.req_id, r.stage_idx, 0, 4, 0, {},
                              "non-f32 dtype to stage>0");
                    continue;
                }
                const int32_t n_in = (pa.dims.size() >= 2)
                    ? (int32_t) pa.dims[1]
                    : (int32_t)(pa.payload.size() / sizeof(float) / engine->n_embd());
                dist::PpEngine::EmbedSeq s;
                s.seq_id      = r.seq_id;
                s.pos         = r.pos;
                s.embd        = reinterpret_cast<const float*>(pa.payload.data());
                s.n           = n_in;
                s.want_logits = r.is_last;
                if (r.is_last) {
                    term_entries.push_back({qi});
                    term_seqs.push_back(std::move(s));
                } else {
                    mid_entries.push_back({qi});
                    mid_seqs.push_back(std::move(s));
                }
            }
        }

        // Dispatch each bucket in a single decode call.
        auto fail_bucket = [&](const std::vector<StageNEntry> & es,
                               const std::string & why) {
            for (auto & e : es) {
                auto & pa = pending_actvs[e.qi];
                send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, why);
            }
        };

        if (!s0_seqs.empty()) {
            if (!engine->decode_tokens_batched(s0_seqs)) {
                std::string why = engine->last_error();
                for (auto & e : s0_entries) {
                    auto & pa = pending_actvs[e.qi];
                    send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, why);
                }
                s0_entries.clear();
                s0_seqs.clear();
            }
        }
        if (!mid_seqs.empty()) {
            if (!engine->decode_embeddings_batched(mid_seqs)) {
                fail_bucket(mid_entries, engine->last_error());
                mid_entries.clear();
                mid_seqs.clear();
            }
        }
        if (!term_seqs.empty()) {
            if (!engine->decode_embeddings_batched(term_seqs)) {
                fail_bucket(term_entries, engine->last_error());
                term_entries.clear();
                term_seqs.clear();
            }
        }

        // Emit per-request outputs.
        for (size_t si = 0; si < s0_entries.size(); ++si) {
            auto & e = s0_entries[si];
            auto & seq = s0_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;
            const bool is_prompt = (pa.flags & 0x01) != 0;
            std::vector<uint32_t> odims = { (uint32_t) engine->n_embd(),
                                            (uint32_t) seq.n };
            std::string hpayload((const char*) seq.out.data(),
                                 seq.out.size() * sizeof(float));
            uint8_t fwd_flags = is_prompt ? (uint8_t)(0x01 | 0x04)
                                          : (uint8_t)(0x02);
            send_actv(0x01, pa.req_id, r.stage_idx,
                      pa.tok_seq, 0, fwd_flags, odims, hpayload);
        }
        for (size_t si = 0; si < mid_entries.size(); ++si) {
            auto & e = mid_entries[si];
            auto & seq = mid_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;
            std::vector<uint32_t> odims = { (uint32_t) engine->n_embd(),
                                            (uint32_t) seq.n };
            std::string hpayload((const char*) seq.out.data(),
                                 seq.out.size() * sizeof(float));
            send_actv(0x01, pa.req_id, r.stage_idx,
                      pa.tok_seq, 0, pa.flags, odims, hpayload);
        }
        for (size_t si = 0; si < term_entries.size(); ++si) {
            auto & e = term_entries[si];
            auto & seq = term_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;

            // Greedy sample over the emitted logits.
            int best = 0; float best_s = seq.out[0];
            for (size_t i = 1; i < seq.out.size(); ++i) {
                if (seq.out[i] > best_s) { best_s = seq.out[i]; best = (int) i; }
            }
            ++r.out_tokens;
            const int32_t eos = engine->eos_token();
            std::string piece = engine->detokenize(best);
            send_actv(0x02, pa.req_id, 0xFFFF, r.out_tokens, 0x04, 0, {}, piece);

            const bool hit_eos = (best == eos);
            const bool hit_cap = (r.out_tokens >= r.max_tokens);
            if (hit_eos || hit_cap) {
                send_actv(0x03, pa.req_id, 0xFFFF, r.out_tokens, 0, 0, {}, "");
                std::cout << "[p2p] req=" << pa.req_id
                          << " done actv_sent_p2p=" << r.actv_sent_p2p
                          << " actv_sent_relay=" << r.actv_sent_relay << "\n";
                // Release KV + bookkeeping for this request.
                engine->reset_seq(r.seq_id);
                reqs.erase(it);
            }
        }
        pending_actvs.clear();
    };

    // Process start clock — drives the "uptime" field in the heartbeat.
    auto process_start = std::chrono::steady_clock::now();
    // GPU info is queried at most every 10s and cached — the popen on
    // nvidia-smi takes ~15ms so we don't want it on every 5s beat.
    GpuInfo gpu_info_cache;
    auto    gpu_info_at = std::chrono::steady_clock::time_point{};

    // NAT-type probe — runs once at startup, blocking up to ~4s.  The
    // result feeds into the heartbeat so the server can pick this rig as
    // a peer relay candidate when applicable.  In stub builds (no P2P) it
    // returns "unknown" instantly.  The reflexive (public) IP we discover
    // here doubles as the auto-fill for --external-ip on the TURN sidecar:
    // rigs behind 1:1 NAT (EC2, GCP) need to advertise their public IPv4
    // in TURN allocations or peers can't reach them.
    std::string  nat_type      = "unknown";
    bool         relay_capable = false;
    std::string  public_ip;
    {
        std::vector<dist::IceServer> probe_ice = {
            {"stun:stun.l.google.com:19302", "", ""},
            {"stun:stun1.l.google.com:19302", "", ""},
        };
        auto np = dist::probe_nat_type(probe_ice, 4000);
        nat_type      = np.type;
        relay_capable = np.relay_capable;
        public_ip     = np.public_ip;
        std::cout << "[nat] type=" << nat_type
                  << " relay_capable=" << (relay_capable ? "yes" : "no");
        if (!public_ip.empty()) std::cout << " public_ip=" << public_ip;
        std::cout << "\n";
    }

    // ── TURN sidecar lifecycle ────────────────────────────────────────────
    // Driven from inside the heartbeat loop so we can react to:
    //   - secret/realm rotation on the next welcome (resume reconnects
    //     re-deliver these and they may have changed),
    //   - the sidecar dying on its own (OOM, port pulled by something else),
    //   - the operator toggling DIST_WITH_TURN by HUP'ing the agent (next
    //     iteration re-evaluates the env var).
    // The first iteration below performs the initial spawn.
    auto compute_turn_port = []() {
        int port = 3478;
        if (const char* env = std::getenv("DIST_TURN_PORT"); env && *env) {
            int v = std::atoi(env);
            if (v > 0 && v < 65536) port = v;
        }
        return port;
    };
    auto turn_opt_in = []() {
        if (const char* env = std::getenv("DIST_WITH_TURN"); env && *env) {
            return env[0] == '1' || env[0] == 't' || env[0] == 'T';
        }
        return false; // default off — operator must opt in
    };
    int  coturn_port = 0;
    bool coturn_announced_state = false; // forces an immediate heartbeat on
                                         // first transition (port becomes
                                         // available or goes away)
    auto reconcile_turn = [&]() {
        bool want = relay_capable && !turn_secret_from_srv.empty()
                  && turn_opt_in();
        int  port = compute_turn_port();
        pid_t pid = ensure_turn_sidecar(want, turn_secret_from_srv,
                                        turn_realm_from_srv, port, public_ip);
        int new_port = (pid > 0) ? port : 0;
        if (new_port != coturn_port) {
            coturn_port = new_port;
            coturn_announced_state = false; // force a heartbeat refresh
            if (coturn_port > 0) {
                std::cout << "[turn] sidecar up pid=" << pid
                          << " port=" << coturn_port
                          << " ext_ip=" << (public_ip.empty() ? "auto" : public_ip)
                          << "\n";
            } else {
                std::cout << "[turn] sidecar not running\n";
            }
        }
    };
    reconcile_turn();

    auto next_beat = std::chrono::steady_clock::now();
    auto next_turn_check = std::chrono::steady_clock::now()
                         + std::chrono::seconds(2);
    while (ws.is_open()) {
        auto now = std::chrono::steady_clock::now();
        // Liveness + reconcile loop for the TURN sidecar.  Cheap when the
        // sidecar is healthy (just a waitpid + a mutex-guarded arg-compare),
        // so we run it every ~2s.  Detects: sidecar crashed (clears
        // coturn_port so the next heartbeat tells the server to stop handing
        // out our URL), opt-in env toggled (spawn/teardown), public_ip
        // changed (we don't re-probe in-loop, but external override picks up
        // on respawn).
        if (now >= next_turn_check) {
            // turn_sidecar_alive() reaps on EOF; reconcile_turn() will
            // respawn if (want && !alive).
            (void) turn_sidecar_alive();
            reconcile_turn();
            next_turn_check = now + std::chrono::seconds(2);
        }
        // Force a heartbeat immediately when the sidecar transitions so the
        // server doesn't hand out a dead URL for a 5s window.
        if (!coturn_announced_state) {
            next_beat = now;
            coturn_announced_state = true;
        }
        if (now >= next_beat) {
            double secs = std::chrono::duration<double>(
                now - dec_window_start).count();
            double tps = (secs > 0.0) ? (double) dec_tokens / secs : 0.0;

            // Refresh GPU stats lazily.
            if (gpu_info_at.time_since_epoch().count() == 0 ||
                std::chrono::duration_cast<std::chrono::seconds>(
                    now - gpu_info_at).count() >= 10) {
                gpu_info_cache = query_gpu_info();
                gpu_info_at = now;
            }

            int64_t uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - process_start).count();
            int inflight = (int) reqs.size();

            std::ostringstream s;
            s << "{\"kind\":\"status\",\"n_gpus\":" << (n_gpu_layers > 0 ? 1 : 0)
              << ",\"tokens_sec\":" << (int) tps
              << ",\"uptime_sec\":" << uptime_sec
              << ",\"inflight\":" << inflight;
            if (gpu_info_cache.ok) {
                s << ",\"gpu_model\":\"" << json_escape(gpu_info_cache.name) << "\""
                  << ",\"vram_total\":" << gpu_info_cache.vram_total
                  << ",\"vram_free\":"  << gpu_info_cache.vram_free;
            }
            if (!nat_type.empty() && nat_type != "unknown") {
                s << ",\"nat_type\":\"" << nat_type << "\"";
            }
            if (relay_capable) {
                s << ",\"relay_capable\":true";
            }
            if (coturn_port > 0) {
                s << ",\"coturn_port\":" << coturn_port;
                // Public IP from our NAT probe (srflx).  The server prefers
                // this over the WS source IP when forming `turn:` URLs —
                // matters for rigs where TCP/443 (WS) and UDP/3478 (TURN)
                // exit via different NAT mappings (1:1 NAT, multi-homed
                // hosts, asymmetric routing).
                if (!public_ip.empty()) {
                    s << ",\"public_ip\":\"" << json_escape(public_ip) << "\"";
                }
            }
            // Roles held — coarse: we report which engine paths are loaded.
            // The swarm dashboard renders these as pills.
            std::vector<std::string> roles;
            if (engine && engine->ready())       roles.push_back("pp_engine");
            if (infer_engine && infer_engine->ready()) roles.push_back("infer");
            if (!roles.empty()) {
                s << ",\"roles\":[";
                for (size_t i = 0; i < roles.size(); ++i) {
                    if (i) s << ",";
                    s << "\"" << json_escape(roles[i]) << "\"";
                }
                s << "]";
            }
            // MaxConcurrent — how many in-flight /v1/chat or /api/infer
            // requests this rig will multiplex.  Today single-slot, since
            // our infer engine is not yet llama.cpp parallel-mode aware;
            // we still advertise it so the server's slot-aware dispatcher
            // sees the value and future builds can bump it to N without
            // a wire-protocol change.  Server treats missing/0 as 1.
            s << ",\"max_concurrent\":1";
            s << "}";
            if (!ws.send_text(s.str())) break;
            // Decay the window — count only the last 5s.
            dec_tokens = 0;
            dec_window_start = now;
            next_beat = now + std::chrono::seconds(5);
        }
        // Peek for server commands (short timeout so we can send heartbeats).
        ws.set_recv_timeout_ms(500);
        std::vector<uint8_t> msg;
        bool is_bin = false;
        if (ws.recv_message(msg, is_bin)) {
            if (is_bin) {
                auto MAGIC = MAGIC_INFR;
                if (msg.size() >= 16 && be32(msg.data()) == MAGIC
                        && msg[4] == 0x01 && msg[5] == 0x01) {
                    // Inference request frame.
                    uint16_t req_id     = be16(msg.data() + 6);
                    uint32_t in_tokens  = be32(msg.data() + 8);
                    uint32_t payload_len = be32(msg.data() + 12);
                    std::string payload;
                    if (msg.size() >= 16 + payload_len) {
                        payload.assign((const char*)(msg.data() + 16), payload_len);
                    }
                    std::cout << "[pair] infer req " << req_id
                              << " in_tokens=" << in_tokens
                              << " payload=" << payload << "\n";

                    auto send_chunk = [&](uint8_t kind, uint32_t tok_in,
                                          uint32_t tok_out, const std::string& text) {
                        std::vector<uint8_t> out(24 + text.size());
                        out[0]=0x49; out[1]=0x4E; out[2]=0x46; out[3]=0x52;
                        out[4] = 0x01;
                        out[5] = 0x02;
                        out[6] = (uint8_t)(req_id >> 8);
                        out[7] = (uint8_t)(req_id & 0xFF);
                        out[8] = kind;
                        out[9]=out[10]=out[11]=0;
                        auto put32lc = [&](size_t off, uint32_t v) {
                            out[off  ] = (uint8_t)(v >> 24);
                            out[off+1] = (uint8_t)(v >> 16);
                            out[off+2] = (uint8_t)(v >>  8);
                            out[off+3] = (uint8_t)(v);
                        };
                        put32lc(12, tok_in);
                        put32lc(16, tok_out);
                        put32lc(20, (uint32_t)text.size());
                        std::memcpy(out.data() + 24, text.data(), text.size());
                        ws.send_binary(out.data(), out.size());
                    };

                    // Minimal JSON field extractors.  The server-side payload
                    // is fully under our control (encoded by encoding/json),
                    // so we get away with positional parsing instead of a
                    // proper parser.
                    auto json_str = [&](const std::string & key) -> std::string {
                        std::string needle = std::string("\"") + key + "\":\"";
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return "";
                        size_t s = p + needle.size();
                        size_t e = s;
                        while (e < payload.size() && payload[e] != '"') {
                            if (payload[e] == '\\' && e + 1 < payload.size()) ++e;
                            ++e;
                        }
                        std::string raw = payload.substr(s, e - s);
                        std::string out; out.reserve(raw.size());
                        auto hexv = [](char c) -> int {
                            if (c >= '0' && c <= '9') return c - '0';
                            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                            return -1;
                        };
                        for (size_t i = 0; i < raw.size(); ++i) {
                            if (raw[i] == '\\' && i + 1 < raw.size()) {
                                char c = raw[i + 1];
                                if (c == '"' || c == '\\' || c == '/') { out += c; ++i; }
                                else if (c == 'n') { out += '\n'; ++i; }
                                else if (c == 'r') { out += '\r'; ++i; }
                                else if (c == 't') { out += '\t'; ++i; }
                                else if (c == 'u' && i + 5 < raw.size()) {
                                    int h0 = hexv(raw[i+2]);
                                    int h1 = hexv(raw[i+3]);
                                    int h2 = hexv(raw[i+4]);
                                    int h3 = hexv(raw[i+5]);
                                    if (h0 >= 0 && h1 >= 0 && h2 >= 0 && h3 >= 0) {
                                        unsigned cp = (h0 << 12) | (h1 << 8) | (h2 << 4) | h3;
                                        if (cp < 0x80) {
                                            out += (char) cp;
                                        } else if (cp < 0x800) {
                                            out += (char) (0xC0 | (cp >> 6));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        } else {
                                            out += (char) (0xE0 | (cp >> 12));
                                            out += (char) (0x80 | ((cp >> 6) & 0x3F));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        }
                                        i += 5;
                                    } else {
                                        out += c; ++i;
                                    }
                                }
                                else { out += c; ++i; }
                            } else {
                                out += raw[i];
                            }
                        }
                        return out;
                    };
                    auto json_int = [&](const std::string & key) -> long long {
                        std::string needle = std::string("\"") + key + "\":";
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return -1;
                        size_t s = p + needle.size();
                        while (s < payload.size() && payload[s] == ' ') ++s;
                        size_t e = s;
                        while (e < payload.size() && (isdigit((unsigned char)payload[e]) || payload[e] == '-')) ++e;
                        if (s == e) return -1;
                        try { return std::stoll(payload.substr(s, e - s)); } catch (...) { return -1; }
                    };
                    auto json_double = [&](const std::string & key, double fallback) -> double {
                        std::string needle = std::string("\"") + key + "\":";
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return fallback;
                        size_t s = p + needle.size();
                        while (s < payload.size() && payload[s] == ' ') ++s;
                        size_t e = s;
                        while (e < payload.size() &&
                               (isdigit((unsigned char)payload[e]) ||
                                payload[e] == '-' || payload[e] == '.' ||
                                payload[e] == 'e' || payload[e] == 'E' ||
                                payload[e] == '+')) ++e;
                        if (s == e) return fallback;
                        try { return std::stod(payload.substr(s, e - s)); } catch (...) { return fallback; }
                    };

                    std::string prompt     = json_str("prompt");
                    std::string shard_url  = json_str("shard_url");
                    std::string shard_file = json_str("shard_file");
                    long long   mt         = json_int("max_tokens");
                    uint32_t    max_tokens = (mt > 0) ? (uint32_t) mt : 128;
                    float       temperature = (float) json_double("temperature", 0.0);

                    if (prompt.empty()) {
                        send_chunk(2 /* error */, in_tokens, 0, "empty prompt");
                        continue;
                    }
                    if (shard_url.empty()) {
                        send_chunk(2, in_tokens, 0, "pool has no model bound");
                        continue;
                    }

                    // (Re)load the single-rig engine if the shard identity changed.
                    std::string key = shard_file + "#full";
                    bool ok = true;
                    if (!infer_engine || infer_engine_key != key) {
                        infer_engine.reset();
                        std::string dest = "/tmp/dist-node-" +
                            std::to_string(::getpid()) + "-infer-" +
                            (shard_file.empty() ? std::string("model.gguf") : shard_file);
                        std::string derr;
                        if (!dist::fetch_shard(shard_url, dest, derr)) {
                            std::cerr << "[pair] infer shard download failed: " << derr << "\n";
                            send_chunk(2, in_tokens, 0, "shard download failed: " + derr);
                            continue;
                        }
                        auto eng = std::make_unique<dist::PpEngine>();
                        dist::PpEngineConfig ecfg;
                        ecfg.shard_path   = dest;
                        ecfg.layer_lo     = 0;
                        ecfg.layer_hi     = 0; // full model
                        ecfg.n_ctx        = 4096;
                        ecfg.n_batch      = 512;
                        ecfg.n_seq_max    = 1;
                        ecfg.n_gpu_layers = n_gpu_layers;
                        ecfg.gpu_lock     = gpu_lock.enabled() ? &gpu_lock : nullptr;
                        ecfg.emit_logits  = true;
                        if (!eng->load(ecfg)) {
                            std::cerr << "[pair] infer engine load failed: "
                                      << eng->last_error() << "\n";
                            send_chunk(2, in_tokens, 0, "engine load failed: " + eng->last_error());
                            continue;
                        }
                        infer_engine      = std::move(eng);
                        infer_engine_key  = key;
                        infer_engine_path = dest;
                    }

                    // Fresh KV for this request.
                    infer_engine->reset_kv();
                    // Wrap as a single user turn using the gguf's chat template
                    // (no-op for base models without one).
                    std::string templated = infer_engine->apply_chat_template(prompt);
                    // parse_special=true so the chat template's <|im_start|> etc.
                    // map to the model's single special-token ids, not byte-level
                    // pieces (which would make the model echo the template back).
                    std::vector<int32_t> toks = infer_engine->tokenize(templated, true, true);
                    if (toks.empty()) {
                        send_chunk(2, in_tokens, 0, "tokenize failed");
                        continue;
                    }
                    uint32_t prompt_tokens = (uint32_t) toks.size();
                    std::cout << "[pair] infer prompt_tokens=" << prompt_tokens
                              << " max_tokens=" << max_tokens << "\n";

                    // Prompt prefill: one llama_decode over all prompt tokens,
                    // logits emitted at the final position.
                    std::vector<float> logits;
                    if (!infer_engine->decode_tokens_logits(toks.data(), (int32_t) toks.size(),
                                                            0, logits)) {
                        std::string why = infer_engine->last_error();
                        send_chunk(2, prompt_tokens, 0, "prefill decode failed: " + why);
                        continue;
                    }
                    int32_t pos = (int32_t) toks.size();
                    const int32_t eos = infer_engine->eos_token();

                    // ─── Parse full sampling param surface from the payload ──
                    // Defaults mirror llama.cpp's common defaults.
                    auto json_bool = [&](const std::string & key, bool fb) -> bool {
                        std::string needle = std::string("\"") + key + "\":";
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return fb;
                        size_t s = p + needle.size();
                        while (s < payload.size() && payload[s] == ' ') ++s;
                        if (payload.compare(s, 4, "true") == 0)  return true;
                        if (payload.compare(s, 5, "false") == 0) return false;
                        return fb;
                    };
                    // Pull a JSON array of strings into a vector.  Minimal parser:
                    // finds "key":[ ... ], reads each "..."-delimited element.
                    auto json_str_array = [&](const std::string & key) -> std::vector<std::string> {
                        std::vector<std::string> out;
                        std::string needle = std::string("\"") + key + "\":[";
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return out;
                        size_t i = p + needle.size();
                        while (i < payload.size() && payload[i] != ']') {
                            while (i < payload.size() && payload[i] != '"' && payload[i] != ']') ++i;
                            if (i >= payload.size() || payload[i] == ']') break;
                            ++i; // past opening quote
                            std::string raw;
                            while (i < payload.size() && payload[i] != '"') {
                                if (payload[i] == '\\' && i + 1 < payload.size()) {
                                    char c = payload[i+1];
                                    if (c == 'n') raw += '\n';
                                    else if (c == 'r') raw += '\r';
                                    else if (c == 't') raw += '\t';
                                    else raw += c;
                                    i += 2;
                                } else {
                                    raw += payload[i++];
                                }
                            }
                            out.push_back(std::move(raw));
                            if (i < payload.size()) ++i; // past closing quote
                            while (i < payload.size() && (payload[i] == ',' || payload[i] == ' ')) ++i;
                        }
                        return out;
                    };
                    // logit_bias: {"token_id": bias, ...}
                    auto json_logit_bias = [&]() -> std::vector<llama_logit_bias> {
                        std::vector<llama_logit_bias> out;
                        std::string needle = std::string("\"logit_bias\":{");
                        size_t p = payload.find(needle);
                        if (p == std::string::npos) return out;
                        size_t i = p + needle.size();
                        while (i < payload.size() && payload[i] != '}') {
                            while (i < payload.size() && payload[i] != '"' && payload[i] != '}') ++i;
                            if (i >= payload.size() || payload[i] == '}') break;
                            ++i;
                            size_t ks = i;
                            while (i < payload.size() && payload[i] != '"') ++i;
                            std::string ks_str = payload.substr(ks, i - ks);
                            if (i < payload.size()) ++i;
                            while (i < payload.size() && (payload[i] == ' ' || payload[i] == ':')) ++i;
                            size_t vs = i;
                            while (i < payload.size() && payload[i] != ',' && payload[i] != '}') ++i;
                            std::string vs_str = payload.substr(vs, i - vs);
                            try {
                                llama_logit_bias lb{};
                                lb.token = (llama_token) std::stoi(ks_str);
                                lb.bias  = std::stof(vs_str);
                                out.push_back(lb);
                            } catch (...) {}
                            while (i < payload.size() && (payload[i] == ',' || payload[i] == ' ')) ++i;
                        }
                        return out;
                    };

                    float    top_p              = (float)  json_double("top_p",            0.95);
                    int      top_k              = (int)    json_int("top_k");
                    if (top_k <= 0) top_k = 40;
                    float    min_p              = (float)  json_double("min_p",            0.05);
                    float    typical_p          = (float)  json_double("typical_p",        1.0);
                    float    repeat_penalty     = (float)  json_double("repeat_penalty",   1.0);
                    int      repeat_last_n      = (int)    json_double("repeat_last_n",    64);
                    float    freq_penalty       = (float)  json_double("frequency_penalty",0.0);
                    float    pres_penalty       = (float)  json_double("presence_penalty", 0.0);
                    float    dry_mult           = (float)  json_double("dry_multiplier",   0.0);
                    float    dry_base           = (float)  json_double("dry_base",         1.75);
                    int      dry_allowed_len    = (int)    json_double("dry_allowed_length", 2);
                    int      dry_penalty_last_n = (int)    json_double("dry_penalty_last_n", -1);
                    float    xtc_prob           = (float)  json_double("xtc_probability",  0.0);
                    float    xtc_thr            = (float)  json_double("xtc_threshold",    0.1);
                    int      mirostat_mode      = (int)    json_double("mirostat",         0);
                    float    mirostat_tau       = (float)  json_double("mirostat_tau",     5.0);
                    float    mirostat_eta       = (float)  json_double("mirostat_eta",     0.1);
                    float    dyn_range          = (float)  json_double("dynatemp_range",   0.0);
                    float    dyn_exp            = (float)  json_double("dynatemp_exponent",1.0);
                    long long seed_ll           =          json_int("seed");
                    bool     ignore_eos         =          json_bool("ignore_eos", false);
                    std::string grammar         =          json_str("grammar");
                    std::vector<std::string> stops = json_str_array("stop");
                    std::vector<llama_logit_bias> lbias = json_logit_bias();

                    uint32_t seed = (seed_ll > 0)
                        ? (uint32_t) seed_ll
                        : (uint32_t) std::chrono::steady_clock::now()
                                         .time_since_epoch().count();

                    // ─── Build the sampler chain in the canonical order ──
                    // logit_bias → penalties → dry → top_k → typical → top_p
                    //   → min_p → xtc → grammar → temp/temp_ext → (mirostat | dist)
                    auto sparams = llama_sampler_chain_default_params();
                    llama_sampler * smpl = llama_sampler_chain_init(sparams);

                    int32_t n_vocab_local = infer_engine->n_vocab();

                    if (temperature <= 0.0f && mirostat_mode == 0 && grammar.empty()) {
                        // Pure greedy.
                        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
                    } else {
                        if (!lbias.empty()) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
                                n_vocab_local, (int32_t) lbias.size(), lbias.data()));
                        }
                        if (repeat_penalty != 1.0f || freq_penalty != 0.0f || pres_penalty != 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                                repeat_last_n, repeat_penalty, freq_penalty, pres_penalty));
                        }
                        if (dry_mult > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                                infer_engine->vocab(),
                                infer_engine->n_ctx_train(),
                                dry_mult, dry_base, dry_allowed_len, dry_penalty_last_n,
                                /*seq_breakers=*/nullptr, /*num_breakers=*/0));
                        }
                        if (top_k > 0) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                        }
                        if (typical_p < 1.0f && typical_p > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_typical(typical_p, 1));
                        }
                        if (top_p < 1.0f && top_p > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
                        }
                        if (min_p > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
                        }
                        if (xtc_prob > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
                                xtc_prob, xtc_thr, 1, seed));
                        }
                        if (!grammar.empty()) {
                            llama_sampler * gs = llama_sampler_init_grammar(
                                infer_engine->vocab(), grammar.c_str(), "root");
                            if (gs) llama_sampler_chain_add(smpl, gs);
                        }
                        if (dyn_range > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(
                                temperature, dyn_range, dyn_exp));
                        } else if (temperature > 0.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
                        }
                        // Final selector: mirostat replaces dist, else dist.
                        if (mirostat_mode == 2) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
                                seed, mirostat_tau, mirostat_eta));
                        } else if (mirostat_mode == 1) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
                                n_vocab_local, seed, mirostat_tau, mirostat_eta, /*m=*/100));
                        } else {
                            llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
                        }
                    }

                    std::vector<llama_token_data> tdata((size_t) n_vocab_local);
                    auto sample_from = [&](const std::vector<float> & ls) -> int32_t {
                        for (int32_t i = 0; i < n_vocab_local; ++i) {
                            tdata[i] = { i, ls[(size_t) i], 0.0f };
                        }
                        // Optionally suppress EOS by sending its logit to -inf.
                        if (ignore_eos && eos >= 0 && eos < n_vocab_local) {
                            tdata[eos].logit = -std::numeric_limits<float>::infinity();
                        }
                        llama_token_data_array cur_p{
                            tdata.data(), (size_t) n_vocab_local, -1, false
                        };
                        llama_sampler_apply(smpl, &cur_p);
                        int32_t id = (cur_p.selected >= 0)
                            ? cur_p.data[cur_p.selected].id
                            : tdata[0].id;
                        llama_sampler_accept(smpl, id);
                        return id;
                    };

                    // Find max stop length so we can keep a sliding suffix.
                    size_t max_stop_len = 0;
                    for (const auto & s : stops) max_stop_len = std::max(max_stop_len, s.size());

                    std::string out_text;
                    uint32_t out_tokens = 0;
                    bool done = false;
                    while (out_tokens < max_tokens && !done && ws.is_open()) {
                        int32_t next = sample_from(logits);
                        ++out_tokens;
                        if (!ignore_eos && next == eos) {
                            done = true;
                            break;
                        }
                        std::string piece = infer_engine->detokenize(next);
                        send_chunk(0 /* token */, prompt_tokens, out_tokens, piece);
                        out_text += piece;

                        // Stop sequence detection.  Match against the recent tail
                        // (longest stop * 2 chars) so we catch boundary crossings.
                        if (!stops.empty()) {
                            size_t tail_lo = out_text.size() > 2 * max_stop_len
                                ? out_text.size() - 2 * max_stop_len : 0;
                            for (const auto & s : stops) {
                                if (!s.empty() && out_text.find(s, tail_lo) != std::string::npos) {
                                    done = true; break;
                                }
                            }
                            if (done) break;
                        }

                        // Decode the just-sampled token to get next-step logits.
                        int32_t one = next;
                        if (!infer_engine->decode_tokens_logits(&one, 1, pos, logits)) {
                            std::string why = infer_engine->last_error();
                            send_chunk(2, prompt_tokens, out_tokens, "decode failed: " + why);
                            done = true;
                            break;
                        }
                        ++pos;
                    }
                    llama_sampler_free(smpl);
                    (void) ok;
                    send_chunk(1 /* done */, prompt_tokens, out_tokens, "");
                } else if (msg.size() >= 20 && be32(msg.data()) == MAGIC_ACTV
                           && msg[4] == 0x01) {
                    if (!accept_actv_bytes(msg.data(), msg.size())) {
                        continue;
                    }
                } else {
                    // Unknown binary frame — echo (keeps M2 relay test working).
                    std::cout << "[pair] ← binary " << msg.size() << " bytes (echo)\n";
                    ws.send_binary(msg.data(), msg.size());
                }
            } else {
                std::string txt((const char*)msg.data(), msg.size());
                std::cout << "[pair] ← " << txt << "\n";

                // ── Peer-relay assignment ─────────────────────────────
                //
                // The server tells this rig "you are the relay between
                // <left> and <right> for session <S>".  We open two
                // PeerConnections (answerer to left, offerer to right)
                // and forward every byte each receives onto the other.
                // The ACTV format is opaque to us — we just shuffle
                // bytes between data channels.
                if (txt.find("\"kind\":\"p2p_relay_assign\"") != std::string::npos) {
                    std::string sess_id = json_peek_string(txt, "session_id");
                    std::string left_id  = json_peek_string(txt, "left_peer");
                    std::string right_id = json_peek_string(txt, "right_peer");
                    if (sess_id.empty() || left_id.empty() || right_id.empty()) {
                        std::cerr << "[relay] bad p2p_relay_assign: missing fields\n";
                        continue;
                    }
                    if (relays.count(sess_id)) {
                        // Duplicate assign for the same session — ignore.
                        continue;
                    }
                    // Build ICE list from the assign payload.  Same
                    // walker as the pp_route handler; we accept STUN/TURN
                    // entries and silently ignore "peer:..." (a relay
                    // doesn't recurse into another relay).
                    auto pull_str_in = [](const char* key, const std::string& body) -> std::string {
                        std::string k = std::string("\"") + key + "\":\"";
                        size_t p = body.find(k);
                        if (p == std::string::npos) return "";
                        size_t s = p + k.size();
                        size_t e = s;
                        while (e < body.size() && body[e] != '"') {
                            if (body[e] == '\\' && e + 1 < body.size()) ++e;
                            ++e;
                        }
                        return body.substr(s, e - s);
                    };
                    std::vector<dist::IceServer> ice;
                    {
                        size_t arr_p = txt.find("\"ice_servers\":[");
                        if (arr_p != std::string::npos) {
                            size_t p = arr_p + std::strlen("\"ice_servers\":[");
                            int depth = 1;
                            size_t end = p;
                            while (end < txt.size() && depth > 0) {
                                if (txt[end] == '[') ++depth;
                                else if (txt[end] == ']') --depth;
                                if (depth > 0) ++end;
                            }
                            std::string arr = txt.substr(p, end - p);
                            size_t i = 0;
                            while (i < arr.size()) {
                                size_t lb = arr.find('{', i);
                                if (lb == std::string::npos) break;
                                size_t rb = arr.find('}', lb);
                                if (rb == std::string::npos) break;
                                std::string obj = arr.substr(lb, rb - lb + 1);
                                dist::IceServer s;
                                s.url        = pull_str_in("urls", obj);
                                if (s.url.empty()) s.url = pull_str_in("url", obj);
                                if (s.url.rfind("peer:", 0) == 0) {
                                    i = rb + 1;
                                    continue;
                                }
                                s.username   = pull_str_in("username",   obj);
                                s.credential = pull_str_in("credential", obj);
                                if (!s.url.empty()) ice.push_back(std::move(s));
                                i = rb + 1;
                            }
                        }
                        if (ice.empty()) {
                            ice.push_back({"stun:stun.l.google.com:19302", "", ""});
                            ice.push_back({"stun:stun1.l.google.com:19302", "", ""});
                        }
                    }

                    auto signal_send = [&ws](const std::string& wire) {
                        ws.send_text(wire);
                    };

                    relay_session rs;
                    rs.session_id    = sess_id;
                    rs.left_peer_id  = left_id;
                    rs.right_peer_id = right_id;

                    // Insert empty entry first so the on_frame callbacks
                    // can locate it via session_id.  ActvPeerPtr fields
                    // are moved in below.
                    relays[sess_id] = std::move(rs);
                    std::string sess_key = sess_id;

                    auto on_open_left = [sess_key]() {
                        std::cout << "[relay] sess=" << sess_key
                                  << " left channel OPEN\n";
                    };
                    auto on_open_right = [sess_key]() {
                        std::cout << "[relay] sess=" << sess_key
                                  << " right channel OPEN\n";
                    };
                    auto on_frame_left = [&relays, sess_key](const std::byte* d, std::size_t n) {
                        auto it = relays.find(sess_key);
                        if (it == relays.end() || !it->second.right_peer) return;
                        if (!dist::send_actv(*it->second.right_peer, d, n)) {
                            // Right channel not open yet — drop.  ACTV is
                            // not retried at the relay layer; the originator
                            // falls back to WS relay if its peer doesn't
                            // ack the frame.
                            return;
                        }
                        it->second.bytes_l2r += n;
                    };
                    auto on_frame_right = [&relays, sess_key](const std::byte* d, std::size_t n) {
                        auto it = relays.find(sess_key);
                        if (it == relays.end() || !it->second.left_peer) return;
                        if (!dist::send_actv(*it->second.left_peer, d, n)) {
                            return;
                        }
                        it->second.bytes_r2l += n;
                    };

                    auto& cell = relays[sess_key];
                    // B is the answerer for A (left): A is offerer (lower
                    // stage index in the original plan), so B waits for
                    // A's offer.
                    cell.left_peer = dist::open_actv_peer(
                        sess_id, left_id, /*is_offerer=*/false,
                        ice, signal_send, on_open_left, on_frame_left);
                    // B is the offerer for C (right): C is answerer.
                    cell.right_peer = dist::open_actv_peer(
                        sess_id, right_id, /*is_offerer=*/true,
                        ice, signal_send, on_open_right, on_frame_right);

                    if (!cell.left_peer || !cell.right_peer) {
                        std::cerr << "[relay] failed to open both legs for sess="
                                  << sess_id << " (P2P not compiled?)\n";
                        relays.erase(sess_key);
                    } else {
                        std::cout << "[relay] sess=" << sess_id
                                  << " assigned left=" << left_id
                                  << " right=" << right_id << "\n";
                    }
                    continue;
                }

                // ── Peer-relay release ─────────────────────────────────
                if (txt.find("\"kind\":\"p2p_relay_release\"") != std::string::npos) {
                    std::string sess_id = json_peek_string(txt, "session_id");
                    auto it = relays.find(sess_id);
                    if (it != relays.end()) {
                        std::cout << "[relay] sess=" << sess_id
                                  << " release l2r=" << it->second.bytes_l2r
                                  << " r2l=" << it->second.bytes_r2l << "\n";
                        // Report stats back so the server can credit this
                        // session's byte count to our reputation row.  The
                        // server uses these as a tiebreaker between rigs
                        // with similar success rates.
                        std::ostringstream stats;
                        stats << "{\"kind\":\"relay_stats\","
                              << "\"session_id\":\"" << sess_id << "\","
                              << "\"bytes_l2r\":" << it->second.bytes_l2r << ","
                              << "\"bytes_r2l\":" << it->second.bytes_r2l << "}";
                        ws.send_text(stats.str());
                        if (it->second.left_peer)  dist::close_actv_peer(std::move(it->second.left_peer));
                        if (it->second.right_peer) dist::close_actv_peer(std::move(it->second.right_peer));
                        relays.erase(it);
                    }
                    continue;
                }

                // ── P2P signaling (offer / answer / ICE) ──────────────
                //
                // The server relays SDP/ICE exchanges between two rigs
                // that the planner paired up.  We don't need to look at
                // the body — actv_p2p::handle_signal parses the SDP /
                // candidate fields itself; our job is to pick the right
                // session and feed it the raw JSON.
                if (txt.find("\"kind\":\"p2p_offer\"")  != std::string::npos ||
                    txt.find("\"kind\":\"p2p_answer\"") != std::string::npos ||
                    txt.find("\"kind\":\"p2p_ice\"")    != std::string::npos) {
                    std::string sess_id = json_peek_string(txt, "session_id");
                    std::string from_id = json_peek_string(txt, "from");
                    std::string kind    = json_peek_string(txt, "kind");
                    if (!sess_id.empty()) {
                        // Relay sessions first — this rig might be a
                        // forwarder for the request that sess_id belongs to.
                        auto rit = relays.find(sess_id);
                        if (rit != relays.end()) {
                            if (rit->second.left_peer && rit->second.left_peer_id == from_id) {
                                dist::handle_signal(*rit->second.left_peer, kind, txt);
                                continue;
                            }
                            if (rit->second.right_peer && rit->second.right_peer_id == from_id) {
                                dist::handle_signal(*rit->second.right_peer, kind, txt);
                                continue;
                            }
                        }
                        for (auto& kv : reqs) {
                            auto& rr = kv.second;
                            if (rr.session_id != sess_id) continue;
                            if (rr.next_peer && rr.next_peer_id == from_id) {
                                dist::handle_signal(*rr.next_peer, kind, txt);
                                break;
                            }
                            if (rr.prev_peer && rr.prev_peer_id == from_id) {
                                dist::handle_signal(*rr.prev_peer, kind, txt);
                                break;
                            }
                        }
                    }
                    continue;
                }

                // Pipeline route assignment.
                if (txt.find("\"kind\":\"pp_route\"") != std::string::npos) {
                    auto pull_int = [&](const char* key) -> long long {
                        std::string k = std::string("\"") + key + "\":";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return -1;
                        size_t s = p + k.size();
                        while (s < txt.size() && txt[s] == ' ') ++s;
                        size_t e = s;
                        while (e < txt.size() && (isdigit((unsigned char)txt[e]) || txt[e]=='-')) ++e;
                        if (s == e) return -1;
                        try { return std::stoll(txt.substr(s, e - s)); } catch (...) { return -1; }
                    };
                    auto pull_bool = [&](const char* key) -> int {
                        std::string k = std::string("\"") + key + "\":";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return -1;
                        size_t s = p + k.size();
                        while (s < txt.size() && txt[s] == ' ') ++s;
                        if (txt.compare(s, 4, "true") == 0) return 1;
                        if (txt.compare(s, 5, "false") == 0) return 0;
                        return -1;
                    };
                    auto pull_str = [&](const char* key) -> std::string {
                        std::string k = std::string("\"") + key + "\":\"";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return "";
                        size_t s = p + k.size();
                        size_t e = s;
                        while (e < txt.size() && txt[e] != '"') {
                            if (txt[e] == '\\' && e + 1 < txt.size()) ++e;
                            ++e;
                        }
                        std::string raw = txt.substr(s, e - s);
                        std::string out; out.reserve(raw.size());
                        auto hex = [](char c) -> int {
                            if (c >= '0' && c <= '9') return c - '0';
                            if (c >= 'a' && c <= 'f') return 10 + c - 'a';
                            if (c >= 'A' && c <= 'F') return 10 + c - 'A';
                            return -1;
                        };
                        for (size_t i = 0; i < raw.size(); ++i) {
                            if (raw[i] == '\\' && i + 1 < raw.size()) {
                                char c = raw[i + 1];
                                if (c == '"' || c == '\\' || c == '/') { out += c; ++i; }
                                else if (c == 'n') { out += '\n'; ++i; }
                                else if (c == 'r') { out += '\r'; ++i; }
                                else if (c == 't') { out += '\t'; ++i; }
                                else if (c == 'u' && i + 5 < raw.size()) {
                                    int h0 = hex(raw[i+2]), h1 = hex(raw[i+3]);
                                    int h2 = hex(raw[i+4]), h3 = hex(raw[i+5]);
                                    if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
                                        out += raw[i]; // invalid — emit as-is
                                    } else {
                                        unsigned cp = (h0<<12)|(h1<<8)|(h2<<4)|h3;
                                        if (cp < 0x80) out += (char) cp;
                                        else if (cp < 0x800) {
                                            out += (char) (0xC0 | (cp >> 6));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        } else {
                                            out += (char) (0xE0 | (cp >> 12));
                                            out += (char) (0x80 | ((cp >> 6) & 0x3F));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        }
                                        i += 5;
                                    }
                                } else { out += c; ++i; }
                            } else {
                                out += raw[i];
                            }
                        }
                        return out;
                    };
                    long long rid = pull_int("req_id");
                    long long si  = pull_int("stage_idx");
                    long long sc  = pull_int("stage_count");
                    long long lo  = pull_int("layer_lo");
                    long long hi  = pull_int("layer_hi");
                    long long mt  = pull_int("max_tokens");
                    int is_first = pull_bool("is_first");
                    int is_last  = pull_bool("is_last");
                    std::string shard_url  = pull_str("shard_url");
                    std::string shard_file = pull_str("shard_file");
                    if (rid >= 0 && si >= 0 && sc > 0) {
                        pp_req r;
                        r.req_id     = (uint16_t)rid;
                        r.stage_idx  = (uint16_t)si;
                        r.stage_count= (uint16_t)sc;
                        r.is_first   = is_first == 1;
                        r.is_last    = is_last == 1;
                        r.layer_lo   = (int32_t)(lo >= 0 ? lo : 0);
                        r.layer_hi   = (int32_t)(hi >= 0 ? hi : 0);
                        r.shard_url  = shard_url;
                        r.shard_file = shard_file;
                        r.max_tokens = (mt > 0) ? (uint32_t) mt : 128;
                        r.pos        = 0;
                        r.out_tokens = 0;
                        // Assign a fresh seq_id; wraps in [0, n_seq_max).
                        // Stored on the request; released in on_request_done.
                        r.seq_id     = next_seq_id;
                        next_seq_id  = (next_seq_id + 1) % 8;

                        // Load the engine for this shard (reused across
                        // requests).  Signed URLs rotate every request, so
                        // we key the cache on content identity instead.
                        bool engine_ok = true;
                        std::string key = (shard_file.empty() ? shard_url : shard_file) +
                                          "#" + std::to_string(r.layer_lo) +
                                          "-" + std::to_string(r.layer_hi);
                        if (!engine || engine_key != key) {
                            engine.reset();
                            std::string dest = "/tmp/dist-node-" +
                                std::to_string(::getpid()) + "-" +
                                (shard_file.empty()
                                    ? ("shard-" + std::to_string(rid) + ".gguf")
                                    : shard_file);
                            std::string derr;
                            if (shard_url.empty()) {
                                std::cerr << "[pair] pp_route missing shard_url\n";
                                engine_ok = false;
                            } else if (!dist::fetch_shard(shard_url, dest, derr)) {
                                std::cerr << "[pair] shard download failed: "
                                          << derr << "\n";
                                engine_ok = false;
                            } else {
                                auto eng = std::make_unique<dist::PpEngine>();
                                dist::PpEngineConfig ecfg;
                                ecfg.shard_path = dest;
                                ecfg.layer_lo   = r.layer_lo;
                                ecfg.layer_hi   = r.layer_hi;
                                ecfg.n_ctx      = 2048;
                                ecfg.n_batch    = 512;
                                ecfg.n_seq_max  = 8;
                                ecfg.n_gpu_layers = n_gpu_layers;
                                ecfg.gpu_lock     = gpu_lock.enabled() ? &gpu_lock : nullptr;
                                if (!eng->load(ecfg)) {
                                    std::cerr << "[pair] engine load failed: "
                                              << eng->last_error() << "\n";
                                    engine_ok = false;
                                } else {
                                    engine = std::move(eng);
                                    engine_key  = key;
                                    engine_path = dest;
                                }
                            }
                        }
                        // Clear this seq's KV so its fresh sequence starts
                        // at pos 0 (other seqs in the same context are left
                        // untouched — they may belong to concurrent requests).
                        if (engine_ok) {
                            engine->reset_seq(r.seq_id);
                        }
                        if (engine_ok) {
                            // P2P ACTV bring-up: parse the peer ids and ICE
                            // hints the planner sent and kick off the
                            // libdatachannel handshake.  When DIST_USE_P2P is
                            // off at build time these calls return null and
                            // we silently stay on the WS relay path.
                            r.session_id   = pull_str("session_id");
                            r.next_peer_id = pull_str("next_peer_id");
                            r.prev_peer_id = pull_str("prev_peer_id");

                            // ICE servers come from the planner's pp_route
                            // frame in the "ice_servers" array.  We hand-roll
                            // a small parser because the array entries are
                            // shallow objects of the form
                            //   {"urls":"...","username":"...","credential":"..."}
                            // and dragging in a real JSON dep here would be
                            // overkill.
                            //
                            // We fall back to baked-in public STUN if the
                            // server didn't ship any — keeps dev-mode and
                            // older servers working.
                            // Minimal in-string extractor: pulls the value of
                            // "key":"..." inside a given string.  We don't
                            // need full JSON unescape here because urls /
                            // usernames / credentials never contain control
                            // chars or unicode escapes in our wire format.
                            auto pull_str_in = [](const char* key, const std::string& body) -> std::string {
                                std::string k = std::string("\"") + key + "\":\"";
                                size_t p = body.find(k);
                                if (p == std::string::npos) return "";
                                size_t s = p + k.size();
                                size_t e = s;
                                while (e < body.size() && body[e] != '"') {
                                    if (body[e] == '\\' && e + 1 < body.size()) ++e;
                                    ++e;
                                }
                                std::string raw = body.substr(s, e - s);
                                // Unescape the handful of sequences we may see.
                                std::string out; out.reserve(raw.size());
                                for (size_t i = 0; i < raw.size(); ++i) {
                                    if (raw[i] == '\\' && i + 1 < raw.size()) {
                                        char c = raw[i + 1];
                                        if (c == '"' || c == '\\' || c == '/') { out += c; ++i; }
                                        else if (c == 'n') { out += '\n'; ++i; }
                                        else if (c == 'r') { out += '\r'; ++i; }
                                        else if (c == 't') { out += '\t'; ++i; }
                                        else { out += c; ++i; }
                                    } else {
                                        out += raw[i];
                                    }
                                }
                                return out;
                            };
                            std::vector<dist::IceServer> ice;
                            {
                                size_t arr_p = txt.find("\"ice_servers\":[");
                                if (arr_p != std::string::npos) {
                                    size_t p = arr_p + std::strlen("\"ice_servers\":[");
                                    int depth = 1;
                                    size_t end = p;
                                    while (end < txt.size() && depth > 0) {
                                        if (txt[end] == '[') ++depth;
                                        else if (txt[end] == ']') --depth;
                                        if (depth > 0) ++end;
                                    }
                                    std::string arr = txt.substr(p, end - p);
                                    size_t i = 0;
                                    while (i < arr.size()) {
                                        size_t lb = arr.find('{', i);
                                        if (lb == std::string::npos) break;
                                        size_t rb = arr.find('}', lb);
                                        if (rb == std::string::npos) break;
                                        std::string obj = arr.substr(lb, rb - lb + 1);
                                        dist::IceServer s;
                                        s.url        = pull_str_in("urls",       obj);
                                        if (s.url.empty()) s.url = pull_str_in("url", obj);
                                        s.username   = pull_str_in("username",   obj);
                                        s.credential = pull_str_in("credential", obj);
                                        if (!s.url.empty()) ice.push_back(std::move(s));
                                        i = rb + 1;
                                    }
                                }
                                if (ice.empty()) {
                                    ice.push_back({"stun:stun.l.google.com:19302", "", ""});
                                    ice.push_back({"stun:stun1.l.google.com:19302", "", ""});
                                }
                            }

                            auto signal_send = [&ws](const std::string& wire) {
                                ws.send_text(wire);
                            };
                            uint16_t this_req = r.req_id;

                            // For the on_frame callback we need to feed
                            // peer-arrived ACTV bytes through the same
                            // pipeline as WS-arrived ones.
                            auto on_frame = [&accept_actv_bytes](const std::byte* d, std::size_t n) {
                                accept_actv_bytes(
                                    reinterpret_cast<const uint8_t*>(d), n);
                            };

                            if (!r.next_peer_id.empty() && !r.is_last) {
                                r.next_peer = dist::open_actv_peer(
                                    r.session_id, r.next_peer_id,
                                    /*is_offerer=*/true,
                                    ice,
                                    signal_send,
                                    [this_req](){
                                        std::cout << "[p2p] req=" << this_req
                                                  << " forward channel OPEN\n";
                                    },
                                    on_frame);
                            }
                            if (!r.prev_peer_id.empty() && !r.is_first) {
                                r.prev_peer = dist::open_actv_peer(
                                    r.session_id, r.prev_peer_id,
                                    /*is_offerer=*/false,
                                    ice,
                                    signal_send,
                                    [this_req](){
                                        std::cout << "[p2p] req=" << this_req
                                                  << " reverse channel OPEN\n";
                                    },
                                    on_frame);
                            }

                            // unique_ptr inside pp_req → must move-insert.
                            reqs[r.req_id] = std::move(r);
                        }
                        std::cout << "[pair] pp_route accepted req=" << rid
                                  << " stage=" << si
                                  << "/" << sc
                                  << " seq=" << r.seq_id
                                  << " layers=[" << r.layer_lo << "," << r.layer_hi << ")"
                                  << " first=" << r.is_first
                                  << " last=" << r.is_last
                                  << " engine=" << (engine_ok ? "ready" : "FAILED")
                                  << "\n";
                    }
                }
                // ── ComfyUI run dispatch ──────────────────────────────────
                //
                // Control plane sends:
                //   {"kind":"comfy_run","job_id":N,"workflow":"image|video","graph":"..."}
                //
                // The `graph` field is a JSON-encoded string holding the
                // ComfyUI graph object.  We submit it to the local ComfyUI
                // and stream each produced file back as a `comfy_result`
                // frame; on completion (success or failure) we always emit
                // a final frame so the server side can release its wait.
                if (txt.find("\"kind\":\"comfy_run\"") != std::string::npos) {
                    auto pull_int = [&](const char* key) -> long long {
                        std::string k = std::string("\"") + key + "\":";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return -1;
                        size_t s = p + k.size();
                        while (s < txt.size() && txt[s] == ' ') ++s;
                        size_t e = s;
                        while (e < txt.size() && (isdigit((unsigned char)txt[e]) || txt[e]=='-')) ++e;
                        if (s == e) return -1;
                        try { return std::stoll(txt.substr(s, e - s)); } catch (...) { return -1; }
                    };
                    auto pull_str = [&](const char* key) -> std::string {
                        // Quoted string extractor with \" escape handling.
                        std::string k = std::string("\"") + key + "\":\"";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return "";
                        size_t s = p + k.size();
                        size_t e = s;
                        while (e < txt.size() && txt[e] != '"') {
                            if (txt[e] == '\\' && e + 1 < txt.size()) e += 2;
                            else ++e;
                        }
                        std::string raw = txt.substr(s, e - s);
                        std::string out; out.reserve(raw.size());
                        for (size_t i = 0; i < raw.size(); ++i) {
                            if (raw[i] == '\\' && i + 1 < raw.size()) {
                                char c = raw[i + 1];
                                if      (c == '"')  { out += '"';  ++i; }
                                else if (c == '\\') { out += '\\'; ++i; }
                                else if (c == '/')  { out += '/';  ++i; }
                                else if (c == 'n')  { out += '\n'; ++i; }
                                else if (c == 'r')  { out += '\r'; ++i; }
                                else if (c == 't')  { out += '\t'; ++i; }
                                else { out += c; ++i; }
                            } else out += raw[i];
                        }
                        return out;
                    };
                    long long job_id = pull_int("job_id");
                    std::string graph = pull_str("graph");
                    std::cout << "[pair] comfy_run job=" << job_id
                              << " graph=" << graph.size() << "B\n";

                    // Run on a worker thread so the WS loop stays responsive
                    // (status heartbeats, llama inference, etc.).  The thread
                    // captures everything by value it needs and writes back
                    // via ws.send_text which is safe from any thread on the
                    // dist::WsClient implementation.
                    std::thread([&comfy, &b64_encode,
                                 &comfy_outbox, &comfy_outbox_mu,
                                 job_id, graph]() {
                        auto send_result = [&](const std::string& fname,
                                               const std::string& b64,
                                               bool final,
                                               const std::string& err) {
                            std::ostringstream o;
                            o << "{\"kind\":\"comfy_result\","
                              << "\"job_id\":" << job_id;
                            if (!fname.empty()) {
                                o << ",\"file\":\"" << json_escape(fname) << "\"";
                            }
                            if (!b64.empty()) {
                                o << ",\"data_b64\":\"" << b64 << "\"";
                            }
                            if (final) o << ",\"final\":true";
                            if (!err.empty()) {
                                o << ",\"error\":\"" << json_escape(err) << "\"";
                            }
                            o << "}";
                            std::lock_guard<std::mutex> lk(comfy_outbox_mu);
                            comfy_outbox.push_back(o.str());
                        };

                        if (graph.empty()) {
                            send_result("", "", true, "empty graph");
                            return;
                        }
                        std::string err = comfy.run(graph, 10 * 60 * 1000,
                            [&](const dist::ComfyResult& r) {
                                std::string b64 = b64_encode(r.data);
                                send_result(r.filename, b64, false, "");
                                return true;
                            });
                        send_result("", "", true, err);
                    }).detach();
                }

                // Diffusion pipeline-parallel route assignment.  Hands the
                // control payload to the dpp adapter, which lazily spawns
                // a Python runtime for (role, model) if needed.
                if (txt.find("\"kind\":\"dpp_route\"") != std::string::npos) {
                    std::string err;
                    if (!dpp.handle_dpp_route(txt, dpp_python_bin,
                                              dpp_module_path, err)) {
                        std::cerr << "[dpp] route rejected: " << err << "\n";
                        std::ostringstream e;
                        e << "{\"kind\":\"dpp_error\",\"message\":\""
                          << json_escape(err) << "\"}";
                        ws.send_text(e.str());
                    } else {
                        std::cout << "[dpp] route accepted\n";
                    }
                }

                // Minimal signalling stub: if the server asks us for a WebRTC
                // answer, respond with signal_error so the client falls back to
                // the relay.  A real agent would terminate the PeerConnection.
                if (txt.find("\"kind\":\"signal\"") != std::string::npos) {
                    size_t p = txt.find("\"req_id\":");
                    if (p != std::string::npos) {
                        size_t start = p + 9;
                        size_t end = start;
                        while (end < txt.size() && (isdigit((unsigned char)txt[end]) || txt[end]=='-')) ++end;
                        std::string idstr = txt.substr(start, end - start);
                        std::ostringstream reply;
                        reply << "{\"kind\":\"signal_error\",\"req_id\":" << idstr
                              << ",\"message\":\"p2p not supported yet\"}";
                        ws.send_text(reply.str());
                    }
                }
            }
        }

        // Non-blocking drain: pull any additional frames already buffered.
        // This gives the flush step a chance to combine back-to-back ACTVs
        // (e.g. two concurrent requests arriving inside one scheduler tick).
        while (ws.is_open()) {
            ws.set_recv_timeout_ms(0);
            std::vector<uint8_t> m2;
            bool b2 = false;
            if (!ws.recv_message(m2, b2)) break;
            if (b2 && m2.size() >= 20 && be32(m2.data()) == MAGIC_ACTV && m2[4] == 0x01) {
                constexpr uint8_t DPP_FLAGS = 0x08 | 0x10 | 0x20;
                if ((m2[16] & DPP_FLAGS) != 0) {
                    dpp.dispatch_actv(m2.data(), m2.size());
                    continue;
                }
                PendingActv pa;
                pa.type     = m2[5];
                pa.req_id   = be16(m2.data() + 6);
                pa.stage_in = be16(m2.data() + 8);
                pa.tok_seq  = be32(m2.data() + 10);
                pa.dtype    = m2[14];
                uint8_t rank = m2[15];
                pa.flags    = m2[16];
                size_t hdr  = 20 + 4 * (size_t)rank;
                pa.dims.resize(rank);
                for (uint8_t i = 0; i < rank; ++i) {
                    pa.dims[i] = be32(m2.data() + 20 + 4*i);
                }
                if (m2.size() < hdr + 4) continue;
                uint32_t payload_len = be32(m2.data() + hdr);
                if (m2.size() >= hdr + 4 + payload_len) {
                    pa.payload.assign((const char*)(m2.data() + hdr + 4), payload_len);
                }
                if (pa.type == 0x01) {
                    pending_actvs.push_back(std::move(pa));
                }
                continue;
            }
            if (!b2) {
                // Text frame consumed off the wire during drain. Once
                // recv_message returns it, the next blocking recv won't see
                // it again, so we must handle dpp_route here too — otherwise
                // back-to-back routes (one per stage) get silently dropped.
                std::string txt((const char*)m2.data(), m2.size());
                std::cout << "[pair] ← " << txt << "\n";
                if (txt.find("\"kind\":\"dpp_route\"") != std::string::npos) {
                    std::string err;
                    if (!dpp.handle_dpp_route(txt, dpp_python_bin,
                                              dpp_module_path, err)) {
                        std::cerr << "[dpp] route rejected: " << err << "\n";
                        std::ostringstream e;
                        e << "{\"kind\":\"dpp_error\",\"message\":\""
                          << json_escape(err) << "\"}";
                        ws.send_text(e.str());
                    } else {
                        std::cout << "[dpp] route accepted\n";
                    }
                }
            }
        }

        flush_pending();
        drain_comfy_outbox();

        // Drain anything the diffusion runtime has produced.  These are
        // already encoded ACTV frames — ship them straight to the WS.
        {
            auto out = dpp.drain_outbox();
            for (auto& f : out) {
                if (!ws.is_open()) break;
                ws.send_binary(f.bytes.data(), f.bytes.size());
            }
        }
    }
    std::cerr << "[pair] connection closed: " << ws.last_error() << "\n";
    return 0;
}

// ─── Subcommand helpers ─────────────────────────────────────────────────────

// Default control-plane URL.  Picked once at install time via `dist-node login`,
// persisted as agent.api_url in state_dir.  Falls back to the prod ACA FQDN.
static std::string default_api_url() {
    if (const char* s = std::getenv("DIST_API_URL"); s && *s) return s;
    std::string saved = state_read("agent.api_url");
    if (!saved.empty()) return saved;
    return "https://distpool-server.gentlegrass-360d3389.centralindia.azurecontainerapps.io";
}

// Convert an https://host/… URL to a wss://host/ws/agent URL for the WS resume.
static std::string ws_url_from_api(const std::string& api_url) {
    if (api_url.rfind("https://", 0) == 0) return "wss://" + api_url.substr(8) + "/ws/agent";
    if (api_url.rfind("http://",  0) == 0) return "ws://"  + api_url.substr(7) + "/ws/agent";
    return api_url;
}

// Try to open `url` in the user's browser.  Best-effort; fails silently.
static void open_browser(const std::string& url) {
#if defined(__APPLE__)
    std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
#elif defined(_WIN32)
    std::string cmd = "start \"\" \"" + url + "\"";
#else
    std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
    (void)std::system(cmd.c_str());
}

static std::string pid_file_path() { return state_path("dist-node.pid"); }

static int cmd_login(int argc, char** argv) {
    std::string api_url;
    int n_gpus = 0; int64_t vram_bytes = 0;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-s" || a == "--server") && i + 1 < argc) api_url = argv[++i];
        else if (a == "--n-gpus" && i + 1 < argc)           n_gpus = std::atoi(argv[++i]);
        else if (a == "--vram"   && i + 1 < argc)           vram_bytes = std::atoll(argv[++i]);
    }
    if (api_url.empty()) api_url = default_api_url();

    char hostname[256] = {};
    ::gethostname(hostname, sizeof(hostname));

    std::ostringstream body;
    body << "{\"hostname\":\""    << json_escape(hostname) << "\","
         << "\"n_gpus\":"          << n_gpus               << ","
         << "\"vram_bytes\":"      << vram_bytes           << "}";

    HttpResp rsp; std::string err;
    if (!http_request(api_url, "/api/device/code", "POST", body.str(),
                      {}, rsp, err) || rsp.status != 200) {
        std::cerr << "[login] mint device code failed: " << err
                  << " status=" << rsp.status << " body=" << rsp.body << "\n";
        return 1;
    }
    std::string device_code = json_peek_string(rsp.body, "device_code");
    std::string user_code   = json_peek_string(rsp.body, "user_code");
    std::string verif       = json_peek_string(rsp.body, "verification_url");
    std::string verif_full  = json_peek_string(rsp.body, "verification_url_complete");
    if (device_code.empty() || user_code.empty() || verif_full.empty()) {
        std::cerr << "[login] malformed response: " << rsp.body << "\n";
        return 1;
    }

    std::cout << "\n  ┌──────────────────────────────────────────────────────┐\n"
              << "  │  Visit:  " << verif      << std::string(std::max<int>(0, 41 - (int)verif.size()), ' ') << " │\n"
              << "  │  Code:   " << user_code  << std::string(std::max<int>(0, 41 - (int)user_code.size()), ' ') << " │\n"
              << "  └──────────────────────────────────────────────────────┘\n\n"
              << "  Opening browser… (if it doesn't open, paste the URL above)\n\n";
    open_browser(verif_full);

    // Poll /api/device/token every 3s until approved or expired.
    std::ostringstream pollbody;
    pollbody << "{\"device_code\":\"" << json_escape(device_code) << "\"}";
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        HttpResp pr; std::string perr;
        if (!http_request(api_url, "/api/device/token", "POST", pollbody.str(),
                          {}, pr, perr)) {
            std::cerr << "[login] poll error: " << perr << " — retrying\n";
            continue;
        }
        if (pr.status == 428) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "  Waiting for approval… (" << el << "s)\r" << std::flush;
            continue;
        }
        if (pr.status == 410) {
            std::cerr << "\n[login] code expired — run `dist-node login` again\n";
            return 1;
        }
        if (pr.status != 200) {
            std::cerr << "\n[login] poll failed: status=" << pr.status
                      << " body=" << pr.body << "\n";
            return 1;
        }
        std::string agent_id  = json_peek_string(pr.body, "agent_id");
        std::string agent_key = json_peek_string(pr.body, "agent_key");
        std::string server    = json_peek_string(pr.body, "server");
        if (agent_id.empty() || agent_key.empty() || server.empty()) {
            std::cerr << "\n[login] malformed token reply: " << pr.body << "\n";
            return 1;
        }
        bool ok = state_write("agent.id",      agent_id)
               && state_write("agent.key",     agent_key)
               && state_write("agent.server",  server)
               && state_write("agent.api_url", api_url);
        if (!ok) {
            std::cerr << "\n[login] could not persist to " << state_dir() << "\n";
            return 1;
        }
        std::cout << "\n  ✓ Logged in. agent_id=" << agent_id
                  << "\n    Saved to " << state_dir()
                  << "\n    Run `dist-node connect` to join the pool.\n\n";
        return 0;
    }
}

static int cmd_connect(int argc, char** argv) {
    bool daemonize = false;
    int n_gpu_layers = 999;
    std::string id_override;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--daemon" || a == "-D")             daemonize = true;
        else if (a == "-g" && i + 1 < argc)            n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--id" && i + 1 < argc)          id_override = argv[++i];
    }
    std::string srv = state_read("agent.server");
    std::string key = state_read("agent.key");
    if (srv.empty() || key.empty()) {
        std::cerr << "[connect] no saved login — run `dist-node login` first\n";
        return 1;
    }
#ifndef _WIN32
    if (daemonize) {
        pid_t pid = fork();
        if (pid < 0) { std::cerr << "fork failed\n"; return 1; }
        if (pid > 0) {
            // Parent: write PID file and exit.
            std::ofstream f(pid_file_path()); f << pid; f.close();
            std::cout << "[connect] daemon pid=" << pid << " (saved to "
                      << pid_file_path() << ")\n"
                      << "[connect] Use `dist-node disconnect` to stop.\n";
            return 0;
        }
        // Child continues.
        ::setsid();
    }
#else
    (void)daemonize;
#endif
    return run_pair_mode(std::string(), srv, id_override, n_gpu_layers);
}

static int cmd_disconnect() {
#ifdef _WIN32
    std::cerr << "[disconnect] not supported on Windows yet\n";
    return 1;
#else
    std::ifstream f(pid_file_path());
    if (!f.good()) {
        std::cerr << "[disconnect] no PID file at " << pid_file_path()
                  << " — is the daemon running?\n";
        return 1;
    }
    pid_t pid = 0; f >> pid;
    if (pid <= 0) {
        std::cerr << "[disconnect] bad PID file\n";
        return 1;
    }
    if (::kill(pid, SIGTERM) != 0) {
        std::cerr << "[disconnect] kill " << pid << " failed: " << std::strerror(errno) << "\n";
        std::error_code ec; std::filesystem::remove(pid_file_path(), ec);
        return 1;
    }
    std::error_code ec; std::filesystem::remove(pid_file_path(), ec);
    std::cout << "[disconnect] sent SIGTERM to pid " << pid << "\n";
    return 0;
#endif
}

static int cmd_logout() {
    std::error_code ec;
    for (const char* n : {"agent.key", "agent.id", "agent.server", "agent.api_url",
                          "agent.api_key", "agent.api_key_id"}) {
        std::filesystem::remove(state_path(n), ec);
    }
    std::cout << "[logout] removed saved login from " << state_dir() << "\n";
    return 0;
}

static int cmd_url(int argc, char** argv) {
    enum Fmt { F_PLAIN, F_OPENAI, F_ANTHROPIC, F_ENV, F_JSON };
    Fmt fmt = F_PLAIN;
    int pool_id = 0;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--openai")    fmt = F_OPENAI;
        else if (a == "--anthropic") fmt = F_ANTHROPIC;
        else if (a == "--env")       fmt = F_ENV;
        else if (a == "--json")      fmt = F_JSON;
        else if (a == "--pool" && i + 1 < argc) pool_id = std::atoi(argv[++i]);
    }

    std::string api_url = default_api_url();
    std::string agent_key = state_read("agent.key");
    if (agent_key.empty()) {
        std::cerr << "[url] no saved login — run `dist-node login` first\n";
        return 1;
    }

    // 1. Pick a pool (cached → first pool returned by /api/agent/pools).
    HttpResp pr; std::string perr;
    if (!http_request(api_url, "/api/agent/pools", "GET", "",
                      {"Authorization: Bearer " + agent_key}, pr, perr) || pr.status != 200) {
        std::cerr << "[url] list pools failed: " << perr
                  << " status=" << pr.status << " body=" << pr.body << "\n";
        return 1;
    }
    // very-light JSON: find first "base_url":"…" (or one whose "id":<pool_id>).
    std::string base_url;
    {
        const std::string& s = pr.body;
        size_t p = 0;
        while (true) {
            size_t pa = s.find("\"id\":", p);
            if (pa == std::string::npos) break;
            int    id = std::atoi(s.c_str() + pa + 5);
            size_t bu = s.find("\"base_url\":\"", pa);
            if (bu == std::string::npos) break;
            bu += 12;
            size_t end = s.find('"', bu);
            if (end == std::string::npos) break;
            std::string url = s.substr(bu, end - bu);
            if (pool_id == 0 || pool_id == id) { base_url = url; break; }
            p = end + 1;
        }
    }
    if (base_url.empty()) {
        std::cerr << "[url] no pools available for this account.\n"
                  << "      Visit " << api_url << " and create or join a pool.\n";
        return 1;
    }

    // 2. Reuse a cached API key, or mint a new one.
    std::string api_key = state_read("agent.api_key");
    if (api_key.empty() || api_key.rfind("sk-dist-", 0) != 0) {
        HttpResp mr; std::string merr;
        std::string body_s = "{\"label\":\"dist-node/" + std::string(state_read("agent.id").empty() ? "rig" : state_read("agent.id")) + "\"}";
        if (!http_request(api_url, "/api/agent/api_key", "POST", body_s,
                          {"Authorization: Bearer " + agent_key}, mr, merr)
            || mr.status != 200) {
            std::cerr << "[url] mint api_key failed: " << merr
                      << " status=" << mr.status << " body=" << mr.body << "\n";
            return 1;
        }
        api_key = json_peek_string(mr.body, "key");
        if (api_key.empty()) {
            std::cerr << "[url] api_key missing in response: " << mr.body << "\n";
            return 1;
        }
        state_write("agent.api_key", api_key);
    }

    // 3. Print in the requested format.
    switch (fmt) {
        case F_OPENAI:
            std::cout << "# OpenAI-compatible endpoint\n"
                      << "export OPENAI_API_BASE=\"" << base_url << "\"\n"
                      << "export OPENAI_BASE_URL=\""  << base_url << "\"\n"
                      << "export OPENAI_API_KEY=\""   << api_key  << "\"\n";
            break;
        case F_ANTHROPIC:
            std::cout << "# Anthropic-style env (uses the OpenAI-compat path)\n"
                      << "export ANTHROPIC_API_URL=\"" << base_url << "\"\n"
                      << "export ANTHROPIC_API_KEY=\"" << api_key  << "\"\n";
            break;
        case F_ENV:
            std::cout << "DIST_API_BASE=" << base_url << "\n"
                      << "DIST_API_KEY="  << api_key  << "\n";
            break;
        case F_JSON:
            std::cout << "{\"base_url\":\"" << base_url
                      << "\",\"api_key\":\"" << api_key << "\"}\n";
            break;
        case F_PLAIN:
        default:
            std::cout << "Endpoint: " << base_url << "\n"
                      << "API Key:  " << api_key << "\n"
                      << "\nTest with:\n"
                      << "  curl " << base_url << "/chat/completions \\\n"
                      << "    -H 'Authorization: Bearer " << api_key << "' \\\n"
                      << "    -H 'Content-Type: application/json' \\\n"
                      << "    -d '{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'\n";
            break;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // Line-buffer stdout/stderr so the user sees `[pair]` progress lines in
    // real time even when piped to a log file or systemd journal.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
    std::cout << std::unitbuf;

    // Subcommand dispatch — runs before any flag parsing.
    if (argc >= 2) {
        std::string sub = argv[1];
        if (sub == "login")      return cmd_login(argc - 2, argv + 2);
        if (sub == "connect")    return cmd_connect(argc - 2, argv + 2);
        if (sub == "disconnect") return cmd_disconnect();
        if (sub == "logout")     return cmd_logout();
        if (sub == "url")        return cmd_url(argc - 2, argv + 2);
    }
    dist::net_startup();
    dist::NodeAgentConfig cfg;
    std::string pair_url;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if (arg == "--pair") {
            pair_url = next();
        } else if (arg == "-s" || arg == "--server") {
            cfg.coordinator_host = next();
        } else if (arg == "-p" || arg == "--control-port") {
            cfg.coordinator_port = (uint16_t)std::stoi(next());
        } else if (arg == "-d" || arg == "--data-port") {
            cfg.data_port = (uint16_t)std::stoi(next());
        } else if (arg == "-g" || arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = std::stoi(next());
        } else if (arg == "-c" || arg == "--context") {
            cfg.n_ctx = (uint32_t)std::stoi(next());
        } else if (arg == "-b" || arg == "--batch") {
            cfg.n_batch = (uint32_t)std::stoi(next());
        } else if (arg == "--id") {
            cfg.node_id = next();
        } else if (arg == "--token-id") {
            cfg.token_id = next();
        } else if (arg == "--token-secret") {
            cfg.token_secret_hex = next();
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Deep-link pair mode: talks to the control-plane server over WebSocket.
    // Skip if the arg was passed but empty (common from systemd templates
    // where DIST_PAIR is cleared after the first successful pair — we then
    // fall through to auto-resume below).
    if (!pair_url.empty() && pair_url != "\"\"") {
        std::string tok, srv;
        if (!parse_pair_url(pair_url, tok, srv)) {
            fprintf(stderr, "Error: invalid --pair URL (expected distpool://pair?token=...&server=...)\n");
            return 1;
        }
        return run_pair_mode(tok, srv, cfg.node_id, cfg.n_gpu_layers);
    }

    // Auto-resume: --pair was omitted but a prior successful pair left an
    // agent_key + server URL behind.  This is how the per-user service unit
    // relaunches across reboots — the installer wrote --pair once, we dropped
    // state, and now every restart reads state and reconnects silently.
    {
        const std::string saved_key    = state_read("agent.key");
        const std::string saved_server = state_read("agent.server");
        if (!saved_key.empty() && !saved_server.empty()) {
            std::cout << "[pair] no --pair given, auto-resuming from "
                      << state_dir() << "\n";
            return run_pair_mode(std::string(), saved_server, cfg.node_id, cfg.n_gpu_layers);
        }
    }

    if (cfg.coordinator_host.empty()) {
        fprintf(stderr, "Error: --server is required (or pass --pair the first time)\n");
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== llama-distributed Node Agent ===\n";
    std::cout << "coordinator=" << cfg.coordinator_host << ":"
              << cfg.coordinator_port
              << " data_port=" << cfg.data_port << "\n";

    try {
        dist::NodeAgent agent(cfg);
        agent.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
