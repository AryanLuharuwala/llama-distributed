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

#include "platform_compat.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <signal.h>
#include <unistd.h>

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

    std::ostringstream hello;
    if (have_resume) {
        std::cout << "[pair] resuming as agent_id=" << agent_id
                  << " (using saved agent_key)\n";
        hello << "{\"kind\":\"resume\","
              << "\"agent_key\":\"" << json_escape(saved_agent_key) << "\","
              << "\"agent_id\":\""  << json_escape(agent_id)        << "\","
              << "\"hostname\":\""  << json_escape(hostname)        << "\","
              << "\"n_gpus\":0,\"vram_bytes\":0}";
    } else {
        hello << "{\"kind\":\"hello\","
              << "\"token\":\""    << json_escape(token)    << "\","
              << "\"agent_id\":\"" << json_escape(agent_id) << "\","
              << "\"hostname\":\"" << json_escape(hostname) << "\","
              << "\"n_gpus\":0,\"vram_bytes\":0}";
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
    };
    std::map<uint16_t, pp_req> reqs;

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
        ws.send_binary(out.data(), out.size());
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
                // Release KV + bookkeeping for this request.
                engine->reset_seq(r.seq_id);
                reqs.erase(it);
            }
        }
        pending_actvs.clear();
    };

    auto next_beat = std::chrono::steady_clock::now();
    while (ws.is_open()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next_beat) {
            double secs = std::chrono::duration<double>(
                now - dec_window_start).count();
            double tps = (secs > 0.0) ? (double) dec_tokens / secs : 0.0;
            std::ostringstream s;
            s << "{\"kind\":\"status\",\"n_gpus\":" << (n_gpu_layers > 0 ? 1 : 0)
              << ",\"tokens_sec\":" << (int) tps << "}";
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
                              << " payload=" << payload.substr(0, 80) << "...\n";

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

                    std::string prompt     = json_str("prompt");
                    std::string shard_url  = json_str("shard_url");
                    std::string shard_file = json_str("shard_file");
                    long long   mt         = json_int("max_tokens");
                    uint32_t    max_tokens = (mt > 0) ? (uint32_t) mt : 128;

                    if (prompt.empty()) {
                        send_chunk(2 /* error */, in_tokens, 0, "empty prompt");
                        send_chunk(1, in_tokens, 0, "");
                        continue;
                    }
                    if (shard_url.empty()) {
                        send_chunk(2, in_tokens, 0, "pool has no model bound");
                        send_chunk(1, in_tokens, 0, "");
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
                            send_chunk(1, in_tokens, 0, "");
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
                            send_chunk(1, in_tokens, 0, "");
                            continue;
                        }
                        infer_engine      = std::move(eng);
                        infer_engine_key  = key;
                        infer_engine_path = dest;
                    }

                    // Fresh KV for this request.
                    infer_engine->reset_kv();
                    std::vector<int32_t> toks = infer_engine->tokenize(prompt, true);
                    if (toks.empty()) {
                        send_chunk(2, in_tokens, 0, "tokenize failed");
                        send_chunk(1, in_tokens, 0, "");
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
                        send_chunk(1, prompt_tokens, 0, "");
                        continue;
                    }
                    int32_t pos = (int32_t) toks.size();
                    const int32_t eos = infer_engine->eos_token();

                    auto sample_greedy = [&](const std::vector<float> & ls) -> int32_t {
                        int32_t best = 0; float best_s = ls[0];
                        for (size_t i = 1; i < ls.size(); ++i) {
                            if (ls[i] > best_s) { best_s = ls[i]; best = (int32_t) i; }
                        }
                        return best;
                    };

                    uint32_t out_tokens = 0;
                    bool done = false;
                    while (out_tokens < max_tokens && !done && ws.is_open()) {
                        int32_t next = sample_greedy(logits);
                        ++out_tokens;
                        if (next == eos) {
                            done = true;
                            break;
                        }
                        std::string piece = infer_engine->detokenize(next);
                        send_chunk(0 /* token */, prompt_tokens, out_tokens, piece);

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
                    (void) ok;
                    send_chunk(1 /* done */, prompt_tokens, out_tokens, "");
                } else if (msg.size() >= 20 && be32(msg.data()) == MAGIC_ACTV
                           && msg[4] == 0x01) {
                    // ACTV frame.  See server/activation.go for layout.
                    // Enqueue now, batch-decode later.
                    PendingActv pa;
                    pa.type     = msg[5];
                    pa.req_id   = be16(msg.data() + 6);
                    pa.stage_in = be16(msg.data() + 8);
                    pa.tok_seq  = be32(msg.data() + 10);
                    pa.dtype    = msg[14];
                    uint8_t rank = msg[15];
                    pa.flags    = msg[16];
                    size_t hdr  = 20 + 4 * (size_t)rank;
                    pa.dims.resize(rank);
                    for (uint8_t i = 0; i < rank; ++i) {
                        pa.dims[i] = be32(msg.data() + 20 + 4*i);
                    }
                    if (msg.size() < hdr + 4) { continue; }
                    uint32_t payload_len = be32(msg.data() + hdr);
                    if (msg.size() >= hdr + 4 + payload_len) {
                        pa.payload.assign((const char*)(msg.data() + hdr + 4), payload_len);
                    }
                    if (pa.type == 0x01) {
                        pending_actvs.push_back(std::move(pa));
                    }
                } else {
                    // Unknown binary frame — echo (keeps M2 relay test working).
                    std::cout << "[pair] ← binary " << msg.size() << " bytes (echo)\n";
                    ws.send_binary(msg.data(), msg.size());
                }
            } else {
                std::string txt((const char*)msg.data(), msg.size());
                std::cout << "[pair] ← " << txt << "\n";
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
                            reqs[r.req_id] = r;
                        }
                        std::cout << "[pair] pp_route accepted req=" << r.req_id
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
            // Recurse the parse logic by re-injecting the frame via a second
            // pass: we only need to handle ACTV here since pp_route is rare
            // enough to wait for the next iteration.
            if (b2 && m2.size() >= 20 && be32(m2.data()) == MAGIC_ACTV && m2[4] == 0x01) {
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
            }
            // Non-ACTV frames in the drain are skipped; they'll be picked up
            // by the next blocking recv iteration.
        }

        flush_pending();
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
