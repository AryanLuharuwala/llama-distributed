#include "cli_common.h"
#include "platform_compat.h"

#include <openssl/err.h>
#include <openssl/ssl.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace dist::cli {

// ── State store ──────────────────────────────────────────────────────────

// Shared base — same path gpunet-node uses, so on a brand-new install we can
// migrate the legacy files into the /cli subdir below.  Never write directly
// to this path from gpunet-cli.
static std::string state_base() {
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

// gpunet-cli has its own subdir so an operator login can't clobber a compute
// rig's agent.id/agent.key.  On first access we copy any pre-existing files
// from the base dir into /cli so prior `gpunet-cli login` state isn't lost.
std::string state_dir() {
#ifdef _WIN32
    const std::string cli = state_base() + "\\cli";
#else
    const std::string cli = state_base() + "/cli";
#endif
    static std::once_flag migrate_once;
    std::call_once(migrate_once, [&]{
        std::error_code ec;
        std::filesystem::create_directories(cli, ec);
        // One-shot migration: pull legacy files into /cli when /cli is empty.
        // Leave the originals in place so gpunet-node keeps working.
        const auto base = state_base();
        const char* keys[] = {"agent.id", "agent.key", "agent.api_key",
                              "agent.server", "agent.api_url"};
        for (const char* k : keys) {
            const std::string src = base + "/" + k;
            const std::string dst = cli  + "/" + k;
            if (std::filesystem::exists(dst, ec)) continue;
            if (!std::filesystem::exists(src, ec)) continue;
            std::filesystem::copy_file(src, dst,
                std::filesystem::copy_options::skip_existing, ec);
        }
    });
    return cli;
}

std::string state_path(const std::string& name) {
    return state_dir() + "/" + name;
}

std::string state_read(const std::string& name) {
    std::ifstream f(state_path(name), std::ios::binary);
    if (!f.good()) return "";
    std::ostringstream buf;
    buf << f.rdbuf();
    std::string s = buf.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

bool state_write(const std::string& name, const std::string& value) {
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

// ── HTTP ─────────────────────────────────────────────────────────────────

namespace {
std::once_flag g_ssl_once;
void ssl_init() {
    std::call_once(g_ssl_once, []{
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
    });
}

bool parse_url(const std::string& url, bool& tls, std::string& host,
               uint16_t& port, std::string& root_path) {
    std::string u = url;
    if      (u.rfind("https://", 0) == 0) { tls = true;  u = u.substr(8); port = 443; }
    else if (u.rfind("http://",  0) == 0) { tls = false; u = u.substr(7); port = 80;  }
    else return false;
    size_t slash = u.find('/');
    std::string hp = (slash == std::string::npos) ? u : u.substr(0, slash);
    root_path = (slash == std::string::npos) ? "/" : u.substr(slash);
    size_t colon = hp.find(':');
    if (colon == std::string::npos) host = hp;
    else { host = hp.substr(0, colon); port = static_cast<uint16_t>(std::stoi(hp.substr(colon + 1))); }
    return !host.empty();
}
} // namespace

bool http_request(const std::string& base_url, const std::string& path,
                  const std::string& method, const std::string& body,
                  const std::vector<std::string>& extra_headers,
                  HttpResp& out, std::string& err) {
    bool tls = false; std::string host; uint16_t port = 0; std::string root_path;
    if (!parse_url(base_url, tls, host, port, root_path)) {
        err = "bad base URL: " + base_url;
        return false;
    }
    (void)root_path;

    addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16]; std::snprintf(ps, sizeof(ps), "%u", static_cast<unsigned>(port));
    if (::getaddrinfo(host.c_str(), ps, &hints, &res) != 0 || !res) {
        err = "dns: " + host;
        return false;
    }
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { ::freeaddrinfo(res); err = "socket"; return false; }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        ::freeaddrinfo(res); dist::close_sock(fd);
        err = "connect " + host;
        return false;
    }
    ::freeaddrinfo(res);

    SSL_CTX* ctx = nullptr;
    SSL*     ssl = nullptr;
    if (tls) {
        ssl_init();
        ctx = SSL_CTX_new(TLS_client_method());
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        SSL_CTX_set_default_verify_paths(ctx);
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host.c_str());
        SSL_set1_host(ssl, host.c_str());
        SSL_set_fd(ssl, fd);
        if (SSL_connect(ssl) != 1) {
            unsigned long e = ERR_get_error();
            char buf[256] = {0};
            ERR_error_string_n(e, buf, sizeof(buf));
            err = std::string("TLS: ") + buf;
            SSL_free(ssl);
            SSL_CTX_free(ctx);
            dist::close_sock(fd);
            return false;
        }
    }

    const bool default_port = (tls && port == 443) || (!tls && port == 80);
    std::string host_hdr = default_port ? host : host + ":" + std::to_string(port);

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n"
        << "Host: " << host_hdr << "\r\n"
        << "User-Agent: gpunet-cli/0.1\r\n"
        << "Accept: application/json\r\n"
        << "Connection: close\r\n";
    if (!body.empty()) {
        req << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    }
    for (const auto& h : extra_headers) req << h << "\r\n";
    req << "\r\n" << body;
    const std::string r = req.str();

    auto sendall = [&](const char* p, size_t n) -> bool {
        while (n > 0) {
            int k = tls ? SSL_write(ssl, p, static_cast<int>(n))
                        : static_cast<int>(::send(fd, p, n, 0));
            if (k <= 0) return false;
            p += k;
            n -= static_cast<size_t>(k);
        }
        return true;
    };

    auto cleanup = [&]{
        if (tls) {
            if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
            if (ctx) SSL_CTX_free(ctx);
        }
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
            dist::close_sock(fd);
        }
    };

    if (!sendall(r.data(), r.size())) {
        err = "send";
        cleanup();
        return false;
    }

    std::string resp;
    char rb[4096];
    while (true) {
        int k = tls ? SSL_read(ssl, rb, sizeof(rb))
                    : static_cast<int>(::recv(fd, rb, sizeof(rb), 0));
        if (k <= 0) break;
        resp.append(rb, static_cast<size_t>(k));
    }
    cleanup();

    size_t sp1 = resp.find(' ');
    size_t sp2 = (sp1 == std::string::npos) ? std::string::npos
                                            : resp.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        err = "bad status";
        return false;
    }
    out.status = std::atoi(resp.substr(sp1 + 1, sp2 - sp1 - 1).c_str());

    size_t hdr_end = resp.find("\r\n\r\n");
    std::string body_raw = (hdr_end == std::string::npos) ? "" : resp.substr(hdr_end + 4);

    bool chunked = false;
    {
        size_t te = resp.find("Transfer-Encoding:");
        if (te != std::string::npos &&
            te < (hdr_end == std::string::npos ? resp.size() : hdr_end) &&
            resp.substr(te, (hdr_end == std::string::npos ? resp.size() : hdr_end) - te)
                .find("chunked") != std::string::npos) {
            chunked = true;
        }
    }
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
            i += len + 2;
        }
        out.body = decoded;
    } else {
        out.body = body_raw;
    }
    return true;
}

// ── JSON peek ────────────────────────────────────────────────────────────

std::string json_peek_string(const std::string& msg, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    size_t p = msg.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    std::string out;
    while (p < msg.size() && msg[p] != '"') {
        if (msg[p] == '\\' && p + 1 < msg.size()) {
            out += msg[p + 1];
            p += 2;
            continue;
        }
        out += msg[p++];
    }
    return out;
}

std::string json_peek_int(const std::string& msg, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t p = msg.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    while (p < msg.size() && (msg[p] == ' ' || msg[p] == '\t')) ++p;
    if (p >= msg.size() || msg[p] == '"') return "";
    std::string out;
    if (msg[p] == '-') { out += '-'; ++p; }
    while (p < msg.size() && msg[p] >= '0' && msg[p] <= '9') out += msg[p++];
    return out;
}

// ── Auth ─────────────────────────────────────────────────────────────────

bool load_auth(AuthCtx& out, std::string& err) {
    // agent.api_url is the HTTPS REST base when stored separately by
    // gpunet-node login.  Fall back to deriving it from agent.server (the
    // WSS URL the agent connects to) for older state directories.
    out.server_url = state_read("agent.api_url");
    if (out.server_url.empty()) {
        std::string ws = state_read("agent.server");
        // wss://host[/path]  -> https://host
        // ws://host[/path]   -> http://host
        // Strip any /ws/agent or similar trailing path; the API lives at
        // origin/api/... regardless.
        if (ws.rfind("wss://", 0) == 0)      out.server_url = "https://" + ws.substr(6);
        else if (ws.rfind("ws://", 0) == 0)  out.server_url = "http://"  + ws.substr(5);
        else                                 out.server_url = ws;
        // Drop everything after the host (and port if any).
        size_t scheme_end = out.server_url.find("://");
        if (scheme_end != std::string::npos) {
            size_t slash = out.server_url.find('/', scheme_end + 3);
            if (slash != std::string::npos) out.server_url.erase(slash);
        }
    }
    out.agent_key  = state_read("agent.key");
    out.api_key    = state_read("agent.api_key");
    out.agent_id   = state_read("agent.id");

    // Strip a trailing slash so callers can always concatenate "/api/..."
    while (!out.server_url.empty() && out.server_url.back() == '/') out.server_url.pop_back();

    if (out.server_url.empty() || out.agent_key.empty()) {
        err = "not logged in — run `gpunet-node login` first";
        return false;
    }

    // Lazily mint an API key if we have an agent_key but no api_key cached.
    // Same shape as `gpunet-node url` uses.
    if (out.api_key.empty()) {
        const std::string label = "gpunet-cli/" + (out.agent_id.empty() ? std::string("rig") : out.agent_id);
        const std::string body  = "{\"label\":\"" + label + "\"}";
        HttpResp r;
        std::string h;
        if (http_request(out.server_url, "/api/agent/api_key", "POST", body,
                         {"Authorization: Bearer " + out.agent_key}, r, h) &&
            r.status == 200) {
            std::string k = json_peek_string(r.body, "key");
            if (k.empty()) k = json_peek_string(r.body, "api_key");
            if (k.empty()) k = json_peek_string(r.body, "token");
            if (!k.empty()) {
                out.api_key = k;
                state_write("agent.api_key", k);
            }
        }
    }

    return true;
}

} // namespace dist::cli
