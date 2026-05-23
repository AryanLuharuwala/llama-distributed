#include "dist_ws_client.h"
#include "auth.h"   // reuse sha256() + random_bytes() + to_hex()

#include "platform_compat.h"

#include <cstring>
#include <errno.h>
#include <sstream>

#ifndef _WIN32
#include <poll.h>
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>

namespace dist {

static bool g_ssl_inited = false;
static void ssl_global_init() {
    if (g_ssl_inited) return;
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    g_ssl_inited = true;
}

// ─── Base64 encode (for the RFC 6455 handshake only) ──────────────────────

static std::string base64(const uint8_t* data, size_t len) {
    static const char* A =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i+2];
        out.push_back(A[(v >> 18) & 0x3F]);
        out.push_back(A[(v >> 12) & 0x3F]);
        out.push_back(i + 1 < len ? A[(v >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < len ? A[v & 0x3F]        : '=');
    }
    return out;
}

// ─── URL parsing ──────────────────────────────────────────────────────────

bool WsClient::parse_url(const std::string& url,
                         std::string& host, uint16_t& port, std::string& path) {
    std::string u = url;
    bool default_tls = false;
    if      (u.rfind("ws://",    0) == 0) { u = u.substr(5);  tls_ = false; }
    else if (u.rfind("wss://",   0) == 0) { u = u.substr(6);  tls_ = true;  default_tls = true; }
    else if (u.rfind("https://", 0) == 0) { u = u.substr(8);  tls_ = true;  default_tls = true; }
    else if (u.rfind("http://",  0) == 0) { u = u.substr(7);  tls_ = false; }
    else { err_ = "unsupported scheme: " + url; return false; }

    size_t slash = u.find('/');
    std::string hp = (slash == std::string::npos) ? u : u.substr(0, slash);
    path = (slash == std::string::npos) ? "/" : u.substr(slash);

    size_t colon = hp.find(':');
    if (colon == std::string::npos) {
        host = hp; port = default_tls ? 443 : 80;
    } else {
        host = hp.substr(0, colon);
        port = (uint16_t)std::stoi(hp.substr(colon + 1));
    }
    return true;
}

// ─── TCP ──────────────────────────────────────────────────────────────────

bool WsClient::tcp_connect(const std::string& host, uint16_t port) {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
        err_ = "getaddrinfo failed for " + host;
        return false;
    }
    struct sockaddr_in addr{};
    std::memcpy(&addr,
                (struct sockaddr_in*)res->ai_addr,
                sizeof(addr));
    addr.sin_port = htons(port);
    freeaddrinfo(res);

    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) { err_ = "socket() failed"; return false; }

    int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));

    if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        err_ = "connect() failed: " + std::string(std::strerror(errno));
        dist::close_sock(fd_); fd_ = -1;
        return false;
    }
    return true;
}

// ─── Handshake ────────────────────────────────────────────────────────────

static std::string sha1_base64(const std::string& s) {
    // SHA-1 is not implemented in auth.cpp (we only have SHA-256).
    // RFC 6455 requires the response Sec-WebSocket-Accept field; we don't
    // actually *verify* it for M1 because we trust our own server.  Return a
    // placeholder — we only send our own key.
    (void)s;
    return "";
}

bool WsClient::http_upgrade(const std::string& host, uint16_t port,
                             const std::string& path) {
    // 16 random bytes, base64'd — the client key per RFC 6455.
    uint8_t nonce[16];
    random_bytes(nonce, sizeof(nonce));
    std::string key = base64(nonce, sizeof(nonce));

    // ACA's HTTP/1.1 ingress demands the Host header match the FQDN without
    // a port (since 443/80 are implicit for https/http).  Omit the default
    // port for TLS and plain HTTP so the gateway doesn't reject us.
    const bool default_port = (tls_ && port == 443) || (!tls_ && port == 80);
    std::string host_hdr = default_port ? host : host + ":" + std::to_string(port);
    const char* origin_scheme = tls_ ? "https://" : "http://";
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host_hdr << "\r\n"
        << "Upgrade: websocket\r\n"
        << "Connection: Upgrade\r\n"
        << "Sec-WebSocket-Key: " << key << "\r\n"
        << "Sec-WebSocket-Version: 13\r\n"
        << "Origin: " << origin_scheme << host_hdr << "\r\n"
        << "\r\n";

    std::string r = req.str();
    if (!send_all(r.data(), r.size())) { err_ = "send handshake failed"; return false; }

    // Read the response headers until we see \r\n\r\n.  Must go through SSL
    // when TLS is active — calling ::recv on a TLS socket returns encrypted
    // bytes (or, post-handshake, often nothing because OpenSSL buffered the
    // first record).
    std::string resp;
    char buf[512];
    while (resp.find("\r\n\r\n") == std::string::npos && resp.size() < 32 * 1024) {
        int n;
        if (ssl_) {
            n = SSL_read((SSL*)ssl_, buf, sizeof(buf));
        } else {
            n = (int)::recv(fd_, buf, sizeof(buf), 0);
        }
        if (n <= 0) { err_ = "recv handshake failed"; return false; }
        resp.append(buf, (size_t)n);
    }

    if (resp.rfind("HTTP/1.1 101", 0) != 0) {
        err_ = "ws upgrade refused: " + resp.substr(0, 80);
        return false;
    }

    // Stash any bytes that arrived after the \r\n\r\n header terminator.
    // These belong to subsequent WS frames (server may push the welcome
    // text or even a binary frame in the same TLS record as the 101
    // response). recv_all consumes leftover_ before falling back to SSL_read.
    size_t hdr_end = resp.find("\r\n\r\n");
    if (hdr_end != std::string::npos) {
        size_t body_off = hdr_end + 4;
        if (body_off < resp.size()) {
            leftover_.assign(
                (const uint8_t*)resp.data() + body_off,
                (const uint8_t*)resp.data() + resp.size());
            leftover_off_ = 0;
        }
    }

    // We don't validate the Accept header for M1.  (Same machine, trusted.)
    (void)sha1_base64;
    return true;
}

// ─── Public API ───────────────────────────────────────────────────────────

bool WsClient::connect(const std::string& url) {
    std::string host, path;
    uint16_t port;
    if (!parse_url(url, host, port, path))  return false;
    if (!tcp_connect(host, port))            return false;
    if (tls_ && !tls_handshake(host))        { close(); return false; }
    if (!http_upgrade(host, port, path))     { close(); return false; }
    return true;
}

bool WsClient::tls_handshake(const std::string& host) {
    ssl_global_init();
    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) { err_ = "SSL_CTX_new failed"; return false; }
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
    // We don't ship a CA bundle with the agent; trust the system default trust
    // store (Linux: /etc/ssl/certs; macOS: SecureTransport via OpenSSL).
    SSL_CTX_set_default_verify_paths(ctx);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);

    SSL* ssl = SSL_new(ctx);
    if (!ssl) { SSL_CTX_free(ctx); err_ = "SSL_new failed"; return false; }
    // SNI is mandatory for ACA and most CDN-fronted hosts.
    SSL_set_tlsext_host_name(ssl, host.c_str());
    // Hostname verification against the server cert.
    SSL_set1_host(ssl, host.c_str());
    SSL_set_fd(ssl, fd_);

    int rc = SSL_connect(ssl);
    if (rc != 1) {
        unsigned long e = ERR_get_error();
        char buf[256] = {0};
        ERR_error_string_n(e, buf, sizeof(buf));
        err_ = std::string("TLS handshake failed: ") + buf;
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        return false;
    }
    ssl_     = ssl;
    ssl_ctx_ = ctx;
    return true;
}

void WsClient::close() {
    if (fd_ >= 0) {
        // Send a CLOSE frame best-effort.
        uint8_t empty[2] = { 0x03, 0xE8 };   // status 1000
        send_frame(OP_CLOSE, empty, 2);
        if (ssl_) {
            SSL_shutdown((SSL*)ssl_);
            SSL_free((SSL*)ssl_);
            ssl_ = nullptr;
        }
        if (ssl_ctx_) {
            SSL_CTX_free((SSL_CTX*)ssl_ctx_);
            ssl_ctx_ = nullptr;
        }
        ::shutdown(fd_, SHUT_RDWR);
        dist::close_sock(fd_);
        fd_ = -1;
    }
}

bool WsClient::set_recv_timeout_ms(uint32_t ms) {
    if (fd_ < 0) return false;
    // ms == 0 → non-blocking drain (return immediately if no data).
    // ms  > 0 → wait up to ms milliseconds in recv_all via poll().
    recv_timeout_ms_ = (int)ms;
#ifdef _WIN32
    DWORD tv = ms;
    return ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO,
                        (const char*)&tv, sizeof(tv)) == 0;
#else
    // Always keep the socket non-blocking once the heartbeat phase starts.
    // recv_all drives the actual wait via poll() with recv_timeout_ms_.
    int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags < 0) {
        err_ = "fcntl F_GETFL failed";
        return false;
    }
    int want = flags | O_NONBLOCK;
    if (want != flags) {
        if (::fcntl(fd_, F_SETFL, want) < 0) {
            err_ = "fcntl F_SETFL failed";
            return false;
        }
    }
    return true;
#endif
}

bool WsClient::send_all(const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        int k;
        if (ssl_) {
            k = SSL_write((SSL*)ssl_, p, (int)n);
            if (k <= 0) {
                int ssl_err = SSL_get_error((SSL*)ssl_, k);
                if (ssl_err == SSL_ERROR_WANT_READ ||
                    ssl_err == SSL_ERROR_WANT_WRITE) {
#ifndef _WIN32
                    struct pollfd pfd{};
                    pfd.fd = fd_;
                    pfd.events = (ssl_err == SSL_ERROR_WANT_READ) ? POLLIN : POLLOUT;
                    int pr = ::poll(&pfd, 1, 5000);
                    if (pr <= 0) return false;
                    continue;
#else
                    return false;
#endif
                }
                return false;
            }
        } else {
            k = (int)::send(fd_, p, n, MSG_NOSIGNAL);
            if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
#ifndef _WIN32
                struct pollfd pfd{};
                pfd.fd = fd_;
                pfd.events = POLLOUT;
                int pr = ::poll(&pfd, 1, 5000);
                if (pr <= 0) return false;
                continue;
#else
                return false;
#endif
            }
            if (k <= 0) return false;
        }
        p += k; n -= (size_t)k;
    }
    return true;
}

bool WsClient::recv_all(void* buf, size_t n) {
    char* p = (char*)buf;
    // Drain bytes captured during http_upgrade past the \r\n\r\n boundary
    // before issuing any new read on the socket / SSL session.
    if (!leftover_.empty() && leftover_off_ < leftover_.size()) {
        size_t avail = leftover_.size() - leftover_off_;
        size_t take  = (avail < n) ? avail : n;
        std::memcpy(p, leftover_.data() + leftover_off_, take);
        leftover_off_ += take;
        p += take; n -= take;
        if (leftover_off_ == leftover_.size()) {
            leftover_.clear();
            leftover_off_ = 0;
        }
    }
    while (n > 0) {
        int k;
        if (ssl_) {
            // Drive SSL_read in non-blocking-ish fashion: rely on the
            // socket's O_NONBLOCK + poll() for actual timeout enforcement.
            // SO_RCVTIMEO + AUTO_RETRY interact poorly; explicit poll()
            // keeps the loop deterministic.
            k = SSL_read((SSL*)ssl_, p, (int)n);
            if (k <= 0) {
                int ssl_err = SSL_get_error((SSL*)ssl_, k);
                if (ssl_err == SSL_ERROR_WANT_READ ||
                    ssl_err == SSL_ERROR_WANT_WRITE) {
#ifndef _WIN32
                    // ms == 0 → don't wait; caller asked for an immediate
                    // drain.  Otherwise wait up to recv_timeout_ms_.
                    if (recv_timeout_ms_ == 0) return false;
                    struct pollfd pfd{};
                    pfd.fd = fd_;
                    pfd.events = (ssl_err == SSL_ERROR_WANT_WRITE) ? POLLOUT : POLLIN;
                    int pr = ::poll(&pfd, 1, recv_timeout_ms_);
                    if (pr <= 0) return false;
                    continue;   // retry SSL_read
#else
                    return false;
#endif
                }
                return false;
            }
        } else {
            k = (int)::recv(fd_, p, n, 0);
            if (k < 0) {
#ifndef _WIN32
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                    // Socket is O_NONBLOCK once the heartbeat phase starts.
                    // Wait up to recv_timeout_ms_ via poll() instead of
                    // spinning the CPU at 100%.  ms == 0 means caller asked
                    // for an immediate drain — just bail.
                    if (recv_timeout_ms_ == 0) return false;
                    struct pollfd pfd{};
                    pfd.fd = fd_;
                    pfd.events = POLLIN;
                    int pr = ::poll(&pfd, 1, recv_timeout_ms_);
                    if (pr <= 0) return false;
                    continue;   // retry recv
                }
#endif
                return false;
            }
            if (k == 0) return false;   // peer closed
        }
        p += k; n -= (size_t)k;
    }
    return true;
}

// Frame format (client→server always masked):
//
//   byte 0: FIN(1) RSV(3) OPCODE(4)
//   byte 1: MASK(1) LEN7(7)
//   [len=126 → 2-byte big-endian length]
//   [len=127 → 8-byte big-endian length]
//   [mask key — 4 bytes]
//   [masked payload]
bool WsClient::send_frame(uint8_t opcode, const uint8_t* payload, size_t n) {
    if (fd_ < 0) return false;
    uint8_t hdr[14];
    size_t  hlen = 2;
    hdr[0] = 0x80 | (opcode & 0x0F);  // FIN=1
    if (n <= 125) {
        hdr[1] = 0x80 | (uint8_t)n;
    } else if (n <= 0xFFFF) {
        hdr[1] = 0x80 | 126;
        hdr[2] = (uint8_t)(n >> 8);
        hdr[3] = (uint8_t)(n);
        hlen = 4;
    } else {
        hdr[1] = 0x80 | 127;
        for (int i = 0; i < 8; ++i) hdr[2 + i] = (uint8_t)(n >> (8 * (7 - i)));
        hlen = 10;
    }
    uint8_t mask[4];
    random_bytes(mask, 4);
    std::memcpy(hdr + hlen, mask, 4);
    hlen += 4;

    if (!send_all(hdr, hlen)) return false;

    if (n > 0) {
        // Mask into a small buffer (chunked to avoid big allocs).
        uint8_t buf[2048];
        size_t off = 0;
        while (off < n) {
            size_t c = std::min(sizeof(buf), n - off);
            for (size_t i = 0; i < c; ++i)
                buf[i] = payload[off + i] ^ mask[(off + i) & 3];
            if (!send_all(buf, c)) return false;
            off += c;
        }
    }
    return true;
}

bool WsClient::recv_frame(uint8_t& opcode, std::vector<uint8_t>& payload, bool& fin) {
    uint8_t h[2];
    if (!recv_all(h, 2)) return false;
    fin    = (h[0] & 0x80) != 0;
    opcode = h[0] & 0x0F;
    bool masked = (h[1] & 0x80) != 0;
    uint64_t len = h[1] & 0x7F;
    if (len == 126) {
        uint8_t e[2]; if (!recv_all(e, 2)) return false;
        len = ((uint64_t)e[0] << 8) | e[1];
    } else if (len == 127) {
        uint8_t e[8]; if (!recv_all(e, 8)) return false;
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
    }
    uint8_t mkey[4] = {};
    if (masked) { if (!recv_all(mkey, 4)) return false; }
    payload.resize((size_t)len);
    if (len > 0 && !recv_all(payload.data(), (size_t)len)) return false;
    if (masked) {
        for (size_t i = 0; i < payload.size(); ++i)
            payload[i] ^= mkey[i & 3];
    }
    return true;
}

bool WsClient::send_text(const std::string& data) {
    return send_frame(OP_TEXT, (const uint8_t*)data.data(), data.size());
}

bool WsClient::send_binary(const uint8_t* data, size_t n) {
    return send_frame(OP_BIN, data, n);
}

bool WsClient::recv_message(std::vector<uint8_t>& out, bool& is_binary) {
    std::vector<uint8_t> buf;
    uint8_t cur_opcode = 0;
    out.clear();
    is_binary = false;
    while (fd_ >= 0) {
        uint8_t op; std::vector<uint8_t> frame; bool fin;
        if (!recv_frame(op, frame, fin)) return false;
        if (op == OP_PING) {
            send_frame(OP_PONG, frame.data(), frame.size());
            continue;
        }
        if (op == OP_PONG)  continue;
        if (op == OP_CLOSE) { close(); return false; }
        if (op == OP_TEXT || op == OP_BIN) {
            buf.clear();
            cur_opcode = op;
            buf.insert(buf.end(), frame.begin(), frame.end());
        } else if (op == OP_CONT) {
            buf.insert(buf.end(), frame.begin(), frame.end());
        }
        if (fin) {
            out = std::move(buf);
            is_binary = (cur_opcode == OP_BIN);
            return true;
        }
    }
    return false;
}

bool WsClient::recv_text(std::string& out) {
    // Simple loop: absorb pings / closes / continuations transparently.
    std::vector<uint8_t> buf;
    uint8_t cur_opcode = 0;

    while (fd_ >= 0) {
        uint8_t op; std::vector<uint8_t> frame; bool fin;
        if (!recv_frame(op, frame, fin)) return false;

        if (op == OP_PING) {
            send_frame(OP_PONG, frame.data(), frame.size());
            continue;
        }
        if (op == OP_PONG) continue;
        if (op == OP_CLOSE) { close(); return false; }

        if (op == OP_TEXT || op == OP_BIN) {
            buf.clear();
            cur_opcode = op;
            buf.insert(buf.end(), frame.begin(), frame.end());
        } else if (op == OP_CONT) {
            buf.insert(buf.end(), frame.begin(), frame.end());
        }
        if (fin) {
            if (cur_opcode == OP_TEXT) {
                out.assign((const char*)buf.data(), buf.size());
                return true;
            }
            // Binary frame — skip.
            continue;
        }
    }
    return false;
}

} // namespace dist
