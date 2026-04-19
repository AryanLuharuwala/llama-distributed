#include "dist_ws_client.h"
#include "auth.h"   // reuse sha256() + random_bytes() + to_hex()

#include "platform_compat.h"

#include <cstring>
#include <errno.h>
#include <sstream>

namespace dist {

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
    if (u.rfind("ws://", 0) == 0)        u = u.substr(5);
    else if (u.rfind("wss://", 0) == 0)  { err_ = "wss:// not supported yet"; return false; }
    else if (u.rfind("http://", 0) == 0) u = u.substr(7);
    else { err_ = "unsupported scheme: " + url; return false; }

    size_t slash = u.find('/');
    std::string hp = (slash == std::string::npos) ? u : u.substr(0, slash);
    path = (slash == std::string::npos) ? "/" : u.substr(slash);

    size_t colon = hp.find(':');
    if (colon == std::string::npos) {
        host = hp; port = 80;
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
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

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

    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Upgrade: websocket\r\n"
        << "Connection: Upgrade\r\n"
        << "Sec-WebSocket-Key: " << key << "\r\n"
        << "Sec-WebSocket-Version: 13\r\n"
        << "Origin: http://" << host << ":" << port << "\r\n"
        << "\r\n";

    std::string r = req.str();
    if (!send_all(r.data(), r.size())) { err_ = "send handshake failed"; return false; }

    // Read the response headers until we see \r\n\r\n.
    std::string resp;
    char buf[512];
    while (resp.find("\r\n\r\n") == std::string::npos && resp.size() < 32 * 1024) {
        ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0) { err_ = "recv handshake failed"; return false; }
        resp.append(buf, n);
    }

    if (resp.rfind("HTTP/1.1 101", 0) != 0) {
        err_ = "ws upgrade refused: " + resp.substr(0, 80);
        return false;
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
    if (!http_upgrade(host, port, path))     { close(); return false; }
    return true;
}

void WsClient::close() {
    if (fd_ >= 0) {
        // Send a CLOSE frame best-effort.
        uint8_t empty[2] = { 0x03, 0xE8 };   // status 1000
        send_frame(OP_CLOSE, empty, 2);
        ::shutdown(fd_, SHUT_RDWR);
        dist::close_sock(fd_);
        fd_ = -1;
    }
}

bool WsClient::set_recv_timeout_ms(uint32_t ms) {
    if (fd_ < 0) return false;
    struct timeval tv{};
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    return ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

bool WsClient::send_all(const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t k = ::send(fd_, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k; n -= (size_t)k;
    }
    return true;
}

bool WsClient::recv_all(void* buf, size_t n) {
    char* p = (char*)buf;
    while (n > 0) {
        ssize_t k = ::recv(fd_, p, n, 0);
        if (k <= 0) return false;
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
