#pragma once
/**
 * dist_conn.h
 *
 * Persistent TCP connection with length-prefixed framing.
 * Thread-safe send; single-reader recv (caller provides threading).
 *
 * Usage:
 *   Connection conn;
 *   conn.connect("192.168.1.5", 7701);
 *   conn.send_msg(header, payload, payload_len);
 *   conn.recv_msg(header, buf);  // blocks until full message arrives
 */

#include "dist_protocol.h"

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "platform_compat.h"

// Back-compat aliases — existing call sites use unqualified sock_t / SOCK_INVALID.
using sock_t = dist::sock_t;
static constexpr sock_t SOCK_INVALID = dist::SOCK_INVALID_V;

namespace dist {

class Connection {
public:
    Connection() = default;
    ~Connection() { close(); }

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // ── Client side ──────────────────────────────────────────────────────────

    void connect(const std::string& host, uint16_t port) {
        sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == SOCK_INVALID) throw std::runtime_error("socket() failed");

        _set_tcp_options(s);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(port);

        // resolve host
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res)
            throw std::runtime_error("getaddrinfo failed for " + host);
        memcpy(&addr.sin_addr,
               &((sockaddr_in*)res->ai_addr)->sin_addr,
               sizeof(addr.sin_addr));
        freeaddrinfo(res);

        if (::connect(s, (sockaddr*)&addr, sizeof(addr)) < 0) {
            _close_sock(s);
            throw std::runtime_error("connect() failed to " + host + ":" + std::to_string(port));
        }

        fd_.store(s);
        seq_.store(0);
        connected_.store(true);
    }

    // ── Server side: wrap an accepted fd ─────────────────────────────────────

    void accept_fd(sock_t s) {
        _set_tcp_options(s);
        fd_.store(s);
        seq_.store(0);
        connected_.store(true);
    }

    bool is_connected() const { return connected_.load(); }

    // Set the SO_RCVTIMEO on the underlying socket.  0 = no timeout (blocking).
    // Returns true if the option was applied.
    bool set_recv_timeout_ms(uint32_t ms) {
        sock_t s = fd_.load();
        if (s == SOCK_INVALID) return false;
#ifdef _WIN32
        DWORD tv = ms;
        return setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,
                          (const char*)&tv, sizeof(tv)) == 0;
#else
        struct timeval tv{};
        tv.tv_sec  = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        return setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,
                          (char*)&tv, sizeof(tv)) == 0;
#endif
    }

    void close() {
        connected_.store(false);
        sock_t s = fd_.exchange(SOCK_INVALID);
        if (s != SOCK_INVALID) _close_sock(s);
    }

    // ── Send ─────────────────────────────────────────────────────────────────

    // Send header + optional payload atomically (mutex-protected for thread safety).
    void send_msg(MsgType type,
                  const void* payload  = nullptr,
                  uint32_t    pay_len  = 0) {
        MsgHeader hdr = make_header(type, pay_len, seq_.fetch_add(1));
        std::lock_guard<std::mutex> lk(send_mu_);
        _send_all(&hdr, sizeof(hdr));
        if (payload && pay_len) _send_all(payload, pay_len);
    }

    // Convenience: send header + two discontiguous buffers (avoids copy for tensor data).
    void send_msg2(MsgType     type,
                   const void* buf1, uint32_t len1,
                   const void* buf2, uint32_t len2) {
        MsgHeader hdr = make_header(type, len1 + len2, seq_.fetch_add(1));
        std::lock_guard<std::mutex> lk(send_mu_);
        _send_all(&hdr, sizeof(hdr));
        if (buf1 && len1) _send_all(buf1, len1);
        if (buf2 && len2) _send_all(buf2, len2);
    }

    // ── Recv ─────────────────────────────────────────────────────────────────

    // Blocking: reads next complete message. Returns false on disconnect.
    bool recv_msg(MsgHeader& hdr_out, std::vector<uint8_t>& payload_out) {
        if (!_recv_all(&hdr_out, sizeof(MsgHeader))) return false;
        if (hdr_out.magic != PROTO_MAGIC) {
            connected_.store(false);
            return false;
        }
        payload_out.resize(hdr_out.payload_len);
        if (hdr_out.payload_len > 0) {
            if (!_recv_all(payload_out.data(), hdr_out.payload_len)) return false;
        }
        return true;
    }

    // Recv directly into a pre-allocated buffer (zero-copy path for tensors).
    bool recv_msg_into(MsgHeader& hdr_out, void* buf, size_t buf_capacity) {
        if (!_recv_all(&hdr_out, sizeof(MsgHeader))) return false;
        if (hdr_out.magic != PROTO_MAGIC) { connected_.store(false); return false; }
        if (hdr_out.payload_len > buf_capacity) { connected_.store(false); return false; }
        if (hdr_out.payload_len > 0) {
            if (!_recv_all(buf, hdr_out.payload_len)) return false;
        }
        return true;
    }

private:
    std::atomic<sock_t>  fd_        { SOCK_INVALID };
    std::atomic<bool>    connected_ { false };
    std::atomic<uint64_t> seq_      { 0 };
    std::mutex           send_mu_;

    static void _set_tcp_options(sock_t s) {
        int one = 1;
#ifdef TCP_NODELAY
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(one));
#endif
        // Large socket buffers for tensor streaming (4 MiB)
        int bufsize = 4 * 1024 * 1024;
        setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(bufsize));
        setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(bufsize));
    }

    static void _close_sock(sock_t s) { dist::close_sock(s); }

    void _send_all(const void* data, size_t len) {
        const char* p = static_cast<const char*>(data);
        sock_t s = fd_.load();
        while (len > 0) {
            ssize_t n = ::send(s, p, len, MSG_NOSIGNAL);
            if (n <= 0) { connected_.store(false); throw std::runtime_error("send failed"); }
            p   += n;
            len -= n;
        }
    }

    bool _recv_all(void* data, size_t len) {
        char* p = static_cast<char*>(data);
        sock_t s = fd_.load();
        while (len > 0) {
            ssize_t n = ::recv(s, p, len, 0);
            if (n <= 0) { connected_.store(false); return false; }
            p   += n;
            len -= n;
        }
        return true;
    }
};

// ─── Listener: accepts incoming connections ──────────────────────────────────

class Listener {
public:
    Listener() = default;
    ~Listener() { stop(); }

    void bind_and_listen(uint16_t port, int backlog = 32) {
        sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == SOCK_INVALID) throw std::runtime_error("socket() failed");

        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (char*)&one, sizeof(one));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (::bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) {
            _close_sock(s);
            throw std::runtime_error("bind() failed on port " + std::to_string(port));
        }
        if (::listen(s, backlog) < 0) {
            _close_sock(s);
            throw std::runtime_error("listen() failed");
        }
        listen_fd_.store(s);
    }

    // Blocks until a client connects. Returns a Connection wrapping the accepted fd.
    std::unique_ptr<Connection> accept_one() {
        sockaddr_in peer{};
        socklen_t   len = sizeof(peer);
        sock_t s = ::accept(listen_fd_.load(), (sockaddr*)&peer, &len);
        if (s == SOCK_INVALID) return nullptr;
        auto conn = std::make_unique<Connection>();
        conn->accept_fd(s);
        return conn;
    }

    void stop() {
        sock_t s = listen_fd_.exchange(SOCK_INVALID);
        if (s != SOCK_INVALID) _close_sock(s);
    }

private:
    std::atomic<sock_t> listen_fd_ { SOCK_INVALID };

    static void _close_sock(sock_t s) { dist::close_sock(s); }
};

} // namespace dist
