#pragma once
/**
 * dist_ws_client.h
 *
 * Minimal, single-threaded RFC 6455 WebSocket client.
 *
 * Scope: just enough to speak text-JSON to our control-plane server.
 *   - No TLS (M1 uses ws://; plain HTTPS support is a later step).
 *   - Text frames only.  Close, ping/pong handled transparently.
 *   - Client-side masking (required by spec).
 *
 * Usage:
 *   dist::WsClient ws;
 *   if (!ws.connect("ws://localhost:8080/ws/agent")) return 1;
 *   ws.send_text("{\"kind\":\"hello\",...}");
 *   std::string msg;
 *   while (ws.recv_text(msg)) {  use msg  }
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dist {

class WsClient {
public:
    WsClient()  = default;
    ~WsClient() { close(); }

    WsClient(const WsClient&)            = delete;
    WsClient& operator=(const WsClient&) = delete;

    // Connect to a ws://host[:port]/path URL.  Returns false on failure.
    bool connect(const std::string& url);

    // Send one text frame (the whole string).  Returns false on write failure.
    bool send_text(const std::string& data);

    // Send one binary frame.
    bool send_binary(const uint8_t* data, size_t n);

    // Block until the next text frame arrives.  `out` is set to the payload.
    // Returns false on disconnect or protocol error.
    // Ping frames are handled transparently (pong sent back, loop continues).
    bool recv_text(std::string& out);

    // Block until the next message arrives (text or binary).
    // is_binary is set according to the opcode.
    // Ping frames are handled transparently.  Binary payload is written to `out`.
    bool recv_message(std::vector<uint8_t>& out, bool& is_binary);

    bool is_open() const { return fd_ >= 0; }
    void close();

    // Optional: set receive timeout in ms.  0 = blocking.
    bool set_recv_timeout_ms(uint32_t ms);

    // Last-error message (human-readable).
    const std::string& last_error() const { return err_; }

private:
    int         fd_ = -1;
    std::string err_;

    // RFC6455 opcodes we care about
    enum : uint8_t {
        OP_CONT  = 0x0,
        OP_TEXT  = 0x1,
        OP_BIN   = 0x2,
        OP_CLOSE = 0x8,
        OP_PING  = 0x9,
        OP_PONG  = 0xA,
    };

    bool parse_url(const std::string& url,
                   std::string& host, uint16_t& port, std::string& path);

    bool tcp_connect(const std::string& host, uint16_t port);
    bool http_upgrade(const std::string& host, uint16_t port, const std::string& path);

    bool recv_all(void* buf, size_t n);
    bool send_all(const void* buf, size_t n);

    bool send_frame(uint8_t opcode, const uint8_t* payload, size_t n);
    bool recv_frame(uint8_t& opcode, std::vector<uint8_t>& payload, bool& fin);
};

} // namespace dist
