// Shared building blocks for gpunet-cli (and other CLI consumers).
//
// Three concerns live here, each small enough that splitting further would
// add more includes than it saves:
//   * State directory — reads/writes the same on-disk layout gpunet-node uses
//     so a user who has already run `gpunet-node login` is automatically
//     authenticated for `gpunet-cli`.
//   * HTTP client — minimal TLS-capable client built on OpenSSL, the same
//     shape as dist_node_main.cpp's http_request.
//   * JSON peek — quick string/int field extraction from flat JSON; the
//     dashboard API surface is small enough that we don't pull in a parser.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dist::cli {

// ── State store ──────────────────────────────────────────────────────────
// Mirrors src/dist_node_main.cpp::state_dir() exactly — uses DIST_STATE_DIR,
// then XDG_STATE_HOME/llama-distributed, then $HOME/.local/state/...
std::string state_dir();
std::string state_path(const std::string& name);
std::string state_read(const std::string& name);
bool        state_write(const std::string& name, const std::string& value);

// ── HTTP ─────────────────────────────────────────────────────────────────
struct HttpResp {
    int         status = 0;
    std::string body;
};

// Synchronous HTTP/1.1 request.  base_url is the origin (https://host[:port]).
// path includes the query string.  Returns true on transport success; check
// status for HTTP-level result.  extra_headers are passed through verbatim
// (no trailing CRLF needed).  body is sent as-is with Content-Type: application/json.
bool http_request(const std::string& base_url,
                  const std::string& path,
                  const std::string& method,
                  const std::string& body,
                  const std::vector<std::string>& extra_headers,
                  HttpResp& out,
                  std::string& err);

// ── JSON helpers ─────────────────────────────────────────────────────────
// Crude single-field extractors.  Only safe for flat fields whose values do
// not contain the key name as a substring.  Sufficient for the dashboard
// envelope shape ({"kind":"...", "data": [...]}).
std::string json_peek_string(const std::string& msg, const std::string& key);
std::string json_peek_int   (const std::string& msg, const std::string& key);

// ── Auth context ─────────────────────────────────────────────────────────
// Resolved view of "who is this CLI acting as".  Populated by load_auth().
struct AuthCtx {
    std::string server_url;  // e.g. https://distpool-server.example.com
    std::string agent_key;   // bearer for /api/agent/* endpoints
    std::string api_key;     // bearer for /api/* user endpoints (OpenAI-style)
    std::string agent_id;
};

// Loads server_url + agent_key from saved gpunet-node state, and optionally
// fetches/refreshes the API key from /api/agent/api_key.  Returns false with
// a human-readable hint in `err` if the user hasn't logged in yet.
bool load_auth(AuthCtx& out, std::string& err);

} // namespace dist::cli
