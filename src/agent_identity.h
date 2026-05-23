#pragma once

// Signed agent identities — see server/agent_identity.go for the
// counterpart and the wire protocol.
//
// The rig generates an ed25519 keypair on first boot and persists the
// private key (raw 32 bytes) to the agent state directory under
// "agent.priv".  The public key (raw 32 bytes) is sent to the server in
// the hello frame's `pubkey` field (base64-url, no padding).  On every
// subsequent reconnect, the server replies to the resume with a
// challenge nonce + ts; the rig signs the string
//
//   "dist-agent-v1|" + agent_id + "|" + nonce + "|" + ts
//
// and sends the 64-byte signature back as base64-url-no-padding in a
// "sig" frame.  The server verifies and either welcomes or closes.
//
// All functions return false on failure and leave the output buffers
// untouched.

#include <cstdint>
#include <string>
#include <vector>

namespace dist {

constexpr size_t ED25519_PRIV_SIZE = 32;
constexpr size_t ED25519_PUB_SIZE  = 32;
constexpr size_t ED25519_SIG_SIZE  = 64;

// Generate a fresh ed25519 keypair using OpenSSL.  priv and pub are
// raw bytes (32 + 32).
bool ed25519_generate(std::vector<uint8_t>& priv_out,
                      std::vector<uint8_t>& pub_out);

// Derive the public key from a stored private key (so we don't need to
// persist both halves).
bool ed25519_pub_from_priv(const std::vector<uint8_t>& priv,
                           std::vector<uint8_t>& pub_out);

// Sign `msg` with the raw 32-byte ed25519 private key.
bool ed25519_sign(const std::vector<uint8_t>& priv,
                  const std::string& msg,
                  std::vector<uint8_t>& sig_out);

// Base64-URL-without-padding encode.  The Go server's decodePubkeyB64 /
// decodeSigB64 helpers accept this form first.
std::string b64url_encode(const uint8_t* data, size_t len);
inline std::string b64url_encode(const std::vector<uint8_t>& v) {
    return b64url_encode(v.data(), v.size());
}

} // namespace dist
