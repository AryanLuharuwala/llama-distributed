#pragma once
/**
 * auth.h
 *
 * Pool-gate authentication + signed contribution receipts.
 *
 *   Token     = (token_id, secret[32], scope, expires_at).  Distributed
 *               out-of-band by the pool operator.  Secret never touches the
 *               wire — only HMACs derived from it do.
 *
 *   Handshake: coord sends 32-byte nonce (AUTH_CHALLENGE).
 *              node replies with token_id + HMAC-SHA256(secret, nonce||token_id)
 *              (AUTH_RESPONSE).
 *              coord verifies and replies AUTH_RESULT with granted scope.
 *
 *   Receipt   = MsgContribReceipt signed with the coordinator's own key so the
 *               node can present the receipt later (cross-coordinator audit).
 *
 * The HMAC-SHA256 implementation is hand-rolled (no OpenSSL dep).
 */

#include "dist_protocol.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

// ─── SHA-256 and HMAC primitives ─────────────────────────────────────────────

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest sha256(const void* data, size_t len);

// HMAC-SHA256(key, msg).  Key may be any length.
Sha256Digest hmac_sha256(const uint8_t* key, size_t key_len,
                         const uint8_t* msg, size_t msg_len);

// Constant-time comparison.
bool ct_equal(const uint8_t* a, const uint8_t* b, size_t n);

// Hex helpers (lowercase, no separators).
std::string  to_hex(const uint8_t* p, size_t n);
std::vector<uint8_t> from_hex(const std::string& s);

// ─── Tokens ──────────────────────────────────────────────────────────────────

struct Token {
    std::string             token_id;      // public identifier
    std::array<uint8_t, 32> secret;        // shared with holder, never on wire
    uint32_t                scope = 0;     // AuthScope bitfield
    uint64_t                expires_at_us = 0; // 0 = never
    std::string             tenant;        // e.g. customer/pool name
    std::string             note;          // free-form metadata
};

class TokenStore {
public:
    // Add / replace a token.
    void put(const Token& t);

    // Look up by token_id.
    std::optional<Token> get(const std::string& token_id) const;

    // Remove (revoke).
    void revoke(const std::string& token_id);

    // Number of tokens.
    size_t size() const;

    // Persist to / load from a flat file.
    //   Format: one line per token, fields '|'-separated:
    //     token_id|hex(secret)|scope|expires_at_us|tenant|note
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // Issue a cryptographically-random token and add it.  Returns the full
    // token (with plaintext secret) so the operator can hand it out once.
    Token issue(const std::string& token_id,
                uint32_t scope,
                uint64_t expires_at_us,
                const std::string& tenant);

private:
    mutable std::mutex                       mtx_;
    std::unordered_map<std::string, Token>   by_id_;
};

// ─── Challenge / response helpers ────────────────────────────────────────────

// Fill `nonce` with CSPRNG bytes.  Uses /dev/urandom, falls back to std::random.
void random_bytes(uint8_t* buf, size_t n);

// Compute the HMAC a node would send in response to `challenge` using `token`.
void compute_auth_response(const uint8_t challenge[AUTH_CHALLENGE_BYTES],
                            const std::string& token_id,
                            const std::array<uint8_t, 32>& secret,
                            uint8_t mac_out[AUTH_MAC_BYTES]);

// Verify an AUTH_RESPONSE.  Returns true if MAC matches and token not expired.
bool verify_auth_response(const uint8_t challenge[AUTH_CHALLENGE_BYTES],
                          const MsgAuthResponse& resp,
                          const TokenStore& store,
                          uint64_t now_us,
                          uint32_t* out_granted_scope = nullptr,
                          std::string* out_reason = nullptr);

// ─── Receipts ────────────────────────────────────────────────────────────────

// Canonical byte sequence that gets HMACed to produce the receipt signature.
// Deterministic — same inputs always hash to the same bytes.
std::vector<uint8_t> serialize_receipt_for_mac(const MsgContribReceipt& r);

// Fill r.mac using the issuer's secret.
void sign_receipt(MsgContribReceipt& r,
                  const std::array<uint8_t, 32>& issuer_secret);

// Verify a receipt signature.  Caller supplies the issuer's secret.
bool verify_receipt(const MsgContribReceipt& r,
                    const std::array<uint8_t, 32>& issuer_secret);

} // namespace dist
