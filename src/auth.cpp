#include "auth.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <sys/stat.h>

namespace dist {

// ─── SHA-256 (FIPS 180-4) ────────────────────────────────────────────────────

namespace {

constexpr uint32_t K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (uint32_t)block[i*4]   << 24
             | (uint32_t)block[i*4+1] << 16
             | (uint32_t)block[i*4+2] << 8
             | (uint32_t)block[i*4+3];
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(W[i-15],7)  ^ rotr(W[i-15],18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr(W[i-2],17)  ^ rotr(W[i-2],19)  ^ (W[i-2]  >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],
             e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K256[i] + W[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

} // namespace

Sha256Digest sha256(const void* data, size_t len) {
    uint32_t state[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    const uint8_t* p = (const uint8_t*)data;
    size_t remaining = len;

    while (remaining >= 64) {
        sha256_compress(state, p);
        p += 64; remaining -= 64;
    }

    uint8_t final_block[128] = {};
    std::memcpy(final_block, p, remaining);
    final_block[remaining] = 0x80;

    size_t pad_len = (remaining < 56) ? 64 : 128;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; ++i)
        final_block[pad_len - 1 - i] = (uint8_t)(bit_len >> (i*8));

    sha256_compress(state, final_block);
    if (pad_len == 128) sha256_compress(state, final_block + 64);

    Sha256Digest out;
    for (int i = 0; i < 8; ++i) {
        out[i*4]   = (uint8_t)(state[i] >> 24);
        out[i*4+1] = (uint8_t)(state[i] >> 16);
        out[i*4+2] = (uint8_t)(state[i] >> 8);
        out[i*4+3] = (uint8_t)(state[i]);
    }
    return out;
}

// ─── HMAC-SHA256 (RFC 2104) ──────────────────────────────────────────────────

Sha256Digest hmac_sha256(const uint8_t* key, size_t key_len,
                         const uint8_t* msg, size_t msg_len) {
    uint8_t k0[64] = {};
    if (key_len > 64) {
        auto d = sha256(key, key_len);
        std::memcpy(k0, d.data(), 32);
    } else {
        std::memcpy(k0, key, key_len);
    }

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = k0[i] ^ 0x36;
        opad[i] = k0[i] ^ 0x5c;
    }

    // inner = SHA256(ipad || msg)
    std::vector<uint8_t> inner_in(64 + msg_len);
    std::memcpy(inner_in.data(), ipad, 64);
    if (msg_len) std::memcpy(inner_in.data() + 64, msg, msg_len);
    auto inner = sha256(inner_in.data(), inner_in.size());

    // outer = SHA256(opad || inner)
    uint8_t outer_in[64 + 32];
    std::memcpy(outer_in,      opad, 64);
    std::memcpy(outer_in + 64, inner.data(), 32);
    return sha256(outer_in, sizeof(outer_in));
}

bool ct_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= (uint8_t)(a[i] ^ b[i]);
    return acc == 0;
}

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string s; s.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s[i*2]   = H[(p[i] >> 4) & 0xF];
        s[i*2+1] = H[p[i] & 0xF];
    }
    return s;
}

std::vector<uint8_t> from_hex(const std::string& s) {
    std::vector<uint8_t> out;
    out.reserve(s.size() / 2);
    auto dec = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        int hi = dec(s[i]), lo = dec(s[i+1]);
        if (hi < 0 || lo < 0) break;
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return out;
}

// ─── Randomness ──────────────────────────────────────────────────────────────

void random_bytes(uint8_t* buf, size_t n) {
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (f) {
        size_t r = std::fread(buf, 1, n, f);
        std::fclose(f);
        if (r == n) return;
    }
    std::random_device rd;
    for (size_t i = 0; i < n; ++i) buf[i] = (uint8_t)(rd() & 0xFF);
}

// ─── TokenStore ──────────────────────────────────────────────────────────────

void TokenStore::put(const Token& t) {
    std::lock_guard<std::mutex> g(mtx_);
    by_id_[t.token_id] = t;
}

std::optional<Token> TokenStore::get(const std::string& id) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return std::nullopt;
    return it->second;
}

void TokenStore::revoke(const std::string& id) {
    std::lock_guard<std::mutex> g(mtx_);
    by_id_.erase(id);
}

size_t TokenStore::size() const {
    std::lock_guard<std::mutex> g(mtx_);
    return by_id_.size();
}

bool TokenStore::save(const std::string& path) const {
    std::lock_guard<std::mutex> g(mtx_);
    std::ofstream f(path);
    if (!f) return false;
    for (const auto& [id, t] : by_id_) {
        f << t.token_id << '|'
          << to_hex(t.secret.data(), t.secret.size()) << '|'
          << t.scope << '|'
          << t.expires_at_us << '|'
          << t.tenant << '|'
          << t.note << '\n';
    }
    f.close();
    ::chmod(path.c_str(), 0600); // secrets — owner-only
    return true;
}

bool TokenStore::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::lock_guard<std::mutex> g(mtx_);
    by_id_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> parts;
        std::string cur;
        for (char c : line) {
            if (c == '|') { parts.push_back(std::move(cur)); cur.clear(); }
            else cur.push_back(c);
        }
        parts.push_back(std::move(cur));
        if (parts.size() < 4) continue;
        Token t;
        t.token_id      = parts[0];
        auto sec        = from_hex(parts[1]);
        if (sec.size() != 32) continue;
        std::memcpy(t.secret.data(), sec.data(), 32);
        t.scope         = (uint32_t)std::stoul(parts[2]);
        t.expires_at_us = std::stoull(parts[3]);
        if (parts.size() > 4) t.tenant = parts[4];
        if (parts.size() > 5) t.note   = parts[5];
        by_id_[t.token_id] = std::move(t);
    }
    return true;
}

Token TokenStore::issue(const std::string& token_id,
                         uint32_t scope,
                         uint64_t expires_at_us,
                         const std::string& tenant) {
    Token t;
    t.token_id      = token_id;
    t.scope         = scope;
    t.expires_at_us = expires_at_us;
    t.tenant        = tenant;
    random_bytes(t.secret.data(), t.secret.size());
    put(t);
    return t;
}

// ─── Challenge / response ────────────────────────────────────────────────────

void compute_auth_response(const uint8_t challenge[AUTH_CHALLENGE_BYTES],
                            const std::string& token_id,
                            const std::array<uint8_t, 32>& secret,
                            uint8_t mac_out[AUTH_MAC_BYTES]) {
    std::vector<uint8_t> msg;
    msg.reserve(AUTH_CHALLENGE_BYTES + token_id.size());
    msg.insert(msg.end(), challenge, challenge + AUTH_CHALLENGE_BYTES);
    msg.insert(msg.end(), token_id.begin(), token_id.end());
    auto mac = hmac_sha256(secret.data(), secret.size(),
                           msg.data(), msg.size());
    std::memcpy(mac_out, mac.data(), AUTH_MAC_BYTES);
}

bool verify_auth_response(const uint8_t challenge[AUTH_CHALLENGE_BYTES],
                          const MsgAuthResponse& resp,
                          const TokenStore& store,
                          uint64_t now_us,
                          uint32_t* out_granted_scope,
                          std::string* out_reason) {
    std::string id(resp.token_id,
                   strnlen(resp.token_id, MAX_TOKEN_ID_LEN));
    auto t = store.get(id);
    if (!t) {
        if (out_reason) *out_reason = "unknown token_id";
        return false;
    }
    if (t->expires_at_us != 0 && t->expires_at_us < now_us) {
        if (out_reason) *out_reason = "token expired";
        return false;
    }

    uint8_t expected[AUTH_MAC_BYTES];
    compute_auth_response(challenge, id, t->secret, expected);
    if (!ct_equal(expected, resp.mac, AUTH_MAC_BYTES)) {
        if (out_reason) *out_reason = "bad MAC";
        return false;
    }

    uint32_t granted = t->scope & resp.scope_requested;
    if (granted == 0 && resp.scope_requested != 0) {
        if (out_reason) *out_reason = "requested scope not granted";
        return false;
    }
    if (out_granted_scope) *out_granted_scope = granted;
    return true;
}

// ─── Receipts ────────────────────────────────────────────────────────────────

std::vector<uint8_t> serialize_receipt_for_mac(const MsgContribReceipt& r) {
    // Serialize everything EXCEPT the mac field, in a fixed canonical order.
    std::vector<uint8_t> buf;
    auto push_bytes = [&](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        buf.insert(buf.end(), b, b + n);
    };
    push_bytes(r.node_id,  sizeof(r.node_id));
    push_bytes(r.tenant,   sizeof(r.tenant));
    push_bytes(&r.window_start_us,      sizeof(r.window_start_us));
    push_bytes(&r.window_end_us,        sizeof(r.window_end_us));
    push_bytes(&r.tokens_processed,     sizeof(r.tokens_processed));
    push_bytes(&r.layer_bytes_forwarded, sizeof(r.layer_bytes_forwarded));
    push_bytes(&r.layer_seconds,        sizeof(r.layer_seconds));
    push_bytes(r.issuer_id, sizeof(r.issuer_id));
    return buf;
}

void sign_receipt(MsgContribReceipt& r,
                  const std::array<uint8_t, 32>& issuer_secret) {
    auto msg = serialize_receipt_for_mac(r);
    auto mac = hmac_sha256(issuer_secret.data(), issuer_secret.size(),
                           msg.data(), msg.size());
    std::memcpy(r.mac, mac.data(), AUTH_MAC_BYTES);
}

bool verify_receipt(const MsgContribReceipt& r,
                    const std::array<uint8_t, 32>& issuer_secret) {
    auto msg = serialize_receipt_for_mac(r);
    auto expected = hmac_sha256(issuer_secret.data(), issuer_secret.size(),
                                msg.data(), msg.size());
    return ct_equal(expected.data(), r.mac, AUTH_MAC_BYTES);
}

} // namespace dist
