#include "agent_identity.h"

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cstring>

namespace dist {

bool ed25519_generate(std::vector<uint8_t>& priv_out,
                      std::vector<uint8_t>& pub_out) {
    EVP_PKEY_CTX* pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    if (!pctx) return false;
    if (EVP_PKEY_keygen_init(pctx) <= 0) {
        EVP_PKEY_CTX_free(pctx);
        return false;
    }
    EVP_PKEY* pkey = nullptr;
    if (EVP_PKEY_keygen(pctx, &pkey) <= 0 || !pkey) {
        EVP_PKEY_CTX_free(pctx);
        return false;
    }
    EVP_PKEY_CTX_free(pctx);

    priv_out.assign(ED25519_PRIV_SIZE, 0);
    pub_out.assign(ED25519_PUB_SIZE, 0);
    size_t plen = ED25519_PRIV_SIZE;
    size_t qlen = ED25519_PUB_SIZE;
    if (EVP_PKEY_get_raw_private_key(pkey, priv_out.data(), &plen) <= 0 ||
        EVP_PKEY_get_raw_public_key(pkey, pub_out.data(), &qlen) <= 0 ||
        plen != ED25519_PRIV_SIZE || qlen != ED25519_PUB_SIZE) {
        EVP_PKEY_free(pkey);
        priv_out.clear();
        pub_out.clear();
        return false;
    }
    EVP_PKEY_free(pkey);
    return true;
}

bool ed25519_pub_from_priv(const std::vector<uint8_t>& priv,
                           std::vector<uint8_t>& pub_out) {
    if (priv.size() != ED25519_PRIV_SIZE) return false;
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr,
                                                  priv.data(), priv.size());
    if (!pkey) return false;
    pub_out.assign(ED25519_PUB_SIZE, 0);
    size_t qlen = ED25519_PUB_SIZE;
    bool ok = EVP_PKEY_get_raw_public_key(pkey, pub_out.data(), &qlen) > 0
              && qlen == ED25519_PUB_SIZE;
    EVP_PKEY_free(pkey);
    if (!ok) pub_out.clear();
    return ok;
}

bool ed25519_sign(const std::vector<uint8_t>& priv,
                  const std::string& msg,
                  std::vector<uint8_t>& sig_out) {
    if (priv.size() != ED25519_PRIV_SIZE) return false;
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr,
                                                  priv.data(), priv.size());
    if (!pkey) return false;

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (!mdctx) { EVP_PKEY_free(pkey); return false; }

    bool ok = false;
    // Ed25519 uses one-shot EVP_DigestSign (no init-update-final).
    if (EVP_DigestSignInit(mdctx, nullptr, nullptr, nullptr, pkey) > 0) {
        size_t slen = ED25519_SIG_SIZE;
        sig_out.assign(ED25519_SIG_SIZE, 0);
        if (EVP_DigestSign(mdctx, sig_out.data(), &slen,
                           reinterpret_cast<const unsigned char*>(msg.data()),
                           msg.size()) > 0 && slen == ED25519_SIG_SIZE) {
            ok = true;
        }
    }
    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    if (!ok) sig_out.clear();
    return ok;
}

// Standard base64-url alphabet: A-Z a-z 0-9 - _ (no padding).
static const char kB64UrlAlphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

std::string b64url_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i+1]) << 8) | data[i+2];
        out.push_back(kB64UrlAlphabet[(v >> 18) & 0x3f]);
        out.push_back(kB64UrlAlphabet[(v >> 12) & 0x3f]);
        out.push_back(kB64UrlAlphabet[(v >> 6) & 0x3f]);
        out.push_back(kB64UrlAlphabet[v & 0x3f]);
        i += 3;
    }
    size_t rem = len - i;
    if (rem == 1) {
        uint32_t v = uint32_t(data[i]) << 16;
        out.push_back(kB64UrlAlphabet[(v >> 18) & 0x3f]);
        out.push_back(kB64UrlAlphabet[(v >> 12) & 0x3f]);
    } else if (rem == 2) {
        uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i+1]) << 8);
        out.push_back(kB64UrlAlphabet[(v >> 18) & 0x3f]);
        out.push_back(kB64UrlAlphabet[(v >> 12) & 0x3f]);
        out.push_back(kB64UrlAlphabet[(v >> 6) & 0x3f]);
    }
    return out;
}

} // namespace dist
