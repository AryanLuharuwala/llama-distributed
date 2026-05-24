#pragma once
//
// P15: FP8 (E4M3) activation compression for the ACTV wire.
//
// FP8 E4M3 packs each element into one byte (1 sign + 4 exponent + 3
// mantissa).  Combined with a per-tensor float32 scale factor, it
// reduces hidden-state bandwidth by 4× vs fp32 and 2× vs fp16 with a
// round-trip error that lands well below the noise floor of the
// inference itself (≤ ~1% relative on normalized hidden states).
//
// Format follows the OCP MX-FP8 / NVIDIA E4M3 convention:
//   bias = 7
//   exponent = 0b1111 + mantissa = 0b111 → NaN (only NaN; no Inf)
//   all other exponent = 0b1111 → normal numbers, max = 448
//
// Wire encoding for dtype=3 (`fp8_e4m3`):
//
//     TensorHeader { dtype=3, n_tokens, n_embd, ... }
//     float32        scale
//     uint8_t[n_tokens * n_embd]   E4M3 bytes
//
// To recover, the receiver multiplies the decoded fp32 value by the
// scale.  The scale is computed from the per-tensor amax so that the
// largest element lands at ±448 (the max-normal E4M3 value).
//
// We do not use __nv_fp8 or any NVIDIA intrinsic — this is plain C++17
// scalar code so it builds on CPU-only hosts.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dist {
namespace fp8 {

constexpr float E4M3_MAX = 448.0f;

// ─── scalar f32 → E4M3 (saturating, round-to-nearest-even) ─────────────
//
// Inputs above ±448 saturate to ±448.  NaN inputs saturate to 0x7F (the
// E4M3 NaN encoding).
inline uint8_t encode_e4m3(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    uint32_t bits  = v.u;
    uint32_t sign  = (bits >> 31) & 1u;
    int32_t  e_in  = (int32_t)((bits >> 23) & 0xFFu) - 127;
    uint32_t m_in  = bits & 0x7FFFFFu;

    // ±0 (and tiny denormals from upstream)
    if ((bits & 0x7FFFFFFFu) == 0u) return (uint8_t)(sign << 7);

    // NaN / Inf input
    if (e_in == 128) {
        if (m_in != 0) return 0x7Fu;        // NaN
        return (uint8_t)((sign << 7) | 0x7Eu); // ±Inf → saturate to ±448
    }

    // Saturate large magnitudes
    float ax = std::fabs(x);
    if (ax >= E4M3_MAX) {
        return (uint8_t)((sign << 7) | 0x7Eu); // 0.1111.110 = ±448
    }

    int32_t e_out = e_in + 7;
    uint32_t m_out;

    if (e_out <= 0) {
        // E4M3 subnormal: implicit leading 1 becomes explicit and we
        // shift mantissa right to align under exp=0.
        uint32_t mant = (1u << 23) | m_in;  // 24-bit significand
        int rshift = (1 - e_out) + 20;       // bring down to 3-bit field
        if (rshift > 31) return (uint8_t)(sign << 7); // underflow to zero
        uint32_t round_bit = (rshift > 0) ? ((mant >> (rshift - 1)) & 1u) : 0u;
        uint32_t sticky    = (rshift > 1) ? ((mant & ((1u << (rshift - 1)) - 1u)) ? 1u : 0u) : 0u;
        m_out              = mant >> rshift;
        // Round to nearest, ties to even
        if (round_bit && (sticky || (m_out & 1u))) m_out++;
        if (m_out >= 8u) {
            // Carried into the normal range
            return (uint8_t)((sign << 7) | (1u << 3));
        }
        return (uint8_t)((sign << 7) | (m_out & 0x7u));
    }

    // E4M3 normal range: exp field 1..15 (15 with mantissa 6 = max).
    // f32 mantissa is 23 bits; truncate to 3 bits with RTNE.
    uint32_t round_bit = (m_in >> 19) & 1u;
    uint32_t sticky    = (m_in & 0x7FFFFu) ? 1u : 0u;
    m_out              = (m_in >> 20) & 0x7u;
    uint32_t exp_field = (uint32_t)e_out;

    if (round_bit && (sticky || (m_out & 1u))) {
        m_out++;
        if (m_out == 8u) {
            m_out = 0u;
            exp_field++;
        }
    }

    if (exp_field >= 0xFu) {
        if (exp_field == 0xFu && m_out <= 6u) {
            return (uint8_t)((sign << 7) | (0xFu << 3) | m_out);
        }
        return (uint8_t)((sign << 7) | 0x7Eu); // saturate
    }
    return (uint8_t)((sign << 7) | (exp_field << 3) | m_out);
}

// ─── scalar E4M3 → f32 ────────────────────────────────────────────────
inline float decode_e4m3(uint8_t b) {
    uint32_t sign      = (b >> 7) & 1u;
    uint32_t exp_field = (b >> 3) & 0xFu;
    uint32_t mant      = b & 0x7u;

    // NaN: exp=15, mant=7
    if (exp_field == 0xFu && mant == 0x7u) {
        union { uint32_t u; float f; } v;
        v.u = 0x7FC00000u;
        return v.f;
    }

    float out;
    if (exp_field == 0u) {
        // Subnormal: value = mant * 2^(-9)
        out = (float)mant * (1.0f / 512.0f);
    } else {
        // Normal: value = (1 + mant/8) * 2^(exp-7)
        int e = (int)exp_field - 7;
        float frac = 1.0f + (float)mant * (1.0f / 8.0f);
        out = std::ldexp(frac, e);
    }
    return sign ? -out : out;
}

// ─── tensor-level encode ──────────────────────────────────────────────
//
// `src` is `n_elems` float32 values.
// On output, `*out_scale` holds the per-tensor scale and
// `out_bytes` is filled with `n_elems` E4M3 bytes.
inline void encode_tensor(const float* src, size_t n_elems,
                          float* out_scale, uint8_t* out_bytes) {
    float amax = 0.0f;
    for (size_t i = 0; i < n_elems; ++i) {
        float a = std::fabs(src[i]);
        if (a > amax) amax = a;
    }
    // Scale so the largest magnitude maps to ±E4M3_MAX exactly.
    // Clamp to a small floor to avoid division by zero on all-zero tensors.
    float scale = (amax > 1e-30f) ? (amax / E4M3_MAX) : 1.0f;
    float inv   = 1.0f / scale;

    for (size_t i = 0; i < n_elems; ++i) {
        out_bytes[i] = encode_e4m3(src[i] * inv);
    }
    *out_scale = scale;
}

// ─── tensor-level decode ──────────────────────────────────────────────
inline void decode_tensor(const uint8_t* src, size_t n_elems,
                          float scale, float* dst) {
    // F-WIRE-06: the scale is peer-supplied; NaN/Inf would poison every
    // downstream multiplication. Clamp to a finite, non-negative value.
    if (!std::isfinite(scale) || scale < 0.0f) scale = 0.0f;
    for (size_t i = 0; i < n_elems; ++i) {
        dst[i] = decode_e4m3(src[i]) * scale;
    }
}

// ─── convenience: byte-vector wrappers ────────────────────────────────
inline std::vector<uint8_t> pack_e4m3(const float* src, size_t n_elems,
                                      float* out_scale) {
    std::vector<uint8_t> bytes(n_elems);
    encode_tensor(src, n_elems, out_scale, bytes.data());
    return bytes;
}

inline std::vector<float> unpack_e4m3(const uint8_t* src, size_t n_elems,
                                      float scale) {
    std::vector<float> out(n_elems);
    decode_tensor(src, n_elems, scale, out.data());
    return out;
}

} // namespace fp8
} // namespace dist
