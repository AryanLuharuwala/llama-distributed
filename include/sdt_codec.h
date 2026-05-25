// sdt_codec.h — on-wire tensor formats for the sd.cpp per-role workers
// (CF12-W1a / CF12-W2).
//
// Three formats live here:
//
//   SDT  — single tensor.  Self-describing header + raw little-endian bytes.
//          Used as the universal "one tensor" hop on a P2P channel (e.g. the
//          latent handed from UNet → VAE, or an image returned from VAE).
//
//   SDCD — named container.  A bag of (name → SDT) plus an optional small
//          key/value string table.  Used for the TE role's output: a single
//          frame carrying prompt_embeds + pooled + per-backbone extras
//          (e.g. SDXL `text_embeds` + `time_ids`; Flux `t5_attn_mask`).
//
//   UPLD — wire-compatible C++ encoder/decoder for python's
//          dpp_runtime/unet_payload.UNetStepPayload.  Lets a sd.cpp UNet
//          block-stage hop into / out of a python block-stage mid-pipeline.
//
// All three are big-endian on the header so a C++ rig and a Python rig agree
// even when one is on a little-endian host and writes via struct.pack(">…").
// Payload bytes are LE fp16 (sample/residuals) or LE fp32 (TE prompt_embeds
// when the backbone wants fp32) — matching the convention used by python.
//
// Tensor dtype is encoded explicitly so a sd.cpp rig can hand fp32 to a
// python rig that downcasts on receive, or vice-versa, without guessing.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dist {

// ---- dtype tags ----------------------------------------------------------
// Stored as u8 on the wire.  Numeric values are stable; do not renumber.
enum class SdtDType : uint8_t {
    F32  = 0,
    F16  = 1,
    BF16 = 2,
    I32  = 3,
    I64  = 4,
    U8   = 5,
};

inline uint32_t sdt_dtype_bytes(SdtDType t) {
    switch (t) {
        case SdtDType::F32:  return 4;
        case SdtDType::I32:  return 4;
        case SdtDType::I64:  return 8;
        case SdtDType::F16:  return 2;
        case SdtDType::BF16: return 2;
        case SdtDType::U8:   return 1;
    }
    return 0;
}

inline const char* sdt_dtype_name(SdtDType t) {
    switch (t) {
        case SdtDType::F32:  return "f32";
        case SdtDType::F16:  return "f16";
        case SdtDType::BF16: return "bf16";
        case SdtDType::I32:  return "i32";
        case SdtDType::I64:  return "i64";
        case SdtDType::U8:   return "u8";
    }
    return "?";
}

// ─── SDT ──────────────────────────────────────────────────────────────────
// Header (big-endian):
//   magic        u32  = "SDT1"  (0x53445431)
//   ver          u8   = 1
//   dtype        u8   (SdtDType)
//   rank         u8
//   reserved     u8   = 0
//   dims         u32 × rank   (row-major, NCHW for image tensors)
//   nbytes       u32
//   payload      nbytes
// Total header = 12 + 4*rank bytes.

constexpr uint32_t SDT_MAGIC = 0x53445431u;  // "SDT1"
constexpr uint8_t  SDT_VER   = 1;

struct SdtTensor {
    SdtDType              dtype = SdtDType::F16;
    std::vector<uint32_t> dims;
    std::vector<uint8_t>  data;  // LE host bytes; size must equal product(dims) * dtype_bytes

    // Convenience: bytes that the payload SHOULD occupy.
    uint64_t expected_nbytes() const {
        uint64_t n = sdt_dtype_bytes(dtype);
        for (uint32_t d : dims) n *= d;
        return n;
    }
};

// Returns true on success.  On failure, `err` is set to a short reason.
bool sdt_encode(const SdtTensor& t, std::vector<uint8_t>& out, std::string& err);
bool sdt_decode(const uint8_t* buf, size_t n, SdtTensor& out, std::string& err);

// Convenience: decode/encode to/from a std::vector slice.
inline bool sdt_decode(const std::vector<uint8_t>& buf, SdtTensor& out, std::string& err) {
    return sdt_decode(buf.data(), buf.size(), out, err);
}

// ─── SDCD ─────────────────────────────────────────────────────────────────
// Named container.  Used by encode_text to ship prompt_embeds + pooled +
// per-backbone extras in a single frame.
//
// Header (big-endian):
//   magic        u32  = "SDCD"  (0x53444344)
//   ver          u8   = 1
//   flags        u8   = 0       (reserved)
//   n_kv         u16            number of metadata key/value strings
//   n_tensors    u16            number of named tensors
//   reserved     u16  = 0
//   for each kv:
//     klen       u16
//     vlen       u16
//     key bytes  (klen)
//     val bytes  (vlen)
//   for each tensor:
//     nlen       u16
//     name bytes (nlen)
//     full SDT frame (header + payload)

constexpr uint32_t SDCD_MAGIC = 0x53444344u;  // "SDCD"
constexpr uint8_t  SDCD_VER   = 1;

struct SdcdKV {
    std::string key;
    std::string val;
};

struct SdcdNamed {
    std::string name;
    SdtTensor   tensor;
};

struct SdcdFrame {
    std::vector<SdcdKV>     kv;       // small metadata (model id, role, etc.)
    std::vector<SdcdNamed>  tensors;  // ordered, name-addressed

    // Lookup helpers — return nullptr if missing.
    const SdtTensor* find(const char* name) const {
        for (const auto& t : tensors) if (t.name == name) return &t.tensor;
        return nullptr;
    }
    const std::string* find_meta(const char* key) const {
        for (const auto& e : kv) if (e.key == key) return &e.val;
        return nullptr;
    }
};

bool sdcd_encode(const SdcdFrame& f, std::vector<uint8_t>& out, std::string& err);
bool sdcd_decode(const uint8_t* buf, size_t n, SdcdFrame& out, std::string& err);

inline bool sdcd_decode(const std::vector<uint8_t>& buf, SdcdFrame& out, std::string& err) {
    return sdcd_decode(buf.data(), buf.size(), out, err);
}

// ─── UPLD ─────────────────────────────────────────────────────────────────
// C++ counterpart of python dpp_runtime.unet_payload.UNetStepPayload.
// Format is fixed — see python module docstring for the canonical spec.
// We re-implement it here so a sd.cpp UNet block-stage can ship/receive
// running hidden states to/from a python block-stage seamlessly.

constexpr uint32_t UPLD_MAGIC          = 0x55504C44u;  // "UPLD"
constexpr uint8_t  UPLD_VER            = 1;
constexpr uint8_t  UPLD_FLAG_FINAL     = 0x01;

struct UpldTensor {
    std::vector<uint32_t> dims;
    std::vector<uint8_t>  data;  // raw LE fp16 bytes (matches python)
};

struct UpldPayload {
    UpldTensor              sample;
    std::vector<UpldTensor> residuals;  // bottom-up stack order
    uint32_t                step_idx      = 0;
    float                   timestep      = 0.0f;
    bool                    is_final_step = false;
};

bool upld_encode(const UpldPayload& p, std::vector<uint8_t>& out, std::string& err);
bool upld_decode(const uint8_t* buf, size_t n, UpldPayload& out, std::string& err);

inline bool upld_decode(const std::vector<uint8_t>& buf, UpldPayload& out, std::string& err) {
    return upld_decode(buf.data(), buf.size(), out, err);
}

}  // namespace dist
