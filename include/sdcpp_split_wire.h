// sdcpp_split_wire.h — SDCD wire codec for the CF12-W6a UNet block-split
// carry state.  Wraps a sd_split_state_t (h, hs[], emb) into the existing
// SDCD framing so a sd.cpp rig and a python rig can swap intermediate
// half-state without inventing a new on-wire envelope.
//
// SDCD frame layout (kind = "upld_sdcpp_half0"):
//   meta:    kind=upld_sdcpp_half0
//            hs_count=<N>
//   tensors: "h"        fp32, dims = (1, C, H, W)
//            "emb"      fp32, dims = (1, T)
//            "hs.0".."hs.N-1"  fp32, dims = (1, C_i, H_i, W_i)
//
// Both encoder and decoder are header-only so the test binary and the
// dist-sdcpp-worker share a single source of truth.

#pragma once

#include "stable-diffusion.h"
#include "sdt_codec.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace dist {

inline constexpr const char* SDCPP_HALF0_KIND = "upld_sdcpp_half0";

inline bool sdcpp_carry_to_sdcd(const sd_split_state_t* st,
                                std::vector<uint8_t>& out,
                                std::string& err) {
    SdcdFrame f;
    f.kv.push_back({"kind", SDCPP_HALF0_KIND});

    int hs_count = 0;
    sd_split_state_get_carry_count(st, &hs_count);
    f.kv.push_back({"hs_count", std::to_string(hs_count)});

    auto add_tensor = [&](const char* name) -> bool {
        const float*   d  = nullptr;
        const int64_t* sh = nullptr;
        int            nd = 0;
        int rc = sd_split_state_get_carry_tensor(st, name, &d, &sh, &nd);
        if (rc != SD_SPLIT_OK) return false;
        SdcdNamed n;
        n.name = name;
        n.tensor.dtype = SdtDType::F32;
        n.tensor.dims.assign(sh, sh + nd);
        size_t nbytes = sizeof(float);
        for (int i = 0; i < nd; ++i) nbytes *= static_cast<size_t>(sh[i]);
        n.tensor.data.assign(reinterpret_cast<const uint8_t*>(d),
                             reinterpret_cast<const uint8_t*>(d) + nbytes);
        f.tensors.push_back(std::move(n));
        return true;
    };

    if (!add_tensor("h"))   { err = "carry missing h";   return false; }
    if (!add_tensor("emb")) { err = "carry missing emb"; return false; }
    for (int i = 0; i < hs_count; ++i) {
        char name[16];
        std::snprintf(name, sizeof(name), "hs.%d", i);
        if (!add_tensor(name)) {
            err = std::string("carry missing ") + name;
            return false;
        }
    }
    return sdcd_encode(f, out, err);
}

inline bool sdcpp_sdcd_to_carry(const uint8_t* buf, size_t n,
                                sd_split_state_t* st,
                                std::string& err) {
    SdcdFrame f;
    if (!sdcd_decode(buf, n, f, err)) return false;
    const std::string* kv_hs = f.find_meta("hs_count");
    if (kv_hs == nullptr) { err = "missing hs_count meta"; return false; }
    int hs_count = std::atoi(kv_hs->c_str());
    sd_split_state_set_hs_count(st, hs_count);

    auto set_named = [&](const char* name) -> bool {
        const SdtTensor* t = f.find(name);
        if (t == nullptr) return false;
        std::vector<int64_t> shape(t->dims.begin(), t->dims.end());
        int rc = sd_split_state_set_carry_tensor(
            st, name,
            reinterpret_cast<const float*>(t->data.data()),
            shape.data(),
            static_cast<int>(shape.size()));
        return rc == SD_SPLIT_OK;
    };
    if (!set_named("h"))   { err = "wire missing h";   return false; }
    if (!set_named("emb")) { err = "wire missing emb"; return false; }
    for (int i = 0; i < hs_count; ++i) {
        char name[16];
        std::snprintf(name, sizeof(name), "hs.%d", i);
        if (!set_named(name)) {
            err = std::string("wire missing ") + name;
            return false;
        }
    }
    return true;
}

inline bool sdcpp_sdcd_to_carry(const std::vector<uint8_t>& in,
                                sd_split_state_t* st,
                                std::string& err) {
    return sdcpp_sdcd_to_carry(in.data(), in.size(), st, err);
}

}  // namespace dist
