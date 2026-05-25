// sdt_codec.cpp — SDT / SDCD / UPLD encoders & decoders.  See sdt_codec.h
// for the wire-format spec.

#include "sdt_codec.h"

#include <cstring>

namespace dist {

namespace {

// All headers are big-endian on the wire.  Implementations below avoid
// depending on htonl/htons so this builds cleanly on Windows too.

void put_u8(std::vector<uint8_t>& out, uint8_t v)  { out.push_back(v); }
void put_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(uint8_t((v >> 8) & 0xFF));
    out.push_back(uint8_t(v & 0xFF));
}
void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(uint8_t((v >> 24) & 0xFF));
    out.push_back(uint8_t((v >> 16) & 0xFF));
    out.push_back(uint8_t((v >>  8) & 0xFF));
    out.push_back(uint8_t(v & 0xFF));
}
void put_f32(std::vector<uint8_t>& out, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    put_u32(out, bits);
}

bool need(const uint8_t* buf, size_t n, size_t off, size_t want) {
    return off + want <= n;
}
uint8_t get_u8(const uint8_t* b, size_t& off) {
    return b[off++];
}
uint16_t get_u16(const uint8_t* b, size_t& off) {
    uint16_t v = (uint16_t(b[off]) << 8) | uint16_t(b[off + 1]);
    off += 2;
    return v;
}
uint32_t get_u32(const uint8_t* b, size_t& off) {
    uint32_t v = (uint32_t(b[off]) << 24) | (uint32_t(b[off + 1]) << 16) |
                 (uint32_t(b[off + 2]) << 8) | uint32_t(b[off + 3]);
    off += 4;
    return v;
}
float get_f32(const uint8_t* b, size_t& off) {
    uint32_t bits = get_u32(b, off);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

}  // namespace

// ─── SDT ──────────────────────────────────────────────────────────────────

bool sdt_encode(const SdtTensor& t, std::vector<uint8_t>& out, std::string& err) {
    if (t.dims.size() > 8) { err = "sdt_encode: rank > 8"; return false; }
    uint64_t want = t.expected_nbytes();
    if (t.data.size() != want) {
        err = "sdt_encode: data size mismatch (got " +
              std::to_string(t.data.size()) + ", want " +
              std::to_string(want) + ")";
        return false;
    }
    if (want > 0xFFFFFFFFull) { err = "sdt_encode: payload > 4 GiB"; return false; }

    out.clear();
    out.reserve(12 + 4 * t.dims.size() + t.data.size());
    put_u32(out, SDT_MAGIC);
    put_u8(out, SDT_VER);
    put_u8(out, uint8_t(t.dtype));
    put_u8(out, uint8_t(t.dims.size()));
    put_u8(out, 0);  // reserved
    for (uint32_t d : t.dims) put_u32(out, d);
    put_u32(out, uint32_t(want));
    out.insert(out.end(), t.data.begin(), t.data.end());
    return true;
}

bool sdt_decode(const uint8_t* buf, size_t n, SdtTensor& out, std::string& err) {
    size_t off = 0;
    if (!need(buf, n, off, 8)) { err = "sdt_decode: short header"; return false; }
    if (get_u32(buf, off) != SDT_MAGIC) { err = "sdt_decode: bad magic"; return false; }
    uint8_t ver  = get_u8(buf, off);
    if (ver != SDT_VER) { err = "sdt_decode: bad ver"; return false; }
    out.dtype = SdtDType(get_u8(buf, off));
    uint8_t rank = get_u8(buf, off);
    (void)get_u8(buf, off);  // reserved
    if (rank > 8) { err = "sdt_decode: rank > 8"; return false; }
    if (!need(buf, n, off, 4u * rank + 4u)) { err = "sdt_decode: truncated dims"; return false; }

    out.dims.clear();
    out.dims.reserve(rank);
    for (uint8_t i = 0; i < rank; ++i) out.dims.push_back(get_u32(buf, off));
    uint32_t nbytes = get_u32(buf, off);

    uint64_t want = out.expected_nbytes();
    if (want != nbytes) {
        err = "sdt_decode: nbytes(" + std::to_string(nbytes) +
              ") != dims*dtype(" + std::to_string(want) + ")";
        return false;
    }
    if (!need(buf, n, off, nbytes)) { err = "sdt_decode: truncated payload"; return false; }
    out.data.assign(buf + off, buf + off + nbytes);
    return true;
}

// ─── SDCD ─────────────────────────────────────────────────────────────────

bool sdcd_encode(const SdcdFrame& f, std::vector<uint8_t>& out, std::string& err) {
    if (f.kv.size() > 0xFFFFu)      { err = "sdcd: too many kv";       return false; }
    if (f.tensors.size() > 0xFFFFu) { err = "sdcd: too many tensors";  return false; }

    out.clear();
    put_u32(out, SDCD_MAGIC);
    put_u8(out, SDCD_VER);
    put_u8(out, 0);  // flags
    put_u16(out, uint16_t(f.kv.size()));
    put_u16(out, uint16_t(f.tensors.size()));
    put_u16(out, 0);  // reserved

    for (const auto& e : f.kv) {
        if (e.key.size() > 0xFFFFu || e.val.size() > 0xFFFFu) {
            err = "sdcd: kv string > 64 KiB"; return false;
        }
        put_u16(out, uint16_t(e.key.size()));
        put_u16(out, uint16_t(e.val.size()));
        out.insert(out.end(), e.key.begin(), e.key.end());
        out.insert(out.end(), e.val.begin(), e.val.end());
    }

    for (const auto& t : f.tensors) {
        if (t.name.size() > 0xFFFFu) { err = "sdcd: tensor name > 64 KiB"; return false; }
        put_u16(out, uint16_t(t.name.size()));
        out.insert(out.end(), t.name.begin(), t.name.end());
        std::vector<uint8_t> tbuf;
        if (!sdt_encode(t.tensor, tbuf, err)) return false;
        out.insert(out.end(), tbuf.begin(), tbuf.end());
    }
    return true;
}

bool sdcd_decode(const uint8_t* buf, size_t n, SdcdFrame& out, std::string& err) {
    size_t off = 0;
    if (!need(buf, n, off, 12)) { err = "sdcd: short header"; return false; }
    if (get_u32(buf, off) != SDCD_MAGIC) { err = "sdcd: bad magic"; return false; }
    uint8_t ver = get_u8(buf, off);
    if (ver != SDCD_VER) { err = "sdcd: bad ver"; return false; }
    (void)get_u8(buf, off);  // flags
    uint16_t n_kv = get_u16(buf, off);
    uint16_t n_t  = get_u16(buf, off);
    (void)get_u16(buf, off);  // reserved

    out.kv.clear();
    out.tensors.clear();
    out.kv.reserve(n_kv);
    out.tensors.reserve(n_t);

    for (uint16_t i = 0; i < n_kv; ++i) {
        if (!need(buf, n, off, 4)) { err = "sdcd: kv hdr"; return false; }
        uint16_t klen = get_u16(buf, off);
        uint16_t vlen = get_u16(buf, off);
        if (!need(buf, n, off, size_t(klen) + size_t(vlen))) {
            err = "sdcd: kv body"; return false;
        }
        SdcdKV e;
        e.key.assign((const char*)(buf + off), klen);   off += klen;
        e.val.assign((const char*)(buf + off), vlen);   off += vlen;
        out.kv.push_back(std::move(e));
    }

    for (uint16_t i = 0; i < n_t; ++i) {
        if (!need(buf, n, off, 2)) { err = "sdcd: tname len"; return false; }
        uint16_t nlen = get_u16(buf, off);
        if (!need(buf, n, off, nlen)) { err = "sdcd: tname body"; return false; }
        SdcdNamed nt;
        nt.name.assign((const char*)(buf + off), nlen);  off += nlen;
        SdtTensor t;
        size_t before = off;
        if (!sdt_decode(buf + off, n - off, t, err)) return false;
        // sdt_decode doesn't tell us how far it consumed; recompute by re-encoding length.
        // Cheap path: parse header to learn nbytes + dims, then advance.
        if (n - before < 8) { err = "sdcd: nested sdt short"; return false; }
        size_t inner = before;
        inner += 4;                          // magic
        inner += 1;                          // ver
        inner += 1;                          // dtype
        uint8_t rank = buf[inner];           inner += 1;
        inner += 1;                          // reserved
        inner += 4u * rank;                  // dims
        if (n < inner + 4) { err = "sdcd: nested sdt hdr"; return false; }
        uint32_t nbytes = (uint32_t(buf[inner]) << 24) | (uint32_t(buf[inner + 1]) << 16) |
                          (uint32_t(buf[inner + 2]) <<  8) |  uint32_t(buf[inner + 3]);
        inner += 4;
        if (n < inner + nbytes) { err = "sdcd: nested sdt body"; return false; }
        inner += nbytes;
        off = inner;
        nt.tensor = std::move(t);
        out.tensors.push_back(std::move(nt));
    }
    return true;
}

// ─── UPLD ─────────────────────────────────────────────────────────────────

bool upld_encode(const UpldPayload& p, std::vector<uint8_t>& out, std::string& err) {
    if (p.residuals.size() > 0xFFFFu) { err = "upld: too many residuals"; return false; }
    if (p.sample.dims.size() > 0xFFu) { err = "upld: sample rank > 255"; return false; }

    out.clear();
    put_u32(out, UPLD_MAGIC);
    put_u8(out, UPLD_VER);
    put_u8(out, p.is_final_step ? UPLD_FLAG_FINAL : 0);
    put_u16(out, uint16_t(p.residuals.size()));
    put_u32(out, p.step_idx);
    put_f32(out, p.timestep);

    // Sample header: rank u8 / dims u32xR / nbytes u32.  Then residuals' headers
    // in the same shape.  Then residuals' payloads in order, then sample bytes.
    put_u8(out, uint8_t(p.sample.dims.size()));
    for (uint32_t d : p.sample.dims) put_u32(out, d);
    put_u32(out, uint32_t(p.sample.data.size()));

    for (const auto& r : p.residuals) {
        if (r.dims.size() > 0xFFu) { err = "upld: residual rank > 255"; return false; }
        put_u8(out, uint8_t(r.dims.size()));
        for (uint32_t d : r.dims) put_u32(out, d);
        put_u32(out, uint32_t(r.data.size()));
    }

    for (const auto& r : p.residuals) {
        out.insert(out.end(), r.data.begin(), r.data.end());
    }
    out.insert(out.end(), p.sample.data.begin(), p.sample.data.end());
    return true;
}

bool upld_decode(const uint8_t* buf, size_t n, UpldPayload& out, std::string& err) {
    size_t off = 0;
    if (!need(buf, n, off, 16)) { err = "upld: short header"; return false; }
    if (get_u32(buf, off) != UPLD_MAGIC) { err = "upld: bad magic"; return false; }
    uint8_t ver = get_u8(buf, off);
    if (ver != UPLD_VER) { err = "upld: bad ver"; return false; }
    uint8_t flags = get_u8(buf, off);
    uint16_t n_res = get_u16(buf, off);
    out.step_idx  = get_u32(buf, off);
    out.timestep  = get_f32(buf, off);
    out.is_final_step = (flags & UPLD_FLAG_FINAL) != 0;

    if (!need(buf, n, off, 1)) { err = "upld: sample rank"; return false; }
    uint8_t s_rank = get_u8(buf, off);
    if (!need(buf, n, off, 4u * s_rank + 4u)) { err = "upld: sample dims"; return false; }
    out.sample.dims.clear();
    out.sample.dims.reserve(s_rank);
    for (uint8_t i = 0; i < s_rank; ++i) out.sample.dims.push_back(get_u32(buf, off));
    uint32_t s_nbytes = get_u32(buf, off);

    std::vector<std::pair<std::vector<uint32_t>, uint32_t>> res_hdrs;
    res_hdrs.reserve(n_res);
    for (uint16_t i = 0; i < n_res; ++i) {
        if (!need(buf, n, off, 1)) { err = "upld: res rank"; return false; }
        uint8_t r_rank = get_u8(buf, off);
        if (!need(buf, n, off, 4u * r_rank + 4u)) { err = "upld: res dims"; return false; }
        std::vector<uint32_t> dims;
        dims.reserve(r_rank);
        for (uint8_t j = 0; j < r_rank; ++j) dims.push_back(get_u32(buf, off));
        uint32_t r_nbytes = get_u32(buf, off);
        res_hdrs.emplace_back(std::move(dims), r_nbytes);
    }

    out.residuals.clear();
    out.residuals.reserve(n_res);
    for (auto& h : res_hdrs) {
        if (!need(buf, n, off, h.second)) { err = "upld: res body"; return false; }
        UpldTensor t;
        t.dims = std::move(h.first);
        t.data.assign(buf + off, buf + off + h.second);
        off += h.second;
        out.residuals.push_back(std::move(t));
    }
    if (!need(buf, n, off, s_nbytes)) { err = "upld: sample body"; return false; }
    out.sample.data.assign(buf + off, buf + off + s_nbytes);
    return true;
}

}  // namespace dist
