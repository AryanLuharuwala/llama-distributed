// sdcpp_roles.cpp — bridge between the C-callable per-role API
// (sdcpp_roles.h) and the underlying stable-diffusion.cpp pipeline.
//
// Most of the heavy lifting (text encoding, sampling, VAE decode) lives
// behind sd.cpp's `StableDiffusionGGML` class, which is not exposed in
// the public header.  Three of the role entry points (encode_text,
// sample, decode_latent) currently delegate to the monolithic
// `generate_image` and slice the output via SDT/SDCD framing on the way
// out — production-grade for the role-split case (TE+UNet+VAE on the
// same rig but emitted as discrete frames), but the *true* memory-saving
// split (TE-only worker that doesn't load UNet) requires the
// `sd_role_internal_*` upstream patch tracked separately.  Until that
// patch lands, sd_role_t.role_filter is honoured at the worker level by
// loading the full model but routing only the requested role's output.
//
// CF12-W6 block-range sampling (sd_role_sample_blocks) requires
// patching `StableDiffusionGGML::sample()` to be sliceable by transformer
// block index — see CF12-W6a.  Phase 1 (this commit): the C API surface
// is patched into the submodule (patches/sdcpp-block-split.patch) and the
// role bridge calls into sd_unet_block_count / sd_loaded_backbone_tag
// when DIST_HAVE_SDCPP_SPLIT is defined.  The underlying body still
// returns SD_SPLIT_ENOTSUP until the forward_half refactor lands, so
// sd_role_sample_blocks returns SDR_ENOTIMPL and the planner falls back
// to the role chain — CF12-W6c gates on `sdcpp_block_split:1` capability,
// which stays false until the body is wired.

#include "sdcpp_roles.h"

#include "stable-diffusion.h"
#include "sdt_codec.h"
#ifdef DIST_HAVE_SDCPP_SPLIT
#  include "sdcpp_split_wire.h"
#endif

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <mutex>

namespace {

// Error codes — keep stable for wire log/grep.
constexpr int SDR_OK         = 0;
constexpr int SDR_EINVAL     = 1;
constexpr int SDR_EBADSDCD   = 2;
constexpr int SDR_EBADSDT    = 3;
constexpr int SDR_ELOAD      = 4;
constexpr int SDR_EGEN       = 5;
constexpr int SDR_ENOTIMPL   = 9;  // requires upstream sd.cpp patch
constexpr int SDR_EWRITE     = 10;

// Static strings — pointed to by sd_role_status_t.msg.  The struct
// contract says the caller does not free; we keep all messages here.
const char* MSG_OK              = "ok";
const char* MSG_NULL_CTX        = "sd_ctx is NULL";
const char* MSG_NULL_IN         = "input struct is NULL";
const char* MSG_NULL_OUT        = "output buf is NULL";
const char* MSG_BAD_SDCD        = "failed to decode SDCD cond frame";
const char* MSG_BAD_SDT         = "failed to decode SDT frame";
const char* MSG_GEN_NULL        = "generate_image returned NULL";
const char* MSG_NOTIMPL_BLOCKS  = "sd_role_sample_blocks: split path requires DIST_HAVE_SDCPP_SPLIT (sdcpp-block-split.patch)";
const char* MSG_PNG_FAIL        = "in-memory PNG encode failed";
const char* MSG_VAE_FAIL        = "VAE decode bridge unavailable (needs sd_internal_decode_first_stage patch)";

// Per-process scratch lock — the bridge writes to ctx-level resources
// (image_out, latent cache) under generate_image which already holds
// its own locks, but role calls themselves are also serialised so we
// don't race a TE call against a sample call on the same ctx.
std::mutex g_role_mutex;

// ─── Buffer helpers ───────────────────────────────────────────────────────

sd_role_buf_t make_buf(std::vector<uint8_t>&& v) {
    sd_role_buf_t b{};
    if (v.empty()) return b;
    b.nbytes = v.size();
    b.data   = (uint8_t*)std::malloc(b.nbytes);
    if (!b.data) { b.nbytes = 0; return b; }
    std::memcpy(b.data, v.data(), b.nbytes);
    return b;
}

sd_role_status_t ok() {
    return sd_role_status_t{SDR_OK, MSG_OK};
}

sd_role_status_t fail(int code, const char* msg) {
    return sd_role_status_t{code, msg};
}

// (PNG encode lives in the worker — roles ship raw NHWC.)

// ─── Backbone tag / block-count inference ────────────────────────────────
// When the CF12-W6a sdcpp-block-split.patch is applied to the submodule
// (compile-time gate DIST_HAVE_SDCPP_SPLIT), we route to the real C API.
// Without the patch we fall back to a "unknown" tag — the planner then
// uses the role chain instead of trying to schedule a UNet block split.

const char* backbone_tag_from_ctx(sd_ctx_t* ctx) {
#ifdef DIST_HAVE_SDCPP_SPLIT
    if (ctx == nullptr) return "unknown";
    const char* tag = sd_loaded_backbone_tag(ctx);
    return tag ? tag : "unknown";
#else
    (void)ctx;
    return "unknown";
#endif
}

// Returns the number of logical UNet sub-blocks the loaded model supports
// for split-mode.  Phase 1 returns 2 (two halves) for SD1/SDXL; 0 means
// split is not yet wired and the planner falls back to the role chain.
int block_count_from_ctx(sd_ctx_t* ctx) {
#ifdef DIST_HAVE_SDCPP_SPLIT
    if (ctx == nullptr) return 0;
    return sd_unet_block_count(ctx);
#else
    (void)ctx;
    return 0;
#endif
}

}  // namespace

// ─── Public C API ─────────────────────────────────────────────────────────

extern "C" void sd_role_buf_free(sd_role_buf_t* buf) {
    if (!buf) return;
    if (buf->data) std::free(buf->data);
    buf->data = nullptr;
    buf->nbytes = 0;
}

extern "C" sd_role_status_t sd_role_encode_text(
    sd_ctx_t* sd_ctx,
    const sd_role_encode_text_in_t* in,
    sd_role_buf_t* out_frame)
{
    if (!sd_ctx)     return fail(SDR_EINVAL,  MSG_NULL_CTX);
    if (!in)         return fail(SDR_EINVAL,  MSG_NULL_IN);
    if (!out_frame)  return fail(SDR_EINVAL,  MSG_NULL_OUT);
    *out_frame = sd_role_buf_t{};

    std::lock_guard<std::mutex> lk(g_role_mutex);

#ifdef DIST_HAVE_SDCPP_SPLIT
    // Real path — invoke the loaded conditioner directly and ship the
    // resulting cond / uncond tensors as named SDCD tensors so a downstream
    // UNet rig can consume embeds without re-encoding.
    sd_cond_t* cresult = sd_cond_new();
    int rc = sd_encode_condition(
        sd_ctx,
        in->prompt ? in->prompt : "",
        (in->negative_prompt && in->negative_prompt[0]) ? in->negative_prompt : "",
        in->clip_skip,
        /*width=*/ -1, /*height=*/ -1,
        cresult);

    if (rc == SD_SPLIT_OK) {
        dist::SdcdFrame f;
        f.kv.push_back({"role",      "te"});
        f.kv.push_back({"backbone",  backbone_tag_from_ctx(sd_ctx)});
        f.kv.push_back({"deferred",  "0"});
        f.kv.push_back({"cfg_split", in->cfg_split ? "1" : "0"});
        f.kv.push_back({"has_uncond", sd_cond_has_uncond(cresult) ? "1" : "0"});

        auto add = [&](const char* name) {
            const float*   d  = nullptr;
            const int64_t* sh = nullptr;
            int            nd = 0;
            if (sd_cond_get_tensor(cresult, name, &d, &sh, &nd) != SD_SPLIT_OK) return;
            dist::SdcdNamed t;
            t.name           = name;
            t.tensor.dtype   = dist::SdtDType::F32;
            size_t nbytes    = sizeof(float);
            t.tensor.dims.reserve(nd);
            for (int i = 0; i < nd; ++i) {
                t.tensor.dims.push_back(static_cast<uint32_t>(sh[i]));
                nbytes *= static_cast<size_t>(sh[i]);
            }
            t.tensor.data.assign(reinterpret_cast<const uint8_t*>(d),
                                 reinterpret_cast<const uint8_t*>(d) + nbytes);
            f.tensors.push_back(std::move(t));
        };
        add("cond.crossattn");
        add("cond.vector");
        add("cond.concat");
        if (sd_cond_has_uncond(cresult)) {
            add("uncond.crossattn");
            add("uncond.vector");
            add("uncond.concat");
        }

        sd_cond_free(cresult);

        std::vector<uint8_t> enc;
        std::string err;
        if (!dist::sdcd_encode(f, enc, err)) {
            static thread_local std::string m;
            m = "sdcd_encode: " + err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
        *out_frame = make_buf(std::move(enc));
        return ok();
    }

    // Conditioner not loaded (rc == ENOTSUP) or encoder threw — fall
    // through to deferred-prompt so the wire still flows.
    sd_cond_free(cresult);
#endif

    // ── Deferred-prompt fallback ────────────────────────────────────────
    // No split patch (or conditioner not available): ship raw strings so
    // a downstream rig with the same model loaded can re-encode locally.
    // Wire-compatible; just doesn't save VRAM.
    dist::SdcdFrame f;
    f.kv.push_back({"role",      "te"});
    f.kv.push_back({"backbone",  backbone_tag_from_ctx(sd_ctx)});
    f.kv.push_back({"deferred",  "1"});
    f.kv.push_back({"prompt",    in->prompt ? std::string(in->prompt) : ""});
    f.kv.push_back({"neg",       in->negative_prompt ? std::string(in->negative_prompt) : ""});
    f.kv.push_back({"cfg_split", in->cfg_split ? "1" : "0"});
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", in->clip_skip);
        f.kv.push_back({"clip_skip", buf});
    }

    std::vector<uint8_t> enc;
    std::string err;
    if (!dist::sdcd_encode(f, enc, err)) {
        static thread_local std::string m;
        m = "sdcd_encode: " + err;
        return fail(SDR_EBADSDCD, m.c_str());
    }
    *out_frame = make_buf(std::move(enc));
    return ok();
}

extern "C" sd_role_status_t sd_role_sample(
    sd_ctx_t* sd_ctx,
    const sd_role_sample_in_t* in,
    sd_role_buf_t* out_latent)
{
    if (!sd_ctx)    return fail(SDR_EINVAL, MSG_NULL_CTX);
    if (!in)        return fail(SDR_EINVAL, MSG_NULL_IN);
    if (!out_latent) return fail(SDR_EINVAL, MSG_NULL_OUT);
    *out_latent = sd_role_buf_t{};

    std::lock_guard<std::mutex> lk(g_role_mutex);

    // Decode incoming SDCD to recover the prompt strings (deferred-prompt
    // path).  Once real embeds are emitted by encode_text, this will pass
    // them straight into generate_image via the patched fields instead of
    // rehydrating from strings.
    dist::SdcdFrame cond;
    {
        std::string err;
        if (!dist::sdcd_decode(in->sdcd_cond, in->sdcd_cond_nbytes, cond, err)) {
            static thread_local std::string m;
            m = std::string("sample/cond: ") + err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
    }
    auto get_meta = [&](const char* k) -> std::string {
        const auto* s = cond.find_meta(k);
        return s ? *s : std::string{};
    };
    std::string prompt = get_meta("prompt");
    std::string neg    = get_meta("neg");

    // Drive the existing whole-pipeline call.  We unfortunately get an
    // RGB image back, not a latent — this is the same gap the upstream
    // patch will close.  For now: emit an SDT frame tagged with the
    // image bytes, dtype=U8, dims=[1, H, W, 3].  Downstream VAE worker
    // sees the image already-decoded and forwards it as-is.  The path is
    // suboptimal (no inter-rig latent split) but functionally correct.

    sd_img_gen_params_t gen;
    std::memset(&gen, 0, sizeof(gen));
    gen.prompt          = prompt.c_str();
    gen.negative_prompt = neg.c_str();
    gen.width           = in->width  > 0 ? in->width  : 512;
    gen.height          = in->height > 0 ? in->height : 512;
    gen.seed            = in->seed;
    gen.batch_count     = 1;
    sd_sample_params_init(&gen.sample_params);
    gen.sample_params.sample_steps     = in->steps > 0 ? in->steps : 20;
    gen.sample_params.guidance.txt_cfg = in->cfg   > 0 ? in->cfg   : 7.0f;
    gen.sample_params.sample_method    = str_to_sample_method(
        in->sampler ? in->sampler : "euler_a");
    gen.sample_params.scheduler        = sd_get_default_scheduler(
        sd_ctx, gen.sample_params.sample_method);

    sd_image_t* img = generate_image(sd_ctx, &gen);
    if (!img) return fail(SDR_EGEN, MSG_GEN_NULL);

    dist::SdtTensor t;
    t.dtype = dist::SdtDType::U8;
    t.dims  = {1, (uint32_t)img->height, (uint32_t)img->width, (uint32_t)img->channel};
    size_t n = size_t(img->height) * size_t(img->width) * size_t(img->channel);
    t.data.assign(img->data, img->data + n);

    if (img->data) std::free(img->data);
    std::free(img);

    std::vector<uint8_t> enc;
    std::string err;
    if (!dist::sdt_encode(t, enc, err)) {
        static thread_local std::string m;
        m = "sample/sdt_encode: " + err;
        return fail(SDR_EBADSDT, m.c_str());
    }
    *out_latent = make_buf(std::move(enc));
    return ok();
}

// Suppress unused-helper warning when the PNG passthrough path is not
// compiled in this configuration.
[[maybe_unused]] static void _sdrole_unused_silencer() {}

extern "C" sd_role_status_t sd_role_sample_blocks(
    sd_ctx_t* sd_ctx,
    const sd_role_sample_blocks_in_t* in,
    sd_role_buf_t* out_payload)
{
    if (!sd_ctx)      return fail(SDR_EINVAL, MSG_NULL_CTX);
    if (!in)          return fail(SDR_EINVAL, MSG_NULL_IN);
    if (!out_payload) return fail(SDR_EINVAL, MSG_NULL_OUT);
    *out_payload = sd_role_buf_t{};

    // Degenerate full-range case (block_lo=0, block_hi=block_total) —
    // collapses to a normal whole-UNet sample.  This is the path the
    // planner falls back to when a rig advertises block_total=1 (i.e.
    // "I can run the UNet but I can't slice it").  Wire-compatible with
    // a real block-split rig that would emit UPLD on intermediate
    // stages and SDT on the last — here we always emit SDT because the
    // single rig owns the whole denoise loop.
    const bool full_range =
        (in->block_lo == 0 &&
         (in->block_hi <= 0 || in->block_hi >= in->block_total) &&
         (in->upld_in == nullptr || in->upld_in_nbytes == 0));
    if (full_range) {
        sd_role_sample_in_t s{};
        s.sdcd_cond        = in->sdcd_cond;
        s.sdcd_cond_nbytes = in->sdcd_cond_nbytes;
        s.width            = 512;  // caller passes via the worker JSON
        s.height           = 512;
        s.steps            = in->steps;
        s.cfg              = in->cfg;
        s.seed             = in->seed;
        s.sampler          = in->sampler;
        s.scheduler        = in->scheduler;
        s.sdt_init_latent  = nullptr;
        s.sdt_init_nbytes  = 0;
        s.strength         = 0.0f;
        return sd_role_sample(sd_ctx, &s, out_payload);
    }

#ifdef DIST_HAVE_SDCPP_SPLIT
    // ── CF12-W7 per-step N-way block-split dispatch ────────────────────
    // Caller (coordinator) drives the sampler loop; each step issues one
    // call per stage with a contiguous block range [block_lo, block_hi):
    //   • lo==0            : upld_in = SDCD "sdcpp_step_x"  (seeds prelude)
    //   • lo>0             : upld_in = SDCD carry from prev rig
    //   • hi <  block_total: emits an SDCD carry for the next rig
    //   • hi == block_total: emits the SDT noise_pred (last stage)
    // Any tiling of [0, total) works — the down-path skip residuals ride
    // the hs[] stack forward until the up-path pops them.  block_total is
    // the real linear block count (sd_unet_block_count), not 2.
    const int total = sd_unet_block_count(sd_ctx);
    if (total < 1) {
        return fail(SDR_ENOTIMPL, "model not split-capable");
    }
    int lo = in->block_lo;
    int hi = (in->block_hi <= 0 || in->block_hi > total) ? total : in->block_hi;
    if (lo < 0)   lo = 0;
    if (lo >= hi) return fail(SDR_EINVAL, "empty block range");
    const bool is_first = (lo == 0);
    const bool is_last  = (hi == total);
    if (in->upld_in == nullptr || in->upld_in_nbytes == 0) {
        return fail(SDR_EINVAL, "upld_in required for partial range");
    }

    std::lock_guard<std::mutex> lk(g_role_mutex);

    // Decode the conditioner frame — for the real path it must be
    // a non-deferred SDCD with cond.crossattn / cond.vector / cond.concat.
    dist::SdcdFrame cond;
    if (in->sdcd_cond != nullptr && in->sdcd_cond_nbytes > 0) {
        std::string err;
        if (!dist::sdcd_decode(in->sdcd_cond, in->sdcd_cond_nbytes, cond, err)) {
            static thread_local std::string m;
            m = std::string("blocks/sdcd_decode cond: ") + err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
        const std::string* deferred = cond.find_meta("deferred");
        if (deferred && *deferred == "1") {
            return fail(SDR_ENOTIMPL, "blocks/cond: deferred-prompt cond unsupported here — TE must emit real embeds");
        }
    }

    sd_split_state_t* state = sd_split_state_new();
    if (state == nullptr) return fail(SDR_EINVAL, "blocks/state alloc");
    int rc = SD_SPLIT_OK;

    auto set_cond_tensor = [&](const char* sdcd_name, const char* unet_name) {
        const dist::SdtTensor* t = cond.find(sdcd_name);
        if (t == nullptr || t->dims.empty()) return;
        std::vector<int64_t> shape(t->dims.begin(), t->dims.end());
        sd_split_state_set_input(
            state, unet_name,
            reinterpret_cast<const float*>(t->data.data()),
            shape.data(), static_cast<int>(shape.size()));
    };
    set_cond_tensor("cond.crossattn", "context");
    set_cond_tensor("cond.vector",    "y");
    set_cond_tensor("cond.concat",    "c_concat");

    // Stage timesteps as a 1-D tensor [B] with the requested sigma.  The
    // backbone-specific scaling (sigma→t for the diffusion params) happens
    // inside the half0 graph.
    {
        float    ts_data[1]  = { in->timestep };
        int64_t  ts_shape[1] = { 1 };
        sd_split_state_set_input(state, "timesteps", ts_data, ts_shape, 1);
    }

    // ── stage the per-stage input ──────────────────────────────────────
    if (is_first) {
        // upld_in is an SDCD step-x frame carrying x and step metadata.
        dist::SdcdFrame xframe;
        const float* x_data = nullptr;
        std::vector<int64_t> x_shape;
        int step_idx_in = 0;
        float ts_in = 0.f;
        std::string err;
        if (!dist::sdcpp_sdcd_to_x(in->upld_in, in->upld_in_nbytes,
                                   xframe, &x_data, x_shape,
                                   &step_idx_in, &ts_in, err)) {
            sd_split_state_free(state);
            static thread_local std::string m;
            m = std::string("blocks/decode step-x: ") + err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
        sd_split_state_set_input(state, "x", x_data,
                                 x_shape.data(),
                                 static_cast<int>(x_shape.size()));
    } else {
        // upld_in is an SDCD carry frame from the previous stage.
        std::string err;
        if (!dist::sdcpp_sdcd_to_carry(in->upld_in, in->upld_in_nbytes, state, err)) {
            sd_split_state_free(state);
            static thread_local std::string m;
            m = std::string("blocks/decode carry: ") + err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
    }

    rc = sd_compute_unet_split_range(sd_ctx, lo, hi,
                                     in->step_idx, in->steps, state);
    if (rc != SD_SPLIT_OK) {
        sd_split_state_free(state);
        static thread_local std::string m;
        m = "blocks/range compute rc=" + std::to_string(rc) +
            " [" + std::to_string(lo) + "," + std::to_string(hi) + ")";
        return fail(SDR_EGEN, m.c_str());
    }

    // ── intermediate stage: emit an SDCD carry for the next rig ────────
    if (!is_last) {
        std::vector<uint8_t> enc;
        std::string carry_err;
        if (!dist::sdcpp_carry_to_sdcd(state, enc, carry_err)) {
            sd_split_state_free(state);
            static thread_local std::string m;
            m = "blocks/carry_to_sdcd: " + carry_err;
            return fail(SDR_EBADSDCD, m.c_str());
        }
        sd_split_state_free(state);
        *out_payload = make_buf(std::move(enc));
        return ok();
    }

    // ── last stage: emit the SDT noise_pred ────────────────────────────

    const float*   np_data  = nullptr;
    const int64_t* np_shape = nullptr;
    int            np_ndims = 0;
    rc = sd_split_state_get_output(state, &np_data, &np_shape, &np_ndims);
    if (rc != SD_SPLIT_OK || np_data == nullptr) {
        sd_split_state_free(state);
        return fail(SDR_EGEN, "blocks/half1 get_output empty");
    }

    dist::SdtTensor np;
    np.dtype = dist::SdtDType::F32;
    size_t numel = 1;
    np.dims.reserve(np_ndims);
    for (int i = 0; i < np_ndims; ++i) {
        np.dims.push_back(static_cast<uint32_t>(np_shape[i]));
        numel *= static_cast<size_t>(np_shape[i]);
    }
    np.data.assign(reinterpret_cast<const uint8_t*>(np_data),
                   reinterpret_cast<const uint8_t*>(np_data) + numel * sizeof(float));
    sd_split_state_free(state);

    std::vector<uint8_t> enc;
    std::string err;
    if (!dist::sdt_encode(np, enc, err)) {
        static thread_local std::string m;
        m = "blocks/half1 sdt_encode: " + err;
        return fail(SDR_EBADSDT, m.c_str());
    }
    *out_payload = make_buf(std::move(enc));
    return ok();
#else
    // Patch not applied — planner will fall back.
    return fail(SDR_ENOTIMPL, MSG_NOTIMPL_BLOCKS);
#endif
}

extern "C" sd_role_status_t sd_role_decode_latent(
    sd_ctx_t* sd_ctx,
    const sd_role_decode_in_t* in,
    sd_role_buf_t* out_image)
{
    if (!sd_ctx)    return fail(SDR_EINVAL, MSG_NULL_CTX);
    if (!in)        return fail(SDR_EINVAL, MSG_NULL_IN);
    if (!out_image) return fail(SDR_EINVAL, MSG_NULL_OUT);
    *out_image = sd_role_buf_t{};

    std::lock_guard<std::mutex> lk(g_role_mutex);

    // Decode the SDT — for the wire-correct stub this is an image already
    // (sample emitted U8 NHWC pixels rather than a real latent).  Just
    // re-emit it as the decoded image bytes.  Once the real latent path
    // lands, this branch becomes the VAE compute call.
    dist::SdtTensor latent;
    {
        std::string err;
        if (!dist::sdt_decode(in->sdt_latent, in->sdt_latent_nbytes, latent, err)) {
            static thread_local std::string m;
            m = std::string("decode/sdt_decode: ") + err;
            return fail(SDR_EBADSDT, m.c_str());
        }
    }

    if (latent.dtype == dist::SdtDType::U8 && latent.dims.size() == 4 &&
        latent.dims[3] >= 1 && latent.dims[3] <= 4) {
        // Already-decoded image — pass through.
        std::vector<uint8_t> enc;
        std::string err;
        if (!dist::sdt_encode(latent, enc, err)) {
            static thread_local std::string m;
            m = "decode/passthrough: " + err;
            return fail(SDR_EBADSDT, m.c_str());
        }
        *out_image = make_buf(std::move(enc));
        return ok();
    }

#ifdef DIST_HAVE_SDCPP_SPLIT
    // Real VAE decode via the internal bridge.  The incoming SDT carries
    // a rank-4 latent (NCHW) — either fp32 or fp16; we promote to fp32
    // before calling into sd.cpp because the bridge takes a float buffer.
    if (latent.dims.size() != 4) {
        return fail(SDR_EBADSDT, "decode/vae: expected rank-4 latent");
    }
    std::vector<float> latent_f32;
    if (latent.dtype == dist::SdtDType::F32) {
        const float* p = reinterpret_cast<const float*>(latent.data.data());
        size_t n = latent.data.size() / sizeof(float);
        latent_f32.assign(p, p + n);
    } else if (latent.dtype == dist::SdtDType::F16) {
        // Naive f16→f32 fallback so the bridge handles fp16 inputs.
        size_t n = latent.data.size() / 2;
        latent_f32.resize(n);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(latent.data.data());
        for (size_t i = 0; i < n; ++i) {
            uint32_t u = h[i];
            uint32_t sign = (u >> 15) & 0x1;
            uint32_t exp  = (u >> 10) & 0x1F;
            uint32_t mant =  u        & 0x3FF;
            uint32_t f;
            if (exp == 0)       f = sign << 31; // subnormal/zero → ±0
            else if (exp == 31) f = (sign << 31) | (0xFFu << 23) | (mant << 13);
            else                f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
            float ff;
            std::memcpy(&ff, &f, 4);
            latent_f32[i] = ff;
        }
    } else {
        return fail(SDR_EBADSDT, "decode/vae: unsupported latent dtype");
    }

    int64_t shape[4] = {
        latent.dims[0], latent.dims[1], latent.dims[2], latent.dims[3],
    };
    float*   img_data  = nullptr;
    int64_t* img_shape = nullptr;
    int      img_ndims = 0;
    int rc = sd_decode_first_stage_to_floats(
        sd_ctx, latent_f32.data(), shape, 4,
        &img_data, &img_shape, &img_ndims);
    if (rc != SD_SPLIT_OK) {
        return fail(SDR_ENOTIMPL, MSG_VAE_FAIL);
    }
    // Image comes back as fp32 in NCHW (range typically [0,1] after
    // postprocess inside sd.cpp); ship as SDT-f32 so the caller can do
    // PNG encode / further processing.
    dist::SdtTensor img;
    img.dtype = dist::SdtDType::F32;
    img.dims.reserve(img_ndims);
    size_t img_numel = 1;
    for (int i = 0; i < img_ndims; ++i) {
        img.dims.push_back(static_cast<uint32_t>(img_shape[i]));
        img_numel *= static_cast<size_t>(img_shape[i]);
    }
    img.data.assign(reinterpret_cast<uint8_t*>(img_data),
                    reinterpret_cast<uint8_t*>(img_data) + img_numel * sizeof(float));
    sd_vae_image_free(img_data, img_shape);

    std::vector<uint8_t> enc;
    std::string err;
    if (!dist::sdt_encode(img, enc, err)) {
        static thread_local std::string m;
        m = "decode/sdt_encode: " + err;
        return fail(SDR_EBADSDT, m.c_str());
    }
    *out_image = make_buf(std::move(enc));
    return ok();
#else
    // Patch not applied — VAE-only role unsupported; planner will fall
    // back to a single-rig "full" route.
    return fail(SDR_ENOTIMPL, MSG_VAE_FAIL);
#endif
}

extern "C" int sd_role_block_count(sd_ctx_t* sd_ctx) {
    return block_count_from_ctx(sd_ctx);
}

extern "C" const char* sd_role_backbone_tag(sd_ctx_t* sd_ctx) {
    return backbone_tag_from_ctx(sd_ctx);
}
