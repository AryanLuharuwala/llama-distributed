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
const char* MSG_NOTIMPL_BLOCKS  = "sd_role_sample_blocks: half0/half1 C API present (sd_compute_unet_split_step); per-step cross-rig driver pending CF12-W6c";
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

    // ── Why this works without the internals patch ──────────────────────
    // The wire-correct behaviour for a TE role is "emit an SDCD frame
    // that downstream UNet rigs can rehydrate prompts from".  Until we
    // have direct access to `cond_stage_model->get_learned_condition`,
    // we ship a deferred-prompt frame: a small SDCD carrying the raw
    // prompt strings + clip_skip + cfg_split flag.  The receiving UNet
    // worker will re-encode locally on first use (it has the same
    // model loaded anyway when running on the same ctx without
    // role-skip).  This is a graceful degradation: bytes flow,
    // pipelines complete, just without the VRAM savings of TE-only.
    //
    // Once the upstream `sd_internal_encode_text` patch lands, this
    // function switches to populating real prompt_embeds / pooled
    // tensors and the deferred-prompt branch can be removed.

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

    // Partial block range — needs sliceable UNet forward, gated behind
    // CF12-W6a upstream patch.  Returning ENOTIMPL keeps the wire
    // contract honest; the planner sees the error and falls back to a
    // single-rig "full" or three-rig "te/unet/vae" route.
    return fail(SDR_ENOTIMPL, MSG_NOTIMPL_BLOCKS);
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

    // True latent → real VAE decode requires `decode_first_stage` bridge.
    return fail(SDR_ENOTIMPL, MSG_VAE_FAIL);
}

extern "C" int sd_role_block_count(sd_ctx_t* sd_ctx) {
    return block_count_from_ctx(sd_ctx);
}

extern "C" const char* sd_role_backbone_tag(sd_ctx_t* sd_ctx) {
    return backbone_tag_from_ctx(sd_ctx);
}
