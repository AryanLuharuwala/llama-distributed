// sdcpp_roles.h — per-role entry points layered over stable-diffusion.cpp.
//
// These split the monolithic `generate_image` call into three independent
// hops so a sd.cpp diffusion pipeline can fan out across rigs (CF12-W).
//
//   sd_role_encode_text   — text-encoder hop.  Owns the conditioner(s).
//                           Emits prompt_embeds + pooled + per-backbone extras
//                           packed as an SDCD frame.  May run with the UNet
//                           and VAE skipped from the loaded ctx (see
//                           sd_ctx_params_t.role_filter below).
//
//   sd_role_sample        — full-UNet sampling hop.  Drives the resident
//                           StableDiffusionGGML through `steps` denoise
//                           iterations starting from the supplied init
//                           latent (or pure noise), conditioned by the
//                           prior-stage SDCD frame.  Emits the final latent
//                           as an SDT frame.
//
//   sd_role_sample_blocks — block-range UNet hop (CF12-W6).  Runs the UNet
//                           forward graph for a transformer-block subset
//                           [block_lo, block_hi).  Takes a UPLD payload in,
//                           emits one out (sample + residuals + step_idx).
//                           When the worker holds the LAST block range and
//                           the input UPLD has is_final_step=1 the call
//                           runs the cfg-combine + scheduler.step internally
//                           and emits the *next* latent (or, if it's the
//                           final step, an SDT-only latent suitable for the
//                           VAE worker downstream).
//
//   sd_role_decode_latent — VAE decode hop.  Takes a single SDT latent in,
//                           emits an SDT containing the decoded RGB image
//                           bytes (u8, NHWC).
//
// Wire formats:
//   SDT  — single tensor                       (sdt_codec.h)
//   SDCD — named container of tensors          (sdt_codec.h)
//   UPLD — block-range hand-off                (sdt_codec.h, python-wire-compatible)
//
// All entry points are C-callable so the worker daemon can dispatch
// without RTTI/exceptions leaking across the API boundary.  Errors come
// back through the small `sd_role_status_t` struct — never `throw`.
//
// Memory model:
//   - Output buffers are owned by the callee and returned via the
//     `sd_role_buf_t` struct.  Free with `sd_role_buf_free`.
//   - Inputs are borrowed; the call returns before any captured pointer
//     is consumed asynchronously.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward decl — defined by stable-diffusion.h.
struct sd_ctx_t;

// ─── Role filter ──────────────────────────────────────────────────────────
// Set on sd_ctx_params_t (the sd.cpp ctx struct) before new_sd_ctx() so the
// loader knows which sub-models can be elided to save VRAM.  The patched
// sd.cpp loader reads this through an env-var fallback (DIST_SDCPP_ROLE)
// when the field isn't present, so this header stays ABI-stable.

typedef enum {
    SD_ROLE_FULL = 0,  // load everything (default; matches current behaviour)
    SD_ROLE_TE   = 1,  // text encoder(s) only
    SD_ROLE_UNET = 2,  // UNet/DiT only (no TE, no VAE)
    SD_ROLE_VAE  = 3,  // VAE only
} sd_role_t;

// ─── Status ────────────────────────────────────────────────────────────────

typedef struct {
    int          code;            // 0 = ok; nonzero = role-defined error
    const char*  msg;             // pointer to a static or sd-owned string
} sd_role_status_t;

// ─── Returned buffers ─────────────────────────────────────────────────────
// Implementation always uses `std::malloc` underneath; sd_role_buf_free
// matches that.  Do not free with `delete`.

typedef struct {
    uint8_t*  data;
    size_t    nbytes;
} sd_role_buf_t;

void sd_role_buf_free(sd_role_buf_t* buf);

// ─── Encode text ──────────────────────────────────────────────────────────
// `out_frame` receives an SDCD container.  Conventional contents:
//
//   tensors:
//     "prompt_embeds"   — encoder output  (rank 3: [B, T, C])
//     "pooled"          — pooled embed    (rank 2: [B, C], when applicable)
//     "negative_prompt_embeds" / "negative_pooled" — when neg prompt given
//     <backbone extras> — e.g. "time_ids", "attention_mask", "t5_extras"
//   kv:
//     "backbone"        — e.g. "sd15", "sdxl", "sd3", "flux", "pixart"
//     "dtype"           — "fp16" | "fp32"
//
// `cfg_split` controls whether the encoder runs the negative prompt in the
// same call (true) or in a separate call (false).  Either is wire-compatible
// downstream — the UNet stage looks for the negative tensors and adapts.

typedef struct {
    const char*  prompt;
    const char*  negative_prompt;   // may be NULL or "" — produces no neg embed
    bool         cfg_split;
    int          clip_skip;         // -1 = backbone default
} sd_role_encode_text_in_t;

sd_role_status_t sd_role_encode_text(
    struct sd_ctx_t*               sd_ctx,
    const sd_role_encode_text_in_t* in,
    sd_role_buf_t*                  out_frame);  // SDCD bytes

// ─── Sample (whole UNet) ──────────────────────────────────────────────────
// Bridge to StableDiffusionGGML::sample().  Used when the UNet is on a
// single rig — no block split.  Output is an SDT latent.

typedef struct {
    const uint8_t* sdcd_cond;        // SDCD frame from sd_role_encode_text
    size_t         sdcd_cond_nbytes;

    int            width;
    int            height;
    int            steps;
    float          cfg;
    int64_t        seed;
    const char*    sampler;          // e.g. "euler_a"
    const char*    scheduler;        // NULL → backbone default

    const uint8_t* sdt_init_latent;  // optional; if non-NULL used for img2img
    size_t         sdt_init_nbytes;
    float          strength;         // img2img strength (0..1); ignored if no init
} sd_role_sample_in_t;

sd_role_status_t sd_role_sample(
    struct sd_ctx_t*           sd_ctx,
    const sd_role_sample_in_t* in,
    sd_role_buf_t*             out_latent);  // SDT bytes (rank-4 latent, fp16)

// ─── Sample blocks (CF12-W6) ──────────────────────────────────────────────
// Block-range UNet hop.  See header docstring above for semantics.

typedef struct {
    const uint8_t* sdcd_cond;          // broadcast once per request; cached
    size_t         sdcd_cond_nbytes;   // by the worker between block calls

    const uint8_t* upld_in;            // running hidden state from previous block stage
    size_t         upld_in_nbytes;     // (or pure noise if step_idx=0 + lo=0)

    int            block_lo;           // inclusive
    int            block_hi;           // exclusive
    int            block_total;        // backbone block count (sanity)

    int            steps;              // total denoise steps in the schedule
    float          cfg;
    int64_t        seed;
    const char*    sampler;
    const char*    scheduler;
} sd_role_sample_blocks_in_t;

// out_payload: UPLD on intermediate stages; SDT (final latent) on the last
// block of the last step.  Inspect the first 4 bytes for SDT_MAGIC vs UPLD_MAGIC.
sd_role_status_t sd_role_sample_blocks(
    struct sd_ctx_t*                   sd_ctx,
    const sd_role_sample_blocks_in_t*  in,
    sd_role_buf_t*                     out_payload);

// ─── Decode latent (VAE) ──────────────────────────────────────────────────

typedef struct {
    const uint8_t* sdt_latent;
    size_t         sdt_latent_nbytes;
} sd_role_decode_in_t;

sd_role_status_t sd_role_decode_latent(
    struct sd_ctx_t*           sd_ctx,
    const sd_role_decode_in_t* in,
    sd_role_buf_t*             out_image);  // SDT bytes (rank-4 u8 NHWC)

// ─── Capability probes ────────────────────────────────────────────────────
// Used by the worker on startup to fill in the cap-advert and by tests.

// Returns the number of transformer blocks the loaded model exposes for
// CF12-W6 block-range splitting.  -1 if not implemented for this backbone.
int sd_role_block_count(struct sd_ctx_t* sd_ctx);

// Backbone tag — "sd15" | "sdxl" | "sd3" | "flux" | "pixart" | "unknown".
const char* sd_role_backbone_tag(struct sd_ctx_t* sd_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif
