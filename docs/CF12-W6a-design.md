# CF12-W6a-real — sd.cpp UNet block-split design

Status: **Phase-1 in progress**.  See "What's landed" / "What remains".

## Goal

Slice a single sd.cpp UNet forward pass across multiple rigs so the model's
parameters can live on disjoint GPUs the same way the diffusers path
already does (CF12-W6a + CF12-E, see `python/dpp_runtime/worker.py` ::
`_handle_unet_partitioned`).  Today every sd.cpp UNet step runs as one
monolithic graph on one rig — `sd_role_sample_blocks` returns `SDR_ENOTIMPL`
for any partial block range (`src/sdcpp_roles.cpp:302`).

## Constraints

1. **`third_party/stable-diffusion.cpp` is a pristine git submodule** (origin
   `leejet/stable-diffusion.cpp`, HEAD `a397e03`).  No direct edits — patch
   out-of-tree via `patches/sdcpp-block-split.patch`, applied idempotently by
   `CMakeLists.txt` (same pattern as `patches/partial-layer-load.patch`).

2. **No GGML symbol cross-contamination.**  sd.cpp ships its own ggml fork
   that would symbol-collide with llama.cpp's ggml inside `dist_common`.
   That's why the C++ worker is a separate executable (`dist-sdcpp-worker`)
   and the C++/Go boundary is JSON lines on stdin/stdout.  Block-split must
   not breach that — wire format stays at the role boundary.

3. **Per-step pass-around** in diffusers (`_unet_first_loopback` ring in
   `python/dpp_runtime/worker.py:495`) works because `scheduler.step` is
   exposed.  sd.cpp's `StableDiffusionGGML::sample` is a closed denoise
   loop — splitting per-step requires extracting `sample_step` into a public
   entry point.  This is the harder half of the work.

## Two cuts, two phases

### Phase 1 — Two-way split at the middle-block boundary

The natural cut: between `middle_block` and `output_blocks[0]`.  At that
moment the skip-residual stack `hs[]` is full but stationary (input phase
done pushing, output phase about to start popping).  No partial-stack
serialization, no mid-phase boundary.

```
  rig A:  input_blocks.0..N → middle_block            ─┐
                                                       │  UPLD-half(h, hs[])
  rig B:  output_blocks.0..M → out.{0,2}              ─┘
```

For the SDXL UNet (`channel_mult={1,2,4}`, `num_res_blocks=2`), `N=9` input
blocks (1 conv_in + 6 res/attn + 2 downsample) → `hs.size() = 9` skip
residuals.

### Phase 2 — N-way intra-input / intra-output partitioning

Subdivide the input phase or the output phase further across more rigs.
Requires partial `hs[]` serialization (rig boundaries inside the push or
pop sequence).  Significantly more complex; punted until Phase 1 lands and
is exercised on a real cluster.

### Phase 3 — Per-step ring driver

Hoist `StableDiffusionGGML::sample`'s closed loop out of the role call so
each `sd_role_sample_blocks` invocation is one step.  Mirrors the
diffusers `_unet_step_through` flow.  Lets multiple rigs round-trip the
loopback every step instead of one rig owning the whole denoise.

For Phase 1, the role call still owns the whole denoise loop internally —
each call drives all steps, but each step is split across the two halves
via in-process compute_half hops.  This is wire-honest (no spurious
step-level loopback) and ships real per-step VRAM savings.

## Phase-1 API surface (this commit)

```c
// stable-diffusion.h — public additions (added by the patch)

// One-shot two-way denoise: caller hands in the SDCD cond + an init latent
// + (which_half ∈ {0, 1}).  half=0 runs input+middle for each step and
// stops; half=1 takes the half=0 carry-state and runs output+final + the
// scheduler step + cfg combine for each step.
//
// Carry-state lives in `sd_split_state_t*` — opaque pointer the caller
// passes from half=0 to half=1.  On a single rig this is in-process.
// On split rigs the carry-state is serialised by the role layer (see
// docs/CF12-W6a-design.md § Wire format).
struct sd_split_state_t;  // opaque

SD_API sd_split_state_t* sd_split_state_new(void);
SD_API void              sd_split_state_free(sd_split_state_t*);

SD_API int sd_split_state_serialize  (sd_split_state_t*, uint8_t** out, size_t* nbytes);
SD_API int sd_split_state_deserialize(const uint8_t* in, size_t nbytes, sd_split_state_t** out);

// Run `which_half` of the UNet for `step_idx` of the current denoise.
// Returns 0 on success, negative on error (e.g. -EAGAIN, -EINVAL, -ENOTSUP).
SD_API int sd_compute_unet_split_step(
    struct sd_ctx_t*       ctx,
    int                    which_half,   // 0 or 1
    int                    step_idx,
    int                    total_steps,
    sd_split_state_t*      state);

// Block-count probe — number of logical UNet blocks the loaded model
// supports for split.  Phase 1 returns 2 for SD1.x/SDXL (the two halves);
// future N-way returns higher counts.
SD_API int sd_unet_block_count(struct sd_ctx_t* ctx);
```

```cpp
// unet.hpp — non-public additions (linked into sd.cpp's library by the patch)

class UnetModelBlock {
    // Existing forward() stays — calls forward_half(half=0) chained into
    // forward_half(half=1) under the hood, byte-identical graph.
    ggml_tensor* forward_half(GGMLRunnerContext* ctx,
                              int half,                         // 0 or 1
                              ggml_tensor* x, ts, ctx, c_concat, y,
                              int num_video_frames,
                              std::vector<ggml_tensor*>& hs,   // in/out
                              ggml_tensor* h_in,               // half=1 only
                              std::vector<ggml_tensor*> controls,
                              float control_strength);
};

struct UNetModelRunner {
    // Phase 1: compute_split runs the WHOLE denoise loop (all steps) with
    // every step split across in-process calls to forward_half(0/1).  No
    // wire serialisation between halves — this is the single-rig sanity
    // path that proves forward_half is graph-identical to forward().
    sd::Tensor<float> compute_split(int n_threads, ...same args as compute...);
};
```

## Wire format — UPLD-half (between role rigs)

Reuse the existing SDCD container (`include/sdt_codec.h`).  Tensors:

| name      | shape           | dtype | notes                           |
|-----------|-----------------|-------|---------------------------------|
| `h`       | [B, C, Hₘ, Wₘ]  | fp16  | hidden state after middle_block |
| `hs.0`    | [B, C₀, H, W]   | fp16  | first skip residual             |
| ...       | ...             | ...   |                                 |
| `hs.N-1`  | [B, Cₙ₋₁, …]    | fp16  | last skip residual              |

KV:

- `kind` = `upld_sdcpp_half0`
- `step_idx` (int, decimal)
- `sigma_idx` (int, decimal)
- `is_final_step` (`0` or `1`)
- `cfg` (float, decimal)
- `seed` (int, decimal)

The receiver (half=1 rig) decodes, hydrates each tensor onto its backend
buffer, calls `sd_compute_unet_split_step(which_half=1, ...)`, and emits
either another SDCD (intermediate step) or an SDT (final latent on the
last step).

## What's landed in this commit

- `patches/sdcpp-block-split.patch` — adds the `sd_split_state_t` and
  `sd_compute_unet_split_step` C API stubs to `stable-diffusion.h` /
  `stable-diffusion.cpp`.  Stubs return `-ENOTSUP` for now; the real
  body lands in the follow-up patch when the unet.hpp `forward_half`
  refactor is verified against a real model.
- `CMakeLists.txt` — applies the new patch alongside the existing
  llama.cpp patch, idempotently.
- `src/sdcpp_roles.cpp` — calls into the new C API surface but still
  returns `SDR_ENOTIMPL` when the underlying call reports unsupported.
  Once the unet.hpp body lands, only the patch changes — the role
  bridge already handles the new path.
- `include/sdcpp_roles.h` — capability probe (`sd_role_block_count`) now
  returns from `sd_unet_block_count(sd_ctx)` instead of a hardcoded
  filename-based guess.  Phase-1 returns 2 for SD1/SDXL once the body
  lands; until then the C stub returns 0 and the planner keeps falling
  back to the role chain (planSdcppRoleChain).

## What remains (next sessions)

1. **`UnetModelBlock::forward_half` body** — refactor the existing
   `forward()` into `forward_half(half=0)` and `forward_half(half=1)` such
   that the concatenation is byte-identical to the monolithic graph.
   Acceptance: cpu-side equivalence test on a tiny SDXL config — same
   noise pred to fp16 precision.
2. **`UNetModelRunner::compute_split`** — drives `forward_half(0)` →
   `forward_half(1)` in-process, exposes multi-output read-back for the
   intermediate `(h, hs[])` carry-state.  Needs care with ggml's cache_tensor
   map (see `ggml_extend.hpp:1739`) to extract more than one output per
   call.
3. **`sd_split_state_t` real impl** — owns the (h, hs[], emb, sigma stack,
   sampler state) carry between halves.  Serialize/deserialize via the
   SDT codec (already exists for inter-process passage).
4. **Per-step entry point** (Phase 3) — extract `sample_step` from
   `StableDiffusionGGML::sample`.  Largest of the remaining patches.
5. **Daemon protocol** — `dist-sdcpp-worker` learns `sdr_sample_blocks_step`
   as a separate command from `sdr_sample_blocks`, so the Go planner can
   drive a per-step ring.
6. **Planner & cap-advert wiring** — once `sd_unet_block_count` returns
   non-zero, the existing `planSdcppUnetSplit` will start scheduling
   sd.cpp rigs into block-split chains (currently falls through to
   role chain when `unet_blocks=0`).
7. **Cross-runtime equivalence test** — `python/dpp_runtime/test_sdcpp_unet_split.py`
   gains a real-model 2-rig roundtrip case (currently asserts ENOTIMPL).

Phase 1 finish line: items 1–4 land; the C++ test from §1 + the wire
test from §7 both pass on a real SDXL checkpoint.

## Why this scoping

- Doing the unet.hpp refactor without a compile-test cycle (the user's
  laptop can't comfortably build sd.cpp from scratch — see
  `feedback_build_parallelism.md`) would produce unreviewable patches.
  Landing the C API surface + the role-bridge wiring first means the
  follow-up patch only touches unet.hpp internals — small, focused, and
  reviewable.
- The same staging let CF12-W's role split ship across multiple sessions
  cleanly — see commit `7005f80` on `prod`.
