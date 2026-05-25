// test_sdcpp_split — CF12-W6a integration test for the sd.cpp UNet
// block-split C API.  Exercises the surface without loading a model:
//   * input/carry/output setter+getter roundtrip
//   * sd_split_state_serialize → sd_split_state_deserialize byte equality
//   * SDCD UPLD-half encoder/decoder roundtrip (via sdcpp_roles bridge)
//
// Returns 0 on full pass; non-zero on the first failed assert.

#include "stable-diffusion.h"
#include "sdt_codec.h"
#include "sdcpp_split_wire.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

#define CHECK(cond, msg) do {                              \
    if (!(cond)) {                                         \
        std::fprintf(stderr, "FAIL: %s (%s)\n", msg, #cond);\
        ++g_fails;                                         \
    }                                                      \
} while (0)

std::vector<float> fill_random(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = static_cast<float>((int32_t)s) / 2.147483647e9f;
    }
    return v;
}

void test_input_carry_roundtrip() {
    std::printf("== test_input_carry_roundtrip ==\n");
    sd_split_state_t* st = sd_split_state_new();
    CHECK(st != nullptr, "state alloc");

    // stage inputs
    int64_t x_shape[4] = {1, 4, 8, 8};
    auto    x_data    = fill_random(1 * 4 * 8 * 8, 1);
    int rc = sd_split_state_set_input(st, "x", x_data.data(), x_shape, 4);
    CHECK(rc == SD_SPLIT_OK, "set x");

    int64_t t_shape[1] = {1};
    float   t_data[1]  = {999.f};
    rc = sd_split_state_set_input(st, "timesteps", t_data, t_shape, 1);
    CHECK(rc == SD_SPLIT_OK, "set timesteps");

    // unknown name
    rc = sd_split_state_set_input(st, "bogus", nullptr, nullptr, 0);
    CHECK(rc == SD_SPLIT_EINVAL, "set bogus -> EINVAL");

    // set carry tensors manually (simulate post-half0 state)
    rc = sd_split_state_set_hs_count(st, 3);
    CHECK(rc == SD_SPLIT_OK, "set hs_count");

    int64_t h_shape[4] = {1, 16, 4, 4};
    auto    h_data     = fill_random(1 * 16 * 4 * 4, 2);
    rc = sd_split_state_set_carry_tensor(st, "h", h_data.data(), h_shape, 4);
    CHECK(rc == SD_SPLIT_OK, "set h");

    int64_t emb_shape[2] = {1, 64};
    auto    emb_data     = fill_random(64, 3);
    rc = sd_split_state_set_carry_tensor(st, "emb", emb_data.data(), emb_shape, 2);
    CHECK(rc == SD_SPLIT_OK, "set emb");

    for (int i = 0; i < 3; ++i) {
        int64_t hs_shape[4] = {1, 8, 4, 4};
        auto    hs_data     = fill_random(1 * 8 * 4 * 4, 4 + i);
        char name[16];
        std::snprintf(name, sizeof(name), "hs.%d", i);
        rc = sd_split_state_set_carry_tensor(st, name, hs_data.data(), hs_shape, 4);
        CHECK(rc == SD_SPLIT_OK, "set hs.i");
    }

    // read back via getter
    int hs_count = -1;
    rc = sd_split_state_get_carry_count(st, &hs_count);
    CHECK(rc == SD_SPLIT_OK && hs_count == 3, "get_carry_count");

    const float*   gp  = nullptr;
    const int64_t* gsh = nullptr;
    int            gnd = 0;
    rc = sd_split_state_get_carry_tensor(st, "h", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_OK && gnd == 4 && gsh[1] == 16, "get h");
    CHECK(std::memcmp(gp, h_data.data(), h_data.size() * sizeof(float)) == 0,
          "h bytes equal");

    rc = sd_split_state_get_carry_tensor(st, "emb", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_OK && gnd == 2, "get emb");
    CHECK(std::memcmp(gp, emb_data.data(), emb_data.size() * sizeof(float)) == 0,
          "emb bytes equal");

    rc = sd_split_state_get_carry_tensor(st, "hs.2", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_OK && gnd == 4, "get hs.2");

    // out of range
    rc = sd_split_state_get_carry_tensor(st, "hs.9", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_EINVAL, "get hs.9 -> EINVAL");

    sd_split_state_free(st);
}

void test_serialize_roundtrip() {
    std::printf("== test_serialize_roundtrip ==\n");
    sd_split_state_t* st = sd_split_state_new();

    sd_split_state_set_hs_count(st, 2);
    int64_t h_shape[4]   = {1, 16, 4, 4};
    int64_t emb_shape[2] = {1, 64};
    int64_t hs_shape[4]  = {1, 8, 4, 4};
    auto    h_data       = fill_random(16 * 4 * 4, 7);
    auto    emb_data     = fill_random(64, 8);
    auto    hs0_data     = fill_random(8 * 4 * 4, 9);
    auto    hs1_data     = fill_random(8 * 4 * 4, 10);
    sd_split_state_set_carry_tensor(st, "h",   h_data.data(),   h_shape,   4);
    sd_split_state_set_carry_tensor(st, "emb", emb_data.data(), emb_shape, 2);
    sd_split_state_set_carry_tensor(st, "hs.0", hs0_data.data(), hs_shape, 4);
    sd_split_state_set_carry_tensor(st, "hs.1", hs1_data.data(), hs_shape, 4);

    uint8_t* buf = nullptr;
    size_t   nbytes = 0;
    int rc = sd_split_state_serialize(st, &buf, &nbytes);
    CHECK(rc == SD_SPLIT_OK, "serialize");
    CHECK(buf != nullptr && nbytes > 0, "non-empty buffer");

    sd_split_state_t* st2 = nullptr;
    rc = sd_split_state_deserialize(buf, nbytes, &st2);
    CHECK(rc == SD_SPLIT_OK, "deserialize");
    CHECK(st2 != nullptr, "got new state");

    int hs_count2 = 0;
    sd_split_state_get_carry_count(st2, &hs_count2);
    CHECK(hs_count2 == 2, "hs_count2 == 2");

    const float* p; const int64_t* sh; int nd;
    sd_split_state_get_carry_tensor(st2, "h", &p, &sh, &nd);
    CHECK(std::memcmp(p, h_data.data(), h_data.size() * sizeof(float)) == 0,
          "h roundtrip equal");
    sd_split_state_get_carry_tensor(st2, "emb", &p, &sh, &nd);
    CHECK(std::memcmp(p, emb_data.data(), emb_data.size() * sizeof(float)) == 0,
          "emb roundtrip equal");
    sd_split_state_get_carry_tensor(st2, "hs.0", &p, &sh, &nd);
    CHECK(std::memcmp(p, hs0_data.data(), hs0_data.size() * sizeof(float)) == 0,
          "hs.0 roundtrip equal");
    sd_split_state_get_carry_tensor(st2, "hs.1", &p, &sh, &nd);
    CHECK(std::memcmp(p, hs1_data.data(), hs1_data.size() * sizeof(float)) == 0,
          "hs.1 roundtrip equal");

    std::free(buf);
    sd_split_state_free(st);
    sd_split_state_free(st2);
}

// SDCD UPLD-half codec lives in include/sdcpp_split_wire.h — shared with
// the role bridge so the wire format has one source of truth.

void test_sdcd_carry_roundtrip() {
    std::printf("== test_sdcd_carry_roundtrip ==\n");
    sd_split_state_t* st = sd_split_state_new();
    sd_split_state_set_hs_count(st, 2);

    int64_t h_shape[4] = {1, 16, 4, 4};
    auto    h_data     = fill_random(16 * 4 * 4, 11);
    int64_t emb_shape[2] = {1, 64};
    auto    emb_data     = fill_random(64, 12);
    int64_t hs_shape[4] = {1, 8, 4, 4};
    auto    hs0_data    = fill_random(8 * 4 * 4, 13);
    auto    hs1_data    = fill_random(8 * 4 * 4, 14);
    sd_split_state_set_carry_tensor(st, "h",   h_data.data(),   h_shape,   4);
    sd_split_state_set_carry_tensor(st, "emb", emb_data.data(), emb_shape, 2);
    sd_split_state_set_carry_tensor(st, "hs.0", hs0_data.data(), hs_shape, 4);
    sd_split_state_set_carry_tensor(st, "hs.1", hs1_data.data(), hs_shape, 4);

    std::vector<uint8_t> wire;
    std::string err;
    CHECK(dist::sdcpp_carry_to_sdcd(st, wire, err), "carry_to_sdcd");

    sd_split_state_t* st2 = sd_split_state_new();
    CHECK(dist::sdcpp_sdcd_to_carry(wire, st2, err), "sdcd_to_carry");

    const float* p; const int64_t* sh; int nd;
    sd_split_state_get_carry_tensor(st2, "h", &p, &sh, &nd);
    CHECK(std::memcmp(p, h_data.data(), h_data.size() * sizeof(float)) == 0,
          "SDCD-wire h equal");
    sd_split_state_get_carry_tensor(st2, "hs.1", &p, &sh, &nd);
    CHECK(std::memcmp(p, hs1_data.data(), hs1_data.size() * sizeof(float)) == 0,
          "SDCD-wire hs.1 equal");

    sd_split_state_free(st);
    sd_split_state_free(st2);
}

void test_block_count_no_ctx() {
    std::printf("== test_block_count_no_ctx ==\n");
    int n = sd_unet_block_count(nullptr);
    CHECK(n == 0, "null ctx -> 0");
    const char* tag = sd_loaded_backbone_tag(nullptr);
    CHECK(std::string(tag) == "unknown", "null ctx -> unknown");
}

void test_cond_lifecycle() {
    std::printf("== test_cond_lifecycle ==\n");
    sd_cond_t* c = sd_cond_new();
    CHECK(c != nullptr, "sd_cond_new");
    CHECK(sd_cond_has_uncond(c) == 0, "fresh cond -> no uncond");

    // Null-ctx encode should fail with EINVAL, not crash.
    int rc = sd_encode_condition(nullptr, "hello", "world", -1, 512, 512, c);
    CHECK(rc == SD_SPLIT_EINVAL, "null ctx encode -> EINVAL");

    // Get-tensor on fresh state: name unknown -> EINVAL, known-but-empty -> EINVAL.
    const float*   gp  = nullptr;
    const int64_t* gsh = nullptr;
    int            gnd = 0;
    rc = sd_cond_get_tensor(c, "bogus", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_EINVAL, "unknown name -> EINVAL");
    rc = sd_cond_get_tensor(c, "cond.crossattn", &gp, &gsh, &gnd);
    CHECK(rc == SD_SPLIT_EINVAL && gp == nullptr, "empty cond.crossattn -> EINVAL");

    sd_cond_free(c);
    sd_cond_free(nullptr);  // null free must be a no-op
}

void test_x_sdcd_roundtrip() {
    std::printf("== test_x_sdcd_roundtrip ==\n");
    // 1×4×8×8 latent shape (SD1.x at 64×64)
    int64_t shape[4] = {1, 4, 8, 8};
    std::vector<float> x = fill_random(1 * 4 * 8 * 8, 42);

    std::vector<uint8_t> wire;
    std::string err;
    CHECK(dist::sdcpp_x_to_sdcd(x.data(), shape, 4, /*step=*/7, /*ts=*/250.5f,
                                wire, err), "encode x");
    CHECK(wire.size() > 0, "non-empty wire");

    dist::SdcdFrame frame;
    const float* gx = nullptr;
    std::vector<int64_t> gshape;
    int gstep = 0;
    float gts = 0.f;
    CHECK(dist::sdcpp_sdcd_to_x(wire.data(), wire.size(), frame,
                                &gx, gshape, &gstep, &gts, err),
          "decode x");
    CHECK(gstep == 7 && std::abs(gts - 250.5f) < 1e-4f, "step/ts roundtrip");
    CHECK(gshape.size() == 4 && gshape[1] == 4, "x shape roundtrip");
    CHECK(std::memcmp(gx, x.data(), x.size() * sizeof(float)) == 0,
          "x bytes equal");
}

void test_vae_decode_null_ctx() {
    std::printf("== test_vae_decode_null_ctx ==\n");
    float latent[4] = {0.f, 0.f, 0.f, 0.f};
    int64_t shape[4] = {1, 4, 1, 1};
    float*   data = nullptr;
    int64_t* sh   = nullptr;
    int      nd   = 0;
    int rc = sd_decode_first_stage_to_floats(nullptr, latent, shape, 4, &data, &sh, &nd);
    CHECK(rc == SD_SPLIT_EINVAL, "null ctx -> EINVAL");

    rc = sd_decode_first_stage_to_floats(nullptr, nullptr, nullptr, 0, &data, &sh, &nd);
    CHECK(rc == SD_SPLIT_EINVAL, "all null -> EINVAL");

    sd_vae_image_free(nullptr, nullptr);  // null free must be a no-op
}

} // namespace

int main() {
    test_input_carry_roundtrip();
    test_serialize_roundtrip();
    test_sdcd_carry_roundtrip();
    test_block_count_no_ctx();
    test_cond_lifecycle();
    test_x_sdcd_roundtrip();
    test_vae_decode_null_ctx();

    if (g_fails == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("FAILED: %d\n", g_fails);
    return 1;
}
