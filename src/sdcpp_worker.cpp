// dist-sdcpp-worker — C++ diffusion backend (whole-pipeline, single rig).
//
// Forked by DppAdapter when a rig advertises `sdcpp:1` and the dpp_route
// targets the entire pipeline at one agent.  Same launch pattern as
// `python -m dpp_runtime`; different backend (sd.cpp instead of diffusers).
//
// Modes:
//   --probe             one-shot: print {"ok":true,"backend":"..."} and exit
//   --daemon            persistent: line-based JSON cmd loop on stdin/stdout
//   (legacy one-shot)   --model … --prompt … --out … (kept for compatibility
//                       with the Phase A adapter and direct CLI use)
//
// Daemon wire (#254 / phase B):
//   - Requests arrive on stdin, one JSON object per newline-terminated line.
//   - Responses are emitted on stdout, also as newline-terminated JSON.
//   - Stdout is line-buffered (we explicitly fflush) so the adapter's reader
//     thread can demux progress vs final events without parsing partials.
//   - Commands processed serially: progress lines belong to the in-flight req.
//
// Recognised commands (cmd field):
//   {"cmd":"gen",     ...}  generate (loads model on demand / on model change)
//   {"cmd":"unload"}        free the resident sd_ctx_t
//   {"cmd":"quit"}          exit 0
//
// Response kinds (emitted on stdout):
//   {"kind":"sdcpp_progress","req_id":N,"step":i,"steps":S,"t":secs}
//   {"kind":"sdcpp_done",    "req_id":N,"out":"/tmp/…png"}
//   {"kind":"sdcpp_error",   "req_id":N,"error":"…"}
//   {"kind":"sdcpp_ready"}  one-shot, emitted after the daemon's argv is
//                            parsed — the adapter waits for this before it
//                            starts piping requests, same pattern as the
//                            python runtime's DPP_LISTEN handshake.
//
// Phase C (task TBD): per-role entrypoints (text-encode / unet block-range /
// vae-decode) for participation in a DPP chain alongside python rigs.  That
// needs upstream sd.cpp patches to expose the StableDiffusionGGML internals
// — out of scope for #254.

#include "stable-diffusion.h"
#include "sdcpp_roles.h"
#include "sdt_codec.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

// stb_image_write is bundled inside sd.cpp's source tree.  Reuse it for
// PNG encode rather than vendoring our own copy.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

namespace {

// In daemon mode the progress callback emits JSON on stdout instead of a
// human-readable line on stderr.  A per-request id is set before each gen.
std::atomic<int>    g_daemon_mode{0};
std::atomic<int>    g_current_req{0};

void log_cb(sd_log_level_t lvl, const char* text, void* /*data*/) {
    const char* tag = (lvl == SD_LOG_ERROR) ? "ERR" :
                      (lvl == SD_LOG_WARN)  ? "WRN" :
                      (lvl == SD_LOG_INFO)  ? "INF" : "DBG";
    std::fprintf(stderr, "[sdcpp:%s] %s", tag, text ? text : "");
    if (!text || text[0] == '\0' || text[std::strlen(text) - 1] != '\n') {
        std::fputc('\n', stderr);
    }
}

void progress_cb(int step, int steps, float t, void* /*data*/) {
    if (g_daemon_mode.load()) {
        std::fprintf(stdout,
                     "{\"kind\":\"sdcpp_progress\",\"req_id\":%d,"
                     "\"step\":%d,\"steps\":%d,\"t\":%.4f}\n",
                     g_current_req.load(), step, steps, (double)t);
        std::fflush(stdout);
    } else {
        std::fprintf(stderr, "[sdcpp:PROG] %d/%d (%.2fs)\n", step, steps, t);
    }
}

int die(const char* msg) {
    std::fprintf(stderr, "[sdcpp:FATAL] %s\n", msg);
    return 2;
}

bool arg_eq(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

int do_probe() {
    // Just validates that the binary loads, libraries resolve, and the
    // ggml backends initialise.  Used by DppAdapter::probe_local_caps so
    // the rig can advertise `sdcpp:1` without paying a model-load cost.
    std::fprintf(stdout, "{\"ok\":true,\"backend\":\"%s\"}\n",
                 sd_get_system_info() ? sd_get_system_info() : "unknown");
    return 0;
}

// ─── Minimal JSON field extraction ──────────────────────────────────────
// daemon command lines are flat objects we author ourselves.  Avoid pulling
// in a JSON library to keep the worker's link surface tiny.

std::string json_str(const std::string& s, const char* key, const std::string& dflt = {}) {
    std::string k = std::string("\"") + key + "\":";
    auto p = s.find(k);
    if (p == std::string::npos) return dflt;
    p += k.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= s.size() || s[p] != '"') return dflt;
    ++p;
    std::string out;
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) {
            char c = s[p + 1];
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case '"': out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/');  break;
                default:  out.push_back(c);    break;
            }
            p += 2;
        } else {
            out.push_back(s[p++]);
        }
    }
    return out;
}

long long json_int(const std::string& s, const char* key, long long dflt) {
    std::string k = std::string("\"") + key + "\":";
    auto p = s.find(k);
    if (p == std::string::npos) return dflt;
    p += k.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    char* end = nullptr;
    long long v = std::strtoll(s.c_str() + p, &end, 10);
    if (end == s.c_str() + p) return dflt;
    return v;
}

double json_dbl(const std::string& s, const char* key, double dflt) {
    std::string k = std::string("\"") + key + "\":";
    auto p = s.find(k);
    if (p == std::string::npos) return dflt;
    p += k.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p) return dflt;
    return v;
}

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)(unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void emit_done(int req_id, const std::string& out_path) {
    std::fprintf(stdout, "{\"kind\":\"sdcpp_done\",\"req_id\":%d,\"out\":\"%s\"}\n",
                 req_id, json_escape(out_path).c_str());
    std::fflush(stdout);
}

void emit_error(int req_id, const std::string& msg) {
    std::fprintf(stdout, "{\"kind\":\"sdcpp_error\",\"req_id\":%d,\"error\":\"%s\"}\n",
                 req_id, json_escape(msg).c_str());
    std::fflush(stdout);
}

// ─── Daemon state ───────────────────────────────────────────────────────
// One resident sd_ctx_t per worker; reloaded only when the requested model
// path changes.  This is the whole point of phase B — eat the multi-GB
// load once per (model, worker) instead of once per generation.

struct DaemonState {
    sd_ctx_t*   ctx          = nullptr;
    std::string loaded_model;
    std::string loaded_role;            // "", "te", "unet", "vae"
    int         loaded_threads = 0;
    bool        loaded_vae_decode_only = true;
};

// CF12-W1h: role-aware load.  Until the upstream sd.cpp patch exposes a
// per-role skip flag, we still load the full ctx — but track the role
// so future cap probes / metrics know what this worker advertised.
// The `DIST_SDCPP_ROLE` env var is honoured as a soft hint for any
// patched build that does support skipping.
bool load_model(DaemonState& st, const std::string& model_path,
                int n_threads, const std::string& role) {
    if (st.ctx && st.loaded_model == model_path &&
        st.loaded_threads == n_threads && st.loaded_role == role) {
        return true;
    }
    if (st.ctx) {
        free_sd_ctx(st.ctx);
        st.ctx = nullptr;
        st.loaded_model.clear();
        st.loaded_role.clear();
    }
    if (!role.empty()) {
        // soft hint — picked up by a patched sd.cpp loader, ignored otherwise
        setenv("DIST_SDCPP_ROLE", role.c_str(), 1);
    }
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.model_path              = model_path.c_str();
    p.n_threads               = n_threads;
    // VAE-only / TE-only rigs still benefit from vae_decode_only=true
    // (skips encoder weights); UNet/full rigs may want it false for img2img.
    p.vae_decode_only         = (role == "te") ? true : st.loaded_vae_decode_only;
    p.free_params_immediately = false;
    st.ctx = new_sd_ctx(&p);
    if (!st.ctx) return false;
    st.loaded_model   = model_path;
    st.loaded_threads = n_threads;
    st.loaded_role    = role;
    return true;
}

// ── Base64 (for SDCD/SDT/UPLD frames on the JSON wire) ──────────────────
// The daemon's stdin/stdout is line-oriented JSON; binary frames are
// transported as base64 strings.  Adapter ↔ worker payloads are tiny
// (a few KiB per step) so the 33% overhead is acceptable.  The P2P
// data channel between rigs continues to ship the raw bytes.

const char B64A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
std::string b64encode(const uint8_t* p, size_t n) {
    std::string out;
    out.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t(p[i]) << 16);
        if (i + 1 < n) v |= (uint32_t(p[i + 1]) << 8);
        if (i + 2 < n) v |= uint32_t(p[i + 2]);
        out.push_back(B64A[(v >> 18) & 0x3F]);
        out.push_back(B64A[(v >> 12) & 0x3F]);
        out.push_back(i + 1 < n ? B64A[(v >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < n ? B64A[ v       & 0x3F] : '=');
    }
    return out;
}
int b64val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
std::vector<uint8_t> b64decode(const std::string& s) {
    std::vector<uint8_t> out;
    out.reserve((s.size() / 4) * 3);
    int bits = 0, acc = 0;
    for (unsigned char c : s) {
        if (c == '=' || c == ' ' || c == '\r' || c == '\n') continue;
        int v = b64val(c);
        if (v < 0) continue;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t((acc >> bits) & 0xFF));
        }
    }
    return out;
}

void handle_gen(DaemonState& st, const std::string& line) {
    int req_id = (int)json_int(line, "req_id", 0);
    g_current_req.store(req_id);

    std::string model_path = json_str(line, "model_path");
    std::string prompt     = json_str(line, "prompt");
    std::string neg        = json_str(line, "negative_prompt");
    std::string sampler    = json_str(line, "sampler", "euler_a");
    std::string out_path   = json_str(line, "out");
    int w     = (int)json_int(line, "width",  512);
    int h     = (int)json_int(line, "height", 512);
    int steps = (int)json_int(line, "steps",  20);
    float cfg = (float)json_dbl(line, "cfg",  7.0);
    long long seed = json_int(line, "seed", -1);
    int n_threads = (int)json_int(line, "threads", 0);
    if (n_threads <= 0) n_threads = sd_get_num_physical_cores();

    if (model_path.empty()) { emit_error(req_id, "missing model_path"); return; }
    if (prompt.empty())     { emit_error(req_id, "missing prompt");     return; }
    if (out_path.empty())   { emit_error(req_id, "missing out");        return; }

    if (!load_model(st, model_path, n_threads, std::string{})) {
        emit_error(req_id, "new_sd_ctx returned NULL (check model_path)");
        return;
    }

    sd_img_gen_params_t gen;
    std::memset(&gen, 0, sizeof(gen));
    gen.prompt          = prompt.c_str();
    gen.negative_prompt = neg.c_str();
    gen.width           = w;
    gen.height          = h;
    gen.seed            = seed;
    gen.batch_count     = 1;
    sd_sample_params_init(&gen.sample_params);
    gen.sample_params.sample_steps     = steps;
    gen.sample_params.guidance.txt_cfg = cfg;
    gen.sample_params.sample_method    = str_to_sample_method(sampler.c_str());
    gen.sample_params.scheduler        = sd_get_default_scheduler(
        st.ctx, gen.sample_params.sample_method);

    sd_image_t* result = generate_image(st.ctx, &gen);
    if (!result) { emit_error(req_id, "generate_image returned NULL"); return; }

    int rc = stbi_write_png(out_path.c_str(),
                            (int)result->width, (int)result->height,
                            (int)result->channel, result->data,
                            (int)(result->width * result->channel));
    if (result->data) std::free(result->data);
    std::free(result);

    if (rc == 0) { emit_error(req_id, "stbi_write_png failed"); return; }
    emit_done(req_id, out_path);
}

// ─── CF12-W1g: role-aware daemon cmds ────────────────────────────────────
// All three follow the same shape:
//   in : {"cmd":"sdr_<role>","req_id":N,"model_path":"…","role":"…", … }
//        binary payloads carried as base64 strings in `sdcd_b64` / `sdt_b64`.
//   out: {"kind":"sdcpp_role_done","req_id":N,"role":"…","frame_b64":"…"}
//        on failure: {"kind":"sdcpp_error","req_id":N,"error":"…"}

void emit_role_done(int req_id, const char* role, const std::vector<uint8_t>& frame) {
    std::fprintf(stdout,
                 "{\"kind\":\"sdcpp_role_done\",\"req_id\":%d,"
                 "\"role\":\"%s\",\"frame_b64\":\"%s\"}\n",
                 req_id, role, b64encode(frame.data(), frame.size()).c_str());
    std::fflush(stdout);
}

bool ensure_loaded(DaemonState& st, const std::string& line, const std::string& role,
                   int& req_id, std::string& model_path, int& n_threads) {
    req_id   = (int)json_int(line, "req_id", 0);
    model_path = json_str(line, "model_path");
    n_threads  = (int)json_int(line, "threads", 0);
    if (n_threads <= 0) n_threads = sd_get_num_physical_cores();

    if (model_path.empty()) { emit_error(req_id, "missing model_path"); return false; }
    if (!load_model(st, model_path, n_threads, role)) {
        emit_error(req_id, "new_sd_ctx returned NULL (check model_path)");
        return false;
    }
    return true;
}

void handle_role_encode_text(DaemonState& st, const std::string& line) {
    int req_id, n_threads; std::string model;
    if (!ensure_loaded(st, line, "te", req_id, model, n_threads)) return;
    g_current_req.store(req_id);

    std::string prompt = json_str(line, "prompt");
    std::string neg    = json_str(line, "negative_prompt");
    bool cfg_split     = json_int(line, "cfg_split", 0) != 0;
    int clip_skip      = (int)json_int(line, "clip_skip", -1);

    sd_role_encode_text_in_t in{};
    in.prompt          = prompt.c_str();
    in.negative_prompt = neg.c_str();
    in.cfg_split       = cfg_split;
    in.clip_skip       = clip_skip;

    sd_role_buf_t buf{};
    sd_role_status_t s = sd_role_encode_text(st.ctx, &in, &buf);
    if (s.code != 0) { emit_error(req_id, s.msg ? s.msg : "encode_text failed"); return; }
    std::vector<uint8_t> v(buf.data, buf.data + buf.nbytes);
    sd_role_buf_free(&buf);
    emit_role_done(req_id, "te", v);
}

void handle_role_sample(DaemonState& st, const std::string& line) {
    int req_id, n_threads; std::string model;
    if (!ensure_loaded(st, line, "unet", req_id, model, n_threads)) return;
    g_current_req.store(req_id);

    std::string sdcd_b64 = json_str(line, "sdcd_b64");
    std::vector<uint8_t> sdcd = b64decode(sdcd_b64);

    int  w     = (int)json_int(line, "width",  512);
    int  h     = (int)json_int(line, "height", 512);
    int  steps = (int)json_int(line, "steps",  20);
    float cfg  = (float)json_dbl(line, "cfg",  7.0);
    long long seed = json_int(line, "seed", -1);
    std::string sampler = json_str(line, "sampler", "euler_a");
    std::string scheduler = json_str(line, "scheduler");

    sd_role_sample_in_t in{};
    in.sdcd_cond         = sdcd.data();
    in.sdcd_cond_nbytes  = sdcd.size();
    in.width             = w;
    in.height            = h;
    in.steps             = steps;
    in.cfg               = cfg;
    in.seed              = seed;
    in.sampler           = sampler.c_str();
    in.scheduler         = scheduler.empty() ? nullptr : scheduler.c_str();
    in.sdt_init_latent   = nullptr;
    in.sdt_init_nbytes   = 0;
    in.strength          = 0.0f;

    sd_role_buf_t buf{};
    sd_role_status_t s = sd_role_sample(st.ctx, &in, &buf);
    if (s.code != 0) { emit_error(req_id, s.msg ? s.msg : "sample failed"); return; }
    std::vector<uint8_t> v(buf.data, buf.data + buf.nbytes);
    sd_role_buf_free(&buf);
    emit_role_done(req_id, "unet", v);
}

void handle_role_decode(DaemonState& st, const std::string& line) {
    int req_id, n_threads; std::string model;
    if (!ensure_loaded(st, line, "vae", req_id, model, n_threads)) return;
    g_current_req.store(req_id);

    std::vector<uint8_t> sdt = b64decode(json_str(line, "sdt_b64"));

    sd_role_decode_in_t in{};
    in.sdt_latent        = sdt.data();
    in.sdt_latent_nbytes = sdt.size();

    sd_role_buf_t buf{};
    sd_role_status_t s = sd_role_decode_latent(st.ctx, &in, &buf);
    if (s.code != 0) { emit_error(req_id, s.msg ? s.msg : "decode failed"); return; }
    std::vector<uint8_t> v(buf.data, buf.data + buf.nbytes);
    sd_role_buf_free(&buf);
    emit_role_done(req_id, "vae", v);
}

void handle_role_sample_blocks(DaemonState& st, const std::string& line) {
    int req_id, n_threads; std::string model;
    if (!ensure_loaded(st, line, "unet", req_id, model, n_threads)) return;
    g_current_req.store(req_id);

    std::vector<uint8_t> sdcd = b64decode(json_str(line, "sdcd_b64"));
    std::vector<uint8_t> upld = b64decode(json_str(line, "upld_b64"));

    sd_role_sample_blocks_in_t in{};
    in.sdcd_cond        = sdcd.data();
    in.sdcd_cond_nbytes = sdcd.size();
    in.upld_in          = upld.data();
    in.upld_in_nbytes   = upld.size();
    in.block_lo         = (int)json_int(line, "block_lo", 0);
    in.block_hi         = (int)json_int(line, "block_hi", 0);
    in.block_total      = (int)json_int(line, "block_total", 0);
    in.steps            = (int)json_int(line, "steps", 20);
    in.cfg              = (float)json_dbl(line, "cfg", 7.0);
    in.seed             = json_int(line, "seed", -1);
    std::string sampler = json_str(line, "sampler", "euler_a");
    std::string sched   = json_str(line, "scheduler");
    in.sampler          = sampler.c_str();
    in.scheduler        = sched.empty() ? nullptr : sched.c_str();

    sd_role_buf_t buf{};
    sd_role_status_t s = sd_role_sample_blocks(st.ctx, &in, &buf);
    if (s.code != 0) { emit_error(req_id, s.msg ? s.msg : "sample_blocks failed"); return; }
    std::vector<uint8_t> v(buf.data, buf.data + buf.nbytes);
    sd_role_buf_free(&buf);
    emit_role_done(req_id, "unet_blocks", v);
}

void handle_role_caps(const std::string& line) {
    int req_id = (int)json_int(line, "req_id", 0);
    // Static caps until backbone inference lands.
    std::fprintf(stdout,
                 "{\"kind\":\"sdcpp_caps\",\"req_id\":%d,"
                 "\"roles\":[\"te\",\"unet\",\"vae\"],"
                 "\"block_split\":false,"
                 "\"backbone\":\"unknown\",\"sdt_ver\":1,\"upld_ver\":1}\n",
                 req_id);
    std::fflush(stdout);
}

int daemon_loop() {
    g_daemon_mode.store(1);

    // Hand the parent a ready marker so it can start piping commands.
    std::fprintf(stdout, "{\"kind\":\"sdcpp_ready\"}\n");
    std::fflush(stdout);

    DaemonState st;
    std::string buf;
    char chunk[4096];

    while (true) {
        // Use the raw fd; fread() on a pipe blocks until the full buffer
        // is filled, which deadlocks our line-oriented protocol.
        ssize_t n = ::read(STDIN_FILENO, chunk, sizeof(chunk));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        buf.append(chunk, chunk + n);

        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (line.empty()) continue;

            std::string cmd = json_str(line, "cmd");
            if (cmd == "gen") {
                handle_gen(st, line);
            } else if (cmd == "sdr_encode_text") {
                handle_role_encode_text(st, line);
            } else if (cmd == "sdr_sample") {
                handle_role_sample(st, line);
            } else if (cmd == "sdr_sample_blocks") {
                handle_role_sample_blocks(st, line);
            } else if (cmd == "sdr_decode_latent") {
                handle_role_decode(st, line);
            } else if (cmd == "sdr_caps") {
                handle_role_caps(line);
            } else if (cmd == "unload") {
                if (st.ctx) { free_sd_ctx(st.ctx); st.ctx = nullptr; st.loaded_model.clear(); st.loaded_role.clear(); }
                int req_id = (int)json_int(line, "req_id", 0);
                std::fprintf(stdout, "{\"kind\":\"sdcpp_unloaded\",\"req_id\":%d}\n", req_id);
                std::fflush(stdout);
            } else if (cmd == "quit") {
                if (st.ctx) { free_sd_ctx(st.ctx); st.ctx = nullptr; }
                return 0;
            } else {
                int req_id = (int)json_int(line, "req_id", 0);
                emit_error(req_id, std::string("unknown cmd: ") + cmd);
            }
        }
    }

    if (st.ctx) free_sd_ctx(st.ctx);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);

    if (argc < 2) {
        std::fprintf(stderr,
            "usage: dist-sdcpp-worker --probe\n"
            "       dist-sdcpp-worker --daemon\n"
            "       dist-sdcpp-worker --model <path> --prompt <text> "
            "--out <path.png> [--neg <text>] [--w 512] [--h 512] "
            "[--steps 20] [--cfg 7.0] [--seed -1] [--sampler euler_a] "
            "[--threads N]\n");
        return 1;
    }

    std::string model_path, prompt, neg_prompt, out_path, sampler_name = "euler_a";
    int width = 512, height = 512, steps = 20, n_threads = 0;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    bool probe_only = false;
    bool daemon_only = false;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--probe")) {
            probe_only = true;
        } else if (arg_eq(argv[i], "--daemon")) {
            daemon_only = true;
        } else if (i + 1 < argc && arg_eq(argv[i], "--model")) {
            model_path = argv[++i];
        } else if (i + 1 < argc && arg_eq(argv[i], "--prompt")) {
            prompt = argv[++i];
        } else if (i + 1 < argc && arg_eq(argv[i], "--neg")) {
            neg_prompt = argv[++i];
        } else if (i + 1 < argc && arg_eq(argv[i], "--out")) {
            out_path = argv[++i];
        } else if (i + 1 < argc && arg_eq(argv[i], "--w")) {
            width = std::atoi(argv[++i]);
        } else if (i + 1 < argc && arg_eq(argv[i], "--h")) {
            height = std::atoi(argv[++i]);
        } else if (i + 1 < argc && arg_eq(argv[i], "--steps")) {
            steps = std::atoi(argv[++i]);
        } else if (i + 1 < argc && arg_eq(argv[i], "--cfg")) {
            cfg_scale = (float)std::atof(argv[++i]);
        } else if (i + 1 < argc && arg_eq(argv[i], "--seed")) {
            seed = (int64_t)std::atoll(argv[++i]);
        } else if (i + 1 < argc && arg_eq(argv[i], "--sampler")) {
            sampler_name = argv[++i];
        } else if (i + 1 < argc && arg_eq(argv[i], "--threads")) {
            n_threads = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "[sdcpp] unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (probe_only)  return do_probe();
    if (daemon_only) return daemon_loop();

    if (model_path.empty()) return die("--model is required");
    if (prompt.empty())     return die("--prompt is required");
    if (out_path.empty())   return die("--out is required");
    if (n_threads <= 0)     n_threads = sd_get_num_physical_cores();

    // ── Load context ──────────────────────────────────────────────────
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path                = model_path.c_str();
    ctx_params.n_threads                 = n_threads;
    ctx_params.vae_decode_only           = true;
    ctx_params.free_params_immediately   = false;

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (!sd_ctx) return die("new_sd_ctx returned NULL — check model_path");

    // ── Build gen request ─────────────────────────────────────────────
    sd_img_gen_params_t gen;
    std::memset(&gen, 0, sizeof(gen));
    gen.prompt          = prompt.c_str();
    gen.negative_prompt = neg_prompt.c_str();
    gen.width           = width;
    gen.height          = height;
    gen.seed            = seed;
    gen.batch_count     = 1;

    sd_sample_params_init(&gen.sample_params);
    gen.sample_params.sample_steps              = steps;
    gen.sample_params.guidance.txt_cfg          = cfg_scale;
    gen.sample_params.sample_method             = str_to_sample_method(sampler_name.c_str());
    gen.sample_params.scheduler                 = sd_get_default_scheduler(sd_ctx, gen.sample_params.sample_method);

    // ── Generate ──────────────────────────────────────────────────────
    sd_image_t* result = generate_image(sd_ctx, &gen);
    if (!result) {
        free_sd_ctx(sd_ctx);
        return die("generate_image returned NULL");
    }

    // ── PNG encode ────────────────────────────────────────────────────
    int rc = stbi_write_png(out_path.c_str(),
                            (int)result->width,
                            (int)result->height,
                            (int)result->channel,
                            result->data,
                            (int)(result->width * result->channel));
    if (result->data) std::free(result->data);
    std::free(result);
    free_sd_ctx(sd_ctx);

    if (rc == 0) return die("stbi_write_png failed");

    std::fprintf(stdout, "{\"ok\":true,\"out\":\"%s\"}\n", out_path.c_str());
    return 0;
}
