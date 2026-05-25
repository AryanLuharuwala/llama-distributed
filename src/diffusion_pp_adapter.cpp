#include "diffusion_pp_adapter.h"

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

extern char** environ;

namespace dist {

// ─── DppRuntime ────────────────────────────────────────────────────────────

DppRuntime::~DppRuntime() {
    close();
}

void DppRuntime::close() {
    stop.store(true);
    if (sock_fd >= 0) {
        ::shutdown(sock_fd, SHUT_RDWR);
        ::close(sock_fd);
        sock_fd = -1;
    }
    if (reader.joinable()) reader.join();
    if (launcher.joinable()) launcher.join();
    if (pid > 0) {
        ::kill(pid, SIGTERM);
        // Best-effort reap with short timeout — caller is in dtor, no policy.
        for (int i = 0; i < 50; ++i) {
            int status = 0;
            pid_t r = ::waitpid(pid, &status, WNOHANG);
            if (r == pid) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        pid = -1;
    }
}

bool DppRuntime::send_or_buffer(const DppFrame& f) {
    if (failed.load()) return false;
    if (!ready.load()) {
        std::lock_guard<std::mutex> lk(pending_mu);
        // Re-check under the lock — launcher drains pending under the same
        // mutex right before flipping ready, so this avoids a torn drop.
        if (!ready.load()) {
            pending.push_back(f);
            return true;
        }
    }
    return raw_send(f);
}

bool DppRuntime::raw_send(const DppFrame& f) {
    if (sock_fd < 0) return false;
    uint32_t n = (uint32_t) f.bytes.size();
    uint8_t hdr[4] = {
        (uint8_t)(n >> 24), (uint8_t)(n >> 16),
        (uint8_t)(n >> 8),  (uint8_t)n,
    };
    ssize_t w = ::send(sock_fd, hdr, 4, MSG_NOSIGNAL);
    if (w != 4) return false;
    size_t off = 0;
    while (off < f.bytes.size()) {
        ssize_t k = ::send(sock_fd, f.bytes.data() + off,
                           f.bytes.size() - off, MSG_NOSIGNAL);
        if (k <= 0) return false;
        off += (size_t)k;
    }
    return true;
}

// ─── DppAdapter ────────────────────────────────────────────────────────────

DppAdapter::DppAdapter() = default;

DppAdapter::~DppAdapter() {
    {
        std::lock_guard<std::mutex> lk(sdcpp_daemon_mu_);
        sdcpp_daemon_.reset();  // sends quit + reaps the worker
    }
    std::lock_guard<std::mutex> lk(runtimes_mu_);
    runtimes_.clear();
}

std::vector<DppFrame> DppAdapter::drain_outbox() {
    std::vector<DppFrame> out;
    std::lock_guard<std::mutex> lk(outbox_mu_);
    out.swap(outbox_);
    return out;
}

std::vector<std::string> DppAdapter::drain_text_outbox() {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lk(text_outbox_mu_);
    out.swap(text_outbox_);
    return out;
}

DppRuntime* DppAdapter::find_runtime_locked(const std::string& key) {
    auto it = runtimes_.find(key);
    if (it == runtimes_.end()) return nullptr;
    return it->second.get();
}

// Tiny JSON field puller — same minimalist style as dist_node_main.cpp's
// pp_route parser.  We deliberately avoid pulling in a full JSON library
// here; dpp_route messages are flat and well-known.
static long long json_int(const std::string& s, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    auto p = s.find(k);
    if (p == std::string::npos) return -1;
    size_t i = p + k.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    size_t j = i;
    while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '-')) ++j;
    if (i == j) return -1;
    try { return std::stoll(s.substr(i, j - i)); } catch (...) { return -1; }
}

static std::string json_str(const std::string& s, const char* key) {
    std::string k = std::string("\"") + key + "\":\"";
    auto p = s.find(k);
    if (p == std::string::npos) return "";
    size_t i = p + k.size();
    size_t j = i;
    while (j < s.size() && s[j] != '"') {
        if (s[j] == '\\' && j + 1 < s.size()) ++j;
        ++j;
    }
    // We accept these unescaped — model names don't contain control bytes.
    return s.substr(i, j - i);
}

// Extract a JSON *object* value (the substring including outer braces).  Used
// for the nested "config" object on dpp_route messages — flat json_str can't
// span braces.  Returns "" if key absent or malformed.
static std::string json_obj(const std::string& s, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    auto p = s.find(k);
    if (p == std::string::npos) return "";
    size_t i = p + k.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i >= s.size() || s[i] != '{') return "";
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    size_t start = i;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}') {
            if (--depth == 0) return s.substr(start, i - start + 1);
        }
    }
    return "";
}

// Build a FLAG_DPP_CONFIG=0x80 ACTV frame carrying UTF-8 JSON config bytes.
// Matches Python wire.py decode: 20-byte header + 0 dim bytes (rank=0) +
// u32 payload_len + payload.
static DppFrame make_config_frame(uint16_t req_id, uint16_t stage_idx,
                                  const std::string& cfg_json) {
    DppFrame f;
    const uint32_t magic = 0x41435456u;  // "ACTV"
    const uint8_t  ver   = 1;
    const uint8_t  typ   = 1;            // TYPE_ACT
    const uint8_t  dtype = 4;            // DTYPE_BYTES
    const uint8_t  rank  = 0;
    const uint8_t  flags = 0x80;         // FLAG_DPP_CONFIG
    const uint32_t plen  = (uint32_t)cfg_json.size();
    f.bytes.resize(20 + 4 + plen);
    auto put_u16 = [&](size_t off, uint16_t v) {
        f.bytes[off]   = (uint8_t)(v >> 8);
        f.bytes[off+1] = (uint8_t)v;
    };
    auto put_u32 = [&](size_t off, uint32_t v) {
        f.bytes[off]   = (uint8_t)(v >> 24);
        f.bytes[off+1] = (uint8_t)(v >> 16);
        f.bytes[off+2] = (uint8_t)(v >> 8);
        f.bytes[off+3] = (uint8_t)v;
    };
    put_u32(0, magic);
    f.bytes[4] = ver;
    f.bytes[5] = typ;
    put_u16(6, req_id);
    put_u16(8, stage_idx);
    put_u32(10, 0);    // tok_seq
    f.bytes[14] = dtype;
    f.bytes[15] = rank;
    f.bytes[16] = flags;
    f.bytes[17] = 0;
    f.bytes[18] = 0;
    f.bytes[19] = 0;
    put_u32(20, plen);
    if (plen) std::memcpy(f.bytes.data() + 24, cfg_json.data(), plen);
    return f;
}

static bool set_nonblock(int fd, bool on) {
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl < 0) return false;
    if (on) fl |= O_NONBLOCK; else fl &= ~O_NONBLOCK;
    return fcntl(fd, F_SETFL, fl) == 0;
}

bool DppAdapter::handle_dpp_route(const std::string& json_msg,
                                  const std::string& python_bin,
                                  const std::string& module_path,
                                  std::string& err_out) {
    long long rid       = json_int(json_msg, "req_id");
    long long stage_idx = json_int(json_msg, "stage_idx");
    std::string role  = json_str(json_msg, "role");
    std::string model = json_str(json_msg, "model");
    if (rid < 0 || role.empty()) {
        err_out = "dpp_route: missing req_id or role";
        return false;
    }
    if (stage_idx < 0) stage_idx = 0; // tolerate older controllers
    std::string key = role + ":" + model;

    // Register the ((req_id, stage_idx) → key) mapping immediately so any
    // in-flight ACTV arriving before the launcher finishes routes to this
    // runtime's pending queue rather than getting dropped.  The stage_idx
    // dimension is what lets a single agent host a multi-stage chain
    // (e.g. text_encoder→unet→vae all local) — outputs from one local
    // runtime get re-dispatched to the next local runtime instead of
    // round-tripping through the control plane.
    {
        std::lock_guard<std::mutex> lk(req_mu_);
        req_to_runtime_[reqStageKey((uint16_t)rid, (uint16_t)stage_idx)] = key;
    }

    // Extract the per-request config blob ({steps, cfg_scale, width, …}) so
    // explicit request values override the worker's env-var defaults.  The
    // config frame is sent BEFORE any ACTV — workers stash it by req_id and
    // pop on kickoff.
    std::string cfg_json = json_obj(json_msg, "config");

    DppRuntime* target = nullptr;
    {
        std::lock_guard<std::mutex> lk(runtimes_mu_);
        if (DppRuntime* existing = find_runtime_locked(key); existing != nullptr) {
            target = existing;
        } else {
            auto up = std::make_unique<DppRuntime>();
            up->role  = role;
            up->model = model;
            target = up.get();
            target->launcher = std::thread(&DppAdapter::launch_python, this,
                                           target, python_bin, module_path);
            runtimes_.emplace(key, std::move(up));
        }
    }
    // Only the UNet stage consumes per-req config (steps/cfg/w/h/seed/frames).
    // TE/VAE handlers ignore it — gate the send to avoid leaking entries in
    // their _cfg_buf dict (req_id space is uint16; bounded but pointless).
    if (!cfg_json.empty() && target != nullptr && role == "unet") {
        DppFrame f = make_config_frame((uint16_t)rid, (uint16_t)stage_idx, cfg_json);
        target->send_or_buffer(f);
    }
    return true;
}

bool DppAdapter::dispatch_actv(const uint8_t* data, size_t n) {
    // Pull req_id (6-7) and stage_idx (8-9) out of the ACTV header — BE.
    if (n < 20) return false;
    uint32_t magic = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
                     ((uint32_t)data[2] << 8)  |  (uint32_t)data[3];
    if (magic != 0x41435456u) return false; // "ACTV"
    uint16_t rid    = ((uint16_t)data[6] << 8) | (uint16_t)data[7];
    uint16_t stage  = ((uint16_t)data[8] << 8) | (uint16_t)data[9];

    std::string key;
    {
        std::lock_guard<std::mutex> lk(req_mu_);
        auto it = req_to_runtime_.find(reqStageKey(rid, stage));
        if (it == req_to_runtime_.end()) return false;
        key = it->second;
    }
    DppRuntime* rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(runtimes_mu_);
        rt = find_runtime_locked(key);
    }
    if (!rt) return false;
    DppFrame f;
    f.bytes.assign(data, data + n);
    return rt->send_or_buffer(f);
}

// Background runner: blocking model-load + socket-connect happens here so the
// main loop stays free to service WS pings.
void DppAdapter::launch_python(DppRuntime* rt,
                               const std::string& python_bin,
                               const std::string& module_path) {
    auto fail = [&](const std::string& err) {
        std::cerr << "[dpp] launch failed: " << err << "\n";
        rt->failed.store(true);
        // Emit an error frame for every req_id currently pointed at this
        // runtime so the server can unblock the infer_dpp handler.
        std::vector<uint16_t> rids;
        {
            std::lock_guard<std::mutex> lk(req_mu_);
            std::string key = rt->role + ":" + rt->model;
            for (auto& kv : req_to_runtime_) {
                if (kv.second == key) rids.push_back(kv.first);
            }
        }
        std::lock_guard<std::mutex> ok(outbox_mu_);
        for (uint16_t rid : rids) {
            outbox_.push_back(make_error_frame(rid, err));
        }
    };

    int stdout_pipe[2];
    if (::pipe(stdout_pipe) != 0) { fail(std::string("pipe: ") + strerror(errno)); return; }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);
        fail(std::string("fork: ") + strerror(errno));
        return;
    }
    if (pid == 0) {
        ::dup2(stdout_pipe[1], STDOUT_FILENO);
        ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);
        std::string pp = "PYTHONPATH=" + module_path;
        if (const char* old = std::getenv("PYTHONPATH"); old && *old) {
            pp += ":"; pp += old;
        }
        ::putenv(const_cast<char*>(strdup(pp.c_str())));
        const char* argv[] = {
            python_bin.c_str(),
            "-m", "dpp_runtime",
            "--role", rt->role.c_str(),
            "--model", rt->model.c_str(),
            "--port", "0",
            nullptr,
        };
        ::execvp(python_bin.c_str(), const_cast<char* const*>(argv));
        std::fprintf(stderr, "[dpp] execvp(%s) failed: %s\n",
                     python_bin.c_str(), strerror(errno));
        _exit(127);
    }
    rt->pid = pid;
    ::close(stdout_pipe[1]);
    set_nonblock(stdout_pipe[0], true);

    std::string buf;
    int port = -1;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (std::chrono::steady_clock::now() < deadline && port < 0 && !rt->stop.load()) {
        char tmp[256];
        ssize_t k = ::read(stdout_pipe[0], tmp, sizeof(tmp));
        if (k > 0) {
            buf.append(tmp, tmp + k);
            auto p = buf.find("DPP_LISTEN ");
            if (p != std::string::npos) {
                auto end = buf.find('\n', p);
                if (end != std::string::npos) {
                    std::string num = buf.substr(p + 11, end - p - 11);
                    try { port = std::stoi(num); } catch (...) { port = -1; }
                }
            }
        } else if (k == 0) {
            break;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            break;
        }
    }
    ::close(stdout_pipe[0]);
    if (port < 0) {
        fail("dpp_runtime did not announce listen port within 300s; last stdout: " + buf);
        ::kill(pid, SIGTERM);
        ::waitpid(pid, nullptr, 0);
        return;
    }

    int sk = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sk < 0) {
        fail(std::string("socket: ") + strerror(errno));
        ::kill(pid, SIGTERM); ::waitpid(pid, nullptr, 0);
        return;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(sk, (sockaddr*)&addr, sizeof(addr)) != 0) {
        fail(std::string("connect: ") + strerror(errno));
        ::close(sk);
        ::kill(pid, SIGTERM); ::waitpid(pid, nullptr, 0);
        return;
    }
    rt->sock_fd = sk;
    rt->port = (uint16_t)port;
    rt->reader = std::thread(&DppAdapter::reader_loop, this, rt);

    // Drain pending frames, then flip ready under the same lock so any
    // dispatch_actv racing with us either appends to pending (we'll catch it)
    // or sees ready=true and writes directly.
    {
        std::lock_guard<std::mutex> lk(rt->pending_mu);
        for (auto& f : rt->pending) rt->raw_send(f);
        rt->pending.clear();
        rt->ready.store(true);
    }
    std::cerr << "[dpp] runtime up: role=" << rt->role
              << " model=" << rt->model
              << " pid=" << pid << " port=" << port << "\n";
}

DppFrame DppAdapter::make_error_frame(uint16_t req_id, const std::string& msg) {
    DppFrame f;
    const uint32_t magic = 0x41435456u;
    const uint8_t  ver   = 1;
    const uint8_t  typ   = 4;   // TYPE_ERROR
    const uint8_t  dtype = 4;   // BYTES
    const uint8_t  rank  = 0;
    const uint8_t  flags = 0;
    const uint32_t plen  = (uint32_t)msg.size();
    f.bytes.resize(20 + 4 + plen);
    auto put_u16 = [&](size_t off, uint16_t v) {
        f.bytes[off]   = (uint8_t)(v >> 8);
        f.bytes[off+1] = (uint8_t)v;
    };
    auto put_u32 = [&](size_t off, uint32_t v) {
        f.bytes[off]   = (uint8_t)(v >> 24);
        f.bytes[off+1] = (uint8_t)(v >> 16);
        f.bytes[off+2] = (uint8_t)(v >> 8);
        f.bytes[off+3] = (uint8_t)v;
    };
    put_u32(0, magic);
    f.bytes[4] = ver;
    f.bytes[5] = typ;
    put_u16(6, req_id);
    put_u16(8, 0);            // stage
    put_u32(10, 0);           // tok_seq
    f.bytes[14] = dtype;
    f.bytes[15] = rank;
    f.bytes[16] = flags;
    f.bytes[17] = 0;
    f.bytes[18] = 0;
    f.bytes[19] = 0;
    put_u32(20, plen);
    std::memcpy(f.bytes.data() + 24, msg.data(), plen);
    return f;
}

void DppAdapter::reader_loop(DppRuntime* rt) {
    while (!rt->stop.load()) {
        uint8_t hdr[4];
        size_t off = 0;
        while (off < 4) {
            ssize_t k = ::recv(rt->sock_fd, hdr + off, 4 - off, 0);
            if (k <= 0) return; // EOF or error
            off += (size_t)k;
        }
        uint32_t n = ((uint32_t)hdr[0] << 24) | ((uint32_t)hdr[1] << 16) |
                     ((uint32_t)hdr[2] << 8)  |  (uint32_t)hdr[3];
        if (n == 0 || n > (32u << 20)) return; // sanity
        std::vector<uint8_t> body(n);
        off = 0;
        while (off < n) {
            ssize_t k = ::recv(rt->sock_fd, body.data() + off, n - off, 0);
            if (k <= 0) return;
            off += (size_t)k;
        }
        DppFrame f;
        f.bytes = std::move(body);

        // Sideband: a "PROG" magic prefix means the body (after the first
        // 4 bytes) is a UTF-8 JSON progress event.  Divert to text outbox
        // for ws.send_text — never sent as a binary ACTV frame.
        if (f.bytes.size() >= 4) {
            const uint8_t* d = f.bytes.data();
            uint32_t magic = ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
                             ((uint32_t)d[2] << 8)  |  (uint32_t)d[3];
            if (magic == 0x50524F47u) { // "PROG"
                std::string line(reinterpret_cast<const char*>(d + 4),
                                 f.bytes.size() - 4);
                std::lock_guard<std::mutex> lk(text_outbox_mu_);
                text_outbox_.push_back(std::move(line));
                continue;
            }
        }

        // Decide local-pipeline vs WS-out.  If this frame's stage maps to a
        // runtime we host (the next role lives on this same agent), forward
        // it to that runtime directly.  Otherwise, push to the outbox for the
        // main loop to ship over the WS.
        bool routed_local = false;
        if (f.bytes.size() >= 20) {
            const uint8_t* d = f.bytes.data();
            uint32_t magic = ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
                             ((uint32_t)d[2] << 8)  |  (uint32_t)d[3];
            if (magic == 0x41435456u) {
                uint16_t rid   = ((uint16_t)d[6] << 8) | (uint16_t)d[7];
                uint16_t stage = ((uint16_t)d[8] << 8) | (uint16_t)d[9];
                std::string next_key;
                {
                    std::lock_guard<std::mutex> lk(req_mu_);
                    auto it = req_to_runtime_.find(reqStageKey(rid, stage));
                    if (it != req_to_runtime_.end()) next_key = it->second;
                }
                if (!next_key.empty() && next_key != (rt->role + ":" + rt->model)) {
                    DppRuntime* next_rt = nullptr;
                    {
                        std::lock_guard<std::mutex> lk(runtimes_mu_);
                        next_rt = find_runtime_locked(next_key);
                    }
                    if (next_rt) {
                        next_rt->send_or_buffer(f);
                        routed_local = true;
                    }
                }
            }
        }
        if (!routed_local) {
            std::lock_guard<std::mutex> lk(outbox_mu_);
            outbox_.push_back(std::move(f));
        }
    }
}

// Minimalist JSON helpers (no third-party dep). We only need to extract a
// handful of fields from sdcpp_route messages — same single-quote-tolerant
// brittle-but-bounded parsers used elsewhere in this file's wire-handling.
namespace {

std::string json_extract_string(const std::string& src, const std::string& key) {
    auto needle = "\"" + key + "\"";
    auto p = src.find(needle);
    if (p == std::string::npos) return "";
    p = src.find(':', p);
    if (p == std::string::npos) return "";
    p = src.find('"', p);
    if (p == std::string::npos) return "";
    auto q = src.find('"', p + 1);
    while (q != std::string::npos && src[q - 1] == '\\') q = src.find('"', q + 1);
    if (q == std::string::npos) return "";
    return src.substr(p + 1, q - p - 1);
}

long long json_extract_int(const std::string& src, const std::string& key,
                           long long fallback) {
    auto needle = "\"" + key + "\"";
    auto p = src.find(needle);
    if (p == std::string::npos) return fallback;
    p = src.find(':', p);
    if (p == std::string::npos) return fallback;
    ++p;
    while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) ++p;
    char* end = nullptr;
    long long v = std::strtoll(src.c_str() + p, &end, 10);
    return end == src.c_str() + p ? fallback : v;
}

double json_extract_double(const std::string& src, const std::string& key,
                           double fallback) {
    auto needle = "\"" + key + "\"";
    auto p = src.find(needle);
    if (p == std::string::npos) return fallback;
    p = src.find(':', p);
    if (p == std::string::npos) return fallback;
    ++p;
    while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) ++p;
    char* end = nullptr;
    double v = std::strtod(src.c_str() + p, &end);
    return end == src.c_str() + p ? fallback : v;
}

// Base64 (RFC 4648, no line breaks) for embedding the PNG bytes in a JSON
// progress event the existing dpp dispatch surface can ingest unchanged.
std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t t = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8) | data[i + 2];
        out.push_back(tbl[(t >> 18) & 0x3F]);
        out.push_back(tbl[(t >> 12) & 0x3F]);
        out.push_back(tbl[(t >> 6) & 0x3F]);
        out.push_back(tbl[t & 0x3F]);
    }
    if (i < data.size()) {
        uint32_t t = (uint32_t)data[i] << 16;
        if (i + 1 < data.size()) t |= (uint32_t)data[i + 1] << 8;
        out.push_back(tbl[(t >> 18) & 0x3F]);
        out.push_back(tbl[(t >> 12) & 0x3F]);
        out.push_back(i + 1 < data.size() ? tbl[(t >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

// JSON string-field extractor that decodes escapes — used to lift the
// `out` path back out of an sdcpp_done line so the adapter can slurp the
// PNG.  The line is authored by us (the worker) so the escape set is the
// strict set we emit; no general-purpose Unicode unescape needed.
std::string json_extract_string_decoded(const std::string& src, const std::string& key) {
    auto needle = "\"" + key + "\":";
    auto p = src.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) ++p;
    if (p >= src.size() || src[p] != '"') return "";
    ++p;
    std::string out;
    while (p < src.size() && src[p] != '"') {
        if (src[p] == '\\' && p + 1 < src.size()) {
            char c = src[p + 1];
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
            out.push_back(src[p++]);
        }
    }
    return out;
}

std::string json_escape_str(const std::string& in) {
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

}  // namespace

// ─── SdcppDaemon ───────────────────────────────────────────────────────────
//
// Resident dist-sdcpp-worker child driven over stdin/stdout JSON lines.
// One per DppAdapter — model identity is internal to the worker, so the
// same daemon can serve N different sdcpp_route requests; sd.cpp reloads
// only when the model_path changes.
//
// We deliberately don't bother with TCP loopback here (the python runtime
// uses it for header symmetry but we own both ends of this wire) — pipes
// are simpler, faster, and the only consumer is the adapter itself.
struct SdcppDaemon {
    pid_t      pid       = -1;
    int        in_fd     = -1;   // write here: stdin of the worker
    int        out_fd    = -1;   // read here:  stdout of the worker
    std::atomic<bool> ready{false};
    std::atomic<bool> stop{false};
    std::thread       reader;
    std::mutex        write_mu;  // serialises stdin writes from N threads

    ~SdcppDaemon() {
        stop.store(true);
        if (in_fd >= 0) {
            // Best-effort polite quit; even if the worker is mid-gen the
            // close() on its stdin will unblock the dtor on shutdown.
            const char* q = "{\"cmd\":\"quit\"}\n";
            (void)::write(in_fd, q, std::strlen(q));
            ::close(in_fd);
            in_fd = -1;
        }
        if (out_fd >= 0) { ::close(out_fd); out_fd = -1; }
        if (reader.joinable()) reader.join();
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 50; ++i) {
                int s = 0;
                if (::waitpid(pid, &s, WNOHANG) == pid) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            pid = -1;
        }
    }

    bool write_line(const std::string& s) {
        std::lock_guard<std::mutex> lk(write_mu);
        if (in_fd < 0) return false;
        size_t off = 0;
        while (off < s.size()) {
            ssize_t k = ::write(in_fd, s.data() + off, s.size() - off);
            if (k <= 0) return false;
            off += (size_t)k;
        }
        return true;
    }
};

namespace {

// Fork dist-sdcpp-worker with --daemon, wiring two pipes (parent→child stdin,
// child→parent stdout).  Stderr is left untouched so sd.cpp's chatty banner
// reaches the agent log the same way python dpp_runtime logs do.
bool spawn_sdcpp_daemon(const std::string& worker_bin,
                        dist::SdcppDaemon& d,
                        std::string& err) {
    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (::pipe(in_pipe) != 0)  { err = std::string("pipe in: ")  + strerror(errno); return false; }
    if (::pipe(out_pipe) != 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        err = std::string("pipe out: ") + strerror(errno); return false;
    }
    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        err = std::string("fork: ") + strerror(errno); return false;
    }
    if (pid == 0) {
        // child: stdin from in_pipe[0], stdout to out_pipe[1]
        ::dup2(in_pipe[0],  STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[0]);  ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::execlp(worker_bin.c_str(), worker_bin.c_str(), "--daemon", (char*)nullptr);
        std::fprintf(stderr, "[sdcpp] execlp(%s --daemon) failed: %s\n",
                     worker_bin.c_str(), strerror(errno));
        _exit(127);
    }
    // parent
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    d.pid    = pid;
    d.in_fd  = in_pipe[1];
    d.out_fd = out_pipe[0];
    return true;
}

}  // namespace

bool DppAdapter::probe_local_sdcpp_caps(const std::string& worker_bin,
                                        std::string& backend_info,
                                        std::string& err) {
    int out_pipe[2];
    if (::pipe(out_pipe) != 0) {
        err = std::string("pipe: ") + strerror(errno);
        return false;
    }
    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        err = std::string("fork: ") + strerror(errno);
        return false;
    }
    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        // stderr to /dev/null — sd.cpp's banner is chatty and we only
        // want the `{"ok":true,...}` line from --probe on stdout.
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) { ::dup2(devnull, STDERR_FILENO); ::close(devnull); }
        ::execlp(worker_bin.c_str(), worker_bin.c_str(), "--probe", (char*)nullptr);
        _exit(127);
    }
    ::close(out_pipe[1]);
    std::string buf;
    char tmp[512];
    while (true) {
        ssize_t k = ::read(out_pipe[0], tmp, sizeof(tmp));
        if (k <= 0) break;
        buf.append(tmp, tmp + k);
        if (buf.size() > (64 << 10)) break;
    }
    ::close(out_pipe[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        err = "dist-sdcpp-worker --probe failed (exit=" +
              std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")";
        return false;
    }
    backend_info = buf;
    // Trim trailing newline for cleaner log lines.
    while (!backend_info.empty() &&
           (backend_info.back() == '\n' || backend_info.back() == '\r')) {
        backend_info.pop_back();
    }
    return true;
}

// Reader thread: parses newline-terminated JSON lines from the worker's
// stdout and routes them.  For sdcpp_done lines, slurp the PNG the worker
// wrote, base64-encode it inline, and emit the canonical
// {"kind":"sdcpp_done","req_id":N,"png_b64":"..."} text-outbox event the
// server's existing dispatch surface already understands.  Other kinds
// (sdcpp_progress, sdcpp_error, sdcpp_ready) pass straight through.
void DppAdapter::sdcpp_daemon_reader(DppAdapter* self, SdcppDaemon* d) {
    std::string buf;
    char chunk[8192];
    while (!d->stop.load()) {
        ssize_t n = ::read(d->out_fd, chunk, sizeof(chunk));
        if (n <= 0) break;
        buf.append(chunk, chunk + n);
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (line.empty()) continue;

            std::string kind = json_extract_string(line, "kind");
            if (kind == "sdcpp_ready") {
                d->ready.store(true);
                continue;
            }
            if (kind == "sdcpp_done") {
                // Slurp PNG and ship as inline base64.
                std::string out_path = json_extract_string_decoded(line, "out");
                long long req_id = json_extract_int(line, "req_id", 0);
                std::vector<uint8_t> png;
                if (!out_path.empty()) {
                    FILE* fp = std::fopen(out_path.c_str(), "rb");
                    if (fp) {
                        uint8_t bb[8192];
                        size_t k;
                        while ((k = std::fread(bb, 1, sizeof(bb), fp)) > 0) {
                            png.insert(png.end(), bb, bb + k);
                        }
                        std::fclose(fp);
                    }
                    ::unlink(out_path.c_str());
                }
                std::string msg;
                if (png.empty()) {
                    msg = std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
                          std::to_string(req_id) + ",\"error\":\"empty PNG\"}";
                } else {
                    msg.reserve(png.size() * 2);
                    msg += "{\"kind\":\"sdcpp_done\",\"req_id\":";
                    msg += std::to_string(req_id);
                    msg += ",\"png_b64\":\"";
                    msg += base64_encode(png);
                    msg += "\"}";
                }
                std::lock_guard<std::mutex> lk(self->text_outbox_mu_);
                self->text_outbox_.push_back(std::move(msg));
                continue;
            }
            // sdcpp_progress / sdcpp_error / anything else: forward verbatim.
            std::lock_guard<std::mutex> lk(self->text_outbox_mu_);
            self->text_outbox_.push_back(std::move(line));
        }
    }
}

bool DppAdapter::handle_sdcpp_route(const std::string& json_msg,
                                    const std::string& worker_bin,
                                    std::string& err_out) {
    // Required fields.
    std::string model_path = json_extract_string(json_msg, "model_path");
    std::string prompt     = json_extract_string(json_msg, "prompt");
    if (model_path.empty()) { err_out = "missing model_path"; return false; }
    if (prompt.empty())     { err_out = "missing prompt";     return false; }

    // Optional fields with sane defaults that match dpp_runtime.
    std::string neg_prompt = json_extract_string(json_msg, "negative_prompt");
    std::string sampler    = json_extract_string(json_msg, "sampler");
    if (sampler.empty()) sampler = "euler_a";
    int w     = (int)json_extract_int(json_msg, "width",  512);
    int h     = (int)json_extract_int(json_msg, "height", 512);
    int steps = (int)json_extract_int(json_msg, "steps",  20);
    float cfg = (float)json_extract_double(json_msg, "cfg", 7.0);
    long long seed = json_extract_int(json_msg, "seed", -1);
    uint16_t req_id = (uint16_t)json_extract_int(json_msg, "req_id", 0);

    // Lazy-spawn the shared worker daemon.
    {
        std::lock_guard<std::mutex> lk(sdcpp_daemon_mu_);
        if (!sdcpp_daemon_) {
            auto d = std::make_unique<SdcppDaemon>();
            std::string spawn_err;
            if (!spawn_sdcpp_daemon(worker_bin, *d, spawn_err)) {
                err_out = "spawn sdcpp daemon: " + spawn_err;
                std::lock_guard<std::mutex> elk(text_outbox_mu_);
                text_outbox_.push_back(
                    std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
                    std::to_string(req_id) + ",\"error\":\"" +
                    json_escape_str(err_out) + "\"}");
                return false;
            }
            d->reader = std::thread(&DppAdapter::sdcpp_daemon_reader, this, d.get());
            sdcpp_daemon_ = std::move(d);

            // Wait briefly for the ready handshake — model load is lazy
            // (paid on the first gen) so this only blocks on exec().
            for (int i = 0; i < 100; ++i) {
                if (sdcpp_daemon_->ready.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
    }

    // Pick a per-request output path; the worker writes here and the reader
    // thread slurps + unlinks it on sdcpp_done.
    char tmpl[] = "/tmp/sdcpp-out-XXXXXX.png";
    int fd = ::mkstemps(tmpl, 4);
    if (fd < 0) {
        err_out = std::string("mkstemps: ") + strerror(errno);
        std::lock_guard<std::mutex> lk(text_outbox_mu_);
        text_outbox_.push_back(
            std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
            std::to_string(req_id) + ",\"error\":\"" +
            json_escape_str(err_out) + "\"}");
        return false;
    }
    ::close(fd);
    std::string out_path = tmpl;

    // Build the JSON gen command.
    std::string cmd;
    cmd.reserve(model_path.size() + prompt.size() + 256);
    cmd += "{\"cmd\":\"gen\"";
    cmd += ",\"req_id\":";   cmd += std::to_string(req_id);
    cmd += ",\"model_path\":\""; cmd += json_escape_str(model_path); cmd += "\"";
    cmd += ",\"prompt\":\"";     cmd += json_escape_str(prompt);     cmd += "\"";
    if (!neg_prompt.empty()) {
        cmd += ",\"negative_prompt\":\""; cmd += json_escape_str(neg_prompt); cmd += "\"";
    }
    cmd += ",\"sampler\":\"";    cmd += json_escape_str(sampler);    cmd += "\"";
    cmd += ",\"out\":\"";        cmd += json_escape_str(out_path);   cmd += "\"";
    cmd += ",\"width\":";  cmd += std::to_string(w);
    cmd += ",\"height\":"; cmd += std::to_string(h);
    cmd += ",\"steps\":";  cmd += std::to_string(steps);
    char cfg_buf[32]; std::snprintf(cfg_buf, sizeof(cfg_buf), "%.4f", cfg);
    cmd += ",\"cfg\":";    cmd += cfg_buf;
    cmd += ",\"seed\":";   cmd += std::to_string(seed);
    cmd += "}\n";

    bool sent = false;
    {
        std::lock_guard<std::mutex> lk(sdcpp_daemon_mu_);
        if (sdcpp_daemon_) sent = sdcpp_daemon_->write_line(cmd);
    }
    if (!sent) {
        ::unlink(out_path.c_str());
        err_out = "sdcpp daemon write failed";
        std::lock_guard<std::mutex> lk(text_outbox_mu_);
        text_outbox_.push_back(
            std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
            std::to_string(req_id) + ",\"error\":\"" +
            json_escape_str(err_out) + "\"}");
        return false;
    }

    err_out.clear();
    return true;
}

// CF12-W3: per-role dispatch over the resident sdcpp daemon.
//
// The server hands us a JSON line with {role, req_id, model_path, ...}.  We
// pick the matching `sdr_<role>` command, copy through the parameters the
// worker needs (binary payloads are already base64 strings on the way in),
// and write the line into the daemon's stdin.  Responses arrive on the
// reader thread as `sdcpp_role_done` / `sdcpp_error` events which already
// flow through `text_outbox_` unchanged — no extra path needed here.
bool DppAdapter::handle_sdcpp_role_route(const std::string& json_msg,
                                         const std::string& worker_bin,
                                         std::string& err_out) {
    std::string role       = json_extract_string(json_msg, "role");
    std::string model_path = json_extract_string(json_msg, "model_path");
    uint16_t    req_id     = (uint16_t)json_extract_int(json_msg, "req_id", 0);

    if (role.empty()) { err_out = "missing role"; return false; }
    if (role != "te" && role != "unet" && role != "unet_blocks" &&
        role != "vae" && role != "caps") {
        err_out = "unknown role: " + role;
        return false;
    }
    if (role != "caps" && model_path.empty()) {
        err_out = "missing model_path";
        return false;
    }

    // Lazy-spawn the shared worker daemon (same handshake as
    // handle_sdcpp_route — model load is paid on the first per-role cmd).
    {
        std::lock_guard<std::mutex> lk(sdcpp_daemon_mu_);
        if (!sdcpp_daemon_) {
            auto d = std::make_unique<SdcppDaemon>();
            std::string spawn_err;
            if (!spawn_sdcpp_daemon(worker_bin, *d, spawn_err)) {
                err_out = "spawn sdcpp daemon: " + spawn_err;
                std::lock_guard<std::mutex> elk(text_outbox_mu_);
                text_outbox_.push_back(
                    std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
                    std::to_string(req_id) + ",\"error\":\"" +
                    json_escape_str(err_out) + "\"}");
                return false;
            }
            d->reader = std::thread(&DppAdapter::sdcpp_daemon_reader, this, d.get());
            sdcpp_daemon_ = std::move(d);
            for (int i = 0; i < 100; ++i) {
                if (sdcpp_daemon_->ready.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
    }

    // Build the per-role command line.
    std::string cmd;
    cmd.reserve(256 + model_path.size());

    auto append_common_head = [&](const char* sdr_cmd) {
        cmd += "{\"cmd\":\"";
        cmd += sdr_cmd;
        cmd += "\",\"req_id\":";
        cmd += std::to_string(req_id);
        if (!model_path.empty()) {
            cmd += ",\"model_path\":\"";
            cmd += json_escape_str(model_path);
            cmd += "\"";
        }
        long long threads = json_extract_int(json_msg, "threads", 0);
        if (threads > 0) {
            cmd += ",\"threads\":";
            cmd += std::to_string(threads);
        }
    };

    auto append_kv_str = [&](const char* key, const std::string& val) {
        cmd += ",\"";
        cmd += key;
        cmd += "\":\"";
        cmd += json_escape_str(val);
        cmd += "\"";
    };

    auto append_kv_int = [&](const char* key, long long val) {
        cmd += ",\"";
        cmd += key;
        cmd += "\":";
        cmd += std::to_string(val);
    };

    auto append_kv_dbl = [&](const char* key, double val) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", val);
        cmd += ",\"";
        cmd += key;
        cmd += "\":";
        cmd += buf;
    };

    if (role == "te") {
        append_common_head("sdr_encode_text");
        append_kv_str("prompt",          json_extract_string(json_msg, "prompt"));
        append_kv_str("negative_prompt", json_extract_string(json_msg, "negative_prompt"));
        append_kv_int("cfg_split",       json_extract_int(json_msg, "cfg_split", 0));
        append_kv_int("clip_skip",       json_extract_int(json_msg, "clip_skip", -1));
    } else if (role == "unet") {
        append_common_head("sdr_sample");
        append_kv_str("sdcd_b64",  json_extract_string(json_msg, "sdcd_b64"));
        append_kv_int("width",     json_extract_int(json_msg, "width",  512));
        append_kv_int("height",    json_extract_int(json_msg, "height", 512));
        append_kv_int("steps",     json_extract_int(json_msg, "steps",  20));
        append_kv_dbl("cfg",       json_extract_double(json_msg, "cfg", 7.0));
        append_kv_int("seed",      json_extract_int(json_msg, "seed", -1));
        std::string sampler = json_extract_string(json_msg, "sampler");
        if (sampler.empty()) sampler = "euler_a";
        append_kv_str("sampler",   sampler);
        std::string sched = json_extract_string(json_msg, "scheduler");
        if (!sched.empty()) append_kv_str("scheduler", sched);
    } else if (role == "unet_blocks") {
        append_common_head("sdr_sample_blocks");
        append_kv_str("sdcd_b64",     json_extract_string(json_msg, "sdcd_b64"));
        append_kv_str("upld_b64",     json_extract_string(json_msg, "upld_b64"));
        append_kv_int("block_lo",     json_extract_int(json_msg, "block_lo", 0));
        append_kv_int("block_hi",     json_extract_int(json_msg, "block_hi", 0));
        append_kv_int("block_total",  json_extract_int(json_msg, "block_total", 0));
        append_kv_int("steps",        json_extract_int(json_msg, "steps", 20));
        append_kv_dbl("cfg",          json_extract_double(json_msg, "cfg", 7.0));
        append_kv_int("seed",         json_extract_int(json_msg, "seed", -1));
        std::string sampler = json_extract_string(json_msg, "sampler");
        if (sampler.empty()) sampler = "euler_a";
        append_kv_str("sampler",      sampler);
        std::string sched = json_extract_string(json_msg, "scheduler");
        if (!sched.empty()) append_kv_str("scheduler", sched);
    } else if (role == "vae") {
        append_common_head("sdr_decode_latent");
        append_kv_str("sdt_b64",  json_extract_string(json_msg, "sdt_b64"));
    } else /* caps */ {
        append_common_head("sdr_caps");
    }
    cmd += "}\n";

    bool sent = false;
    {
        std::lock_guard<std::mutex> lk(sdcpp_daemon_mu_);
        if (sdcpp_daemon_) sent = sdcpp_daemon_->write_line(cmd);
    }
    if (!sent) {
        err_out = "sdcpp daemon write failed";
        std::lock_guard<std::mutex> lk(text_outbox_mu_);
        text_outbox_.push_back(
            std::string("{\"kind\":\"sdcpp_error\",\"req_id\":") +
            std::to_string(req_id) + ",\"error\":\"" +
            json_escape_str(err_out) + "\"}");
        return false;
    }

    err_out.clear();
    return true;
}

bool DppAdapter::probe_local_caps(const std::string& python_bin,
                                  const std::string& module_path,
                                  std::string& err) {
    // Quick: spawn `python -c "import dpp_runtime"` with PYTHONPATH set.
    pid_t pid = ::fork();
    if (pid < 0) {
        err = "fork: ";
        err += strerror(errno);
        return false;
    }
    if (pid == 0) {
        std::string pp = "PYTHONPATH=" + module_path;
        if (const char* old = std::getenv("PYTHONPATH"); old && *old) {
            pp += ":";
            pp += old;
        }
        ::putenv(const_cast<char*>(strdup(pp.c_str())));
        // Discard child output — we only care about exit status.
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        ::execlp(python_bin.c_str(), python_bin.c_str(),
                 "-c", "import dpp_runtime.wire", (char*)nullptr);
        _exit(127);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) {
        err = "waitpid failed";
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        err = "dpp_runtime not importable (exit=" +
              std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")";
        return false;
    }
    return true;
}

} // namespace dist
