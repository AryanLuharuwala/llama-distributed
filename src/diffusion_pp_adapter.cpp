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
    std::lock_guard<std::mutex> lk(runtimes_mu_);
    runtimes_.clear();
}

std::vector<DppFrame> DppAdapter::drain_outbox() {
    std::vector<DppFrame> out;
    std::lock_guard<std::mutex> lk(outbox_mu_);
    out.swap(outbox_);
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

    std::lock_guard<std::mutex> lk(runtimes_mu_);
    if (find_runtime_locked(key) != nullptr) return true;

    auto up = std::make_unique<DppRuntime>();
    up->role  = role;
    up->model = model;
    DppRuntime* raw = up.get();
    raw->launcher = std::thread(&DppAdapter::launch_python, this,
                                raw, python_bin, module_path);
    runtimes_.emplace(key, std::move(up));
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
