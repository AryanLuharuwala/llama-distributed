// sglang_adapter.cpp — IRuntimeAdapter implementation backed by an
// SGLang server (https://github.com/sgl-project/sglang).
//
// Why a separate adapter even though SGLang also speaks OpenAI?
// Because we need `meta_info.cached_tokens` to feed back into the
// control plane for prefix-cache-aware routing.  The OpenAI-compat
// endpoints strip that signal; the native /generate endpoint
// surfaces it directly, so we use that one.
//
// HTTP client is hand-rolled for the same reason as vllm_adapter.cpp.

#include "sglang_adapter.h"

#include "platform_compat.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  using socklen_t_local = int;
  #define DIST_CLOSESOCK closesocket
#else
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <signal.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
  using socklen_t_local = socklen_t;
  #define DIST_CLOSESOCK ::close
#endif

namespace dist {

namespace {

std::string to_lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool getenv_truthy(const char* k) {
    const char* v = std::getenv(k);
    if (!v || !*v) return false;
    std::string s = to_lower(v);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

struct ParsedURL {
    std::string scheme = "http";
    std::string host;
    uint16_t    port   = 80;
    std::string path;
};

ParsedURL parse_url(const std::string& url) {
    ParsedURL p;
    std::string s = url;
    auto sep = s.find("://");
    if (sep != std::string::npos) {
        p.scheme = to_lower(s.substr(0, sep));
        s = s.substr(sep + 3);
    }
    auto slash = s.find('/');
    std::string hp = (slash == std::string::npos) ? s : s.substr(0, slash);
    p.path = (slash == std::string::npos) ? "" : s.substr(slash);
    auto colon = hp.find(':');
    if (colon == std::string::npos) {
        p.host = hp;
        p.port = (p.scheme == "https") ? 443 : 80;
    } else {
        p.host = hp.substr(0, colon);
        try { p.port = (uint16_t)std::stoi(hp.substr(colon + 1)); }
        catch (...) { p.port = (p.scheme == "https") ? 443 : 80; }
    }
    return p;
}

int open_socket(const std::string& host, uint16_t port, int connect_ms) {
#if defined(_WIN32)
    WSADATA wsa{}; static int once = (WSAStartup(MAKEWORD(2,2), &wsa), 0);
    (void)once;
#endif
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char portbuf[16]; std::snprintf(portbuf, sizeof(portbuf), "%u", port);
    if (getaddrinfo(host.c_str(), portbuf, &hints, &res) != 0 || !res) return -1;

    int fd = -1;
    for (auto* p = res; p; p = p->ai_next) {
        fd = (int)::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
#if defined(_WIN32)
        DWORD tv = (DWORD)connect_ms;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
        struct timeval tv{};
        tv.tv_sec  = connect_ms / 1000;
        tv.tv_usec = (connect_ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        if (::connect(fd, p->ai_addr, (socklen_t_local)p->ai_addrlen) == 0) break;
        DIST_CLOSESOCK(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

ssize_t write_all(int fd, const char* p, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, p + sent, n - sent, 0);
        if (r <= 0) return -1;
        sent += (size_t)r;
    }
    return (ssize_t)sent;
}

ssize_t read_some(int fd, char* buf, size_t n) {
    return ::recv(fd, buf, n, 0);
}

std::string json_escape(const std::string& s) {
    std::string out; out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c);
                    out += b;
                } else out += c;
        }
    }
    return out;
}

std::string json_unescape(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c != '\\' || i + 1 >= s.size()) { out.push_back(c); continue; }
        char n = s[i + 1];
        switch (n) {
            case '"':  out.push_back('"');  i++; break;
            case '\\': out.push_back('\\'); i++; break;
            case '/':  out.push_back('/');  i++; break;
            case 'n':  out.push_back('\n'); i++; break;
            case 'r':  out.push_back('\r'); i++; break;
            case 't':  out.push_back('\t'); i++; break;
            case 'b':  out.push_back('\b'); i++; break;
            case 'f':  out.push_back('\f'); i++; break;
            case 'u': {
                if (i + 5 < s.size()) {
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i + 2 + k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                    }
                    i += 5;
                    if (cp < 0x80) out.push_back((char)cp);
                    else if (cp < 0x800) {
                        out.push_back((char)(0xC0 | (cp >> 6)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back((char)(0xE0 | (cp >> 12)));
                        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    }
                } else out.push_back(c);
                break;
            }
            default: out.push_back(c); break;
        }
    }
    return out;
}

bool json_find_str(const std::string& json, const std::string& key, std::string* out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t s = pos + needle.size();
        while (s < json.size() && std::isspace((unsigned char)json[s])) ++s;
        if (s >= json.size()) return false;
        if (json[s] == 'n' && json.compare(s, 4, "null") == 0) { *out = ""; return true; }
        if (json[s] != '"') { pos = s; continue; }
        size_t e = s + 1;
        while (e < json.size() && json[e] != '"') {
            if (json[e] == '\\' && e + 1 < json.size()) e += 2;
            else ++e;
        }
        *out = json.substr(s + 1, e - (s + 1));
        return true;
    }
    return false;
}

bool json_find_int(const std::string& json, const std::string& key, int64_t* out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    size_t s = pos + needle.size();
    while (s < json.size() && std::isspace((unsigned char)json[s])) ++s;
    size_t e = s;
    while (e < json.size() && (std::isdigit((unsigned char)json[e]) || json[e] == '-')) ++e;
    if (e == s) return false;
    try { *out = std::stoll(json.substr(s, e - s)); }
    catch (...) { return false; }
    return true;
}

// Build the JSON body for SGLang /generate.
std::string build_generate_body(const RuntimeRequest& req) {
    std::ostringstream o;
    o << "{";
    o << "\"text\":\"" << json_escape(req.prompt) << "\",";
    o << "\"sampling_params\":{";
    o << "\"max_new_tokens\":" << req.max_tokens;
    o << ",\"temperature\":" << req.temperature;
    o << ",\"top_p\":" << req.top_p;
    if (req.top_k > 0) o << ",\"top_k\":" << req.top_k;
    if (req.repetition_penalty != 1.0f)
        o << ",\"repetition_penalty\":" << req.repetition_penalty;
    if (!req.stop.empty()) {
        o << ",\"stop\":[";
        for (size_t i = 0; i < req.stop.size(); ++i) {
            if (i) o << ",";
            o << "\"" << json_escape(req.stop[i]) << "\"";
        }
        o << "]";
    }
    o << "}";
    if (req.seed >= 0) o << ",\"return_logprob\":false,\"seed\":" << req.seed;
    o << ",\"stream\":true";
    o << "}";
    return o.str();
}

struct ChunkedState {
    int       phase = 0;
    size_t    remaining = 0;
    std::string size_line;
    bool      done = false;
};

bool feed_chunked(ChunkedState& st, const char* data, size_t n,
                  std::string& body) {
    size_t i = 0;
    while (i < n && !st.done) {
        if (st.phase == 0) {
            char c = data[i++];
            if (c == '\n') {
                std::string raw = st.size_line;
                st.size_line.clear();
                while (!raw.empty() && (raw.back() == '\r' || std::isspace((unsigned char)raw.back())))
                    raw.pop_back();
                auto semi = raw.find(';');
                if (semi != std::string::npos) raw = raw.substr(0, semi);
                if (raw.empty()) continue;
                try { st.remaining = (size_t)std::stoull(raw, nullptr, 16); }
                catch (...) { return false; }
                if (st.remaining == 0) { st.done = true; return true; }
                st.phase = 1;
            } else {
                st.size_line.push_back(c);
                if (st.size_line.size() > 32) return false;
            }
        } else if (st.phase == 1) {
            size_t take = std::min(st.remaining, n - i);
            body.append(data + i, take);
            i += take;
            st.remaining -= take;
            if (st.remaining == 0) st.phase = 2;
        } else {
            char c = data[i++];
            if (c == '\n') st.phase = 0;
        }
    }
    return true;
}

struct SimpleHttpResp {
    int status = 0;
    std::string body;
};

SimpleHttpResp http_simple_request(const std::string& host, uint16_t port,
                                   const std::string& path_prefix,
                                   const std::string& method,
                                   const std::string& path,
                                   const std::string& bearer,
                                   int timeout_ms) {
    SimpleHttpResp r;
    int fd = open_socket(host, port, timeout_ms);
    if (fd < 0) return r;

    std::ostringstream req;
    req << method << " " << path_prefix << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Connection: close\r\n";
    if (!bearer.empty()) req << "Authorization: Bearer " << bearer << "\r\n";
    req << "\r\n";
    std::string head = req.str();
    if (write_all(fd, head.data(), head.size()) < 0) { DIST_CLOSESOCK(fd); return r; }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::string acc;
    char buf[4096];
    while (true) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        ssize_t n = read_some(fd, buf, sizeof(buf));
        if (n <= 0) break;
        acc.append(buf, (size_t)n);
        if (acc.size() > 4 * 1024 * 1024) break;
    }
    DIST_CLOSESOCK(fd);

    auto pos = acc.find("\r\n\r\n");
    if (pos == std::string::npos) return r;
    std::string status_line = acc.substr(0, acc.find("\r\n"));
    auto sp = status_line.find(' ');
    if (sp != std::string::npos) {
        try { r.status = std::stoi(status_line.substr(sp + 1, 3)); }
        catch (...) { r.status = 0; }
    }
    r.body = acc.substr(pos + 4);
    return r;
}

} // namespace

// ─── SglangAdapter ───────────────────────────────────────────────────

SglangAdapter::SglangAdapter(SglangAdapterConfig cfg)
    : cfg_(std::move(cfg)) {
    parse_base_url();
}

SglangAdapter::~SglangAdapter() { close(); }

bool SglangAdapter::parse_base_url() {
    ParsedURL u = parse_url(cfg_.base_url);
    host_        = u.host;
    port_        = u.port;
    path_prefix_ = u.path;
    if (!path_prefix_.empty() && path_prefix_.back() == '/')
        path_prefix_.pop_back();
    return !host_.empty();
}

bool SglangAdapter::probe(int timeout_ms) {
    // SGLang exposes /health (HTTP 200 when ready) and /get_model_info.
    auto r = http_simple_request(host_, port_, path_prefix_,
                                 "GET", "/health",
                                 cfg_.api_key, timeout_ms);
    return r.status == 200;
}

std::string SglangAdapter::spawn_server(const std::string& model_path) {
#if defined(_WIN32)
    (void)model_path;
    return "spawn-mode not supported on Windows";
#else
    std::lock_guard<std::mutex> lk(spawn_mu_);
    if (child_pid_ > 0) return "";

    pid_t pid = fork();
    if (pid < 0) return std::string("fork failed: ") + std::strerror(errno);
    if (pid == 0) {
        std::vector<std::string> args;
        args.push_back(cfg_.python_bin);
        args.push_back("-m");
        args.push_back("sglang.launch_server");
        args.push_back("--model-path");
        args.push_back(model_path);
        args.push_back("--host");
        args.push_back("127.0.0.1");
        args.push_back("--port");
        args.push_back(std::to_string(port_));
        if (!cfg_.extra_args.empty()) {
            std::istringstream is(cfg_.extra_args);
            std::string tok;
            while (is >> tok) args.push_back(tok);
        }
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        setsid();
        execvp(cfg_.python_bin.c_str(), argv.data());
        std::perror("execvp(sglang)");
        _exit(127);
    }
    child_pid_ = (int)pid;
    return wait_for_ready(cfg_.spawn_ready_timeout_s);
#endif
}

std::string SglangAdapter::wait_for_ready(int timeout_s) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (probe(500)) return "";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return "sglang /health did not return 200 within " + std::to_string(timeout_s) + "s";
}

std::string SglangAdapter::load_model(const std::string& model_path) {
    served_model_ = model_path;
    if (cfg_.spawn) {
        if (auto err = spawn_server(model_path); !err.empty()) return err;
    }
    if (!probe(cfg_.connect_timeout_ms))
        return "sglang not reachable at " + cfg_.base_url + " — start the server or set DIST_SGLANG_SPAWN=1";
    return "";
}

void SglangAdapter::close() {
    bool was = closed_.exchange(true);
    if (was) return;
#if !defined(_WIN32)
    std::lock_guard<std::mutex> lk(spawn_mu_);
    if (child_pid_ > 0) {
        ::kill(child_pid_, SIGTERM);
        int status = 0;
        for (int i = 0; i < 50; ++i) {
            pid_t r = ::waitpid(child_pid_, &status, WNOHANG);
            if (r == child_pid_) { child_pid_ = -1; return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ::kill(child_pid_, SIGKILL);
        ::waitpid(child_pid_, &status, 0);
        child_pid_ = -1;
    }
#endif
}

std::string SglangAdapter::generate(const RuntimeRequest& req, ChunkCallback cb) {
    if (host_.empty()) return "error: sglang base_url not parsed";

    int fd = open_socket(host_, port_, cfg_.connect_timeout_ms);
    if (fd < 0) return "error: sglang connect failed";

    std::string body = build_generate_body(req);

    std::ostringstream hreq;
    hreq << "POST " << path_prefix_ << "/generate HTTP/1.1\r\n"
         << "Host: " << host_ << ":" << port_ << "\r\n"
         << "Connection: close\r\n"
         << "Accept: text/event-stream\r\n"
         << "Content-Type: application/json\r\n"
         << "Content-Length: " << body.size() << "\r\n";
    if (!cfg_.api_key.empty())
        hreq << "Authorization: Bearer " << cfg_.api_key << "\r\n";
    hreq << "\r\n";
    std::string hs = hreq.str();
    if (write_all(fd, hs.data(), hs.size()) < 0 ||
        write_all(fd, body.data(), body.size()) < 0) {
        DIST_CLOSESOCK(fd);
        return "error: sglang request write failed";
    }

    // Read response headers with the connect timeout.
    {
#if defined(_WIN32)
        DWORD tv = (DWORD)cfg_.connect_timeout_ms;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
        struct timeval tv{};
        tv.tv_sec  = cfg_.connect_timeout_ms / 1000;
        tv.tv_usec = (cfg_.connect_timeout_ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    }
    std::string headbuf;
    char ch;
    while (headbuf.find("\r\n\r\n") == std::string::npos) {
        ssize_t r = read_some(fd, &ch, 1);
        if (r <= 0) { DIST_CLOSESOCK(fd); return "error: sglang header read failed"; }
        headbuf.push_back(ch);
        if (headbuf.size() > 16 * 1024) { DIST_CLOSESOCK(fd); return "error: sglang header bomb"; }
    }
    int status = 0;
    auto sp = headbuf.find(' ');
    if (sp != std::string::npos) {
        try { status = std::stoi(headbuf.substr(sp + 1, 3)); }
        catch (...) { status = 0; }
    }
    if (status != 200) {
        char buf[2048]; ssize_t n = read_some(fd, buf, sizeof(buf));
        std::string err(buf, n > 0 ? (size_t)n : 0);
        DIST_CLOSESOCK(fd);
        return "error: sglang http " + std::to_string(status) + ": " + err;
    }

    bool chunked = to_lower(headbuf).find("transfer-encoding: chunked") != std::string::npos;

    // No read timeout on the body — vLLM/SGLang streams can run long.
#if defined(_WIN32)
    DWORD tv0 = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv0, sizeof(tv0));
#else
    struct timeval tv0{};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv0, sizeof(tv0));
#endif

    ChunkedState cstate;
    std::string sse_event;
    std::string accumulated_text;   // SGLang sends cumulative text; we diff
    std::string finish_reason;
    bool   saw_done = false;
    bool   aborted  = false;

    auto process_event = [&](const std::string& ev) -> bool {
        std::string payload;
        size_t i = 0;
        while (i < ev.size()) {
            size_t nl = ev.find('\n', i);
            std::string line = ev.substr(i, nl == std::string::npos ? ev.size() - i : nl - i);
            i = (nl == std::string::npos) ? ev.size() : nl + 1;
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data:", 0) != 0) continue;
            std::string d = line.substr(5);
            if (!d.empty() && d.front() == ' ') d.erase(0, 1);
            if (!payload.empty()) payload.push_back('\n');
            payload.append(d);
        }
        if (payload.empty()) return true;
        if (payload == "[DONE]") { saw_done = true; return true; }

        // SGLang /generate streams:
        //   { "text": "<cumulative output>", "meta_info": { ... } }
        std::string text_esc;
        if (json_find_str(payload, "text", &text_esc)) {
            std::string decoded = json_unescape(text_esc);
            // Diff against what we've already emitted so the caller sees deltas.
            if (decoded.size() > accumulated_text.size() &&
                decoded.compare(0, accumulated_text.size(), accumulated_text) == 0) {
                std::string delta = decoded.substr(accumulated_text.size());
                accumulated_text  = decoded;
                if (!delta.empty()) {
                    RuntimeChunk c;
                    c.text     = delta;
                    c.token_id = -1;
                    c.final    = false;
                    if (!cb(c)) { aborted = true; return false; }
                }
            } else if (decoded != accumulated_text) {
                // SGLang re-emitted from scratch (rare) — just send what we
                // got and reset the baseline.
                accumulated_text = decoded;
                RuntimeChunk c;
                c.text = decoded; c.token_id = -1; c.final = false;
                if (!cb(c)) { aborted = true; return false; }
            }
        }

        // meta_info has cached_tokens, prompt_tokens, completion_tokens,
        // and finish_reason.{type}.  We read flat — these are unique
        // names so first match wins even though they live inside
        // meta_info specifically.
        int64_t v = 0;
        if (json_find_int(payload, "cached_tokens",     &v)) last_cached_tokens_.store(v);
        if (json_find_int(payload, "prompt_tokens",     &v)) last_prompt_tokens_.store(v);
        if (json_find_int(payload, "completion_tokens", &v)) last_completion_tokens_.store(v);

        std::string fr;
        if (json_find_str(payload, "type", &fr) && !fr.empty()) {
            // meta_info.finish_reason.type — only present on terminal frames.
            finish_reason = fr;
        }
        return true;
    };

    auto feed_raw = [&](const char* data, size_t n) -> bool {
        for (size_t off = 0; off < n; ++off) {
            sse_event.push_back(data[off]);
            size_t sz = sse_event.size();
            bool boundary = false;
            if (sz >= 2 && sse_event[sz - 1] == '\n' && sse_event[sz - 2] == '\n') boundary = true;
            else if (sz >= 4 &&
                     sse_event[sz - 1] == '\n' && sse_event[sz - 2] == '\r' &&
                     sse_event[sz - 3] == '\n' && sse_event[sz - 4] == '\r') boundary = true;
            if (boundary) {
                std::string ev = sse_event;
                sse_event.clear();
                if (!process_event(ev)) return false;
            }
            if (sse_event.size() > 4 * 1024 * 1024) return false;
        }
        return true;
    };

    char buf[4096];
    last_cached_tokens_.store(0);
    last_prompt_tokens_.store(0);
    last_completion_tokens_.store(0);

    while (true) {
        ssize_t n = read_some(fd, buf, sizeof(buf));
        if (n <= 0) break;
        if (chunked) {
            std::string decoded;
            if (!feed_chunked(cstate, buf, (size_t)n, decoded)) break;
            if (!decoded.empty()) {
                if (!feed_raw(decoded.data(), decoded.size())) break;
            }
            if (cstate.done) break;
        } else {
            if (!feed_raw(buf, (size_t)n)) break;
        }
        if (aborted || saw_done) break;
    }
    DIST_CLOSESOCK(fd);

    if (!sse_event.empty()) (void)process_event(sse_event);

    RuntimeChunk last;
    last.text          = "";
    last.token_id      = -1;
    last.final         = true;
    last.finish_reason = aborted ? std::string("cancelled")
                       : (finish_reason.empty() ? std::string("stop") : finish_reason);
    cb(last);
    return aborted ? std::string("cancelled") : last.finish_reason;
}

SglangAdapterConfig sglang_config_from_env() {
    SglangAdapterConfig c;
    if (const char* v = std::getenv("DIST_SGLANG_URL"); v && *v) c.base_url = v;
    if (const char* v = std::getenv("DIST_SGLANG_API_KEY"); v && *v) c.api_key = v;
    if (const char* v = std::getenv("DIST_SGLANG_PYTHON"); v && *v) c.python_bin = v;
    if (const char* v = std::getenv("DIST_SGLANG_EXTRA_ARGS"); v && *v) c.extra_args = v;
    c.spawn = getenv_truthy("DIST_SGLANG_SPAWN");
    if (const char* v = std::getenv("DIST_SGLANG_SPAWN_TIMEOUT"); v && *v) {
        try { c.spawn_ready_timeout_s = std::stoi(v); } catch (...) {}
    }
    if (const char* v = std::getenv("DIST_SGLANG_CONNECT_TIMEOUT_MS"); v && *v) {
        try { c.connect_timeout_ms = std::stoi(v); } catch (...) {}
    }
    return c;
}

} // namespace dist
