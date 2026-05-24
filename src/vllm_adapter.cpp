// vllm_adapter.cpp — IRuntimeAdapter implementation backed by a vLLM
// OpenAI-compatible HTTP server.
//
// Wire shape:
//   POST {base}/v1/completions
//     {
//       "model": "<served-model-name>",
//       "prompt": "<...>",
//       "max_tokens": N,
//       "temperature": T,
//       "top_p": P,
//       "top_k": K,
//       "stop": [...],
//       "stream": true
//     }
//   →  SSE stream of  data: { "choices": [ { "text": "<delta>",
//                                             "finish_reason": "stop"|"length"|null }] }
//      ... terminated by  data: [DONE]
//
// We don't pull in libcurl — the agent already does its own HTTP for
// comfy and the WS layer.  See comfy_adapter.cpp for the sibling
// implementation; we deliberately keep the socket primitives local
// rather than sharing across translation units, so each adapter stays
// self-contained.

#include "vllm_adapter.h"

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

// ─── String / URL helpers ────────────────────────────────────────────

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

// ─── Socket primitives ───────────────────────────────────────────────

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

// ─── JSON helpers ────────────────────────────────────────────────────
//
// We don't bring in a JSON library.  vLLM SSE deltas are flat objects
// with a handful of keys, so a positional scan is enough.

// Escape a string for embedding into a JSON string literal.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Decode a JSON string literal slice (between the outer quotes), unescaping
// the common escapes vLLM emits.  Unknown escapes pass through.
std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
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
                    // UTF-8 encode the BMP codepoint (no surrogate-pair handling
                    // — vLLM token deltas only escape ASCII controls in practice).
                    if (cp < 0x80) {
                        out.push_back((char)cp);
                    } else if (cp < 0x800) {
                        out.push_back((char)(0xC0 | (cp >> 6)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back((char)(0xE0 | (cp >> 12)));
                        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back((char)(0x80 | (cp & 0x3F)));
                    }
                } else {
                    out.push_back(c);
                }
                break;
            }
            default: out.push_back(c); break;
        }
    }
    return out;
}

// Find "key": "<string>" anywhere in `json` and return the (still-escaped)
// value slice.  Returns true if found.  Doesn't recurse — first match wins.
bool json_find_str(const std::string& json, const std::string& key, std::string* out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t s = pos + needle.size();
        while (s < json.size() && std::isspace((unsigned char)json[s])) ++s;
        if (s >= json.size()) return false;
        if (json[s] == 'n' && json.compare(s, 4, "null") == 0) {
            *out = "";
            return true;
        }
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

// Build the JSON request body for /v1/completions in streaming mode.
std::string build_completions_body(const std::string& model, const RuntimeRequest& req) {
    std::ostringstream o;
    o << "{";
    o << "\"model\":\"" << json_escape(model.empty() ? std::string("dist-default") : model) << "\",";
    o << "\"prompt\":\"" << json_escape(req.prompt) << "\",";
    o << "\"max_tokens\":" << req.max_tokens << ",";
    o << "\"temperature\":" << req.temperature << ",";
    o << "\"top_p\":" << req.top_p << ",";
    if (req.top_k > 0) o << "\"top_k\":" << req.top_k << ",";
    if (req.repetition_penalty != 1.0f)
        o << "\"repetition_penalty\":" << req.repetition_penalty << ",";
    if (req.seed >= 0) o << "\"seed\":" << req.seed << ",";
    if (!req.stop.empty()) {
        o << "\"stop\":[";
        for (size_t i = 0; i < req.stop.size(); ++i) {
            if (i) o << ",";
            o << "\"" << json_escape(req.stop[i]) << "\"";
        }
        o << "],";
    }
    if (!req.request_id.empty())
        o << "\"user\":\"" << json_escape(req.request_id) << "\",";
    o << "\"stream\":true";
    o << "}";
    return o.str();
}

// ─── HTTP response parser (non-streaming) ────────────────────────────

struct SimpleHttpResp {
    int         status = 0;
    std::string body;
};

SimpleHttpResp http_simple_request(const std::string& host, uint16_t port,
                                   const std::string& path_prefix,
                                   const std::string& method,
                                   const std::string& path,
                                   const std::string& bearer,
                                   const std::string& body,
                                   const std::string& content_type,
                                   int timeout_ms) {
    SimpleHttpResp r;
    int fd = open_socket(host, port, timeout_ms);
    if (fd < 0) return r;

    std::ostringstream req;
    req << method << " " << path_prefix << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Connection: close\r\n";
    if (!bearer.empty())
        req << "Authorization: Bearer " << bearer << "\r\n";
    if (!body.empty()) {
        req << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    }
    req << "\r\n";
    std::string head = req.str();
    if (write_all(fd, head.data(), head.size()) < 0) { DIST_CLOSESOCK(fd); return r; }
    if (!body.empty()) {
        if (write_all(fd, body.data(), body.size()) < 0) { DIST_CLOSESOCK(fd); return r; }
    }

    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    std::string acc;
    char buf[4096];
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        ssize_t n = read_some(fd, buf, sizeof(buf));
        if (n <= 0) break;
        acc.append(buf, (size_t)n);
        if (acc.size() > 8 * 1024 * 1024) break;
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

// ─── Streaming SSE consumer ──────────────────────────────────────────

// Parses an SSE chunk delivered by vLLM and forwards deltas to `cb`.
// State machine: we accumulate bytes in `buf`, peel off lines on \n,
// peel off events on the blank line separator, and route each event's
// `data: …` payload through the callback.  Returns the finish_reason
// observed from the final delta ("" if none) and sets *aborted=true if
// the callback returned false.
struct SseResult {
    std::string finish_reason;
    bool        aborted = false;
    bool        saw_done = false;
};

// Decode HTTP chunked-transfer framing in `chunk` into `out` (raw body
// bytes).  `state` is a tiny inline parser: 0=size line, 1=data, 2=CRLF
// after data.  Returns false if framing is malformed.
struct ChunkedState {
    int       phase = 0;       // 0: size, 1: data, 2: trailing CRLF
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
                // size_line is hex possibly followed by ';' extensions
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
        } else { // 2: consume trailing CRLF
            char c = data[i++];
            if (c == '\n') st.phase = 0;
        }
    }
    return true;
}

} // namespace

// ─── VllmAdapter ─────────────────────────────────────────────────────

VllmAdapter::VllmAdapter(VllmAdapterConfig cfg)
    : cfg_(std::move(cfg)) {
    parse_base_url();
}

VllmAdapter::~VllmAdapter() {
    close();
}

bool VllmAdapter::parse_base_url() {
    ParsedURL u = parse_url(cfg_.base_url);
    host_        = u.host;
    port_        = u.port;
    path_prefix_ = u.path;
    if (!path_prefix_.empty() && path_prefix_.back() == '/')
        path_prefix_.pop_back();
    return !host_.empty();
}

bool VllmAdapter::probe(int timeout_ms) {
    auto r = http_simple_request(host_, port_, path_prefix_,
                                 "GET", "/health",
                                 cfg_.api_key, "", "",
                                 timeout_ms);
    return r.status == 200;
}

std::string VllmAdapter::spawn_server(const std::string& model_path) {
#if defined(_WIN32)
    (void)model_path;
    return "spawn-mode not supported on Windows";
#else
    std::lock_guard<std::mutex> lk(spawn_mu_);
    if (child_pid_ > 0) return "";

    pid_t pid = fork();
    if (pid < 0) return std::string("fork failed: ") + std::strerror(errno);
    if (pid == 0) {
        // Child.  Build argv: python -m vllm.entrypoints.openai.api_server
        //                    --model <path> --host 0.0.0.0 --port <port>
        //                    [extra args]
        std::vector<std::string> args;
        args.push_back(cfg_.python_bin);
        args.push_back("-m");
        args.push_back("vllm.entrypoints.openai.api_server");
        args.push_back("--model");
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

        // Detach from controlling terminal so a Ctrl-C on the agent
        // doesn't kill vLLM mid-request — we manage termination via
        // SIGTERM in close().
        setsid();
        execvp(cfg_.python_bin.c_str(), argv.data());
        std::perror("execvp(vllm)");
        _exit(127);
    }
    child_pid_ = (int)pid;
    return wait_for_ready(cfg_.spawn_ready_timeout_s);
#endif
}

std::string VllmAdapter::wait_for_ready(int timeout_s) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (probe(500)) return "";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return "vllm /health did not return 200 within "
         + std::to_string(timeout_s) + "s";
}

std::string VllmAdapter::load_model(const std::string& model_path) {
    served_model_ = model_path;
    if (cfg_.spawn) {
        if (auto err = spawn_server(model_path); !err.empty()) return err;
    }
    if (!probe(cfg_.connect_timeout_ms))
        return "vllm not reachable at " + cfg_.base_url + " — start the OpenAI server or set DIST_VLLM_SPAWN=1";
    return "";
}

void VllmAdapter::close() {
    bool was = closed_.exchange(true);
    if (was) return;
#if !defined(_WIN32)
    std::lock_guard<std::mutex> lk(spawn_mu_);
    if (child_pid_ > 0) {
        ::kill(child_pid_, SIGTERM);
        int status = 0;
        // Best-effort wait up to 5s, then SIGKILL.
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

std::string VllmAdapter::generate(const RuntimeRequest& req, ChunkCallback cb) {
    if (host_.empty()) return "error: vllm base_url not parsed";
    if (cfg_.spawn && !probe(cfg_.connect_timeout_ms))
        return "error: vllm not ready (spawn-mode probe failed)";

    int fd = open_socket(host_, port_, cfg_.connect_timeout_ms);
    if (fd < 0) return "error: vllm connect failed";

    std::string body = build_completions_body(served_model_, req);

    std::ostringstream hreq;
    hreq << "POST " << path_prefix_ << "/v1/completions HTTP/1.1\r\n"
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
        return "error: vllm request write failed";
    }

    // Read response headers — disable any read timeout for long-running
    // streams beyond the initial header read.
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
    {
        char ch;
        while (headbuf.find("\r\n\r\n") == std::string::npos) {
            ssize_t r = read_some(fd, &ch, 1);
            if (r <= 0) { DIST_CLOSESOCK(fd); return "error: vllm header read failed"; }
            headbuf.push_back(ch);
            if (headbuf.size() > 16 * 1024) { DIST_CLOSESOCK(fd); return "error: vllm header bomb"; }
        }
    }
    int status = 0;
    auto eol = headbuf.find("\r\n");
    if (eol != std::string::npos) {
        auto sp = headbuf.find(' ');
        if (sp != std::string::npos) {
            try { status = std::stoi(headbuf.substr(sp + 1, 3)); }
            catch (...) { status = 0; }
        }
    }
    if (status != 200) {
        // Read body for the error message so callers can log it.
        char buf[2048]; ssize_t n = read_some(fd, buf, sizeof(buf));
        std::string err(buf, n > 0 ? (size_t)n : 0);
        DIST_CLOSESOCK(fd);
        return "error: vllm http " + std::to_string(status) + ": " + err;
    }

    // Look for Transfer-Encoding: chunked — vLLM uses chunked for SSE.
    bool chunked = false;
    {
        std::string lower_head = to_lower(headbuf);
        chunked = lower_head.find("transfer-encoding: chunked") != std::string::npos;
    }

    // Clear read timeout for the stream itself; let the caller cancel
    // via the callback when they need to abort.
#if defined(_WIN32)
    DWORD tv0 = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv0, sizeof(tv0));
#else
    struct timeval tv0{};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv0, sizeof(tv0));
#endif

    // Stream loop.
    std::string raw;         // unchunked body buffer
    std::string sse_event;   // current SSE event accumulator (data:.. lines)
    ChunkedState cstate;
    SseResult sse;

    auto process_event = [&](const std::string& ev) -> bool {
        // Each event is one or more `data: ...` lines.  Concatenate the
        // payloads (per SSE spec) and parse as JSON.
        std::string payload;
        size_t i = 0;
        while (i < ev.size()) {
            size_t nl = ev.find('\n', i);
            std::string line = ev.substr(i, nl == std::string::npos ? ev.size() - i : nl - i);
            i = (nl == std::string::npos) ? ev.size() : nl + 1;
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data:", 0) != 0) continue;
            std::string data_part = line.substr(5);
            if (!data_part.empty() && data_part.front() == ' ') data_part.erase(0, 1);
            if (!payload.empty()) payload.push_back('\n');
            payload.append(data_part);
        }
        if (payload.empty()) return true;
        if (payload == "[DONE]") { sse.saw_done = true; return true; }

        // Extract text + finish_reason from the first `choices[0]` entry.
        std::string text_esc;
        bool has_text = json_find_str(payload, "text", &text_esc);
        std::string fr;
        json_find_str(payload, "finish_reason", &fr);
        if (!fr.empty()) sse.finish_reason = fr;

        if (has_text && !text_esc.empty()) {
            RuntimeChunk ch;
            ch.text     = json_unescape(text_esc);
            ch.token_id = -1;
            ch.final    = false;
            if (!cb(ch)) {
                sse.aborted = true;
                return false;
            }
        }
        return true;
    };

    auto feed_raw = [&](const char* data, size_t n) -> bool {
        size_t off = 0;
        while (off < n) {
            // Find the SSE event terminator (\n\n or \r\n\r\n).
            // We scan forward in `sse_event + new bytes` for "\n\n".
            // Append byte-by-byte so we can spot the terminator.
            char c = data[off++];
            sse_event.push_back(c);
            size_t sz = sse_event.size();
            bool boundary = false;
            if (sz >= 2 && sse_event[sz - 1] == '\n' && sse_event[sz - 2] == '\n') {
                boundary = true;
            } else if (sz >= 4 &&
                       sse_event[sz - 1] == '\n' && sse_event[sz - 2] == '\r' &&
                       sse_event[sz - 3] == '\n' && sse_event[sz - 4] == '\r') {
                boundary = true;
            }
            if (boundary) {
                std::string ev = sse_event;
                sse_event.clear();
                if (!process_event(ev)) return false;
            }
            if (sse_event.size() > 1 * 1024 * 1024) return false; // event bomb
        }
        return true;
    };

    char buf[4096];
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
        if (sse.aborted || sse.saw_done) break;
    }
    DIST_CLOSESOCK(fd);

    // Flush any tail event without a trailing blank line.
    if (!sse_event.empty()) (void)process_event(sse_event);

    // Final chunk with the determined finish_reason.
    RuntimeChunk last;
    last.text          = "";
    last.token_id      = -1;
    last.final         = true;
    last.finish_reason = sse.aborted ? std::string("cancelled")
                       : (sse.finish_reason.empty() ? std::string("stop") : sse.finish_reason);
    cb(last);

    return sse.aborted ? std::string("cancelled") : last.finish_reason;
}

// ─── Env helpers ─────────────────────────────────────────────────────

VllmAdapterConfig vllm_config_from_env() {
    VllmAdapterConfig c;
    if (const char* v = std::getenv("DIST_VLLM_URL"); v && *v) c.base_url = v;
    if (const char* v = std::getenv("DIST_VLLM_API_KEY"); v && *v) c.api_key = v;
    if (const char* v = std::getenv("DIST_VLLM_PYTHON"); v && *v) c.python_bin = v;
    if (const char* v = std::getenv("DIST_VLLM_EXTRA_ARGS"); v && *v) c.extra_args = v;
    c.spawn = getenv_truthy("DIST_VLLM_SPAWN");
    if (const char* v = std::getenv("DIST_VLLM_SPAWN_TIMEOUT"); v && *v) {
        try { c.spawn_ready_timeout_s = std::stoi(v); } catch (...) {}
    }
    if (const char* v = std::getenv("DIST_VLLM_CONNECT_TIMEOUT_MS"); v && *v) {
        try { c.connect_timeout_ms = std::stoi(v); } catch (...) {}
    }
    return c;
}

} // namespace dist
