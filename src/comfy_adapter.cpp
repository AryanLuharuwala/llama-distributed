// comfy_adapter.cpp — minimal HTTP client + ComfyUI workflow runner.
//
// Why hand-rolled HTTP?  The agent already builds without libcurl; pulling
// in another runtime dep just to talk to a localhost endpoint is overkill.
// ComfyUI exposes plain HTTP on a single port, so a ~200-line client is
// enough.  We support: GET, POST (with body), connect+read timeouts,
// HTTP/1.1 with `Connection: close`, Content-Length and chunked transfer.
//
// Workflow loop:
//   1. POST /prompt { "prompt": <graph>, "client_id": "<uuid>" }
//      → returns { "prompt_id": "..." }
//   2. Poll GET /history/<prompt_id> every 500ms until the entry shows up
//      AND outputs are populated (or we time out).
//   3. For each `outputs[node].images[*]` (or `.gifs`, `.videos`) entry,
//      GET /view?filename=...&subfolder=...&type=output and forward.

#include "comfy_adapter.h"

#include "platform_compat.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <thread>

#if defined(_WIN32)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socklen_t_local = int;
  #define DIST_CLOSESOCK closesocket
  #define DIST_SOCK_ERRNO WSAGetLastError()
#else
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/types.h>
  #include <unistd.h>
  using socklen_t_local = socklen_t;
  #define DIST_CLOSESOCK ::close
  #define DIST_SOCK_ERRNO errno
#endif

namespace dist {

namespace {

bool ci_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b - a);
}

// Parse "[scheme://]host[:port][/path]" into pieces.  Defaults: scheme=http,
// port=80, path="".  Used by ComfyClient to decode DIST_COMFY_URL.
struct ParsedURL {
    std::string scheme = "http";
    std::string host;
    uint16_t    port = 80;
    std::string path;
};
ParsedURL parse_url(const std::string& url) {
    ParsedURL p;
    std::string s = url;
    auto pos = s.find("://");
    if (pos != std::string::npos) {
        p.scheme = lower(s.substr(0, pos));
        s = s.substr(pos + 3);
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
        catch (...) { p.port = 80; }
    }
    return p;
}

// Read exactly `n` bytes from `fd` with a single overall deadline.
// Returns the number of bytes actually read (0 on timeout/EOF).
ssize_t read_n_deadline(int fd, char* buf, size_t n,
                        std::chrono::steady_clock::time_point deadline) {
    size_t got = 0;
    while (got < n) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return (ssize_t)got;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (ms <= 0) ms = 1;
#if defined(_WIN32)
        DWORD tv = (DWORD)ms;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
        struct timeval tv{};
        tv.tv_sec  = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        ssize_t r = ::recv(fd, buf + got, n - got, 0);
        if (r < 0) return (ssize_t)got;
        if (r == 0) return (ssize_t)got;
        got += (size_t)r;
    }
    return (ssize_t)got;
}

// Read until the response stream closes or deadline elapses.
std::string read_to_close(int fd, std::chrono::steady_clock::time_point deadline) {
    std::string out;
    char buf[4096];
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (ms <= 0) ms = 1;
#if defined(_WIN32)
        DWORD tv = (DWORD)ms;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
        struct timeval tv{};
        tv.tv_sec  = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
        if (r <= 0) break;
        out.append(buf, (size_t)r);
    }
    return out;
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

// Decode HTTP/1.1 response: status, headers (lowercased keys), body.  Handles
// Content-Length and chunked encoding; falls back to read-to-close.
struct HttpResponse {
    int status = 0;
    std::string body;
};
HttpResponse read_http_response(int fd, std::chrono::steady_clock::time_point deadline) {
    HttpResponse out;
    // Read until we see CRLFCRLF for headers.
    std::string head;
    char ch;
    while (head.find("\r\n\r\n") == std::string::npos) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return out;
        if (read_n_deadline(fd, &ch, 1, deadline) != 1) return out;
        head.push_back(ch);
        if (head.size() > 64 * 1024) return out;   // header bomb guard
    }
    // Parse status line.
    auto eol = head.find("\r\n");
    if (eol == std::string::npos) return out;
    std::string status_line = head.substr(0, eol);
    // "HTTP/1.1 200 OK"
    auto sp1 = status_line.find(' ');
    if (sp1 != std::string::npos) {
        auto sp2 = status_line.find(' ', sp1 + 1);
        try { out.status = std::stoi(status_line.substr(sp1 + 1, sp2 - sp1 - 1)); }
        catch (...) { out.status = 0; }
    }
    // Parse headers.
    std::string headers_block = head.substr(eol + 2, head.size() - eol - 4);
    long long content_length = -1;
    bool chunked = false;
    size_t p = 0;
    while (p < headers_block.size()) {
        auto nl = headers_block.find("\r\n", p);
        if (nl == std::string::npos) nl = headers_block.size();
        std::string line = headers_block.substr(p, nl - p);
        p = nl + 2;
        auto col = line.find(':');
        if (col == std::string::npos) continue;
        std::string k = lower(trim(line.substr(0, col)));
        std::string v = trim(line.substr(col + 1));
        if (k == "content-length") {
            try { content_length = std::stoll(v); } catch (...) {}
        } else if (k == "transfer-encoding" && ci_equal(v, "chunked")) {
            chunked = true;
        }
    }
    // Read body.
    if (chunked) {
        while (true) {
            // Read chunk size line.
            std::string sz_line;
            while (true) {
                if (read_n_deadline(fd, &ch, 1, deadline) != 1) return out;
                if (ch == '\r') {
                    char lf;
                    if (read_n_deadline(fd, &lf, 1, deadline) != 1) return out;
                    break;
                }
                sz_line.push_back(ch);
                if (sz_line.size() > 64) return out;
            }
            auto semi = sz_line.find(';');
            if (semi != std::string::npos) sz_line = sz_line.substr(0, semi);
            sz_line = trim(sz_line);
            size_t chunk_size = 0;
            try { chunk_size = (size_t)std::stoull(sz_line, nullptr, 16); }
            catch (...) { return out; }
            if (chunk_size == 0) {
                // Trailing CRLF after last chunk.
                char crlf[2];
                read_n_deadline(fd, crlf, 2, deadline);
                break;
            }
            std::vector<char> chunk(chunk_size);
            if ((size_t)read_n_deadline(fd, chunk.data(), chunk_size, deadline) != chunk_size) return out;
            out.body.append(chunk.data(), chunk.size());
            char crlf[2];
            read_n_deadline(fd, crlf, 2, deadline);
            // Hard cap on body to avoid runaway: 256 MB per response.
            if (out.body.size() > 256 * 1024 * 1024) return out;
        }
    } else if (content_length >= 0) {
        out.body.resize((size_t)content_length);
        ssize_t got = read_n_deadline(fd, out.body.data(), (size_t)content_length, deadline);
        out.body.resize(got > 0 ? (size_t)got : 0);
    } else {
        out.body = read_to_close(fd, deadline);
    }
    return out;
}

// Find the value for a key in a JSON object string.  This is a positional
// scan, not a real parser — enough for the small surface area we hit on
// the ComfyUI side.  Returns "" if missing.
std::string json_find_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t pos = 0;
    while (true) {
        pos = json.find(needle, pos);
        if (pos == std::string::npos) return "";
        size_t s = pos + needle.size();
        while (s < json.size() && std::isspace((unsigned char)json[s])) ++s;
        if (s < json.size() && json[s] == '"') {
            size_t e = s + 1;
            while (e < json.size() && json[e] != '"') {
                if (json[e] == '\\' && e + 1 < json.size()) e += 2;
                else ++e;
            }
            return json.substr(s + 1, e - (s + 1));
        }
        pos = s;
    }
}

// Extract a JSON sub-object (top-level only) that follows "key": ...
// Returns the slice including the opening { and closing }, or "" if missing.
std::string json_find_object(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    size_t s = pos + needle.size();
    while (s < json.size() && std::isspace((unsigned char)json[s])) ++s;
    if (s >= json.size() || json[s] != '{') return "";
    int depth = 0;
    size_t e = s;
    bool in_str = false;
    while (e < json.size()) {
        char c = json[e];
        if (in_str) {
            if (c == '\\' && e + 1 < json.size()) { e += 2; continue; }
            if (c == '"') in_str = false;
        } else {
            if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') { --depth; if (depth == 0) return json.substr(s, e - s + 1); }
        }
        ++e;
    }
    return "";
}

std::string make_client_id() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto r1 = gen(), r2 = gen();
    char buf[48];
    std::snprintf(buf, sizeof(buf), "dist-rig-%016llx%016llx",
                  (unsigned long long)r1, (unsigned long long)r2);
    return buf;
}

std::string sniff_mime(const std::string& filename) {
    auto dot = filename.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = lower(filename.substr(dot + 1));
    if (ext == "png")  return "image/png";
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "gif")  return "image/gif";
    if (ext == "webp") return "image/webp";
    if (ext == "mp4")  return "video/mp4";
    if (ext == "webm") return "video/webm";
    if (ext == "json") return "application/json";
    return "application/octet-stream";
}

std::string url_encode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    auto hex = "0123456789ABCDEF";
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out.push_back(c);
        else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0xF]);
        }
    }
    return out;
}

}  // namespace

ComfyClient::ComfyClient(std::string base_url) : base_(std::move(base_url)) {
    auto p = parse_url(base_);
    host_ = p.host.empty() ? "127.0.0.1" : p.host;
    port_ = p.port ? p.port : 8188;
    path_prefix_ = p.path;
    if (!path_prefix_.empty() && path_prefix_.back() == '/') {
        path_prefix_.pop_back();
    }
}

std::string ComfyClient::http_get(const std::string& path, int timeout_ms, int* status) {
    int fd = open_socket(host_, port_, std::min(timeout_ms, 5000));
    if (fd < 0) { if (status) *status = 0; return ""; }
    std::ostringstream req;
    req << "GET " << path_prefix_ << path << " HTTP/1.1\r\n"
        << "Host: " << host_ << ":" << port_ << "\r\n"
        << "User-Agent: gpunet-node/1.0\r\n"
        << "Accept: */*\r\n"
        << "Connection: close\r\n"
        << "\r\n";
    std::string s = req.str();
    ::send(fd, s.data(), (int)s.size(), 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto resp = read_http_response(fd, deadline);
    DIST_CLOSESOCK(fd);
    if (status) *status = resp.status;
    return resp.body;
}

std::string ComfyClient::http_post(const std::string& path,
                                    const std::string& content_type,
                                    const std::string& body,
                                    int timeout_ms, int* status) {
    int fd = open_socket(host_, port_, std::min(timeout_ms, 5000));
    if (fd < 0) { if (status) *status = 0; return ""; }
    std::ostringstream req;
    req << "POST " << path_prefix_ << path << " HTTP/1.1\r\n"
        << "Host: " << host_ << ":" << port_ << "\r\n"
        << "User-Agent: gpunet-node/1.0\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Accept: */*\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;
    std::string s = req.str();
    // Send in a single shot for small payloads; for large bodies the kernel
    // will fragment as needed.  We don't bother with partial-write retries
    // because the deadline path will surface the failure.
    ::send(fd, s.data(), (int)s.size(), 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto resp = read_http_response(fd, deadline);
    DIST_CLOSESOCK(fd);
    if (status) *status = resp.status;
    return resp.body;
}

// proxy forwards a coordinator-originated comfy_meta request to the
// local ComfyUI.  The set of permitted paths is mirrored from
// server/comfy_meta.go::comfyMetaAllowed — both sides enforce it so a
// future server bug that lets a bad path through still hits this
// agent-side fence.  Returns body; *status holds the HTTP status (0
// on transport failure or disallowed path).
//
// Anything mutating that isn't /interrupt or /free is blocked because
// the coordinator owns the /prompt + /upload + /history surface and
// applies user/pool authorisation there.
static bool comfy_meta_path_allowed(const std::string& method,
                                    const std::string& path) {
    // GET reads are introspection-only.
    if (method == "GET") {
        if (path == "/system_stats" ||
            path == "/features" ||
            path == "/embeddings" ||
            path == "/models" ||
            path == "/object_info" ||
            path == "/prompt" ||
            path == "/queue" ||
            path == "/history") return true;
        if (path.rfind("/object_info/", 0) == 0) return true;
        if (path.rfind("/models/", 0) == 0)      return true;
        if (path.rfind("/history/", 0) == 0)     return true;
        return false;
    }
    // POST is locked to two control endpoints.
    if (method == "POST") {
        return path == "/interrupt" || path == "/free";
    }
    return false;
}

std::string ComfyClient::proxy(const std::string& method,
                               const std::string& path,
                               const std::string& body,
                               int timeout_ms,
                               int* status) {
    if (!comfy_meta_path_allowed(method, path)) {
        if (status) *status = 0;
        return "";
    }
    if (method == "GET") {
        return http_get(path, timeout_ms, status);
    }
    // POST.  ComfyUI accepts application/json for both /interrupt and
    // /free; an empty body is fine — ComfyUI defaults the unload flags
    // to false (matching its own behaviour when called with no body).
    std::string b = body.empty() ? std::string("{}") : body;
    return http_post(path, "application/json", b, timeout_ms, status);
}

ComfyProbe ComfyClient::probe(int timeout_ms) {
    ComfyProbe out;
    int st = 0;
    auto stats = http_get("/system_stats", timeout_ms, &st);
    if (st != 200) {
        out.error = stats.empty()
            ? ("comfyui unreachable at " + host_ + ":" + std::to_string(port_))
            : ("comfyui /system_stats returned " + std::to_string(st));
        return out;
    }
    out.ok = true;
    out.version = json_find_string(stats, "comfyui_version");
    if (out.version.empty()) out.version = json_find_string(stats, "version");

    // Best-effort model list: pull the CheckpointLoaderSimple ckpt_name enum
    // out of /object_info.  ComfyUI tucks the choices in a deeply nested
    // structure; we sniff just enough characters to extract the array.
    int st2 = 0;
    auto obj = http_get("/object_info/CheckpointLoaderSimple", timeout_ms, &st2);
    if (st2 == 200 && !obj.empty()) {
        // Look for a JSON array that's followed by ", " or "], " — the
        // first such array under "ckpt_name" is the list of checkpoints.
        auto p = obj.find("ckpt_name");
        if (p != std::string::npos) {
            p = obj.find('[', p);
            if (p != std::string::npos) {
                // Walk to the matching close-bracket.
                int depth = 0;
                size_t e = p;
                while (e < obj.size()) {
                    char c = obj[e];
                    if (c == '[') ++depth;
                    else if (c == ']') { if (--depth == 0) break; }
                    ++e;
                }
                if (e < obj.size()) {
                    std::string arr = obj.substr(p + 1, e - p - 1);
                    // Pull "..." strings out of the array.
                    size_t i = 0;
                    while (i < arr.size()) {
                        if (arr[i] == '"') {
                            size_t j = i + 1;
                            while (j < arr.size() && arr[j] != '"') {
                                if (arr[j] == '\\' && j + 1 < arr.size()) j += 2;
                                else ++j;
                            }
                            out.models.push_back(arr.substr(i + 1, j - i - 1));
                            i = j + 1;
                        } else {
                            ++i;
                        }
                        if (out.models.size() >= 64) break;   // cap for transport
                    }
                }
            }
        }
    }
    return out;
}

std::string ComfyClient::run(const std::string& graph_json,
                              int total_timeout_ms,
                              ComfyOnResult on_result) {
    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::milliseconds(total_timeout_ms);

    std::string client_id = make_client_id();

    // 1. Submit prompt.
    //    Body shape: { "prompt": <graph>, "client_id": "<id>" }
    //    `graph_json` is already the serialized ComfyUI graph object.
    std::ostringstream body;
    body << "{\"prompt\":" << graph_json
         << ",\"client_id\":\"" << client_id << "\"}";
    int st = 0;
    auto submit = http_post("/prompt", "application/json", body.str(), 30000, &st);
    if (st != 200) {
        return std::string("comfyui /prompt failed: status=") + std::to_string(st)
               + " body=" + submit.substr(0, 240);
    }
    std::string prompt_id = json_find_string(submit, "prompt_id");
    if (prompt_id.empty()) {
        return "comfyui /prompt missing prompt_id: " + submit.substr(0, 240);
    }

    // 2. Poll /history/<prompt_id>.  ComfyUI returns {} until the run completes;
    //    on success the response is a single-key object keyed by prompt_id.
    std::string history;
    while (true) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return "comfyui run timed out (no completion within "
                   + std::to_string(total_timeout_ms) + "ms)";
        }
        int hst = 0;
        history = http_get("/history/" + url_encode(prompt_id), 10000, &hst);
        if (hst == 200 && history.find(prompt_id) != std::string::npos) {
            // Make sure outputs key is populated, not just queue position.
            auto entry = json_find_object(history, prompt_id);
            if (!entry.empty() && entry.find("outputs") != std::string::npos) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    auto entry = json_find_object(history, prompt_id);
    auto outputs = json_find_object(entry, "outputs");
    if (outputs.empty()) {
        return "comfyui run finished with no outputs: " + entry.substr(0, 240);
    }

    // 3. Walk the outputs object — each value is a per-node object that may
    //    contain images / gifs / videos arrays.  Pull every {filename,
    //    subfolder, type} triple and download it.
    //
    // Positional walk: find every "filename":"..." and look back to the
    // nearest enclosing object for subfolder + type.  Simple and good enough
    // since ComfyUI's output schema is small and shallow.
    struct OutRef { std::string filename, subfolder, type; };
    std::vector<OutRef> refs;
    {
        size_t pos = 0;
        while (true) {
            auto p = outputs.find("\"filename\"", pos);
            if (p == std::string::npos) break;
            // Find enclosing object: scan backwards for the matching '{'.
            int depth = 0;
            size_t start = p;
            while (start > 0) {
                char c = outputs[start];
                if (c == '}') ++depth;
                else if (c == '{') { if (depth == 0) break; --depth; }
                if (start == 0) break;
                --start;
            }
            // Find matching close-brace.
            int d2 = 0;
            size_t end = start;
            bool in_str = false;
            while (end < outputs.size()) {
                char c = outputs[end];
                if (in_str) {
                    if (c == '\\' && end + 1 < outputs.size()) { end += 2; continue; }
                    if (c == '"') in_str = false;
                } else {
                    if (c == '"') in_str = true;
                    else if (c == '{') ++d2;
                    else if (c == '}') { if (--d2 == 0) { ++end; break; } }
                }
                ++end;
            }
            std::string obj = outputs.substr(start, end - start);
            OutRef r;
            r.filename  = json_find_string(obj, "filename");
            r.subfolder = json_find_string(obj, "subfolder");
            r.type      = json_find_string(obj, "type");
            if (!r.filename.empty()) refs.push_back(std::move(r));
            pos = end;
            if (refs.size() >= 64) break;   // cap
        }
    }
    if (refs.empty()) {
        return "comfyui run finished but no files were produced";
    }

    // 4. Download each file via /view and forward.
    for (const auto& r : refs) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return "comfyui run timed out while downloading outputs";
        }
        std::string path = "/view?filename=" + url_encode(r.filename)
                         + "&subfolder=" + url_encode(r.subfolder)
                         + "&type=" + url_encode(r.type.empty() ? "output" : r.type);
        int fst = 0;
        auto body_str = http_get(path, 60000, &fst);
        if (fst != 200 || body_str.empty()) {
            return "comfyui /view failed for " + r.filename
                   + " status=" + std::to_string(fst);
        }
        ComfyResult cr;
        cr.filename = r.filename;
        cr.mime     = sniff_mime(r.filename);
        cr.data.assign(body_str.begin(), body_str.end());
        if (on_result && !on_result(cr)) return "";   // caller aborted
    }
    return "";
}

ComfyClient make_default_comfy_client() {
    const char* env = std::getenv("DIST_COMFY_URL");
    std::string url = (env && *env) ? env : "http://127.0.0.1:8188";
    return ComfyClient(std::move(url));
}

bool comfy_force_enabled() {
    auto truthy = [](const char* v) {
        if (!v || !*v) return false;
        std::string s = v;
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        return s == "1" || s == "true" || s == "yes" || s == "on";
    };
    return truthy(std::getenv("DIST_WITH_COMFYUI")) ||
           truthy(std::getenv("DIST_COMFY_FORCE"));
}

}  // namespace dist
