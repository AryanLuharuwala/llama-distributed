#include "shard_download.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace dist {

namespace {

struct ParsedUrl {
    std::string host;
    uint16_t    port = 80;
    std::string path;  // includes query string
};

bool parse_http_url(const std::string & url, ParsedUrl & out, std::string & err) {
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) {
        err = "shard URL must be http://";
        return false;
    }
    size_t i = prefix.size();
    size_t slash = url.find('/', i);
    std::string hostport = url.substr(i, slash == std::string::npos
                                             ? std::string::npos
                                             : slash - i);
    out.path = (slash == std::string::npos) ? "/" : url.substr(slash);
    size_t colon = hostport.find(':');
    if (colon == std::string::npos) {
        out.host = hostport;
        out.port = 80;
    } else {
        out.host = hostport.substr(0, colon);
        out.port = (uint16_t) std::atoi(hostport.substr(colon + 1).c_str());
        if (out.port == 0) out.port = 80;
    }
    return true;
}

bool send_all(int fd, const char * data, size_t n) {
    while (n > 0) {
        ssize_t w = ::send(fd, data, n, 0);
        if (w <= 0) return false;
        data += w; n -= (size_t) w;
    }
    return true;
}

} // namespace

bool fetch_shard(const std::string & url, const std::string & dest_path,
                 std::string & err) {
    ParsedUrl u;
    if (!parse_http_url(url, u, err)) return false;

    // Resolve.
    struct addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", (unsigned) u.port);
    int rc = ::getaddrinfo(u.host.c_str(), port_str, &hints, &res);
    if (rc != 0 || !res) {
        err = "getaddrinfo: " + std::string(gai_strerror(rc));
        return false;
    }

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { ::freeaddrinfo(res); err = "socket failed"; return false; }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        ::freeaddrinfo(res); ::close(fd);
        err = "connect failed to " + u.host + ":" + std::to_string(u.port);
        return false;
    }
    ::freeaddrinfo(res);

    std::string req =
        "GET " + u.path + " HTTP/1.1\r\n"
        "Host: " + u.host + ":" + std::to_string(u.port) + "\r\n"
        "User-Agent: dist-node/0.1\r\n"
        "Connection: close\r\n"
        "\r\n";
    if (!send_all(fd, req.data(), req.size())) {
        ::close(fd); err = "send request failed"; return false;
    }

    // Read until we see \r\n\r\n (end of headers), then stream body to file.
    std::string hdr;
    char buf[8192];
    ssize_t n;
    size_t header_end = std::string::npos;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
        hdr.append(buf, n);
        header_end = hdr.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
        if (hdr.size() > 65536) {
            ::close(fd); err = "HTTP headers exceed 64KB"; return false;
        }
    }
    if (header_end == std::string::npos) {
        ::close(fd); err = "no HTTP header terminator"; return false;
    }

    // Parse status line.
    size_t sp1 = hdr.find(' ');
    size_t sp2 = hdr.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        ::close(fd); err = "bad HTTP status line"; return false;
    }
    int status = std::atoi(hdr.substr(sp1 + 1, sp2 - sp1 - 1).c_str());
    if (status != 200) {
        ::close(fd); err = "HTTP status " + std::to_string(status); return false;
    }

    FILE * fp = std::fopen(dest_path.c_str(), "wb");
    if (!fp) {
        ::close(fd); err = "open " + dest_path + " for write failed"; return false;
    }

    // Flush any body already in `hdr` past the header terminator.
    size_t body_start = header_end + 4;
    if (body_start < hdr.size()) {
        std::fwrite(hdr.data() + body_start, 1, hdr.size() - body_start, fp);
    }

    // Stream the rest.
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
        if (std::fwrite(buf, 1, (size_t) n, fp) != (size_t) n) {
            std::fclose(fp); ::close(fd);
            err = "write to " + dest_path + " failed";
            return false;
        }
    }
    std::fclose(fp);
    ::close(fd);
    if (n < 0) { err = "recv failed mid-body"; return false; }
    return true;
}

} // namespace dist
