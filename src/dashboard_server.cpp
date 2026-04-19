/**
 * dashboard_server.cpp
 *
 * Hand-rolled HTTP/1.1 server — no external library.
 *
 * Protocol assumptions (sufficient for a local dashboard):
 *   - Only GET is handled.
 *   - Request parsing reads until the first blank line.
 *   - No TLS (intended for LAN/localhost use).
 *   - SSE clients stay connected indefinitely; we push every second.
 */

#include "dashboard_server.h"
#include "dashboard_html.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "platform_compat.h"

namespace dist {

// ─── Constructor / Destructor ─────────────────────────────────────────────────

DashboardServer::DashboardServer(DashboardConfig cfg, ClusterMonitor& monitor)
    : cfg_(std::move(cfg))
    , monitor_(monitor)
{}

DashboardServer::~DashboardServer() { stop(); }

// ─── Start / Stop ─────────────────────────────────────────────────────────────

void DashboardServer::start() {
    if (running_.exchange(true)) return;

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        std::cerr << "[Dashboard] socket() failed\n"; return;
    }
    int yes = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(cfg_.http_port);
    sa.sin_addr.s_addr = INADDR_ANY;

    if (::bind(srv, (sockaddr*)&sa, sizeof(sa)) < 0) {
        std::cerr << "[Dashboard] bind() failed on port " << cfg_.http_port << "\n";
        dist::close_sock(srv);
        return;
    }
    ::listen(srv, 16);
    std::cout << "[Dashboard] http://0.0.0.0:" << cfg_.http_port << "\n";

    sse_broadcast_thread_ = std::thread([this]{ sse_broadcast_loop(); });

    accept_thread_ = std::thread([this, srv]() mutable {
        while (running_.load()) {
            sockaddr_in ca{};
            socklen_t clen = sizeof(ca);
            int fd = ::accept(srv, (sockaddr*)&ca, &clen);
            if (fd < 0) { if (running_.load()) dist::sleep_ms(5); continue; }
            // Fire and forget — non-SSE requests are quick; SSE gets its own thread
            std::thread([this, fd]{ handle_client(fd); }).detach();
        }
        dist::close_sock(srv);
    });
}

void DashboardServer::stop() {
    if (!running_.exchange(false)) return;
    // Wake up SSE broadcast thread
    sse_broadcast_thread_.join();
    if (accept_thread_.joinable()) accept_thread_.join();
}

// ─── Client handler ───────────────────────────────────────────────────────────

void DashboardServer::handle_client(int fd) {
    // Read request until \r\n\r\n
    std::string req;
    req.reserve(512);
    char buf[256];
    while (req.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) { dist::close_sock(fd); return; }
        req.append(buf, (size_t)n);
        if (req.size() > 8192) break;
    }

    // Parse first line
    auto nl = req.find("\r\n");
    std::string line = (nl != std::string::npos) ? req.substr(0, nl) : req;
    // "GET /path HTTP/1.1"
    std::string path;
    {
        size_t s = line.find(' ');
        size_t e = (s != std::string::npos) ? line.find(' ', s+1) : std::string::npos;
        if (s != std::string::npos && e != std::string::npos)
            path = line.substr(s+1, e-s-1);
    }
    // Strip query string
    auto qpos = path.find('?');
    if (qpos != std::string::npos) path = path.substr(0, qpos);

    if (path == "/" || path.empty()) {
        auto resp = serve_index();
        ::send(fd, resp.data(), resp.size(), 0);
        dist::close_sock(fd);
    } else if (path == "/stats") {
        auto resp = serve_stats();
        ::send(fd, resp.data(), resp.size(), 0);
        dist::close_sock(fd);
    } else if (path == "/join") {
        auto resp = serve_join();
        ::send(fd, resp.data(), resp.size(), 0);
        dist::close_sock(fd);
    } else if (path == "/events") {
        serve_sse(fd);
        // fd closed inside serve_sse
    } else {
        auto resp = http_404();
        ::send(fd, resp.data(), resp.size(), 0);
        dist::close_sock(fd);
    }
}

// ─── Route handlers ───────────────────────────────────────────────────────────

std::string DashboardServer::serve_index() {
    return http_200("text/html; charset=utf-8", DASHBOARD_HTML);
}

std::string DashboardServer::serve_stats() {
    return http_200("application/json", monitor_.to_json());
}

std::string DashboardServer::serve_join() {
    // Produce the exact command a new node should run to join.
    std::string host = cfg_.public_host.empty() ? "YOUR_COORDINATOR_IP" : cfg_.public_host;

    std::string cmd;
    if (cfg_.vm_mode) {
        cmd = "dist-vm-node --server " + host
            + " --control-port " + std::to_string(cfg_.ctrl_port)
            + " --vm-port "      + std::to_string(cfg_.vm_ctrl_port);
    } else {
        cmd = "dist-node --server " + host
            + " --control-port " + std::to_string(cfg_.ctrl_port);
    }
    return http_200("text/plain", cmd);
}

void DashboardServer::serve_sse(int fd) {
    // Send SSE headers
    std::string hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    if (::send(fd, hdr.data(), hdr.size(), 0) < 0) {
        dist::close_sock(fd); return;
    }

    {
        std::lock_guard<std::mutex> lk(sse_mu_);
        sse_clients_.push_back(fd);
    }

    // Block until client disconnects (detected by sse_push returning false)
    // We park the thread here; sse_broadcast_loop does the actual sending.
    // Just wait.
    while (running_.load()) {
        bool found = false;
        {
            std::lock_guard<std::mutex> lk(sse_mu_);
            found = std::find(sse_clients_.begin(), sse_clients_.end(), fd)
                    != sse_clients_.end();
        }
        if (!found) break; // removed by broadcast loop on error
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    dist::close_sock(fd);
}

// ─── SSE broadcast ────────────────────────────────────────────────────────────

void DashboardServer::sse_broadcast_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!running_.load()) break;

        std::string json = monitor_.to_json();
        std::string frame = "data: " + json + "\n\n";

        std::lock_guard<std::mutex> lk(sse_mu_);
        std::vector<int> dead;
        for (int fd : sse_clients_) {
            if (::send(fd, frame.data(), frame.size(), MSG_NOSIGNAL) < 0)
                dead.push_back(fd);
        }
        for (int fd : dead) {
            sse_clients_.erase(
                std::remove(sse_clients_.begin(), sse_clients_.end(), fd),
                sse_clients_.end());
            dist::close_sock(fd);
        }
    }
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

std::string DashboardServer::http_200(const std::string& ct,
                                       const std::string& body) {
    std::ostringstream o;
    o << "HTTP/1.1 200 OK\r\n"
      << "Content-Type: " << ct << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
    return o.str();
}

std::string DashboardServer::http_404() {
    std::string body = "Not found";
    std::ostringstream o;
    o << "HTTP/1.1 404 Not Found\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
    return o.str();
}

} // namespace dist
