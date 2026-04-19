#pragma once
/**
 * dashboard_server.h
 *
 * DashboardServer — minimal HTTP server embedded in the coordinator.
 *
 * Serves:
 *   GET  /           → the dashboard SPA (HTML + inline CSS + JS)
 *   GET  /stats      → JSON snapshot of ClusterMonitor (for polling)
 *   GET  /events     → SSE stream; pushes "data: <json>\n\n" every second
 *   GET  /join       → returns plain-text join command for new nodes
 *
 * Implementation: hand-rolled TCP accept loop, no external HTTP library.
 * Keeps it to one header + one .cpp with zero new dependencies.
 *
 * Threading: one accept thread + one thread per SSE subscriber (typically
 * just a few browser tabs). Non-SSE requests are handled synchronously on
 * the accept thread.
 */

#include "cluster_monitor.h"
#include "dist_conn.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dist {

struct DashboardConfig {
    std::string bind_host   = "0.0.0.0";
    uint16_t    http_port   = 7780;
    std::string public_host;          // host shown in the join command
    uint16_t    ctrl_port   = PORT_CONTROL;
    uint16_t    vm_ctrl_port = 7703;
    bool        vm_mode     = false;  // true → show VM mode join command
};

class DashboardServer {
public:
    DashboardServer(DashboardConfig cfg, ClusterMonitor& monitor);
    ~DashboardServer();

    void start();
    void stop();

    uint16_t port() const { return cfg_.http_port; }

private:
    void accept_loop();
    void handle_client(int fd);

    // HTTP response helpers
    static std::string http_200(const std::string& content_type,
                                const std::string& body);
    static std::string http_404();

    // Route handlers — return full HTTP response string
    std::string serve_index();
    std::string serve_stats();
    std::string serve_join();
    void        serve_sse(int fd);

    // SSE broadcaster
    void sse_broadcast_loop();
    void sse_push(int fd, const std::string& json);

    DashboardConfig  cfg_;
    ClusterMonitor&  monitor_;
    std::atomic<bool> running_ { false };

    Listener          listener_;
    std::thread       accept_thread_;
    std::thread       sse_broadcast_thread_;

    // SSE subscriber file descriptors
    std::mutex        sse_mu_;
    std::vector<int>  sse_clients_;
};

} // namespace dist
