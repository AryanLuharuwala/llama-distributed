// gpunet-cli top — live TUI dashboard.
//
// FTXUI is fetched at configure time via CMake FetchContent and linked into
// the gpunet-cli target.  The dashboard polls the server's REST API every
// second and re-renders.  Tabs: Rigs / Pools / Logs.  Keybinds: q quit,
// 1/2/3 tab switch, ↑/↓ row highlight, r force refresh.
//
// The whole file is conditionally compiled when DIST_BUILD_CLI_TUI is on.
// A stub dist_cli_run_top() exists below for builds without FTXUI so the
// `top` subcommand at least prints a friendly message.

#include "cli_common.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dc = dist::cli;

#ifdef DIST_HAVE_FTXUI

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

namespace {

// JSON array walker — same helper as dist_cli_main.cpp.  Duplicated here
// rather than dragged through a header because both files need it
// privately and the surface is too small to justify another module.
std::vector<std::string> json_array_objects(const std::string& body) {
    std::vector<std::string> out;
    int depth = 0;
    bool in_str = false, esc = false;
    size_t start = std::string::npos;
    size_t array_depth = 0;
    bool started = false;
    for (size_t i = 0; i < body.size(); ++i) {
        char c = body[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '[') { ++array_depth; started = true; continue; }
        if (c == ']') { if (array_depth) --array_depth; continue; }
        if (!started || array_depth == 0) continue;
        if (c == '{') { if (depth == 0) start = i; ++depth; }
        else if (c == '}') {
            if (--depth == 0 && start != std::string::npos) {
                out.emplace_back(body.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
    }
    return out;
}

bool find_top_array(const std::string& body, const std::string& key,
                    std::string& slice) {
    std::string needle = "\"" + key + "\":";
    size_t p = body.find(needle);
    if (p == std::string::npos) return false;
    p += needle.size();
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t' || body[p] == '\n')) ++p;
    if (p >= body.size() || body[p] != '[') return false;
    int depth = 0;
    bool in_str = false, esc = false;
    size_t start = p;
    for (; p < body.size(); ++p) {
        char c = body[p];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '[') ++depth;
        else if (c == ']') { if (--depth == 0) { slice = body.substr(start, p - start + 1); return true; } }
    }
    return false;
}

struct Snapshot {
    std::vector<std::string> rigs;
    std::vector<std::string> pools;
    std::vector<std::string> logs;
    std::deque<int>          throughput;   // requests/sec, last 60 samples
    std::chrono::steady_clock::time_point fetched;
    std::string              error;
};

class Poller {
public:
    Poller(const dc::AuthCtx& ctx) : ctx_(ctx) {}

    void start() {
        running_.store(true);
        th_ = std::thread([this]{ run(); });
    }
    void stop() {
        running_.store(false);
        if (th_.joinable()) th_.join();
    }
    Snapshot snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        return snap_;
    }

private:
    void fetch_into(Snapshot& s) {
        const std::string bearer = "Authorization: Bearer " + ctx_.api_key;
        dc::HttpResp r; std::string err;

        s.error.clear();

        if (dc::http_request(ctx_.server_url, "/api/me/rigs", "GET", "",
                             {bearer}, r, err) && r.status == 200) {
            std::string arr;
            if (!find_top_array(r.body, "rigs", arr)) arr = r.body;
            s.rigs = json_array_objects(arr);
        } else if (s.error.empty()) {
            s.error = "rigs: " + (err.empty() ? std::to_string(r.status) : err);
        }

        if (dc::http_request(ctx_.server_url, "/api/pools", "GET", "",
                             {bearer}, r, err) && r.status == 200) {
            std::string arr;
            if (!find_top_array(r.body, "pools", arr)) arr = r.body;
            s.pools = json_array_objects(arr);
        }

        if (dc::http_request(ctx_.server_url, "/api/inference_log?limit=20",
                             "GET", "", {bearer}, r, err) && r.status == 200) {
            std::string arr;
            if (!find_top_array(r.body, "log", arr) &&
                !find_top_array(r.body, "entries", arr)) arr = r.body;
            s.logs = json_array_objects(arr);

            // Sample throughput: rough count of entries in the last second.
            // Server doesn't expose a clean RPS metric so this is just an
            // ambient pulse.
            int rps = static_cast<int>(s.logs.size());
            s.throughput.push_back(rps);
            while (s.throughput.size() > 60) s.throughput.pop_front();
        }

        s.fetched = std::chrono::steady_clock::now();
    }

    void run() {
        while (running_.load()) {
            Snapshot working;
            {
                std::lock_guard<std::mutex> lk(mu_);
                working = snap_;
            }
            fetch_into(working);
            {
                std::lock_guard<std::mutex> lk(mu_);
                snap_ = std::move(working);
            }
            for (int i = 0; i < 10 && running_.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    const dc::AuthCtx&        ctx_;
    std::atomic<bool>         running_{false};
    std::thread               th_;
    std::mutex                mu_;
    Snapshot                  snap_;
};

Element ascii_sparkline(const std::deque<int>& samples, int width = 40) {
    if (samples.empty()) return text("(no data)") | dim;
    int hi = 1;
    for (int v : samples) if (v > hi) hi = v;
    const char* bars[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    std::string out;
    int n = static_cast<int>(samples.size());
    int start = std::max(0, n - width);
    for (int i = start; i < n; ++i) {
        int idx = (samples[i] * 8) / hi;
        if (idx < 0) idx = 0; if (idx > 8) idx = 8;
        out += bars[idx];
    }
    return text(out);
}

Element render_rigs(const Snapshot& s) {
    Elements rows;
    rows.push_back(hbox({
        text(" AGENT_ID            ") | bold,
        text(" STATUS     ") | bold,
        text(" NAT        ") | bold,
        text(" SLOTS  ") | bold,
        text(" RTT   ") | bold,
        text(" NICK") | bold,
    }) | inverted);
    for (const auto& obj : s.rigs) {
        std::string agent = dc::json_peek_string(obj, "agent_id");
        std::string status= dc::json_peek_string(obj, "status");
        std::string nat   = dc::json_peek_string(obj, "nat_type");
        std::string su    = dc::json_peek_int(obj, "slots_used");
        std::string sm    = dc::json_peek_int(obj, "slots_max");
        std::string rtt   = dc::json_peek_int(obj, "rtt_ms");
        std::string nick  = dc::json_peek_string(obj, "nick");
        Color st = (status == "online") ? Color::Green
                 : (status == "idle")   ? Color::Yellow
                                        : Color::GrayDark;
        rows.push_back(hbox({
            text(" " + agent) | size(WIDTH, EQUAL, 22),
            text(" " + status) | color(st) | size(WIDTH, EQUAL, 12),
            text(" " + nat) | size(WIDTH, EQUAL, 12),
            text(" " + su + "/" + sm) | size(WIDTH, EQUAL, 8),
            text(" " + rtt + "ms") | size(WIDTH, EQUAL, 8),
            text(" " + nick),
        }));
    }
    if (s.rigs.empty()) rows.push_back(text(" (no rigs paired yet — run `gpunet-node connect`)") | dim);
    return vbox(rows) | border;
}

Element render_pools(const Snapshot& s) {
    Elements rows;
    rows.push_back(hbox({
        text(" ID  ") | bold,
        text(" NAME                          ") | bold,
        text(" VIS      ") | bold,
        text(" ROLE   ") | bold,
        text(" SLUG ") | bold,
    }) | inverted);
    for (const auto& obj : s.pools) {
        std::string id   = dc::json_peek_int(obj, "id");
        std::string name = dc::json_peek_string(obj, "name");
        std::string vis  = dc::json_peek_string(obj, "visibility");
        std::string role = dc::json_peek_string(obj, "my_role");
        std::string slug = dc::json_peek_string(obj, "slug");
        rows.push_back(hbox({
            text(" " + id) | size(WIDTH, EQUAL, 5),
            text(" " + name) | size(WIDTH, EQUAL, 31),
            text(" " + vis) | size(WIDTH, EQUAL, 10),
            text(" " + (role.empty() ? std::string("-") : role)) | size(WIDTH, EQUAL, 8),
            text(" " + slug),
        }));
    }
    if (s.pools.empty()) rows.push_back(text(" (no pools — create one with `gpunet-cli pools create`)") | dim);
    return vbox(rows) | border;
}

Element render_logs(const Snapshot& s) {
    Elements rows;
    rows.push_back(hbox({
        text(" TS                 ") | bold,
        text(" MODEL                          ") | bold,
        text(" AGENT       ") | bold,
        text(" STATUS  ") | bold,
        text(" P_TOK  ") | bold,
        text(" C_TOK") | bold,
    }) | inverted);
    int shown = 0;
    for (auto it = s.logs.rbegin(); it != s.logs.rend() && shown < 30; ++it, ++shown) {
        const auto& obj = *it;
        rows.push_back(hbox({
            text(" " + dc::json_peek_string(obj, "ts")) | size(WIDTH, EQUAL, 20),
            text(" " + dc::json_peek_string(obj, "model")) | size(WIDTH, EQUAL, 31),
            text(" " + dc::json_peek_string(obj, "agent_id")) | size(WIDTH, EQUAL, 13),
            text(" " + dc::json_peek_int(obj, "status")) | size(WIDTH, EQUAL, 8),
            text(" " + dc::json_peek_int(obj, "prompt_tokens")) | size(WIDTH, EQUAL, 7),
            text(" " + dc::json_peek_int(obj, "completion_tokens")),
        }));
    }
    if (s.logs.empty()) rows.push_back(text(" (no inference activity)") | dim);
    return vbox(rows) | border;
}

} // namespace

int dist_cli_run_top(const dc::AuthCtx& ctx) {
    Poller poller(ctx);
    poller.start();

    auto screen = ScreenInteractive::Fullscreen();
    int tab_index = 0;
    std::vector<std::string> tab_titles = {"Rigs", "Pools", "Logs"};

    auto tab_toggle = Toggle(&tab_titles, &tab_index);

    auto renderer = Renderer(tab_toggle, [&]{
        Snapshot s = poller.snapshot();
        auto ms_since = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - s.fetched).count();
        std::string status = "updated " + std::to_string(ms_since) + "ms ago";
        if (!s.error.empty()) status += "   (err: " + s.error + ")";

        Element body;
        switch (tab_index) {
            case 0: body = render_rigs(s); break;
            case 1: body = render_pools(s); break;
            default: body = render_logs(s); break;
        }

        return vbox({
            hbox({
                text("DISTPOOL") | bold | color(Color::White),
                text("   "),
                text(ctx.server_url) | dim,
                filler(),
                text("q quit  1/2/3 tabs  r refresh") | dim,
            }) | border,
            hbox({
                tab_toggle->Render() | border | flex,
                hbox({text(" rps "), ascii_sparkline(s.throughput, 30)}) | border,
            }),
            body | flex,
            hbox({text(status) | dim, filler()}),
        });
    });

    auto with_keys = CatchEvent(renderer, [&](Event e){
        if (e == Event::Character('q') || e == Event::Escape) {
            screen.ExitLoopClosure()();
            return true;
        }
        if (e == Event::Character('1')) { tab_index = 0; return true; }
        if (e == Event::Character('2')) { tab_index = 1; return true; }
        if (e == Event::Character('3')) { tab_index = 2; return true; }
        return false;
    });

    // Repaint loop — FTXUI doesn't poll for us; we tick the screen at ~10Hz.
    std::atomic<bool> ticking{true};
    std::thread ticker([&]{
        while (ticking.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(with_keys);
    ticking.store(false);
    if (ticker.joinable()) ticker.join();
    poller.stop();
    return 0;
}

#else // !DIST_HAVE_FTXUI

int dist_cli_run_top(const dc::AuthCtx&) {
    std::fprintf(stderr,
        "gpunet-cli top: TUI dashboard was disabled at build time.\n"
        "Rebuild with -DDIST_BUILD_CLI_TUI=ON to enable.\n");
    return 1;
}

#endif
