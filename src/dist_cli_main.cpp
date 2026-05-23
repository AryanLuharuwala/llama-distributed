// dist-cli — operator-facing CLI for distpool.
//
// Standalone binary.  Does NOT require dist-node to be installed — a laptop
// with no GPU can `dist-cli login` to manage pools, watch rigs, and stream
// logs without ever joining the compute fabric.  When dist-node IS installed
// on the same box, both binaries share the state dir so `dist-node login`
// also authenticates dist-cli.
//
// Subcommands:
//   dist-cli login [--server URL]         Browser device-code login.
//   dist-cli logout                       Wipe local credentials.
//   dist-cli status                       Show pairing + auth info.
//   dist-cli pools list                   List pools the user belongs to.
//   dist-cli pools create <name> [--public]
//   dist-cli pools join <invite-token>
//   dist-cli pools members <pool-id>
//   dist-cli pools invite <pool-id>
//   dist-cli pools kick <pool-id> <rig-id>
//   dist-cli models list
//   dist-cli models import <hf-repo-id>
//   dist-cli models search <query> [--tag X] [--library Y]
//   dist-cli rigs list                    Show all rigs belonging to me.
//   dist-cli rigs watch                   Polling tail of my rig fleet.
//   dist-cli logs [--follow] [--tail N]   Inference log.
//   dist-cli top                          Live TUI dashboard (see dist_cli_top.cpp).
//
// Every subcommand prints human-readable text by default and accepts --json
// to dump the raw server response unfiltered (handy for scripts).

#include "cli_common.h"
#include "platform_compat.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#  include <unistd.h>
#endif

namespace dc = dist::cli;

// Forward declarations — implemented in dist_cli_top.cpp / dist_cli_split.cpp.
int dist_cli_run_top(const dc::AuthCtx& ctx);
int dist_cli_run_split(const dc::AuthCtx& ctx, int64_t pool_id);

namespace {

std::atomic<bool> g_quit{false};
void on_sigint(int) { g_quit.store(true); }

// ── Mini JSON array iterator ─────────────────────────────────────────────
// The cli_common json_peek_* helpers handle flat fields.  For lists we walk
// the array and yield each object's text span — callers then re-apply the
// flat helpers.  This avoids pulling in a real JSON library.
std::vector<std::string> json_array_objects(const std::string& body) {
    std::vector<std::string> out;
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    size_t start = std::string::npos;
    size_t array_depth = 0;
    bool started = false;
    for (size_t i = 0; i < body.size(); ++i) {
        char c = body[i];
        if (in_str) {
            if (esc) { esc = false; }
            else if (c == '\\') { esc = true; }
            else if (c == '"') { in_str = false; }
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '[') { ++array_depth; if (!started) started = true; continue; }
        if (c == ']') { if (array_depth) --array_depth; continue; }
        if (!started || array_depth == 0) continue;
        if (c == '{') {
            if (depth == 0) start = i;
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0 && start != std::string::npos) {
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
        else if (c == ']') {
            if (--depth == 0) { slice = body.substr(start, p - start + 1); return true; }
        }
    }
    return false;
}

// JSON-escape a string for embedding in a request body.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

bool flag(const std::vector<std::string>& args, const std::string& f) {
    for (const auto& a : args) if (a == f) return true;
    return false;
}

std::string opt(const std::vector<std::string>& args, const std::string& key,
                const std::string& dflt = "") {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == key) return args[i + 1];
    return dflt;
}

// ── Subcommands ──────────────────────────────────────────────────────────

int cmd_status(const dc::AuthCtx& ctx, const std::vector<std::string>&) {
    std::cout << "server:    " << ctx.server_url << "\n";
    std::cout << "agent_id:  " << (ctx.agent_id.empty() ? "(unset)" : ctx.agent_id) << "\n";
    std::cout << "auth:      "
              << (ctx.agent_key.empty() ? "none" : "agent_key=ok")
              << (ctx.api_key.empty()   ? "" : "  api_key=ok") << "\n";

    dc::HttpResp r; std::string err;
    if (dc::http_request(ctx.server_url, "/api/me", "GET", "",
                         {"Authorization: Bearer " + ctx.api_key}, r, err) &&
        r.status == 200) {
        std::cout << "user:      "
                  << dc::json_peek_string(r.body, "display_name")
                  << " (id=" << dc::json_peek_int(r.body, "id") << ")\n";
    } else {
        std::cout << "user:      <not reachable>  " << err << "\n";
    }
    return 0;
}

int cmd_pools(const dc::AuthCtx& ctx, std::vector<std::string> args) {
    if (args.empty()) {
        std::cerr << "usage: dist-cli pools <list|create|join|members|invite|kick> ...\n";
        return 1;
    }
    std::string sub = args[0];
    args.erase(args.begin());
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;

    auto fmt_pool_row = [](const std::string& obj) {
        std::string id   = dc::json_peek_int(obj, "id");
        std::string name = dc::json_peek_string(obj, "name");
        std::string vis  = dc::json_peek_string(obj, "visibility");
        std::string slug = dc::json_peek_string(obj, "slug");
        std::string role = dc::json_peek_string(obj, "my_role");
        std::printf("  %-5s  %-30s  %-9s  %-7s  %s\n",
                    id.c_str(), name.c_str(), vis.c_str(),
                    role.empty() ? "-" : role.c_str(),
                    slug.c_str());
    };

    if (sub == "list") {
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, "/api/pools", "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "list failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::printf("  %-5s  %-30s  %-9s  %-7s  %s\n",
                    "ID", "NAME", "VIS", "ROLE", "SLUG");
        std::string arr;
        if (!find_top_array(r.body, "pools", arr)) arr = r.body;
        for (const auto& obj : json_array_objects(arr)) fmt_pool_row(obj);
        return 0;
    }

    if (sub == "create") {
        if (args.empty() || args[0].rfind("--", 0) == 0) {
            std::cerr << "usage: dist-cli pools create <name> [--public]\n";
            return 1;
        }
        std::string name = args[0];
        std::string vis  = flag(args, "--public") ? "public" : "private";
        std::string body = "{\"name\":\"" + json_escape(name) +
                           "\",\"visibility\":\"" + vis + "\"}";
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, "/api/pools", "POST", body,
                              {bearer}, r, err) || r.status >= 300) {
            std::cerr << "create failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        std::cout << "created pool id=" << dc::json_peek_int(r.body, "id")
                  << " slug=" << dc::json_peek_string(r.body, "slug") << "\n";
        return 0;
    }

    if (sub == "join") {
        if (args.empty()) {
            std::cerr << "usage: dist-cli pools join <invite-token>\n";
            return 1;
        }
        std::string body = "{\"invite\":\"" + json_escape(args[0]) + "\"}";
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, "/api/pools/join", "POST", body,
                              {bearer}, r, err) || r.status >= 300) {
            std::cerr << "join failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        std::cout << "joined pool id=" << dc::json_peek_int(r.body, "id")
                  << " name=" << dc::json_peek_string(r.body, "name") << "\n";
        return 0;
    }

    if (sub == "members") {
        if (args.empty()) {
            std::cerr << "usage: dist-cli pools members <pool-id>\n";
            return 1;
        }
        dc::HttpResp r; std::string err;
        std::string path = "/api/pools/" + args[0];
        if (!dc::http_request(ctx.server_url, path, "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "members failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::string arr;
        if (!find_top_array(r.body, "members", arr)) arr = r.body;
        std::printf("  %-12s  %-24s  %s\n", "USER_ID", "DISPLAY_NAME", "ROLE");
        for (const auto& obj : json_array_objects(arr)) {
            std::printf("  %-12s  %-24s  %s\n",
                        dc::json_peek_int(obj, "user_id").c_str(),
                        dc::json_peek_string(obj, "display_name").c_str(),
                        dc::json_peek_string(obj, "role").c_str());
        }
        return 0;
    }

    if (sub == "invite") {
        if (args.empty()) {
            std::cerr << "usage: dist-cli pools invite <pool-id>\n";
            return 1;
        }
        dc::HttpResp r; std::string err;
        std::string path = "/api/pools/" + args[0] + "/invite";
        if (!dc::http_request(ctx.server_url, path, "POST", "{}",
                              {bearer}, r, err) || r.status >= 300) {
            std::cerr << "invite failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        std::string tok = dc::json_peek_string(r.body, "token");
        std::string url = dc::json_peek_string(r.body, "url");
        std::cout << "invite: " << (url.empty() ? tok : url) << "\n";
        return 0;
    }

    if (sub == "kick") {
        if (args.size() < 2) {
            std::cerr << "usage: dist-cli pools kick <pool-id> <rig-id>\n";
            return 1;
        }
        std::string path = "/api/pools/" + args[0] + "/rigs/" + args[1];
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, path, "DELETE", "",
                              {bearer}, r, err) || r.status >= 300) {
            std::cerr << "kick failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        std::cout << "kicked rig " << args[1] << " from pool " << args[0] << "\n";
        return 0;
    }

    std::cerr << "unknown pools subcommand: " << sub << "\n";
    return 1;
}

int cmd_models(const dc::AuthCtx& ctx, std::vector<std::string> args) {
    if (args.empty()) {
        std::cerr << "usage: dist-cli models <list|import|search|discover> ...\n";
        return 1;
    }
    std::string sub = args[0];
    args.erase(args.begin());
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;

    if (sub == "list") {
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, "/api/models", "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "list failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::string arr;
        if (!find_top_array(r.body, "models", arr)) arr = r.body;
        auto human_bytes = [](long long n) -> std::string {
            if (n <= 0) return "—";
            const char* units[] = {"B","KB","MB","GB","TB","PB"};
            double v = static_cast<double>(n);
            int u = 0;
            while (v >= 1024.0 && u < 5) { v /= 1024.0; ++u; }
            char buf[32];
            if (u == 0) std::snprintf(buf, sizeof(buf), "%lld %s", n, units[u]);
            else        std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
            return buf;
        };
        std::printf("  %-44s  %-7s  %-7s  %-10s  %s\n",
                    "NAME", "LAYERS", "SHARDS", "SIZE", "REPO");
        for (const auto& obj : json_array_objects(arr)) {
            long long sz = 0;
            try { sz = std::stoll(dc::json_peek_int(obj, "size_bytes")); } catch (...) {}
            std::printf("  %-44s  %-7s  %-7s  %-10s  %s\n",
                        dc::json_peek_string(obj, "name").c_str(),
                        dc::json_peek_int(obj, "n_layers").c_str(),
                        dc::json_peek_int(obj, "n_shards").c_str(),
                        human_bytes(sz).c_str(),
                        dc::json_peek_string(obj, "repo_id").c_str());
        }
        return 0;
    }

    if (sub == "import") {
        if (args.empty()) {
            std::cerr << "usage: dist-cli models import <hf-repo-id>\n";
            return 1;
        }
        std::string body = "{\"repo_id\":\"" + json_escape(args[0]) + "\"}";
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, "/api/hf/import", "POST", body,
                              {bearer}, r, err) || r.status >= 300) {
            std::cerr << "import failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        std::cout << "import queued: job_id="
                  << dc::json_peek_string(r.body, "job_id")
                  << "  repo=" << args[0] << "\n";
        return 0;
    }

    if (sub == "search") {
        if (args.empty()) {
            std::cerr << "usage: dist-cli models search <query> [--tag X] [--library Y] [--limit N]\n";
            return 1;
        }
        std::string q = args[0];
        std::string tag     = opt(args, "--tag");
        std::string lib     = opt(args, "--library");
        std::string limit   = opt(args, "--limit", "20");
        auto urlenc = [](const std::string& s) {
            std::string o; o.reserve(s.size());
            for (unsigned char c : s) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
                    o += static_cast<char>(c);
                } else {
                    char b[4]; std::snprintf(b, sizeof(b), "%%%02X", c);
                    o += b;
                }
            }
            return o;
        };
        std::string path = "/api/hf/search?q=" + urlenc(q) + "&limit=" + limit;
        if (!tag.empty()) path += "&tags=" + urlenc(tag);
        if (!lib.empty()) path += "&library=" + urlenc(lib);

        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, path, "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "search failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::string arr;
        if (!find_top_array(r.body, "results", arr)) arr = r.body;
        std::printf("  %-60s  %-9s  %-7s  %s\n", "ID", "DOWNLOADS", "LIKES", "LIBRARY");
        for (const auto& obj : json_array_objects(arr)) {
            std::printf("  %-60s  %-9s  %-7s  %s\n",
                        dc::json_peek_string(obj, "id").c_str(),
                        dc::json_peek_int(obj, "downloads").c_str(),
                        dc::json_peek_int(obj, "likes").c_str(),
                        dc::json_peek_string(obj, "library").c_str());
        }
        return 0;
    }

    if (sub == "discover") {
        // Trending GGUF browse, cross-referenced with what's already imported
        // so the user sees green ✓ next to repos they don't need to download
        // again.  Defaults: library=gguf, sort=trendingScore, limit=20.
        std::string q     = args.empty() ? std::string{} : args[0];
        std::string lib   = opt(args, "--library", "gguf");
        std::string sort_ = opt(args, "--sort", "trendingScore");
        std::string limit = opt(args, "--limit", "20");
        auto urlenc = [](const std::string& s) {
            std::string o; o.reserve(s.size());
            for (unsigned char c : s) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
                    o += static_cast<char>(c);
                } else {
                    char b[4]; std::snprintf(b, sizeof(b), "%%%02X", c);
                    o += b;
                }
            }
            return o;
        };

        // Snapshot the registered set first so we can mark hits.
        std::set<std::string> imported;
        {
            dc::HttpResp lr; std::string lerr;
            if (dc::http_request(ctx.server_url, "/api/models", "GET", "",
                                 {bearer}, lr, lerr) && lr.status == 200) {
                std::string lst;
                if (!find_top_array(lr.body, "models", lst)) lst = lr.body;
                for (const auto& obj : json_array_objects(lst)) {
                    std::string id = dc::json_peek_string(obj, "repo_id");
                    if (id.empty()) id = dc::json_peek_string(obj, "name");
                    for (auto& c : id) c = static_cast<char>(std::tolower(c));
                    if (!id.empty()) imported.insert(id);
                }
            }
        }

        std::string path = "/api/hf/search?q=" + urlenc(q)
                         + "&limit=" + limit
                         + "&sort=" + urlenc(sort_);
        if (!lib.empty()) path += "&library=" + urlenc(lib);

        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, path, "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "discover failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::string arr;
        if (!find_top_array(r.body, "results", arr)) arr = r.body;
        std::printf("  %-2s  %-54s  %-9s  %-6s  %s\n",
                    "", "ID", "DOWNLOADS", "LIKES", "LIBRARY");
        for (const auto& obj : json_array_objects(arr)) {
            std::string id = dc::json_peek_string(obj, "id");
            std::string lower = id;
            for (auto& c : lower) c = static_cast<char>(std::tolower(c));
            bool have = imported.count(lower) > 0;
            std::printf("  %-2s  %-54s  %-9s  %-6s  %s\n",
                        have ? "✓" : " ",
                        id.c_str(),
                        dc::json_peek_int(obj, "downloads").c_str(),
                        dc::json_peek_int(obj, "likes").c_str(),
                        dc::json_peek_string(obj, "library").c_str());
        }
        std::printf("\n  ✓ = already imported.  dist-cli models import <id> to pull.\n");
        return 0;
    }

    std::cerr << "unknown models subcommand: " << sub << "\n";
    return 1;
}

int cmd_rigs(const dc::AuthCtx& ctx, std::vector<std::string> args) {
    if (args.empty()) {
        std::cerr << "usage: dist-cli rigs <list|watch|forget AGENT_ID>\n";
        return 1;
    }
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;

    auto fetch = [&](dc::HttpResp& r, std::string& err) {
        return dc::http_request(ctx.server_url, "/api/me/rigs", "GET", "",
                                {bearer}, r, err) && r.status == 200;
    };

    auto print_table = [](const std::string& body) {
        std::string arr;
        if (!find_top_array(body, "rigs", arr)) arr = body;
        auto human_bytes = [](long long n) -> std::string {
            if (n <= 0) return "—";
            const char* units[] = {"B","KB","MB","GB","TB"};
            double v = static_cast<double>(n);
            int u = 0;
            while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
            char buf[32];
            if (u == 0) std::snprintf(buf, sizeof(buf), "%lld%s", n, units[u]);
            else        std::snprintf(buf, sizeof(buf), "%.1f%s", v, units[u]);
            return buf;
        };
        std::printf("  %-22s  %-9s  %-12s  %-7s  %-7s  %s\n",
                    "AGENT_ID", "STATUS", "GPU", "VRAM", "SLOTS", "HOSTNAME");
        int n_rows = 0;
        for (const auto& obj : json_array_objects(arr)) {
            ++n_rows;
            // health is the server-classified bucket; falls back to online flag.
            std::string st = dc::json_peek_string(obj, "health");
            if (st.empty()) {
                st = dc::json_peek_string(obj, "online") == "true" ? "online" : "offline";
            }
            std::string inflight = dc::json_peek_int(obj, "inflight");
            std::string maxconc  = dc::json_peek_int(obj, "max_concurrent");
            if (inflight.empty()) inflight = "0";
            if (maxconc.empty() || maxconc == "0") maxconc = "—";
            std::string slots = inflight + "/" + maxconc;
            long long vram = 0;
            try { vram = std::stoll(dc::json_peek_int(obj, "vram_total")); } catch (...) {}
            std::string gpu = dc::json_peek_string(obj, "gpu_model");
            if (gpu.empty()) gpu = "—";
            if (gpu.size() > 12) gpu = gpu.substr(0, 11) + "…";
            std::printf("  %-22s  %-9s  %-12s  %-7s  %-7s  %s\n",
                        dc::json_peek_string(obj, "agent_id").c_str(),
                        st.c_str(),
                        gpu.c_str(),
                        human_bytes(vram).c_str(),
                        slots.c_str(),
                        dc::json_peek_string(obj, "hostname").c_str());
        }
        if (n_rows == 0) {
            std::printf("  (no rigs paired — run `dist-node login` on a GPU box)\n");
        }
    };

    if (args[0] == "list") {
        dc::HttpResp r; std::string err;
        if (!fetch(r, err)) {
            std::cerr << "list failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        print_table(r.body);
        return 0;
    }

    if (args[0] == "watch") {
        std::signal(SIGINT, on_sigint);
        while (!g_quit.load()) {
            dc::HttpResp r; std::string err;
            if (fetch(r, err)) {
                std::printf("\x1b[2J\x1b[H");          // clear + home
                std::printf("dist-cli rigs watch   %s\n", ctx.server_url.c_str());
                print_table(r.body);
                std::fflush(stdout);
            } else {
                std::cerr << "[poll] " << err << "\n";
            }
            for (int i = 0; i < 20 && !g_quit.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return 0;
    }

    if (args[0] == "forget") {
        if (args.size() < 2) {
            std::cerr << "usage: dist-cli rigs forget <AGENT_ID>\n"
                         "  (use `dist-cli rigs list` to find the agent_id)\n";
            return 1;
        }
        const std::string agent_id = args[1];
        const std::string path = "/api/rigs/" + agent_id;
        dc::HttpResp r; std::string err;
        if (!dc::http_request(ctx.server_url, path, "DELETE", "",
                              {bearer}, r, err)) {
            std::cerr << "[forget] request failed: " << err << "\n";
            return 1;
        }
        if (r.status == 409) {
            std::cerr << "[forget] rig is online — run `dist-node logout` on it first, "
                         "or wait for the heartbeat to lapse.\n";
            return 1;
        }
        if (r.status != 200) {
            std::cerr << "[forget] failed (" << r.status << "): " << r.body << "\n";
            return 1;
        }
        if (dc::json_peek_string(r.body, "deleted") == "true") {
            std::cout << "  ✓ forgot rig " << agent_id << "\n";
        } else {
            std::cout << "  (rig " << agent_id << " not found — nothing to do)\n";
        }
        return 0;
    }

    std::cerr << "unknown rigs subcommand: " << args[0] << "\n";
    return 1;
}

int cmd_logs(const dc::AuthCtx& ctx, std::vector<std::string> args) {
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;
    bool follow = flag(args, "--follow") || flag(args, "-f");
    std::string tail = opt(args, "--tail", "50");

    auto fetch_once = [&]() -> int {
        dc::HttpResp r; std::string err;
        std::string path = "/api/inference_log?limit=" + tail;
        if (!dc::http_request(ctx.server_url, path, "GET", "",
                              {bearer}, r, err) || r.status != 200) {
            std::cerr << "logs failed (" << r.status << "): "
                      << (err.empty() ? r.body : err) << "\n";
            return 1;
        }
        if (flag(args, "--json")) { std::cout << r.body << "\n"; return 0; }
        std::string arr;
        if (!find_top_array(r.body, "log", arr) &&
            !find_top_array(r.body, "entries", arr)) arr = r.body;
        for (const auto& obj : json_array_objects(arr)) {
            std::printf("[%s] %-8s  %-20s  status=%s  prompt_tokens=%s  completion_tokens=%s\n",
                        dc::json_peek_string(obj, "ts").c_str(),
                        dc::json_peek_string(obj, "model").c_str(),
                        dc::json_peek_string(obj, "agent_id").c_str(),
                        dc::json_peek_int(obj, "status").c_str(),
                        dc::json_peek_int(obj, "prompt_tokens").c_str(),
                        dc::json_peek_int(obj, "completion_tokens").c_str());
        }
        return 0;
    };

    if (!follow) return fetch_once();

    std::signal(SIGINT, on_sigint);
    while (!g_quit.load()) {
        if (fetch_once() != 0) return 1;
        for (int i = 0; i < 20 && !g_quit.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!g_quit.load()) std::printf("\x1b[2J\x1b[H");
    }
    return 0;
}

// ── Login / logout ───────────────────────────────────────────────────────

// Default control-plane URL. Same fallback dist-node uses so a user can run
// either binary first and the other inherits the state.
std::string default_api_url() {
    if (const char* s = std::getenv("DIST_API_URL"); s && *s) return s;
    std::string saved = dc::state_read("agent.api_url");
    if (!saved.empty()) return saved;
    return "https://distpool-server.gentlegrass-360d3389.centralindia.azurecontainerapps.io";
}

void open_browser(const std::string& url) {
#if defined(__APPLE__)
    std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
#elif defined(_WIN32)
    std::string cmd = "start \"\" \"" + url + "\"";
#else
    std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
    (void)std::system(cmd.c_str());
}

int cmd_login(std::vector<std::string> args) {
    std::string api_url = opt(args, "--server");
    if (api_url.empty()) api_url = opt(args, "-s");
    if (api_url.empty()) api_url = default_api_url();

    char hostname[256] = {};
#ifndef _WIN32
    ::gethostname(hostname, sizeof(hostname));
#else
    std::strncpy(hostname, "windows-host", sizeof(hostname) - 1);
#endif

    // n_gpus=-1, vram_bytes=-1 — sentinel telling the server this is a
    // management seat, not a compute rig.  The server's upsert in
    // device_code.go preserves any existing rig capabilities instead of
    // overwriting them, so an operator login on a box that's already running
    // dist-node won't zero out the rig's advertised GPU count.
    std::ostringstream body;
    body << "{\"hostname\":\"" << json_escape(hostname) << "\","
         << "\"n_gpus\":-1,\"vram_bytes\":-1}";

    dc::HttpResp rsp; std::string err;
    if (!dc::http_request(api_url, "/api/device/code", "POST", body.str(),
                          {}, rsp, err) || rsp.status != 200) {
        std::cerr << "[login] mint device code failed: " << err
                  << " status=" << rsp.status << " body=" << rsp.body << "\n";
        return 1;
    }
    std::string device_code = dc::json_peek_string(rsp.body, "device_code");
    std::string user_code   = dc::json_peek_string(rsp.body, "user_code");
    std::string verif       = dc::json_peek_string(rsp.body, "verification_url");
    std::string verif_full  = dc::json_peek_string(rsp.body, "verification_url_complete");
    if (device_code.empty() || user_code.empty() || verif_full.empty()) {
        std::cerr << "[login] malformed response: " << rsp.body << "\n";
        return 1;
    }

    std::cout << "\n  ┌──────────────────────────────────────────────────────┐\n"
              << "  │  Visit:  " << verif
              << std::string(std::max<int>(0, 41 - (int)verif.size()), ' ') << " │\n"
              << "  │  Code:   " << user_code
              << std::string(std::max<int>(0, 41 - (int)user_code.size()), ' ') << " │\n"
              << "  └──────────────────────────────────────────────────────┘\n\n"
              << "  Opening browser… (if it doesn't open, paste the URL above)\n\n";
    open_browser(verif_full);

    std::ostringstream pollbody;
    pollbody << "{\"device_code\":\"" << json_escape(device_code) << "\"}";
    auto t0 = std::chrono::steady_clock::now();
    std::string agent_id, agent_key, server;
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        dc::HttpResp pr; std::string perr;
        if (!dc::http_request(api_url, "/api/device/token", "POST", pollbody.str(),
                              {}, pr, perr)) {
            std::cerr << "[login] poll error: " << perr << " — retrying\n";
            continue;
        }
        if (pr.status == 428) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "  Waiting for approval… (" << el << "s)\r" << std::flush;
            continue;
        }
        if (pr.status == 410) {
            std::cerr << "\n[login] code expired — run `dist-cli login` again\n";
            return 1;
        }
        if (pr.status != 200) {
            std::cerr << "\n[login] poll failed: status=" << pr.status
                      << " body=" << pr.body << "\n";
            return 1;
        }
        agent_id  = dc::json_peek_string(pr.body, "agent_id");
        agent_key = dc::json_peek_string(pr.body, "agent_key");
        server    = dc::json_peek_string(pr.body, "server");
        if (agent_id.empty() || agent_key.empty()) {
            std::cerr << "\n[login] malformed token reply: " << pr.body << "\n";
            return 1;
        }
        break;
    }

    // Persist exactly what dist-node would have written, so the same state dir
    // works for both binaries. `agent.server` may be empty for non-compute
    // seats but we save whatever the server returned.
    bool ok = dc::state_write("agent.id",      agent_id)
           && dc::state_write("agent.key",     agent_key)
           && dc::state_write("agent.api_url", api_url);
    if (!server.empty()) ok = ok && dc::state_write("agent.server", server);
    if (!ok) {
        std::cerr << "\n[login] could not persist to " << dc::state_dir() << "\n";
        return 1;
    }

    // Mint the bearer api_key now so the next command doesn't pay the round-trip.
    const std::string label = "dist-cli/" + agent_id;
    const std::string mint_body = "{\"label\":\"" + json_escape(label) + "\"}";
    dc::HttpResp mr; std::string merr;
    if (dc::http_request(api_url, "/api/agent/api_key", "POST", mint_body,
                         {"Authorization: Bearer " + agent_key}, mr, merr) &&
        mr.status == 200) {
        std::string k = dc::json_peek_string(mr.body, "key");
        if (k.empty()) k = dc::json_peek_string(mr.body, "api_key");
        if (k.empty()) k = dc::json_peek_string(mr.body, "token");
        if (!k.empty()) dc::state_write("agent.api_key", k);
    }

    std::cout << "\n  ✓ Logged in. agent_id=" << agent_id
              << "\n    Saved to " << dc::state_dir() << "\n\n";
    return 0;
}

int cmd_logout(const std::vector<std::string>&) {
    // Truncate rather than delete — keeps the state dir well-formed and lets
    // a subsequent `dist-cli login` reuse the same paths.
    for (const char* k : {"agent.id", "agent.key", "agent.api_key",
                          "agent.server", "agent.api_url"}) {
        dc::state_write(k, "");
    }
    std::cout << "logged out — credentials cleared from " << dc::state_dir() << "\n";
    return 0;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "%s — operator CLI for distpool\n\n"
        "  %s login [--server URL]            Browser device-code login.\n"
        "  %s logout                          Wipe local credentials.\n"
        "  %s status                          Pairing + auth info.\n"
        "  %s pools list|create|join|members|invite|kick ...\n"
        "  %s models list|import|search|discover ...\n"
        "  %s rigs list|watch|forget <AGENT_ID>\n"
        "  %s logs [--follow] [--tail N]\n"
        "  %s top                             Live TUI dashboard.\n"
        "  %s split <pool-id>                 Visual layer/VRAM splitter.\n"
        "\n"
        "All commands accept --json to dump the raw server response.\n"
        "If dist-node is installed on the same box it shares this auth.\n",
        prog, prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    dist::net_startup();

    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string sub = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);

    // Login/logout don't need (and must not require) pre-existing auth.
    if (sub == "login")  return cmd_login(args);
    if (sub == "logout") return cmd_logout(args);

    dc::AuthCtx ctx;
    std::string err;
    if (!dc::load_auth(ctx, err)) {
        std::fprintf(stderr, "auth: %s\n", err.c_str());
        std::fprintf(stderr, "hint: run `%s login` to authenticate.\n", argv[0]);
        return 2;
    }

    if (sub == "status") return cmd_status(ctx, args);
    if (sub == "pools")  return cmd_pools(ctx, args);
    if (sub == "models") return cmd_models(ctx, args);
    if (sub == "rigs")   return cmd_rigs(ctx, args);
    if (sub == "logs")   return cmd_logs(ctx, args);
    if (sub == "top")    return dist_cli_run_top(ctx);
    if (sub == "split") {
        if (args.empty()) {
            std::fprintf(stderr, "usage: dist-cli split <pool-id>\n");
            return 1;
        }
        return dist_cli_run_split(ctx, std::atoll(args[0].c_str()));
    }

    std::fprintf(stderr, "unknown subcommand: %s\n\n", sub.c_str());
    usage(argv[0]);
    return 1;
}
