// dist-cli split — visual layer-to-rig assignment for one pool.
//
// Renders the auto-computed pipeline plan as a horizontal layer ribbon with
// one row per stage, then lets the operator nudge boundaries and reassign
// rigs.  The view is purely advisory today — the planner still picks at
// dispatch time — but the edited plan is written to
//   <state_dir>/plans/pool-<id>.json
// so a future planner override (or a manual `dist-cli plan apply`) can pick
// it up without re-typing layer ranges.
//
// VRAM estimate = (size_bytes / n_layers) × layers_in_stage × 1.10  (10% KV
// + activations slack).  size_bytes comes from /api/models; if it's zero
// (older model, backfill missed it) the VRAM column reads "—".
//
// Pressing `w` PUTs the current layout to /api/pools/{id}/plan.  The server
// validates contiguity + membership and persists it in pools.plan_override;
// the planner consumes it at next dispatch.  A local copy is also written
// to <state_dir>/plans/pool-<id>.json for offline review.
//
// Keybinds (active when a stage row is selected):
//   ↑/↓        select stage
//   ←/→        shrink/grow the stage's right boundary by one layer
//   Shift+←/→  shrink/grow the LEFT boundary
//   r          reassign this stage to the next eligible rig
//   w          write current plan to disk
//   q / Esc    quit (prompts if there are unsaved edits)
//
// The whole file is conditionally compiled when DIST_BUILD_CLI_TUI is on.
// Without FTXUI we expose a non-interactive text fallback that prints the
// plan + estimates and exits.

#include "cli_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace dc = dist::cli;

namespace {

// ── JSON helpers (duplicated locally — keeps the surface tight) ─────────

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

bool find_top_array(const std::string& body, const std::string& key, std::string& slice) {
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

std::string human_bytes(int64_t b) {
    if (b <= 0) return "—";
    const char* u[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(b);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, u[i]);
    return buf;
}

struct Rig {
    std::string agent_id;
    std::string hostname;
    int64_t     vram_total = 0;
    int64_t     vram_free  = 0;
    bool        online     = false;
    std::string nat_type;
    int         inflight = 0;
    int         max_concurrent = 0;
};

struct Stage {
    int         stage_idx = 0;
    int         layer_lo  = 0;
    int         layer_hi  = 0;       // inclusive
    std::string agent_id;            // public-fingerprinted (from /topology)
    std::string hostname;
    int64_t     est_bytes = 0;
};

struct Pool {
    int64_t            id = 0;
    std::string        model;
    int                n_layers = 0;
    int64_t            model_bytes = 0;
    std::vector<Stage> stages;
    bool               dirty = false; // user edited
};

// Match a topology stage back to a meRig.  /api/me/rigs returns the raw
// agent_id; /topology returns a public-fingerprinted one to avoid leaking
// rig identities to other pool members.  We can't reverse the fingerprint,
// so we fall back to hostname matching — works because hostnames within a
// user's fleet are usually unique enough for this view.
const Rig* find_rig(const std::vector<Rig>& rigs, const Stage& st) {
    for (const auto& r : rigs)
        if (!st.hostname.empty() && r.hostname == st.hostname) return &r;
    for (const auto& r : rigs)
        if (!st.agent_id.empty() && r.agent_id == st.agent_id) return &r;
    return nullptr;
}

// Fetch pool topology + my rigs + the pool's bound model size.
bool fetch_state(const dc::AuthCtx& ctx, int64_t pool_id, Pool& pool,
                 std::vector<Rig>& rigs, std::string& err) {
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;

    // Rigs first — used by the stage→rig match.
    {
        dc::HttpResp r; std::string e;
        if (!dc::http_request(ctx.server_url, "/api/me/rigs", "GET", "",
                              {bearer}, r, e) || r.status != 200) {
            err = "rigs: " + (e.empty() ? r.body : e);
            return false;
        }
        std::string arr;
        if (!find_top_array(r.body, "rigs", arr)) arr = r.body;
        for (const auto& obj : json_array_objects(arr)) {
            Rig rg;
            rg.agent_id  = dc::json_peek_string(obj, "agent_id");
            rg.hostname  = dc::json_peek_string(obj, "hostname");
            rg.vram_total = std::atoll(dc::json_peek_int(obj, "vram_total").c_str());
            rg.vram_free  = std::atoll(dc::json_peek_int(obj, "vram_free").c_str());
            rg.nat_type   = dc::json_peek_string(obj, "nat_type");
            rg.online     = dc::json_peek_string(obj, "online") == "true" ||
                            dc::json_peek_int(obj, "online") == "1";
            rg.inflight   = std::atoi(dc::json_peek_int(obj, "inflight").c_str());
            rg.max_concurrent = std::atoi(dc::json_peek_int(obj, "max_concurrent").c_str());
            rigs.push_back(std::move(rg));
        }
    }

    // Topology — the current automatic plan.
    {
        dc::HttpResp r; std::string e;
        std::string path = "/api/pools/" + std::to_string(pool_id) + "/topology";
        if (!dc::http_request(ctx.server_url, path, "GET", "",
                              {bearer}, r, e) || r.status != 200) {
            err = "topology: " + (e.empty() ? r.body : e);
            return false;
        }
        pool.id       = pool_id;
        pool.model    = dc::json_peek_string(r.body, "model");
        pool.n_layers = std::atoi(dc::json_peek_int(r.body, "n_layers").c_str());
        std::string arr;
        if (!find_top_array(r.body, "stages", arr)) arr = "[]";
        for (const auto& obj : json_array_objects(arr)) {
            Stage st;
            st.stage_idx = std::atoi(dc::json_peek_int(obj, "stage_idx").c_str());
            st.layer_lo  = std::atoi(dc::json_peek_int(obj, "layer_lo").c_str());
            st.layer_hi  = std::atoi(dc::json_peek_int(obj, "layer_hi").c_str());
            st.agent_id  = dc::json_peek_string(obj, "agent_id");
            st.hostname  = dc::json_peek_string(obj, "hostname");
            pool.stages.push_back(std::move(st));
        }
        std::sort(pool.stages.begin(), pool.stages.end(),
                  [](const Stage& a, const Stage& b){ return a.stage_idx < b.stage_idx; });

        // /topology returns publicAgentID-fingerprinted agent_ids so members
        // of one pool can't enumerate another user's rig fleet.  We need the
        // raw agent_id to push a manual plan back to the server — recover it
        // by matching hostname against /api/me/rigs.  Stages whose hostnames
        // don't match any owned rig keep the fingerprinted id; saving such a
        // plan will fail server-side validation, which is the correct outcome
        // (you can't pin a rig you don't own).
        for (auto& s : pool.stages) {
            if (s.hostname.empty()) continue;
            for (const auto& r : rigs) {
                if (r.hostname == s.hostname) { s.agent_id = r.agent_id; break; }
            }
        }
    }

    // Model size — needed for per-layer bytes.  Best-effort; missing field
    // just leaves the estimate at zero.
    {
        dc::HttpResp r; std::string e;
        if (dc::http_request(ctx.server_url, "/api/models", "GET", "",
                             {bearer}, r, e) && r.status == 200) {
            std::string arr;
            if (!find_top_array(r.body, "models", arr)) arr = r.body;
            for (const auto& obj : json_array_objects(arr)) {
                if (dc::json_peek_string(obj, "name") == pool.model) {
                    pool.model_bytes = std::atoll(dc::json_peek_int(obj, "size_bytes").c_str());
                    if (pool.n_layers == 0) {
                        pool.n_layers = std::atoi(dc::json_peek_int(obj, "n_layers").c_str());
                    }
                    break;
                }
            }
        }
    }

    return true;
}

int64_t per_layer_bytes(const Pool& p) {
    if (p.n_layers <= 0 || p.model_bytes <= 0) return 0;
    return p.model_bytes / p.n_layers;
}

void recompute_estimates(Pool& p) {
    int64_t plb = per_layer_bytes(p);
    for (auto& s : p.stages) {
        int n = std::max(0, s.layer_hi - s.layer_lo + 1);
        // 10% slack for KV cache + activations.
        s.est_bytes = static_cast<int64_t>(plb * n * 1.10);
    }
}

// Server-facing payload — stage list only.  Hostname/est_bytes are dropped;
// the server's planner re-derives those at dispatch time and persisting them
// in plan_override would let an old hostname mismatch a renamed rig.
std::string plan_payload(const Pool& p) {
    std::ostringstream o;
    o << "{\"stages\":[";
    for (size_t i = 0; i < p.stages.size(); ++i) {
        const auto& s = p.stages[i];
        if (i) o << ",";
        o << "{\"stage_idx\":" << s.stage_idx
          << ",\"layer_lo\":" << s.layer_lo
          << ",\"layer_hi\":" << s.layer_hi
          << ",\"agent_id\":\"" << s.agent_id << "\"}";
    }
    o << "]}";
    return o.str();
}

// Local cache — handy when the server is unreachable, or for diffs.
void write_local_plan(const Pool& p) {
    std::error_code ec;
    std::string dir = dc::state_dir() + "/plans";
    std::filesystem::create_directories(dir, ec);
    if (ec) return;
    std::string dest = dir + "/pool-" + std::to_string(p.id) + ".json";
    std::ofstream f(dest, std::ios::binary | std::ios::trunc);
    if (!f.good()) return;
    f << plan_payload(p);
}

bool push_plan(const dc::AuthCtx& ctx, const Pool& p, std::string& err) {
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;
    std::string path = "/api/pools/" + std::to_string(p.id) + "/plan";
    dc::HttpResp r; std::string e;
    if (!dc::http_request(ctx.server_url, path, "PUT", plan_payload(p),
                          {bearer}, r, e)) {
        err = "transport: " + e;
        return false;
    }
    if (r.status >= 300) {
        err = "server " + std::to_string(r.status) + ": " + r.body;
        return false;
    }
    write_local_plan(p);
    return true;
}

bool clear_plan(const dc::AuthCtx& ctx, int64_t pool_id, std::string& err) {
    const std::string bearer = "Authorization: Bearer " + ctx.api_key;
    std::string path = "/api/pools/" + std::to_string(pool_id) + "/plan";
    dc::HttpResp r; std::string e;
    if (!dc::http_request(ctx.server_url, path, "DELETE", "",
                          {bearer}, r, e)) {
        err = "transport: " + e;
        return false;
    }
    if (r.status >= 300) {
        err = "server " + std::to_string(r.status) + ": " + r.body;
        return false;
    }
    return true;
}

// ── Text fallback (always available) ────────────────────────────────────

void render_text(const Pool& p, const std::vector<Rig>& rigs) {
    std::printf("\npool %lld   model %s   layers %d   size %s\n",
                static_cast<long long>(p.id),
                p.model.empty() ? "(unbound)" : p.model.c_str(),
                p.n_layers,
                human_bytes(p.model_bytes).c_str());
    int64_t plb = per_layer_bytes(p);
    std::printf("per-layer ≈ %s\n\n", human_bytes(plb).c_str());

    const int BAR_W = 60;
    for (const auto& s : p.stages) {
        char bar[BAR_W + 1];
        for (int i = 0; i < BAR_W; ++i) {
            int layer = p.n_layers > 0 ? (i * p.n_layers) / BAR_W : 0;
            bar[i] = (layer >= s.layer_lo && layer <= s.layer_hi) ? '#' : '.';
        }
        bar[BAR_W] = 0;
        const Rig* rg = find_rig(rigs, s);
        int n = s.layer_hi - s.layer_lo + 1;
        std::printf("[s%d]  %s  layers %d-%d (%d)\n",
                    s.stage_idx, bar, s.layer_lo, s.layer_hi, n);
        if (rg) {
            const char* fit = (s.est_bytes <= rg->vram_free)        ? "ok"
                            : (s.est_bytes <= rg->vram_free * 11/10)? "tight"
                                                                    : "overflow";
            std::printf("       %-24s  est %s / free %s  %s\n",
                        rg->hostname.c_str(),
                        human_bytes(s.est_bytes).c_str(),
                        human_bytes(rg->vram_free).c_str(),
                        fit);
        } else {
            std::printf("       %-24s  est %s / free ?\n",
                        s.hostname.empty() ? "(unknown)" : s.hostname.c_str(),
                        human_bytes(s.est_bytes).c_str());
        }
    }
    std::printf("\n");
}

} // namespace

#ifdef DIST_HAVE_FTXUI

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

namespace {

Element layer_ribbon(const Pool& p, int sel_idx) {
    const int BAR_W = 64;
    Elements cells;
    cells.reserve(BAR_W);
    // Distinct hue per stage; sombre, low-saturation set.
    static const Color stage_palette[] = {
        Color::RGB(120, 145, 170),
        Color::RGB(165, 130, 110),
        Color::RGB(130, 160, 130),
        Color::RGB(160, 140, 165),
        Color::RGB(145, 155, 110),
        Color::RGB(115, 140, 155),
        Color::RGB(170, 150, 130),
        Color::RGB(125, 130, 150),
    };
    for (int i = 0; i < BAR_W; ++i) {
        int layer = p.n_layers > 0 ? (i * p.n_layers) / BAR_W : 0;
        int owner = -1;
        for (size_t s = 0; s < p.stages.size(); ++s) {
            if (layer >= p.stages[s].layer_lo && layer <= p.stages[s].layer_hi) {
                owner = static_cast<int>(s);
                break;
            }
        }
        Element cell;
        if (owner < 0) {
            cell = text("·") | color(Color::GrayDark);
        } else {
            Color c = stage_palette[owner % (sizeof(stage_palette)/sizeof(stage_palette[0]))];
            const char* glyph = (owner == sel_idx) ? "█" : "▇";
            cell = text(glyph) | color(c);
        }
        cells.push_back(cell);
    }
    return hbox(std::move(cells));
}

Element stage_row(const Pool& p, const std::vector<Rig>& rigs,
                  size_t idx, bool selected) {
    const auto& s = p.stages[idx];
    const Rig* rg = find_rig(rigs, s);

    int n = s.layer_hi - s.layer_lo + 1;

    std::string title = "stage " + std::to_string(s.stage_idx)
                      + "   layers " + std::to_string(s.layer_lo)
                      + "–" + std::to_string(s.layer_hi)
                      + "  (" + std::to_string(n) + ")";

    std::string rigname = rg ? rg->hostname
                             : (s.hostname.empty() ? "(no rig)" : s.hostname);

    std::string est = human_bytes(s.est_bytes);
    std::string free = rg ? human_bytes(rg->vram_free) : std::string("—");
    std::string totl = rg ? human_bytes(rg->vram_total) : std::string("—");

    Color fit_color = Color::GrayLight;
    std::string fit_tag = "·";
    if (rg && rg->vram_free > 0 && s.est_bytes > 0) {
        if (s.est_bytes <= rg->vram_free) {
            fit_color = Color::RGB(120, 170, 130); fit_tag = "fits";
        } else if (s.est_bytes <= rg->vram_free + (rg->vram_free / 10)) {
            fit_color = Color::RGB(190, 170, 110); fit_tag = "tight";
        } else {
            fit_color = Color::RGB(190, 120, 120); fit_tag = "overflow";
        }
    }

    // Per-stage utilization bar — how much of free VRAM the stage uses.
    Element util;
    {
        int64_t denom = rg ? rg->vram_free : 0;
        double frac = (denom > 0)
                    ? std::min(1.5, static_cast<double>(s.est_bytes) / denom)
                    : 0.0;
        const int UW = 16;
        Elements ucells;
        for (int i = 0; i < UW; ++i) {
            double t = (i + 1.0) / UW;
            const char* g = (t <= frac) ? "▰" : "▱";
            Color c = fit_color;
            if (t > 1.0) c = Color::RGB(190, 120, 120);
            ucells.push_back(text(g) | color(c));
        }
        util = hbox(std::move(ucells));
    }

    auto label = text(title);
    if (selected) label = label | bold | color(Color::White);
    else          label = label | color(Color::GrayLight);

    Element row = vbox({
        hbox({
            text(selected ? "▶ " : "  "),
            label | size(WIDTH, EQUAL, 38),
            text(rigname) | size(WIDTH, EQUAL, 22) | color(Color::GrayLight),
            text("est ") | color(Color::GrayDark),
            text(est)    | size(WIDTH, EQUAL, 9)  | color(fit_color),
            text(" / free ") | color(Color::GrayDark),
            text(free)   | size(WIDTH, EQUAL, 9)  | color(Color::GrayLight),
            text("  ") ,
            util,
            text("  "),
            text(fit_tag) | color(fit_color),
        }),
    });
    return row;
}

} // namespace

int dist_cli_run_split(const dc::AuthCtx& ctx, int64_t pool_id) {
    Pool             pool;
    std::vector<Rig> rigs;
    std::string      err;
    if (!fetch_state(ctx, pool_id, pool, rigs, err)) {
        std::fprintf(stderr, "split: %s\n", err.c_str());
        return 1;
    }
    if (pool.stages.empty()) {
        std::fprintf(stderr, "split: pool %lld has no stages — is the pool bound to a model with rigs joined?\n",
                     static_cast<long long>(pool_id));
        return 1;
    }
    if (pool.n_layers <= 0) {
        std::fprintf(stderr, "split: pool's model has n_layers=0 — cannot edit.\n");
        return 1;
    }
    recompute_estimates(pool);

    auto screen = ScreenInteractive::Fullscreen();
    int  sel    = 0;
    std::atomic<bool> quit{false};
    std::string status_msg = "↑↓ stage   ←→ right edge   h/l left edge   r reassign   w push plan   c clear   q quit";

    // Build a candidate rig list (online, has VRAM) so 'r' rotates through them.
    auto candidates = [&] {
        std::vector<size_t> v;
        for (size_t i = 0; i < rigs.size(); ++i)
            if (rigs[i].vram_total > 0) v.push_back(i);
        return v;
    }();

    auto reassign = [&](Stage& s) {
        if (candidates.empty()) return;
        // Find current rig in candidates, advance one.
        size_t cur = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (rigs[candidates[i]].hostname == s.hostname) { cur = i; break; }
        }
        size_t next = (cur == candidates.size()) ? 0 : (cur + 1) % candidates.size();
        s.hostname = rigs[candidates[next]].hostname;
        s.agent_id = rigs[candidates[next]].agent_id;  // raw; only the saved plan sees it
        pool.dirty = true;
    };

    auto nudge = [&](int stage_i, int side, int delta) {
        // side: -1 = left edge (lo), +1 = right edge (hi)
        if (stage_i < 0 || stage_i >= (int)pool.stages.size()) return;
        Stage& s = pool.stages[stage_i];
        if (side > 0) {
            int new_hi = s.layer_hi + delta;
            if (new_hi < s.layer_lo) return;
            if (new_hi >= pool.n_layers) return;
            // Push next stage's lo to match.
            if (stage_i + 1 < (int)pool.stages.size()) {
                Stage& nx = pool.stages[stage_i + 1];
                int new_nx_lo = new_hi + 1;
                if (new_nx_lo > nx.layer_hi) return; // would empty next stage
                s.layer_hi = new_hi;
                nx.layer_lo = new_nx_lo;
            } else {
                s.layer_hi = new_hi;
            }
        } else {
            int new_lo = s.layer_lo + delta;
            if (new_lo > s.layer_hi) return;
            if (new_lo < 0) return;
            if (stage_i - 1 >= 0) {
                Stage& pv = pool.stages[stage_i - 1];
                int new_pv_hi = new_lo - 1;
                if (new_pv_hi < pv.layer_lo) return;
                s.layer_lo = new_lo;
                pv.layer_hi = new_pv_hi;
            } else {
                s.layer_lo = new_lo;
            }
        }
        recompute_estimates(pool);
        pool.dirty = true;
    };

    auto render = Renderer([&] {
        // Header.
        int64_t plb = per_layer_bytes(pool);
        std::string header_l = "dist-cli split   pool " + std::to_string(pool.id);
        std::string header_r = pool.model + "   " + std::to_string(pool.n_layers)
                             + " layers   " + human_bytes(pool.model_bytes)
                             + "   ≈ " + human_bytes(plb) + "/layer";

        // Ribbon + axis.
        Element ribbon = layer_ribbon(pool, sel);
        std::string axis_l = "0";
        std::string axis_r = std::to_string(pool.n_layers - 1);
        Element axis = hbox({
            text(axis_l) | color(Color::GrayDark),
            filler(),
            text(axis_r) | color(Color::GrayDark),
        });

        // Stage rows.
        Elements rows;
        for (size_t i = 0; i < pool.stages.size(); ++i) {
            rows.push_back(stage_row(pool, rigs, i, (int)i == sel));
        }

        // Fleet summary at the bottom — VRAM totals for rigs not in the plan.
        Elements fleet;
        for (const auto& r : rigs) {
            bool in_plan = false;
            for (const auto& s : pool.stages)
                if (s.hostname == r.hostname) { in_plan = true; break; }
            std::string tag = in_plan ? "in plan " : "spare   ";
            Color col = in_plan ? Color::GrayLight : Color::GrayDark;
            std::string line = tag + r.hostname
                             + "   total " + human_bytes(r.vram_total)
                             + "   free "  + human_bytes(r.vram_free);
            fleet.push_back(text(line) | color(col));
        }

        Element body = vbox({
            hbox({
                text(header_l) | bold,
                filler(),
                text(header_r) | color(Color::GrayLight),
            }) | border,
            text(""),
            window(text(" layer ribbon ") | dim, vbox({ ribbon, axis })),
            text(""),
            window(text(" stages ") | dim, vbox(std::move(rows))),
            text(""),
            window(text(" fleet ") | dim, vbox(std::move(fleet))),
            filler(),
            hbox({
                text(pool.dirty ? "● unsaved" : "○ clean ")
                    | color(pool.dirty ? Color::RGB(230, 180, 110) : Color::GrayDark),
                text("   "),
                text(status_msg) | color(Color::GrayLight),
            }) | border,
        });
        return body;
    });

    auto comp = CatchEvent(render, [&](Event e) {
        if (e == Event::Character('q') || e == Event::Escape) {
            quit.store(true);
            screen.ExitLoopClosure()();
            return true;
        }
        if (e == Event::ArrowDown) {
            sel = std::min(sel + 1, (int)pool.stages.size() - 1);
            return true;
        }
        if (e == Event::ArrowUp) {
            sel = std::max(sel - 1, 0);
            return true;
        }
        if (e == Event::ArrowRight) { nudge(sel, +1, +1); return true; }
        if (e == Event::ArrowLeft)  { nudge(sel, +1, -1); return true; }
        // FTXUI doesn't expose Shift+Arrow uniformly; use h/l for left-edge nudges.
        if (e == Event::Character('l')) { nudge(sel, -1, +1); return true; }
        if (e == Event::Character('h')) { nudge(sel, -1, -1); return true; }
        if (e == Event::Character('r')) {
            if (sel >= 0 && sel < (int)pool.stages.size()) {
                reassign(pool.stages[sel]);
            }
            return true;
        }
        if (e == Event::Character('w')) {
            std::string werr;
            if (push_plan(ctx, pool, werr)) {
                status_msg = "plan saved to server (planner will use it on next dispatch)";
                pool.dirty = false;
            } else {
                status_msg = "push failed: " + werr;
            }
            return true;
        }
        if (e == Event::Character('c')) {
            std::string cerr;
            if (clear_plan(ctx, pool.id, cerr)) {
                status_msg = "override cleared — planner reverts to auto";
            } else {
                status_msg = "clear failed: " + cerr;
            }
            return true;
        }
        return false;
    });

    screen.Loop(comp);
    return 0;
}

#else // !DIST_HAVE_FTXUI

int dist_cli_run_split(const dc::AuthCtx& ctx, int64_t pool_id) {
    Pool             pool;
    std::vector<Rig> rigs;
    std::string      err;
    if (!fetch_state(ctx, pool_id, pool, rigs, err)) {
        std::fprintf(stderr, "split: %s\n", err.c_str());
        return 1;
    }
    recompute_estimates(pool);
    render_text(pool, rigs);
    std::fprintf(stderr, "(rebuild with -DDIST_BUILD_CLI_TUI=ON for the interactive splitter)\n");
    return 0;
}

#endif
