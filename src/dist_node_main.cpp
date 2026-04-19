/**
 * dist_node_main.cpp
 *
 * Entry point for a Node Agent — one per GPU server.
 *
 * Usage:
 *   dist-node -s COORDINATOR_HOST [options]
 *
 *   -s, --server HOST           Coordinator host (required)
 *   -p, --control-port PORT     Coordinator control port (default: 7700)
 *   -d, --data-port PORT        This node's data port (default: 7701)
 *   -g, --n-gpu-layers N        GPU layers (default: 999 = all)
 *   -c, --context N             Context window (default: 4096)
 *   -b, --batch N               Batch size (default: 512)
 *   --id NAME                   Node ID override (default: hostname:pid)
 *   -h, --help
 */

#include "node_agent.h"
#include "dist_ws_client.h"
#include "pp_engine.h"
#include "shard_download.h"
#include "gpu_lock.h"

#include "platform_compat.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -s COORDINATOR_HOST [options]\n"
        "   or: %s --pair distpool://pair?token=...&server=ws://...\n"
        "\n"
        "  -s, --server HOST         Coordinator host (direct-join mode)\n"
        "  -p, --control-port PORT   Coordinator control port (default: 7700)\n"
        "  -d, --data-port PORT      This node's data listen port (default: 7701)\n"
        "  -g, --n-gpu-layers N      GPU layers to offload (default: 999=all)\n"
        "  -c, --context N           Context window size (default: 4096)\n"
        "  -b, --batch N             Batch size (default: 512)\n"
        "      --id NAME             Override node ID\n"
        "      --token-id NAME       Auth token id (optional)\n"
        "      --token-secret HEX    Auth token secret, 64 hex chars (optional)\n"
        "      --pair URL            Deep-link pairing URL (distpool://…)\n"
        "  -h, --help\n",
        prog, prog);
}

// ─── Pair mode helpers ──────────────────────────────────────────────────────

// Parse a distpool://pair?token=T&server=WS URL.
// Returns true if the URL was well-formed.
static bool parse_pair_url(const std::string& url,
                            std::string& out_token,
                            std::string& out_server) {
    const std::string prefix = "distpool://pair?";
    if (url.rfind(prefix, 0) != 0) return false;
    std::string qs = url.substr(prefix.size());

    auto pull = [](const std::string& s, const std::string& key) -> std::string {
        size_t p = s.find(key + "=");
        if (p == std::string::npos) return "";
        size_t start = p + key.size() + 1;
        size_t end   = s.find('&', start);
        return s.substr(start, end == std::string::npos ? std::string::npos : end - start);
    };

    out_token  = pull(qs, "token");
    out_server = pull(qs, "server");
    return !out_token.empty() && !out_server.empty();
}

// Minimal JSON string escaper (ASCII, no control chars expected).
static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

static std::string default_agent_id() {
    char host[256] = {};
    ::gethostname(host, sizeof(host));
    return std::string(host) + ":" + std::to_string(::getpid());
}

// Enter pair mode: connect to the server WebSocket, send hello, and loop on
// status updates.  Returns process exit code.
static int run_pair_mode(const std::string& token, const std::string& server,
                         const std::string& agent_id_override,
                         int n_gpu_layers) {
    dist::WsClient ws;
    if (!ws.connect(server)) {
        std::cerr << "[pair] connect failed: " << ws.last_error() << "\n";
        return 1;
    }
    std::cout << "[pair] connected to " << server << "\n";

    // GPU / VRAM introspection is deferred (Milestone 2) — report 0 for M1.
    std::string agent_id = agent_id_override.empty() ? default_agent_id() : agent_id_override;
    char hostname[256] = {};
    ::gethostname(hostname, sizeof(hostname));

    std::ostringstream hello;
    hello << "{\"kind\":\"hello\","
          << "\"token\":\""    << json_escape(token)    << "\","
          << "\"agent_id\":\"" << json_escape(agent_id) << "\","
          << "\"hostname\":\"" << json_escape(hostname) << "\","
          << "\"n_gpus\":0,\"vram_bytes\":0}";

    if (!ws.send_text(hello.str())) {
        std::cerr << "[pair] send hello failed\n";
        return 1;
    }

    std::string msg;
    if (!ws.recv_text(msg)) {
        std::cerr << "[pair] no welcome received\n";
        return 1;
    }
    std::cout << "[pair] server reply: " << msg << "\n";
    if (msg.find("\"kind\":\"welcome\"") == std::string::npos) {
        std::cerr << "[pair] not welcomed — exiting\n";
        return 1;
    }

    std::cout << "[pair] paired successfully; entering heartbeat loop\n";

    // One pp_req per in-flight request assigned to this node.  Keyed on
    // req_id; populated on pp_route, mutated by ACTV frames, dropped when the
    // request finishes (terminal emits `done`, or an error short-circuits).
    struct pp_req {
        uint16_t req_id = 0;
        uint16_t stage_idx = 0;
        uint16_t stage_count = 0;
        bool is_first = false;
        bool is_last  = false;
        int32_t layer_lo = 0;
        int32_t layer_hi = 0;
        std::string shard_url;
        std::string shard_file;
        uint32_t max_tokens  = 128;
        int32_t  pos = 0;
        uint32_t out_tokens = 0;
        // Per-request seq id in the shared llama KV.  Assigned on first
        // pp_route; released on completion.
        int      seq_id = 0;
    };
    std::map<uint16_t, pp_req> reqs;

    // Pool of seq_ids.  llama_batch_init(n_seq_max=1) from load() means we
    // can only use seq_id 0 today; the batched-decode path walks this pool.
    // For a real multi-tenant deployment bump n_seq_max at load().
    int next_seq_id = 0;

    // One engine cached across requests that share the same shard content.
    // Signed URLs change every request, so we key on (file, layer_lo, layer_hi)
    // and reuse the downloaded file + loaded model between requests.
    std::unique_ptr<dist::PpEngine> engine;
    std::string engine_key;   // "<file>#<lo>-<hi>"
    std::string engine_path;  // local path of the downloaded shard

    // Cross-process lock over device 0.  Any two dist-node processes sharing
    // one CUDA device serialise their decodes through this flock; a single
    // ggml CUDA backend cannot host two independent llama_contexts doing
    // concurrent decodes without corrupting each other's state.
    dist::GpuLock gpu_lock;
    if (n_gpu_layers > 0) {
        if (!gpu_lock.open(0)) {
            std::cerr << "[pair] gpu lock open failed: "
                      << gpu_lock.last_error() << "\n";
        }
    }

    // ACTV framing (reused by both the recv path and the batched flush).
    const uint32_t MAGIC_INFR = 0x494E4652u;
    const uint32_t MAGIC_ACTV = 0x41435456u;
    auto be32 = [](const uint8_t* p) -> uint32_t {
        return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
    };
    auto be16 = [](const uint8_t* p) -> uint16_t {
        return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
    };
    auto put32 = [](uint8_t* p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24);
        p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);
        p[3] = (uint8_t)v;
    };
    auto put16 = [](uint8_t* p, uint16_t v) {
        p[0] = (uint8_t)(v >> 8);
        p[1] = (uint8_t)v;
    };
    auto send_actv = [&](uint8_t type, uint16_t req_id, uint16_t stage,
                         uint32_t tok_seq, uint8_t dtype, uint8_t flags,
                         const std::vector<uint32_t>& dims,
                         const std::string& payload) {
        const size_t fixed = 20;
        size_t dims_bytes = 4 * dims.size();
        size_t total_hdr = fixed + dims_bytes + 4;
        std::vector<uint8_t> out(total_hdr + payload.size());
        put32(out.data() + 0, MAGIC_ACTV);
        out[4] = 0x01;
        out[5] = type;
        put16(out.data() + 6, req_id);
        put16(out.data() + 8, stage);
        put32(out.data() + 10, tok_seq);
        out[14] = dtype;
        out[15] = (uint8_t)dims.size();
        out[16] = flags;
        out[17] = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
            put32(out.data() + fixed + 4 * i, dims[i]);
        }
        put32(out.data() + fixed + dims_bytes, (uint32_t)payload.size());
        std::memcpy(out.data() + total_hdr, payload.data(), payload.size());
        ws.send_binary(out.data(), out.size());
    };

    // Rolling throughput — updated by flush_pending below, consumed by the
    // heartbeat.  Declared up here so the flush lambda can capture them by
    // reference.
    uint64_t dec_tokens = 0;
    auto     dec_window_start = std::chrono::steady_clock::now();

    // Pending ACTVs accumulate here between recv→drain cycles, then flush().
    struct PendingActv {
        uint8_t  type = 0;
        uint16_t req_id = 0;
        uint16_t stage_in = 0;
        uint32_t tok_seq = 0;
        uint8_t  dtype = 0;
        uint8_t  flags = 0;
        std::vector<uint32_t> dims;
        std::string payload;
    };
    std::vector<PendingActv> pending_actvs;

    // One shot: group pending ACTVs by (stage-kind), run one batched decode
    // per group, emit output frames.  Splits by stage because stage 0 uses
    // the token path while later stages use the embedding path.
    auto flush_pending = [&]() {
        if (pending_actvs.empty()) return;
        if (!engine || !engine->ready()) {
            for (auto & pa : pending_actvs) {
                send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, "engine not ready");
            }
            pending_actvs.clear();
            return;
        }

        // Bucket entries into three groups.  `idx` tracks the seqs[] slot
        // we hand to the engine so we can locate its output back here.
        struct Stage0Entry { size_t qi; std::vector<int32_t> toks; };
        struct StageNEntry { size_t qi; };
        std::vector<Stage0Entry> s0_entries;
        std::vector<StageNEntry> mid_entries;
        std::vector<StageNEntry> term_entries;
        std::vector<dist::PpEngine::TokenSeq>  s0_seqs;
        std::vector<dist::PpEngine::EmbedSeq>  mid_seqs;
        std::vector<dist::PpEngine::EmbedSeq>  term_seqs;

        for (size_t qi = 0; qi < pending_actvs.size(); ++qi) {
            auto & pa = pending_actvs[qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) {
                std::cout << "[pair] flush: no route for req=" << pa.req_id << " — drop\n";
                continue;
            }
            auto & r = it->second;
            const bool is_prompt = (pa.flags & 0x01) != 0;
            // Reset per-request KV on new prompt.
            if (is_prompt) {
                engine->reset_seq(r.seq_id);
                r.pos = 0;
                if (r.is_last) r.out_tokens = 0;
            }

            if (r.is_first) {
                std::vector<int32_t> toks = engine->tokenize(pa.payload, is_prompt);
                if (toks.empty()) {
                    send_actv(0x04, pa.req_id, r.stage_idx, 0, 4, 0, {},
                              "tokenize failed");
                    continue;
                }
                Stage0Entry e; e.qi = qi; e.toks = std::move(toks);
                s0_entries.push_back(std::move(e));
                dist::PpEngine::TokenSeq s;
                s.seq_id = r.seq_id;
                s.pos    = r.pos;
                s.tokens = s0_entries.back().toks.data();
                s.n      = (int32_t) s0_entries.back().toks.size();
                s0_seqs.push_back(std::move(s));
            } else {
                if (pa.dtype != 0) {
                    send_actv(0x04, pa.req_id, r.stage_idx, 0, 4, 0, {},
                              "non-f32 dtype to stage>0");
                    continue;
                }
                const int32_t n_in = (pa.dims.size() >= 2)
                    ? (int32_t) pa.dims[1]
                    : (int32_t)(pa.payload.size() / sizeof(float) / engine->n_embd());
                dist::PpEngine::EmbedSeq s;
                s.seq_id      = r.seq_id;
                s.pos         = r.pos;
                s.embd        = reinterpret_cast<const float*>(pa.payload.data());
                s.n           = n_in;
                s.want_logits = r.is_last;
                if (r.is_last) {
                    term_entries.push_back({qi});
                    term_seqs.push_back(std::move(s));
                } else {
                    mid_entries.push_back({qi});
                    mid_seqs.push_back(std::move(s));
                }
            }
        }

        // Dispatch each bucket in a single decode call.
        auto fail_bucket = [&](const std::vector<StageNEntry> & es,
                               const std::string & why) {
            for (auto & e : es) {
                auto & pa = pending_actvs[e.qi];
                send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, why);
            }
        };

        if (!s0_seqs.empty()) {
            if (!engine->decode_tokens_batched(s0_seqs)) {
                std::string why = engine->last_error();
                for (auto & e : s0_entries) {
                    auto & pa = pending_actvs[e.qi];
                    send_actv(0x04, pa.req_id, 0, 0, 4, 0, {}, why);
                }
                s0_entries.clear();
                s0_seqs.clear();
            }
        }
        if (!mid_seqs.empty()) {
            if (!engine->decode_embeddings_batched(mid_seqs)) {
                fail_bucket(mid_entries, engine->last_error());
                mid_entries.clear();
                mid_seqs.clear();
            }
        }
        if (!term_seqs.empty()) {
            if (!engine->decode_embeddings_batched(term_seqs)) {
                fail_bucket(term_entries, engine->last_error());
                term_entries.clear();
                term_seqs.clear();
            }
        }

        // Emit per-request outputs.
        for (size_t si = 0; si < s0_entries.size(); ++si) {
            auto & e = s0_entries[si];
            auto & seq = s0_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;
            const bool is_prompt = (pa.flags & 0x01) != 0;
            std::vector<uint32_t> odims = { (uint32_t) engine->n_embd(),
                                            (uint32_t) seq.n };
            std::string hpayload((const char*) seq.out.data(),
                                 seq.out.size() * sizeof(float));
            uint8_t fwd_flags = is_prompt ? (uint8_t)(0x01 | 0x04)
                                          : (uint8_t)(0x02);
            send_actv(0x01, pa.req_id, r.stage_idx,
                      pa.tok_seq, 0, fwd_flags, odims, hpayload);
        }
        for (size_t si = 0; si < mid_entries.size(); ++si) {
            auto & e = mid_entries[si];
            auto & seq = mid_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;
            std::vector<uint32_t> odims = { (uint32_t) engine->n_embd(),
                                            (uint32_t) seq.n };
            std::string hpayload((const char*) seq.out.data(),
                                 seq.out.size() * sizeof(float));
            send_actv(0x01, pa.req_id, r.stage_idx,
                      pa.tok_seq, 0, pa.flags, odims, hpayload);
        }
        for (size_t si = 0; si < term_entries.size(); ++si) {
            auto & e = term_entries[si];
            auto & seq = term_seqs[si];
            auto & pa = pending_actvs[e.qi];
            auto it = reqs.find(pa.req_id);
            if (it == reqs.end()) continue;
            auto & r = it->second;
            r.pos += seq.n;
            dec_tokens += (uint64_t) seq.n;

            // Greedy sample over the emitted logits.
            int best = 0; float best_s = seq.out[0];
            for (size_t i = 1; i < seq.out.size(); ++i) {
                if (seq.out[i] > best_s) { best_s = seq.out[i]; best = (int) i; }
            }
            ++r.out_tokens;
            const int32_t eos = engine->eos_token();
            std::string piece = engine->detokenize(best);
            send_actv(0x02, pa.req_id, 0xFFFF, r.out_tokens, 0x04, 0, {}, piece);

            const bool hit_eos = (best == eos);
            const bool hit_cap = (r.out_tokens >= r.max_tokens);
            if (hit_eos || hit_cap) {
                send_actv(0x03, pa.req_id, 0xFFFF, r.out_tokens, 0, 0, {}, "");
                // Release KV + bookkeeping for this request.
                engine->reset_seq(r.seq_id);
                reqs.erase(it);
            }
        }
        pending_actvs.clear();
    };

    auto next_beat = std::chrono::steady_clock::now();
    while (ws.is_open()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next_beat) {
            double secs = std::chrono::duration<double>(
                now - dec_window_start).count();
            double tps = (secs > 0.0) ? (double) dec_tokens / secs : 0.0;
            std::ostringstream s;
            s << "{\"kind\":\"status\",\"n_gpus\":" << (n_gpu_layers > 0 ? 1 : 0)
              << ",\"tokens_sec\":" << (int) tps << "}";
            if (!ws.send_text(s.str())) break;
            // Decay the window — count only the last 5s.
            dec_tokens = 0;
            dec_window_start = now;
            next_beat = now + std::chrono::seconds(5);
        }
        // Peek for server commands (short timeout so we can send heartbeats).
        ws.set_recv_timeout_ms(500);
        std::vector<uint8_t> msg;
        bool is_bin = false;
        if (ws.recv_message(msg, is_bin)) {
            if (is_bin) {
                auto MAGIC = MAGIC_INFR;
                if (msg.size() >= 16 && be32(msg.data()) == MAGIC
                        && msg[4] == 0x01 && msg[5] == 0x01) {
                    // Inference request frame.
                    uint16_t req_id     = be16(msg.data() + 6);
                    uint32_t in_tokens  = be32(msg.data() + 8);
                    uint32_t payload_len = be32(msg.data() + 12);
                    std::string payload;
                    if (msg.size() >= 16 + payload_len) {
                        payload.assign((const char*)(msg.data() + 16), payload_len);
                    }
                    std::cout << "[pair] infer req " << req_id
                              << " in_tokens=" << in_tokens
                              << " payload=" << payload.substr(0, 80) << "...\n";

                    // Send a stub token stream.  Real agent: run llama_decode.
                    auto send_chunk = [&](uint8_t kind, uint32_t tok_in,
                                          uint32_t tok_out, const std::string& text) {
                        std::vector<uint8_t> out(24 + text.size());
                        // magic
                        out[0]=0x49; out[1]=0x4E; out[2]=0x46; out[3]=0x52;
                        out[4] = 0x01;   // ver
                        out[5] = 0x02;   // type = chunk
                        out[6] = (uint8_t)(req_id >> 8);
                        out[7] = (uint8_t)(req_id & 0xFF);
                        out[8] = kind;
                        out[9]=out[10]=out[11]=0;
                        auto put32 = [&](size_t off, uint32_t v) {
                            out[off  ] = (uint8_t)(v >> 24);
                            out[off+1] = (uint8_t)(v >> 16);
                            out[off+2] = (uint8_t)(v >>  8);
                            out[off+3] = (uint8_t)(v);
                        };
                        put32(12, tok_in);
                        put32(16, tok_out);
                        put32(20, (uint32_t)text.size());
                        std::memcpy(out.data() + 24, text.data(), text.size());
                        ws.send_binary(out.data(), out.size());
                    };

                    const char* tokens[] = {
                        "Hello", ", ", "this ", "is ", "a ", "distributed ",
                        "pool ", "reply."
                    };
                    uint32_t out_tokens = 0;
                    for (const char* t : tokens) {
                        ++out_tokens;
                        send_chunk(0 /* token */, in_tokens, out_tokens, t);
                        std::this_thread::sleep_for(std::chrono::milliseconds(80));
                    }
                    send_chunk(1 /* done */, in_tokens, out_tokens, "");
                } else if (msg.size() >= 20 && be32(msg.data()) == MAGIC_ACTV
                           && msg[4] == 0x01) {
                    // ACTV frame.  See server/activation.go for layout.
                    // Enqueue now, batch-decode later.
                    PendingActv pa;
                    pa.type     = msg[5];
                    pa.req_id   = be16(msg.data() + 6);
                    pa.stage_in = be16(msg.data() + 8);
                    pa.tok_seq  = be32(msg.data() + 10);
                    pa.dtype    = msg[14];
                    uint8_t rank = msg[15];
                    pa.flags    = msg[16];
                    size_t hdr  = 20 + 4 * (size_t)rank;
                    pa.dims.resize(rank);
                    for (uint8_t i = 0; i < rank; ++i) {
                        pa.dims[i] = be32(msg.data() + 20 + 4*i);
                    }
                    if (msg.size() < hdr + 4) { continue; }
                    uint32_t payload_len = be32(msg.data() + hdr);
                    if (msg.size() >= hdr + 4 + payload_len) {
                        pa.payload.assign((const char*)(msg.data() + hdr + 4), payload_len);
                    }
                    if (pa.type == 0x01) {
                        pending_actvs.push_back(std::move(pa));
                    }
                } else {
                    // Unknown binary frame — echo (keeps M2 relay test working).
                    std::cout << "[pair] ← binary " << msg.size() << " bytes (echo)\n";
                    ws.send_binary(msg.data(), msg.size());
                }
            } else {
                std::string txt((const char*)msg.data(), msg.size());
                std::cout << "[pair] ← " << txt << "\n";
                // Pipeline route assignment.
                if (txt.find("\"kind\":\"pp_route\"") != std::string::npos) {
                    auto pull_int = [&](const char* key) -> long long {
                        std::string k = std::string("\"") + key + "\":";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return -1;
                        size_t s = p + k.size();
                        while (s < txt.size() && txt[s] == ' ') ++s;
                        size_t e = s;
                        while (e < txt.size() && (isdigit((unsigned char)txt[e]) || txt[e]=='-')) ++e;
                        if (s == e) return -1;
                        try { return std::stoll(txt.substr(s, e - s)); } catch (...) { return -1; }
                    };
                    auto pull_bool = [&](const char* key) -> int {
                        std::string k = std::string("\"") + key + "\":";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return -1;
                        size_t s = p + k.size();
                        while (s < txt.size() && txt[s] == ' ') ++s;
                        if (txt.compare(s, 4, "true") == 0) return 1;
                        if (txt.compare(s, 5, "false") == 0) return 0;
                        return -1;
                    };
                    auto pull_str = [&](const char* key) -> std::string {
                        std::string k = std::string("\"") + key + "\":\"";
                        size_t p = txt.find(k);
                        if (p == std::string::npos) return "";
                        size_t s = p + k.size();
                        size_t e = s;
                        while (e < txt.size() && txt[e] != '"') {
                            if (txt[e] == '\\' && e + 1 < txt.size()) ++e;
                            ++e;
                        }
                        std::string raw = txt.substr(s, e - s);
                        std::string out; out.reserve(raw.size());
                        auto hex = [](char c) -> int {
                            if (c >= '0' && c <= '9') return c - '0';
                            if (c >= 'a' && c <= 'f') return 10 + c - 'a';
                            if (c >= 'A' && c <= 'F') return 10 + c - 'A';
                            return -1;
                        };
                        for (size_t i = 0; i < raw.size(); ++i) {
                            if (raw[i] == '\\' && i + 1 < raw.size()) {
                                char c = raw[i + 1];
                                if (c == '"' || c == '\\' || c == '/') { out += c; ++i; }
                                else if (c == 'n') { out += '\n'; ++i; }
                                else if (c == 'r') { out += '\r'; ++i; }
                                else if (c == 't') { out += '\t'; ++i; }
                                else if (c == 'u' && i + 5 < raw.size()) {
                                    int h0 = hex(raw[i+2]), h1 = hex(raw[i+3]);
                                    int h2 = hex(raw[i+4]), h3 = hex(raw[i+5]);
                                    if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
                                        out += raw[i]; // invalid — emit as-is
                                    } else {
                                        unsigned cp = (h0<<12)|(h1<<8)|(h2<<4)|h3;
                                        if (cp < 0x80) out += (char) cp;
                                        else if (cp < 0x800) {
                                            out += (char) (0xC0 | (cp >> 6));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        } else {
                                            out += (char) (0xE0 | (cp >> 12));
                                            out += (char) (0x80 | ((cp >> 6) & 0x3F));
                                            out += (char) (0x80 | (cp & 0x3F));
                                        }
                                        i += 5;
                                    }
                                } else { out += c; ++i; }
                            } else {
                                out += raw[i];
                            }
                        }
                        return out;
                    };
                    long long rid = pull_int("req_id");
                    long long si  = pull_int("stage_idx");
                    long long sc  = pull_int("stage_count");
                    long long lo  = pull_int("layer_lo");
                    long long hi  = pull_int("layer_hi");
                    long long mt  = pull_int("max_tokens");
                    int is_first = pull_bool("is_first");
                    int is_last  = pull_bool("is_last");
                    std::string shard_url  = pull_str("shard_url");
                    std::string shard_file = pull_str("shard_file");
                    if (rid >= 0 && si >= 0 && sc > 0) {
                        pp_req r;
                        r.req_id     = (uint16_t)rid;
                        r.stage_idx  = (uint16_t)si;
                        r.stage_count= (uint16_t)sc;
                        r.is_first   = is_first == 1;
                        r.is_last    = is_last == 1;
                        r.layer_lo   = (int32_t)(lo >= 0 ? lo : 0);
                        r.layer_hi   = (int32_t)(hi >= 0 ? hi : 0);
                        r.shard_url  = shard_url;
                        r.shard_file = shard_file;
                        r.max_tokens = (mt > 0) ? (uint32_t) mt : 128;
                        r.pos        = 0;
                        r.out_tokens = 0;
                        // Assign a fresh seq_id; wraps in [0, n_seq_max).
                        // Stored on the request; released in on_request_done.
                        r.seq_id     = next_seq_id;
                        next_seq_id  = (next_seq_id + 1) % 8;

                        // Load the engine for this shard (reused across
                        // requests).  Signed URLs rotate every request, so
                        // we key the cache on content identity instead.
                        bool engine_ok = true;
                        std::string key = (shard_file.empty() ? shard_url : shard_file) +
                                          "#" + std::to_string(r.layer_lo) +
                                          "-" + std::to_string(r.layer_hi);
                        if (!engine || engine_key != key) {
                            engine.reset();
                            std::string dest = "/tmp/dist-node-" +
                                std::to_string(::getpid()) + "-" +
                                (shard_file.empty()
                                    ? ("shard-" + std::to_string(rid) + ".gguf")
                                    : shard_file);
                            std::string derr;
                            if (shard_url.empty()) {
                                std::cerr << "[pair] pp_route missing shard_url\n";
                                engine_ok = false;
                            } else if (!dist::fetch_shard(shard_url, dest, derr)) {
                                std::cerr << "[pair] shard download failed: "
                                          << derr << "\n";
                                engine_ok = false;
                            } else {
                                auto eng = std::make_unique<dist::PpEngine>();
                                dist::PpEngineConfig ecfg;
                                ecfg.shard_path = dest;
                                ecfg.layer_lo   = r.layer_lo;
                                ecfg.layer_hi   = r.layer_hi;
                                ecfg.n_ctx      = 2048;
                                ecfg.n_batch    = 512;
                                ecfg.n_seq_max  = 8;
                                ecfg.n_gpu_layers = n_gpu_layers;
                                ecfg.gpu_lock     = gpu_lock.enabled() ? &gpu_lock : nullptr;
                                if (!eng->load(ecfg)) {
                                    std::cerr << "[pair] engine load failed: "
                                              << eng->last_error() << "\n";
                                    engine_ok = false;
                                } else {
                                    engine = std::move(eng);
                                    engine_key  = key;
                                    engine_path = dest;
                                }
                            }
                        }
                        // Clear this seq's KV so its fresh sequence starts
                        // at pos 0 (other seqs in the same context are left
                        // untouched — they may belong to concurrent requests).
                        if (engine_ok) {
                            engine->reset_seq(r.seq_id);
                        }
                        if (engine_ok) {
                            reqs[r.req_id] = r;
                        }
                        std::cout << "[pair] pp_route accepted req=" << r.req_id
                                  << " stage=" << si
                                  << "/" << sc
                                  << " seq=" << r.seq_id
                                  << " layers=[" << r.layer_lo << "," << r.layer_hi << ")"
                                  << " first=" << r.is_first
                                  << " last=" << r.is_last
                                  << " engine=" << (engine_ok ? "ready" : "FAILED")
                                  << "\n";
                    }
                }
                // Minimal signalling stub: if the server asks us for a WebRTC
                // answer, respond with signal_error so the client falls back to
                // the relay.  A real agent would terminate the PeerConnection.
                if (txt.find("\"kind\":\"signal\"") != std::string::npos) {
                    size_t p = txt.find("\"req_id\":");
                    if (p != std::string::npos) {
                        size_t start = p + 9;
                        size_t end = start;
                        while (end < txt.size() && (isdigit((unsigned char)txt[end]) || txt[end]=='-')) ++end;
                        std::string idstr = txt.substr(start, end - start);
                        std::ostringstream reply;
                        reply << "{\"kind\":\"signal_error\",\"req_id\":" << idstr
                              << ",\"message\":\"p2p not supported yet\"}";
                        ws.send_text(reply.str());
                    }
                }
            }
        }

        // Non-blocking drain: pull any additional frames already buffered.
        // This gives the flush step a chance to combine back-to-back ACTVs
        // (e.g. two concurrent requests arriving inside one scheduler tick).
        while (ws.is_open()) {
            ws.set_recv_timeout_ms(0);
            std::vector<uint8_t> m2;
            bool b2 = false;
            if (!ws.recv_message(m2, b2)) break;
            // Recurse the parse logic by re-injecting the frame via a second
            // pass: we only need to handle ACTV here since pp_route is rare
            // enough to wait for the next iteration.
            if (b2 && m2.size() >= 20 && be32(m2.data()) == MAGIC_ACTV && m2[4] == 0x01) {
                PendingActv pa;
                pa.type     = m2[5];
                pa.req_id   = be16(m2.data() + 6);
                pa.stage_in = be16(m2.data() + 8);
                pa.tok_seq  = be32(m2.data() + 10);
                pa.dtype    = m2[14];
                uint8_t rank = m2[15];
                pa.flags    = m2[16];
                size_t hdr  = 20 + 4 * (size_t)rank;
                pa.dims.resize(rank);
                for (uint8_t i = 0; i < rank; ++i) {
                    pa.dims[i] = be32(m2.data() + 20 + 4*i);
                }
                if (m2.size() < hdr + 4) continue;
                uint32_t payload_len = be32(m2.data() + hdr);
                if (m2.size() >= hdr + 4 + payload_len) {
                    pa.payload.assign((const char*)(m2.data() + hdr + 4), payload_len);
                }
                if (pa.type == 0x01) {
                    pending_actvs.push_back(std::move(pa));
                }
            }
            // Non-ACTV frames in the drain are skipped; they'll be picked up
            // by the next blocking recv iteration.
        }

        flush_pending();
    }
    std::cerr << "[pair] connection closed: " << ws.last_error() << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    dist::net_startup();
    dist::NodeAgentConfig cfg;
    std::string pair_url;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if (arg == "--pair") {
            pair_url = next();
        } else if (arg == "-s" || arg == "--server") {
            cfg.coordinator_host = next();
        } else if (arg == "-p" || arg == "--control-port") {
            cfg.coordinator_port = (uint16_t)std::stoi(next());
        } else if (arg == "-d" || arg == "--data-port") {
            cfg.data_port = (uint16_t)std::stoi(next());
        } else if (arg == "-g" || arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = std::stoi(next());
        } else if (arg == "-c" || arg == "--context") {
            cfg.n_ctx = (uint32_t)std::stoi(next());
        } else if (arg == "-b" || arg == "--batch") {
            cfg.n_batch = (uint32_t)std::stoi(next());
        } else if (arg == "--id") {
            cfg.node_id = next();
        } else if (arg == "--token-id") {
            cfg.token_id = next();
        } else if (arg == "--token-secret") {
            cfg.token_secret_hex = next();
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Deep-link pair mode: talks to the control-plane server over WebSocket.
    if (!pair_url.empty()) {
        std::string tok, srv;
        if (!parse_pair_url(pair_url, tok, srv)) {
            fprintf(stderr, "Error: invalid --pair URL (expected distpool://pair?token=...&server=...)\n");
            return 1;
        }
        return run_pair_mode(tok, srv, cfg.node_id, cfg.n_gpu_layers);
    }

    if (cfg.coordinator_host.empty()) {
        fprintf(stderr, "Error: --server is required\n");
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== llama-distributed Node Agent ===\n";
    std::cout << "coordinator=" << cfg.coordinator_host << ":"
              << cfg.coordinator_port
              << " data_port=" << cfg.data_port << "\n";

    try {
        dist::NodeAgent agent(cfg);
        agent.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
