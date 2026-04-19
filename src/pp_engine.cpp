#include "pp_engine.h"
#include "gpu_lock.h"

#include "llama.h"

#include <cstdio>
#include <cstring>

namespace dist {

PpEngine::PpEngine() = default;

PpEngine::~PpEngine() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

bool PpEngine::load(const PpEngineConfig & cfg) {
    gpu_lock_ = cfg.gpu_lock;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers   = cfg.n_gpu_layers;
    mparams.use_mmap       = true;
    mparams.layer_range_lo = cfg.layer_lo;
    mparams.layer_range_hi = cfg.layer_hi;

    model_ = llama_model_load_from_file(cfg.shard_path.c_str(), mparams);
    if (!model_) {
        err_ = "failed to load shard: " + cfg.shard_path;
        return false;
    }
    vocab_   = llama_model_get_vocab(model_);
    n_embd_  = llama_model_n_embd(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = cfg.n_ctx;
    cparams.n_batch    = cfg.n_batch;
    cparams.n_seq_max  = cfg.n_seq_max;
    cparams.embeddings = true;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        err_ = "failed to init context";
        return false;
    }
    return true;
}

std::vector<int32_t> PpEngine::tokenize(const std::string & text, bool add_bos) const {
    std::vector<int32_t> toks(text.size() + 8);
    int n = llama_tokenize(vocab_, text.c_str(), (int) text.size(),
                           toks.data(), (int) toks.size(), add_bos, false);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab_, text.c_str(), (int) text.size(),
                           toks.data(), (int) toks.size(), add_bos, false);
    }
    if (n < 0) return {};
    toks.resize(n);
    return toks;
}

std::string PpEngine::detokenize(int32_t token) const {
    char buf[64];
    int n = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, false);
    if (n <= 0) return {};
    return std::string(buf, buf + n);
}

int32_t PpEngine::eos_token() const {
    return llama_vocab_eos(vocab_);
}

void PpEngine::reset_kv() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_), true);
    }
    cached_toks_.clear();
}

int32_t PpEngine::prompt_prefix_hit(const int32_t * tokens, int32_t n) const {
    int32_t k = 0;
    const int32_t cap = (int32_t) cached_toks_.size();
    const int32_t lim = (n < cap) ? n : cap;
    while (k < lim && cached_toks_[k] == tokens[k]) ++k;
    // Always leave at least one token to re-decode so we get embeddings for
    // the active position.  A full hit with no tail would produce no output.
    if (k >= n) k = n - 1;
    if (k < 0)  k = 0;
    return k;
}

void PpEngine::trim_kv_to(int32_t keep) {
    if (!ctx_) return;
    if (keep < 0) keep = 0;
    if ((size_t) keep >= cached_toks_.size()) return;
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, keep, -1);
    cached_toks_.resize((size_t) keep);
}

void PpEngine::record_tokens(const int32_t * tokens, int32_t n) {
    cached_toks_.insert(cached_toks_.end(), tokens, tokens + n);
}

bool PpEngine::decode_tokens(const int32_t * tokens, int32_t n, int32_t pos_base,
                             std::vector<float> & out_hidden) {
    if (!ctx_) { err_ = "engine not loaded"; return false; }

    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        b.token[i]     = tokens[i];
        b.pos[i]       = pos_base + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1; // emit embeddings for every position
    }

    int rc;
    {
        GpuLockGuard g(gpu_lock_);
        (void) g;
        rc = llama_decode(ctx_, b);
    }
    if (rc != 0) {
        err_ = "stage-0 llama_decode failed";
        llama_batch_free(b);
        return false;
    }

    out_hidden.resize((size_t) n * n_embd_);
    for (int32_t i = 0; i < n; i++) {
        float * e = llama_get_embeddings_ith(ctx_, i);
        if (!e) {
            err_ = "stage-0 no embeddings at position";
            llama_batch_free(b);
            return false;
        }
        std::memcpy(out_hidden.data() + (size_t) i * n_embd_, e,
                    sizeof(float) * n_embd_);
    }
    llama_batch_free(b);
    return true;
}

bool PpEngine::decode_embeddings(const float * hidden, int32_t n, int32_t pos_base,
                                 bool want_logits, std::vector<float> & out) {
    if (!ctx_) { err_ = "engine not loaded"; return false; }

    llama_batch b = llama_batch_init(n, n_embd_, 1);
    b.n_tokens = n;
    std::memcpy(b.embd, hidden, sizeof(float) * (size_t) n * n_embd_);
    for (int32_t i = 0; i < n; i++) {
        b.pos[i]       = pos_base + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        if (want_logits) {
            b.logits[i] = (i == n - 1) ? 1 : 0;
        } else {
            b.logits[i] = 1; // intermediate: emit hidden for every position
        }
    }

    int rc;
    {
        GpuLockGuard g(gpu_lock_);
        (void) g;
        rc = llama_decode(ctx_, b);
    }
    if (rc != 0) {
        err_ = "stage-N llama_decode failed";
        llama_batch_free(b);
        return false;
    }

    if (want_logits) {
        float * l = llama_get_logits_ith(ctx_, n - 1);
        if (!l) {
            err_ = "no logits at last position";
            llama_batch_free(b);
            return false;
        }
        out.assign(l, l + n_vocab_);
    } else {
        out.resize((size_t) n * n_embd_);
        for (int32_t i = 0; i < n; i++) {
            float * e = llama_get_embeddings_ith(ctx_, i);
            if (!e) {
                err_ = "intermediate stage no embeddings at position";
                llama_batch_free(b);
                return false;
            }
            std::memcpy(out.data() + (size_t) i * n_embd_, e,
                        sizeof(float) * n_embd_);
        }
    }
    llama_batch_free(b);
    return true;
}

bool PpEngine::decode_tokens_batched(std::vector<TokenSeq> & seqs) {
    if (!ctx_) { err_ = "engine not loaded"; return false; }
    if (seqs.empty()) return true;

    int32_t total = 0;
    for (auto & s : seqs) total += s.n;
    if (total == 0) return true;

    llama_batch b = llama_batch_init(total, 0, 1);
    b.n_tokens = total;

    std::vector<int32_t> offsets(seqs.size(), 0);
    int32_t cursor = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        auto & s = seqs[si];
        offsets[si] = cursor;
        for (int32_t i = 0; i < s.n; ++i) {
            int32_t k = cursor + i;
            b.token[k]     = s.tokens[i];
            b.pos[k]       = s.pos + i;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = s.seq_id;
            b.logits[k]    = 1; // stage-0 emits hidden at every position
        }
        cursor += s.n;
    }

    int rc;
    {
        GpuLockGuard g(gpu_lock_);
        (void) g;
        rc = llama_decode(ctx_, b);
    }
    if (rc != 0) {
        err_ = "stage-0 batched llama_decode failed";
        llama_batch_free(b);
        return false;
    }

    for (size_t si = 0; si < seqs.size(); ++si) {
        auto & s = seqs[si];
        s.out.resize((size_t) s.n * n_embd_);
        for (int32_t i = 0; i < s.n; ++i) {
            float * e = llama_get_embeddings_ith(ctx_, offsets[si] + i);
            if (!e) {
                err_ = "stage-0 batched: no embeddings at position";
                llama_batch_free(b);
                return false;
            }
            std::memcpy(s.out.data() + (size_t) i * n_embd_, e,
                        sizeof(float) * n_embd_);
        }
    }
    llama_batch_free(b);
    return true;
}

bool PpEngine::decode_embeddings_batched(std::vector<EmbedSeq> & seqs) {
    if (!ctx_) { err_ = "engine not loaded"; return false; }
    if (seqs.empty()) return true;

    int32_t total = 0;
    for (auto & s : seqs) total += s.n;
    if (total == 0) return true;

    llama_batch b = llama_batch_init(total, n_embd_, 1);
    b.n_tokens = total;

    std::vector<int32_t> offsets(seqs.size(), 0);
    int32_t cursor = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        auto & s = seqs[si];
        offsets[si] = cursor;
        std::memcpy(b.embd + (size_t) cursor * n_embd_, s.embd,
                    sizeof(float) * (size_t) s.n * n_embd_);
        for (int32_t i = 0; i < s.n; ++i) {
            int32_t k = cursor + i;
            b.pos[k]       = s.pos + i;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = s.seq_id;
            if (s.want_logits) {
                b.logits[k] = (i == s.n - 1) ? 1 : 0;
            } else {
                b.logits[k] = 1;
            }
        }
        cursor += s.n;
    }

    int rc;
    {
        GpuLockGuard g(gpu_lock_);
        (void) g;
        rc = llama_decode(ctx_, b);
    }
    if (rc != 0) {
        err_ = "stage-N batched llama_decode failed";
        llama_batch_free(b);
        return false;
    }

    for (size_t si = 0; si < seqs.size(); ++si) {
        auto & s = seqs[si];
        if (s.want_logits) {
            // Last position of this sub-sequence.
            float * l = llama_get_logits_ith(ctx_, offsets[si] + s.n - 1);
            if (!l) {
                err_ = "no logits at last position (batched)";
                llama_batch_free(b);
                return false;
            }
            s.out.assign(l, l + n_vocab_);
        } else {
            s.out.resize((size_t) s.n * n_embd_);
            for (int32_t i = 0; i < s.n; ++i) {
                float * e = llama_get_embeddings_ith(ctx_, offsets[si] + i);
                if (!e) {
                    err_ = "intermediate batched: no embeddings at position";
                    llama_batch_free(b);
                    return false;
                }
                std::memcpy(s.out.data() + (size_t) i * n_embd_, e,
                            sizeof(float) * n_embd_);
            }
        }
    }
    llama_batch_free(b);
    return true;
}

void PpEngine::reset_seq(int seq_id) {
    if (!ctx_) return;
    llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, 0, -1);
}

} // namespace dist
