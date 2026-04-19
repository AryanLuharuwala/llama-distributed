// pp_engine.h
//
// Per-stage wrapper around llama.cpp's partial-layer load + decode for
// pipeline-parallel inference.  One PpEngine instance owns exactly one shard
// (a contiguous layer range) and its llama_context; callers advance it with
// decode_tokens (stage 0) or decode_embeddings (intermediate/terminal).
//
// The shape of the activation tensor produced by a stage is
//   [n_embd, n_tokens]  (row-major, n_embd contiguous per token)
// which matches what llama_get_embeddings_ith returns for each position.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace dist {

class GpuLock;

struct PpEngineConfig {
    std::string shard_path;   // local path to the gguf shard
    int32_t     layer_lo = 0;
    int32_t     layer_hi = 0; // exclusive; 0 means "full model"
    uint32_t    n_ctx    = 2048;
    uint32_t    n_batch  = 512;
    int32_t     n_gpu_layers = 0;
    // Max concurrent sequences sharing this context's KV.  Each in-flight
    // request is assigned a distinct seq_id in [0, n_seq_max).
    uint32_t    n_seq_max = 8;
    // Optional cross-process GPU lock; held across every decode_*.  May be
    // nullptr on CPU-only nodes or when no serialisation is desired.
    GpuLock *   gpu_lock    = nullptr;
};

class PpEngine {
public:
    PpEngine();
    ~PpEngine();

    PpEngine(const PpEngine&) = delete;
    PpEngine& operator=(const PpEngine&) = delete;

    // Load the shard, build a context.  `cfg.n_ctx` is the maximum context
    // length for this request sequence.  Returns false on error; call
    // last_error() for diagnostics.
    bool load(const PpEngineConfig & cfg);

    // True once a shard is loaded and a context is ready.
    bool ready() const { return model_ && ctx_; }

    // Model geometry exposed to callers.
    int  n_embd() const { return n_embd_; }
    int  n_vocab() const { return n_vocab_; }
    const llama_vocab * vocab() const { return vocab_; }

    // Tokenise a UTF-8 string with the shard's vocab.  Only meaningful on the
    // stage that owns the input embedding table (stage 0).  `add_bos` should
    // be true for the very first prompt; false for per-token continuations.
    std::vector<int32_t> tokenize(const std::string & text, bool add_bos) const;

    // Detokenise a single token id to a short utf-8 piece.
    std::string detokenize(int32_t token) const;

    int32_t eos_token() const;

    // Clear the KV cache so this engine can start a fresh sequence.
    // Also drops the prompt-prefix cache.
    void reset_kv();

    // Prompt-prefix cache: on stage 0, given the tokenised prompt for a new
    // request, returns the number of leading tokens that are already in the
    // KV cache (i.e. can be skipped).  Callers are expected to then:
    //   - trim the KV cache to exactly those positions with trim_kv_to(K),
    //   - feed the remaining (n - K) tokens via decode_tokens(pos_base=K).
    // Returns 0 when there is nothing to reuse.  Only meaningful after a
    // previous decode_tokens on the same engine.
    int32_t prompt_prefix_hit(const int32_t * tokens, int32_t n) const;

    // Trim the KV cache of sequence 0 down to exactly `keep` positions.
    // Drops the corresponding suffix of the cached prompt tokens too.
    void trim_kv_to(int32_t keep);

    // Feed the cached-prompt tracker with tokens accepted into the KV.
    // dist_node_main calls this after every successful decode_tokens so the
    // engine knows what sequence is currently materialised.
    void record_tokens(const int32_t * tokens, int32_t n);

    // Stage 0: feed `tokens` at absolute positions [pos_base, pos_base+n),
    // extract the per-position hidden states and return them as a flat vector
    // of floats of length (n * n_embd).  `pos_base` should be 0 for the first
    // call and grow monotonically.
    bool decode_tokens(const int32_t * tokens, int32_t n, int32_t pos_base,
                       std::vector<float> & out_hidden);

    // Intermediate/terminal stages: feed `hidden` (length n * n_embd) at
    // absolute positions [pos_base, pos_base+n), and extract either:
    //   - hidden states out  (intermediate stages; pass want_logits=false)
    //   - logits at the last position  (terminal stage; pass want_logits=true)
    //
    // For intermediate: `out` is filled with n*n_embd floats.
    // For terminal:     `out` is filled with n_vocab floats (last position only).
    bool decode_embeddings(const float * hidden, int32_t n, int32_t pos_base,
                           bool want_logits, std::vector<float> & out);

    // ─── Batched decode — pack multiple independent requests into one llama
    //     decode call.  Each item carries its own seq_id; the engine assigns
    //     and reuses them internally.  Items never share a sequence.

    // Per-request token input for stage-0 batched decode.
    struct TokenSeq {
        int             seq_id  = 0;    // distinct id across the batch
        int32_t         pos     = 0;    // absolute position to write at
        const int32_t * tokens  = nullptr;
        int32_t         n       = 0;
        // Populated on return: hidden states [n * n_embd], row-major per token.
        std::vector<float> out;
    };

    // Per-request activation input for intermediate/terminal batched decode.
    struct EmbedSeq {
        int             seq_id     = 0;
        int32_t         pos        = 0;
        const float   * embd       = nullptr;    // length n * n_embd
        int32_t         n          = 0;
        bool            want_logits = false;     // terminal: emit logits for last pos
        // Populated on return:
        //   - want_logits==true  : n_vocab floats (last position)
        //   - want_logits==false : n * n_embd floats
        std::vector<float> out;
    };

    // One llama_decode over the union of token-sequences.
    bool decode_tokens_batched(std::vector<TokenSeq> & seqs);

    // One llama_decode over the union of activation-sequences.
    bool decode_embeddings_batched(std::vector<EmbedSeq> & seqs);

    // Drop a whole sequence id from the KV cache — use after a request finishes.
    void reset_seq(int seq_id);

    const std::string & last_error() const { return err_; }

private:
    llama_model   * model_   = nullptr;
    llama_context * ctx_     = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int             n_embd_  = 0;
    int             n_vocab_ = 0;
    std::string     err_;
    GpuLock       * gpu_lock_ = nullptr;

    // Tokens currently materialised in sequence 0's KV (stage 0 only).
    // decode_tokens appends; trim_kv_to truncates; reset_kv clears.
    std::vector<int32_t> cached_toks_;
};

} // namespace dist
