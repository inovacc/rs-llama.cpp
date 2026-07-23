// go-llama.cpp binding — pure C llama.h API (no libcommon).
//
// Uses only the C API exported by llama.dll/libllama, so the MinGW cgo host can
// link against an MSVC-built llama.dll (CUDA) as well as a MinGW static build.
// Sampling is a hand-built llama_sampler chain; tokenize/detokenize use the raw
// llama_tokenize / llama_token_to_piece. The C ABI in binding.h is unchanged.
//
// Generation: load_model, eval, llama_predict, llama_tokenize_string.
// Embeddings, state save/load and speculative sampling are stubs.

#include "binding.h"
#include "llama.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct binding_state {
    llama_model       *model  = nullptr;
    llama_context     *ctx    = nullptr;
    const llama_vocab *vocab  = nullptr;
    int                n_past = 0;
};

struct binding_params {
    std::string prompt;
    int         n_predict = 128;
    int         n_keep    = 0;
    int         n_batch   = 512;
    int         n_threads = 0;

    uint32_t seed            = LLAMA_DEFAULT_SEED;
    int32_t  top_k           = 40;
    float    top_p           = 0.95f;
    float    temp            = 0.80f;
    int32_t  penalty_last_n  = 64;
    float    penalty_repeat  = 1.00f;
    float    penalty_freq    = 0.00f;
    float    penalty_present = 0.00f;

    float    min_p           = 0.00f;   // 0 = disabled
    float    typical_p       = 1.00f;   // 1.0 = disabled
    int32_t  mirostat        = 0;       // 0 = off, 1 = v1, 2 = v2
    float    mirostat_eta    = 0.10f;
    float    mirostat_tau    = 5.00f;

    std::string grammar;            // GBNF; "" = unconstrained

    std::vector<std::string>      antiprompt;
    std::vector<llama_logit_bias> logit_bias;
};

int decode_tokens(llama_context *ctx, const std::vector<llama_token> &toks, int n_batch) {
    if (n_batch <= 0) {
        n_batch = 512;
    }
    for (size_t i = 0; i < toks.size(); i += (size_t)n_batch) {
        int n = (int)(toks.size() - i);
        if (n > n_batch) {
            n = n_batch;
        }
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data() + i), n);
        if (llama_decode(ctx, b) != 0) {
            return 1;
        }
    }
    return 0;
}

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special) {
    if (vocab == nullptr) {
        return {};
    }
    int32_t need = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, true);
    if (need < 0) {
        need = -need;
    }
    std::vector<llama_token> toks((size_t)need);
    int32_t got = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), need, add_special, true);
    if (got < 0) {
        return {};
    }
    toks.resize((size_t)got);
    return toks;
}

std::string token_to_piece(const llama_vocab *vocab, llama_token id) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, id, buf, (int32_t)sizeof(buf), 0, false);
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, (size_t)n);
}

llama_sampler *make_sampler(const binding_params *bp, const llama_vocab *vocab) {
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // logit_bias first — bias raw logits before any truncation.
    if (!bp->logit_bias.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            n_vocab, (int32_t)bp->logit_bias.size(), bp->logit_bias.data()));
    }

    // repetition penalties (unchanged condition).
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }

    // grammar constraint — masks tokens that violate the GBNF before any
    // terminal sampler, so mirostat / greedy / the truncation tail all sample
    // from grammar-valid tokens. NULL means the grammar failed to parse: free
    // the chain and fail (generate() returns an error on a nullptr sampler).
    if (!bp->grammar.empty()) {
        llama_sampler *gr = llama_sampler_init_grammar(vocab, bp->grammar.c_str(), "root");
        if (gr == nullptr) {
            llama_sampler_free(smpl);
            return nullptr;
        }
        llama_sampler_chain_add(smpl, gr);
    }

    // mirostat is terminal: it performs its own temperature + selection, so it
    // replaces the truncation + dist tail entirely.
    if (bp->mirostat == 1) {
        const float eta = bp->mirostat_eta > 0.0f ? bp->mirostat_eta : 0.10f;
        const float tau = bp->mirostat_tau > 0.0f ? bp->mirostat_tau : 5.00f;
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(n_vocab, bp->seed, tau, eta, 100));
        return smpl;
    }
    if (bp->mirostat == 2) {
        const float eta = bp->mirostat_eta > 0.0f ? bp->mirostat_eta : 0.10f;
        const float tau = bp->mirostat_tau > 0.0f ? bp->mirostat_tau : 5.00f;
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(bp->seed, tau, eta));
        return smpl;
    }

    if (bp->temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }

    // truncation tail — each entry opt-in; defaults reproduce the original chain.
    if (bp->top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(bp->top_k));
    }
    if (bp->typical_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(bp->typical_p, 1));
    }
    if (bp->top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(bp->top_p, 1));
    }
    if (bp->min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(bp->min_p, 1));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(bp->temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(bp->seed));
    return smpl;
}

// generate runs the prompt-decode + sampling loop, appending the full generated
// text to `out` and setting `n_tokens` to the number of tokens produced.
// Returns 0 on success, 1 on error. Streams each piece via tokenCallback.
int generate(binding_params *bp, binding_state *st, std::string &out, int &n_tokens) {
    n_tokens = 0;
    if (bp->n_threads > 0) {
        llama_set_n_threads(st->ctx, bp->n_threads, bp->n_threads);
    }
    llama_memory_clear(llama_get_memory(st->ctx), true);
    st->n_past = 0;

    std::vector<llama_token> toks = tokenize(st->vocab, bp->prompt, true);
    const int n_ctx = (int)llama_n_ctx(st->ctx);
    if ((int)toks.size() >= n_ctx) {
        return 1;
    }
    if (decode_tokens(st->ctx, toks, bp->n_batch) != 0) {
        return 1;
    }
    st->n_past = (int)toks.size();

    llama_sampler *smpl = make_sampler(bp, st->vocab);
    if (smpl == nullptr) {
        return 1;
    }

    const int n_predict = bp->n_predict > 0 ? bp->n_predict : 128;
    bool stop = false;
    for (int i = 0; i < n_predict && st->n_past < n_ctx && !stop; i++) {
        llama_token id = llama_sampler_sample(smpl, st->ctx, -1);
        if (llama_vocab_is_eog(st->vocab, id)) {
            break;
        }
        std::string piece = token_to_piece(st->vocab, id);
        out += piece;
        n_tokens++;
        if (!piece.empty()) {
            std::vector<char> buf(piece.begin(), piece.end());
            buf.push_back('\0');
            if (tokenCallback(st, buf.data()) == 0) {
                stop = true;
            }
        }
        if (stop) {
            break;
        }
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(st->ctx, b) != 0) {
            break;
        }
        st->n_past++;
    }
    llama_sampler_free(smpl);
    return 0;
}

} // namespace

extern "C" {

void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16,
                 bool mlock, bool embeddings, bool mmap, bool low_vram,
                 int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit,
                 bool numa, float rope_freq_base, float rope_freq_scale,
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity) {
    (void)n_seed; (void)memory_f16; (void)mlock; (void)mmap; (void)low_vram;
    (void)maingpu; (void)tensorsplit; (void)mul_mat_q; (void)lora; (void)lora_base;
    (void)perplexity;

    llama_backend_init();
    llama_numa_init(numa ? GGML_NUMA_STRATEGY_DISTRIBUTE : GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;

    llama_model *model = llama_model_load_from_file(fname, mparams);
    if (model == nullptr) {
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0)   { cparams.n_ctx   = (uint32_t)n_ctx; }
    if (n_batch > 0) { cparams.n_batch = (uint32_t)n_batch; }
    cparams.embeddings = embeddings;
    if (rope_freq_base  > 0.0f) { cparams.rope_freq_base  = rope_freq_base; }
    if (rope_freq_scale > 0.0f) { cparams.rope_freq_scale = rope_freq_scale; }

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        return nullptr;
    }

    binding_state *st = new binding_state();
    st->model = model;
    st->ctx   = ctx;
    st->vocab = llama_model_get_vocab(model);
    return st;
}

void llama_binding_free_model(void *state) {
    binding_state *st = static_cast<binding_state *>(state);
    if (st == nullptr) {
        return;
    }
    if (st->ctx != nullptr)   { llama_free(st->ctx); }
    if (st->model != nullptr) { llama_model_free(st->model); }
    delete st;
}

void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty,
                            float presence_penalty, int mirostat, float mirostat_eta,
                            float mirostat_tau, bool penalize_nl,
                            const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                            const char *maingpu, const char *tensorsplit, bool prompt_cache_ro,
                            const char *grammar, float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char *negative_prompt, int n_draft,
                            float min_p,
                            const int32_t *logit_bias_tokens, const float *logit_bias_values,
                            int logit_bias_count) {
    (void)ignore_eos; (void)memory_f16; (void)tfs_z; (void)penalize_nl;
    (void)session_file; (void)prompt_cache_all; (void)mlock; (void)mmap; (void)maingpu;
    (void)tensorsplit; (void)prompt_cache_ro; (void)rope_freq_base;
    (void)rope_freq_scale; (void)negative_prompt_scale; (void)negative_prompt; (void)n_draft;

    binding_params *p = new binding_params();
    p->prompt    = prompt ? std::string(prompt) : std::string();
    p->n_threads = threads;
    p->n_predict = tokens;
    p->n_batch   = n_batch > 0 ? n_batch : 512;
    p->n_keep    = n_keep;
    if (seed >= 0) {
        p->seed = (uint32_t)seed;
    }
    p->top_k           = top_k;
    p->top_p           = top_p;
    p->temp            = temp;
    p->penalty_last_n  = repeat_last_n;
    p->penalty_repeat  = repeat_penalty;
    p->penalty_freq    = frequency_penalty;
    p->penalty_present = presence_penalty;
    p->typical_p    = typical_p;
    p->mirostat     = mirostat;
    p->mirostat_eta = mirostat_eta;
    p->mirostat_tau = mirostat_tau;
    p->min_p = min_p;
    p->grammar = grammar ? std::string(grammar) : std::string();
    for (int i = 0; i < logit_bias_count; i++) {
        p->logit_bias.push_back(llama_logit_bias{
            (llama_token)logit_bias_tokens[i], logit_bias_values[i] });
    }

    if (antiprompt != nullptr && antiprompt_count > 0) {
        for (int i = 0; i < antiprompt_count; i++) {
            if (antiprompt[i] != nullptr) {
                p->antiprompt.push_back(std::string(antiprompt[i]));
            }
        }
    }
    return p;
}

void llama_free_params(void *params_ptr) {
    delete static_cast<binding_params *>(params_ptr);
}

int eval(void *params_ptr, void *state_pr, char *text) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (st == nullptr || st->ctx == nullptr || text == nullptr) {
        return 1;
    }
    if (bp != nullptr && bp->n_threads > 0) {
        llama_set_n_threads(st->ctx, bp->n_threads, bp->n_threads);
    }
    std::vector<llama_token> toks = tokenize(st->vocab, std::string(text), false);
    int rc = decode_tokens(st->ctx, toks, bp ? bp->n_batch : 512);
    if (rc == 0) {
        st->n_past += (int)toks.size();
    }
    return rc;
}

int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (bp == nullptr || st == nullptr || st->ctx == nullptr || result == nullptr) {
        return 1;
    }
    (void)debug;

    std::string out;
    int nt = 0;
    if (generate(bp, st, out, nt) != 0) {
        result[0] = '\0';
        return 1;
    }

    // Legacy ABI: `result` is n_predict BYTES (Go make([]byte, tokens)), but text
    // can be several bytes per token — bound the copy to avoid overflow. Prefer
    // llama_predict_full for the complete text.
    size_t cap = bp->n_predict > 0 ? (size_t)bp->n_predict : 128;
    if (out.size() > cap - 1) {
        out.resize(cap - 1);
    }
    std::memcpy(result, out.data(), out.size());
    result[out.size()] = '\0';
    return 0;
}

int llama_predict_full(void *params_ptr, void *state_pr, char *result, int result_size,
                       int *n_tokens, bool debug) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (n_tokens != nullptr) {
        *n_tokens = 0;
    }
    if (bp == nullptr || st == nullptr || st->ctx == nullptr || result == nullptr || result_size <= 0) {
        return -1;
    }
    (void)debug;

    std::string out;
    int nt = 0;
    if (generate(bp, st, out, nt) != 0) {
        result[0] = '\0';
        return -1;
    }
    if (n_tokens != nullptr) {
        *n_tokens = nt;
    }

    // Write up to result_size-1 bytes (NUL-terminated); return the FULL length so
    // the caller can detect truncation and resize+retry.
    size_t w = out.size();
    if (w > (size_t)result_size - 1) {
        w = (size_t)result_size - 1;
    }
    std::memcpy(result, out.data(), w);
    result[w] = '\0';
    return (int)out.size();
}

int apply_chat_template(void *state_pr, const char *system, const char *user, char *result, int result_size) {
    binding_state *st = static_cast<binding_state *>(state_pr);
    if (st == nullptr || st->model == nullptr || result == nullptr || result_size <= 0) {
        return -1;
    }
    const char *tmpl = llama_model_chat_template(st->model, nullptr);
    if (tmpl == nullptr) {
        return 0; // no embedded template — caller falls back to raw concatenation
    }

    llama_chat_message msgs[2];
    size_t n = 0;
    if (system != nullptr && system[0] != '\0') {
        msgs[n].role = "system";
        msgs[n].content = system;
        n++;
    }
    msgs[n].role = "user";
    msgs[n].content = user ? user : "";
    n++;

    int32_t len = llama_chat_apply_template(tmpl, msgs, n, /*add_ass*/ true, result, result_size);
    if (len >= 0 && len < result_size) {
        result[len] = '\0';
    }
    return (int)len;
}

int llama_tokenize_string(void *params_ptr, void *state_pr, int *result) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (bp == nullptr || st == nullptr || st->ctx == nullptr || result == nullptr) {
        return -1;
    }
    std::vector<llama_token> toks = tokenize(st->vocab, bp->prompt, true);
    // result holds at most n_predict ints (Go allocates make([]C.int, tokens)).
    size_t cap = bp->n_predict > 0 ? (size_t)bp->n_predict : toks.size();
    size_t w = toks.size() < cap ? toks.size() : cap;
    for (size_t i = 0; i < w; i++) {
        result[i] = (int)toks[i];
    }
    return (int)toks.size();
}

int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)res_embeddings;
    return 1; // TODO: llama_get_embeddings_seq + pooling
}

int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize,
                         float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)tokens; (void)tokenSize; (void)res_embeddings;
    return 1;
}

int speculative_sampling(void *params_ptr, void *target_model, void *draft_model,
                         char *result, bool debug) {
    (void)params_ptr; (void)target_model; (void)draft_model; (void)debug;
    if (result != nullptr) { result[0] = '\0'; }
    return 1;
}

int load_state(void *ctx, char *statefile, char *modes) {
    (void)ctx; (void)statefile; (void)modes;
    return 1;
}

void save_state(void *ctx, char *dst, char *modes) {
    (void)ctx; (void)dst; (void)modes;
}

} // extern "C"

std::vector<std::string> create_vector(const char **strings, int count) {
    std::vector<std::string> v;
    if (count > 0) {
        v.reserve((size_t)count);
        for (int i = 0; i < count; i++) {
            v.push_back(std::string(strings[i]));
        }
    }
    return v;
}

void delete_vector(std::vector<std::string> *vec) {
    delete vec;
}
