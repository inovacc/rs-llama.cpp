#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

extern unsigned char tokenCallback(void *, char *);

int load_state(void *ctx, char *statefile, char*modes);

int eval(void* params_ptr, void *ctx, char*text);

void save_state(void *ctx, char *dst, char*modes);

void* load_model(const char *fname, 
                 int n_ctx, 
                 int n_seed, 
                 bool memory_f16, 
                 bool mlock, 
                 bool embeddings, 
                 bool mmap, 
                 bool low_vram, 
                 int n_gpu, 
                 int n_batch, 
                 const char *maingpu, 
                 const char *tensorsplit, 
                 bool numa, 
                 float rope_freq_base, 
                 float rope_freq_scale,
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity
                 );

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings);

int get_token_embeddings(void* params_ptr, void* state_pr,  int *tokens, int tokenSize, float * res_embeddings);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit ,
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char *negative_prompt, int n_draft,
                            float min_p,
                            const int32_t *logit_bias_tokens, const float *logit_bias_values,
                            int logit_bias_count);

int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char* result, bool debug);

void llama_free_params(void* params_ptr);

void llama_binding_free_model(void* state);

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result);

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug);

// llama_predict_full generates from params->prompt, writing up to result_size-1
// bytes (NUL-terminated) into result and the generated token count into *n_tokens.
// Returns the FULL generated length in bytes (may exceed result_size-1 → the
// caller should resize and retry), or a negative value on error. Unlike
// llama_predict, the output is not capped to the token count.
int llama_predict_full(void* params_ptr, void* state_pr, char* result, int result_size, int* n_tokens, bool debug);

// apply_chat_template formats (system,user) using the model's embedded GGUF
// chat template into result (capacity result_size). Returns the formatted
// length, or a negative/larger-than-capacity value if the buffer is too small.
// If the model has no template, returns 0 (caller should fall back to raw).
int apply_chat_template(void* state_pr, const char* system, const char* user, char* result, int result_size);

#ifdef __cplusplus
}


std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
