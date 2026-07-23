//! Raw `extern "C"` FFI for the go-llama.cpp C shim (`csrc/binding.{h,cpp}`),
//! linked against a static CPU build of llama.cpp.
//!
//! Every declaration below mirrors `csrc/binding.h` 1:1. Pointers that the shim
//! treats as opaque handles (`void *`) are exposed as `*mut c_void`.
//!
//! `tokenCallback` is DECLARED (not defined) here: it is an
//! `extern unsigned char tokenCallback(void *, char *)` that the shim CALLS
//! during generation but which the CONSUMER must DEFINE. The higher-level
//! `llama` crate provides the real implementation via `#[no_mangle]`. This crate
//! never defines it, so it does not collide with the consumer's definition.
//! (A stub is provided only in this crate's own link-probe test — see
//! `tests/link_probe.rs` — purely to satisfy the standalone test-binary link.)

#![allow(non_snake_case)]

use std::os::raw::{c_char, c_float, c_int, c_void};

extern "C" {
    // Streaming callback the shim CALLS; DEFINED by the consumer, not here.
    pub fn tokenCallback(state: *mut c_void, token: *mut c_char) -> u8;

    pub fn load_state(ctx: *mut c_void, statefile: *mut c_char, modes: *mut c_char) -> c_int;

    pub fn eval(params_ptr: *mut c_void, ctx: *mut c_void, text: *mut c_char) -> c_int;

    pub fn save_state(ctx: *mut c_void, dst: *mut c_char, modes: *mut c_char);

    #[allow(clippy::too_many_arguments)]
    pub fn load_model(
        fname: *const c_char,
        n_ctx: c_int,
        n_seed: c_int,
        memory_f16: bool,
        mlock: bool,
        embeddings: bool,
        mmap: bool,
        low_vram: bool,
        n_gpu: c_int,
        n_batch: c_int,
        maingpu: *const c_char,
        tensorsplit: *const c_char,
        numa: bool,
        rope_freq_base: c_float,
        rope_freq_scale: c_float,
        mul_mat_q: bool,
        lora: *const c_char,
        lora_base: *const c_char,
        perplexity: bool,
    ) -> *mut c_void;

    pub fn get_embeddings(
        params_ptr: *mut c_void,
        state_pr: *mut c_void,
        res_embeddings: *mut c_float,
    ) -> c_int;

    pub fn get_token_embeddings(
        params_ptr: *mut c_void,
        state_pr: *mut c_void,
        tokens: *mut c_int,
        tokenSize: c_int,
        res_embeddings: *mut c_float,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn llama_allocate_params(
        prompt: *const c_char,
        seed: c_int,
        threads: c_int,
        tokens: c_int,
        top_k: c_int,
        top_p: c_float,
        temp: c_float,
        repeat_penalty: c_float,
        repeat_last_n: c_int,
        ignore_eos: bool,
        memory_f16: bool,
        n_batch: c_int,
        n_keep: c_int,
        antiprompt: *mut *const c_char,
        antiprompt_count: c_int,
        tfs_z: c_float,
        typical_p: c_float,
        frequency_penalty: c_float,
        presence_penalty: c_float,
        mirostat: c_int,
        mirostat_eta: c_float,
        mirostat_tau: c_float,
        penalize_nl: bool,
        session_file: *const c_char,
        prompt_cache_all: bool,
        mlock: bool,
        mmap: bool,
        maingpu: *const c_char,
        tensorsplit: *const c_char,
        prompt_cache_ro: bool,
        grammar: *const c_char,
        rope_freq_base: c_float,
        rope_freq_scale: c_float,
        negative_prompt_scale: c_float,
        negative_prompt: *const c_char,
        n_draft: c_int,
        min_p: c_float,
        logit_bias_tokens: *const i32,
        logit_bias_values: *const c_float,
        logit_bias_count: c_int,
    ) -> *mut c_void;

    pub fn speculative_sampling(
        params_ptr: *mut c_void,
        target_model: *mut c_void,
        draft_model: *mut c_void,
        result: *mut c_char,
        debug: bool,
    ) -> c_int;

    pub fn llama_free_params(params_ptr: *mut c_void);

    pub fn llama_binding_free_model(state: *mut c_void);

    pub fn llama_tokenize_string(
        params_ptr: *mut c_void,
        state_pr: *mut c_void,
        result: *mut c_int,
    ) -> c_int;

    pub fn llama_predict(
        params_ptr: *mut c_void,
        state_pr: *mut c_void,
        result: *mut c_char,
        debug: bool,
    ) -> c_int;

    pub fn llama_predict_full(
        params_ptr: *mut c_void,
        state_pr: *mut c_void,
        result: *mut c_char,
        result_size: c_int,
        n_tokens: *mut c_int,
        debug: bool,
    ) -> c_int;

    pub fn apply_chat_template(
        state_pr: *mut c_void,
        system: *const c_char,
        user: *const c_char,
        result: *mut c_char,
        result_size: c_int,
    ) -> c_int;

    // ---- mtmd (multimodal vision) ----
    pub fn mtmd_load(state: *mut c_void, mmproj_path: *const c_char) -> *mut c_void;

    pub fn mtmd_describe(
        mtmd_ctx: *mut c_void,
        state: *mut c_void,
        image_path: *const c_char,
        prompt: *const c_char,
        result: *mut c_char,
        result_size: c_int,
        n_tokens: *mut c_int,
    ) -> c_int;

    pub fn mtmd_free_ctx(mtmd_ctx: *mut c_void);
}

/// Forces the linker to retain the C shim (and therefore resolve every
/// llama.cpp symbol it references). Never called at runtime — taking the
/// address of an FFI symbol is enough to make the final link exercise the
/// static archives. See `tests/link_probe.rs` for the executable link proof.
pub fn _link_probe() -> usize {
    let mut n: usize = 0;
    n += load_model as *const () as usize;
    n += llama_allocate_params as *const () as usize;
    n += llama_free_params as *const () as usize;
    n += llama_binding_free_model as *const () as usize;
    n += llama_tokenize_string as *const () as usize;
    n += llama_predict as *const () as usize;
    n += llama_predict_full as *const () as usize;
    n += apply_chat_template as *const () as usize;
    n += eval as *const () as usize;
    n += load_state as *const () as usize;
    n += save_state as *const () as usize;
    n += get_embeddings as *const () as usize;
    n += get_token_embeddings as *const () as usize;
    n += speculative_sampling as *const () as usize;
    n += mtmd_load as *const () as usize;
    n += mtmd_describe as *const () as usize;
    n += mtmd_free_ctx as *const () as usize;
    n
}
