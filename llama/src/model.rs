//! Safe wrapper over the raw llama.cpp FFI (`llama_sys`).
//!
//! Ports the behavior of go-llama.cpp's `llama.go` `LLama` type: `New` →
//! [`Model::load`], `Free` → `Drop`, `TokenizeString` → [`Model::tokenize`],
//! `PredictResult` → [`Model::predict`], `ApplyChatTemplate` →
//! [`Model::apply_chat_template`], plus the not-yet-implemented surfaces
//! (`LoadState`, `SaveState`, `Embeddings`, `TokenEmbeddings`,
//! `SpeculativeSampling`) as `Err(LlamaError::NotImplemented)` stubs.
//!
//! Streaming (`SetTokenCallback` / the `tokenCallback` FFI export) and the
//! end-to-end smoke test are handled by separate work items and are not part
//! of this module.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Mutex;

use llama_sys::{
    apply_chat_template, llama_allocate_params, llama_binding_free_model, llama_free_params,
    llama_predict_full, llama_tokenize_string, load_model, mtmd_describe, mtmd_load,
};

use crate::error::{LlamaError, Result};
use crate::logit_bias;
use crate::options::{ModelOptions, PredictOptions};
use crate::stream::Filter;
use crate::vision::VisionModel;

/// Newtype around the raw trait-object pointer stored in the registry.
/// `*mut dyn FnMut(&str) -> bool` is not `Send`/`Sync` by default (it is a
/// raw pointer to a foreign, possibly-!Send closure); this wrapper asserts
/// the invariant documented on [`CALLBACKS`] so `Mutex<HashMap<_, RawCb>>`
/// itself is `Sync` and usable in a `static`.
#[derive(Clone, Copy)]
struct RawCb(*mut (dyn FnMut(&str) -> bool + 'static));

// SAFETY: the pointee is only ever dereferenced from within `tokenCallback`,
// which runs synchronously on the same thread that is inside the blocking
// `Model::predict_stream` / `llama_predict_full` call that registered it
// (llama.cpp's generation loop is single-threaded w.r.t. one context), so no
// cross-thread access to the pointee ever actually occurs even though the
// pointer value itself is moved between the registering thread and the
// (identical) callback thread while under the `Mutex`.
unsafe impl Send for RawCb {}
// SAFETY: see above; the `Mutex` additionally serializes all access to the
// map holding these pointers.
unsafe impl Sync for RawCb {}

/// Global registry of per-context streaming callbacks, mirroring Go's
/// `callbacks map[uintptr]func(string) bool` in `llama.go` (lines 445-473).
/// Keyed by the `binding_state*` pointer (as `usize`) so the C-called
/// `tokenCallback` can recover the right Rust closure for the context that
/// is currently generating.
///
/// The registry stores a raw pointer to a `dyn FnMut(&str) -> bool` that is
/// owned by the stack frame of `Model::predict_stream` for the duration of
/// the call (never by the registry itself) — `register_callback`/
/// `unregister_callback` are always paired via `CallbackGuard` so the entry
/// never outlives the closure it points at.
static CALLBACKS: Mutex<Option<HashMap<usize, RawCb>>> = Mutex::new(None);

/// Registers `callback` for `state`, mirroring Go's `setCallback`.
///
/// # Safety
/// `callback` must remain valid (i.e. the pointee must not move or be
/// dropped) for as long as it stays registered. Callers must pair this with
/// [`unregister_callback`] before the referent is dropped — `CallbackGuard`
/// does this automatically.
unsafe fn register_callback(state: *mut c_void, callback: *mut (dyn FnMut(&str) -> bool + 'static)) {
    let mut guard = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(HashMap::new).insert(state as usize, RawCb(callback));
}

/// Removes the callback registered for `state`, mirroring Go's
/// `setCallback(state, nil)`.
fn unregister_callback(state: *mut c_void) {
    let mut guard = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = guard.as_mut() {
        map.remove(&(state as usize));
    }
}

/// RAII guard that unregisters a context's streaming callback on drop, even
/// if generation returns early via `?` — mirrors Go's `defer
/// setCallback(l.state, prev)` in `Predict`/`PredictResult`.
struct CallbackGuard {
    state: *mut c_void,
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        unregister_callback(self.state);
    }
}

/// Definition of the `tokenCallback` symbol the C shim CALLS during
/// generation (declared, not defined, in `llama_sys`; see that crate's doc
/// comment). Linking any code path that reaches `generate()` in
/// `binding.cpp` (i.e. `llama_predict`/`llama_predict_full`) requires this
/// symbol to be resolvable.
///
/// Recovers the Rust callback registered for `state` (via
/// `Model::predict_stream`) from the global `CALLBACKS` registry and invokes
/// it with the decoded token piece, mirroring Go's `tokenCallback`
/// (`llama.go` lines 450-460): if no callback is registered for `state`,
/// generation is allowed to continue (returns 1), matching Go's `return
/// true` fallthrough. Returns `1` to continue, `0` to stop, matching the C
/// shim's `unsigned char` contract (Go: `true`/`false`).
///
/// # Safety
///
/// `token` must be a valid, NUL-terminated buffer for the duration of the
/// call. This holds when invoked by the C shim (`binding.cpp`), which writes
/// `buf.push_back('\0')` before calling; this function must never be called
/// directly from Rust with an arbitrary pointer.
#[no_mangle]
pub unsafe extern "C" fn tokenCallback(state: *mut c_void, token: *mut c_char) -> u8 {
    // SAFETY: `token` is a NUL-terminated buffer written by `generate()` in
    // binding.cpp (`buf.push_back('\0')` before the call) and is valid for
    // the duration of this call.
    let piece = unsafe { CStr::from_ptr(token) }.to_string_lossy();

    let guard = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner());
    let cb_ptr = match guard.as_ref().and_then(|m| m.get(&(state as usize))) {
        Some(p) => p.0,
        None => return 1,
    };
    drop(guard);

    // SAFETY: `cb_ptr` was registered by `Model::predict_stream`, which
    // keeps the referent alive (on its own stack frame) and unregisters it
    // via `CallbackGuard` before returning; since `tokenCallback` is only
    // ever invoked synchronously from within that same still-running call,
    // the pointee is guaranteed live here.
    let cb: &mut dyn FnMut(&str) -> bool = unsafe { &mut *cb_ptr };
    if cb(&piece) {
        1
    } else {
        0
    }
}

/// A loaded llama.cpp model + inference context.
///
/// Field-for-field analogue of Go's `LLama` (`llama.go` lines 66-70): an
/// opaque state pointer plus the `embeddings` flag the model was loaded
/// with (needed to reproduce `ErrEmbeddingsDisabled`).
///
/// Not `Send`/`Sync`: llama.cpp's context is not documented as safe for
/// concurrent/cross-thread use (Go's doc comment says the same: "not safe
/// for concurrent use by multiple goroutines"), so this type is left with
/// its natural `*mut c_void`-derived `!Send + !Sync` auto traits.
pub struct Model {
    ptr: *mut c_void,
    embeddings: bool,
}

impl Model {
    /// Loads the GGUF model at `path` per `opts`. Mirrors Go's `New`
    /// (`llama.go` lines 76-105): marshals all 19 `load_model` args,
    /// defaulting `mul_mat_q` to `true` when `opts.mul_mat_q` is `None`
    /// (Go's local `MulMatQ := true` fallback).
    ///
    /// Returns `Err(LlamaError::ModelLoad)` if the underlying call returns a
    /// null pointer, or if any option string contains an embedded NUL (Go's
    /// `C.CString` would panic on that; failing with `ModelLoad` here is the
    /// safe-Rust equivalent).
    pub fn load(path: &str, opts: &ModelOptions) -> Result<Model> {
        let c_path = CString::new(path).map_err(|_| LlamaError::ModelLoad)?;
        let c_main_gpu = CString::new(opts.main_gpu.as_str()).map_err(|_| LlamaError::ModelLoad)?;
        let c_tensor_split =
            CString::new(opts.tensor_split.as_str()).map_err(|_| LlamaError::ModelLoad)?;
        let c_lora_adapter =
            CString::new(opts.lora_adapter.as_str()).map_err(|_| LlamaError::ModelLoad)?;
        let c_lora_base =
            CString::new(opts.lora_base.as_str()).map_err(|_| LlamaError::ModelLoad)?;

        // Go: `MulMatQ := true; if mo.MulMatQ != nil { MulMatQ = *mo.MulMatQ }`.
        let mul_mat_q = opts.mul_mat_q.unwrap_or(true);

        // SAFETY: every pointer passed is either `c_path.as_ptr()` or one of
        // the CStrings above, all of which stay alive for the duration of
        // this call (they are not dropped until after `load_model` returns).
        // `load_model` does not retain any of these pointers past the call.
        let ptr = unsafe {
            load_model(
                c_path.as_ptr(),
                opts.context_size,
                opts.seed,
                opts.f16_memory,
                opts.mlock,
                opts.embeddings,
                opts.mmap,
                opts.low_vram,
                opts.n_gpu_layers,
                opts.n_batch,
                c_main_gpu.as_ptr(),
                c_tensor_split.as_ptr(),
                opts.numa,
                opts.freq_rope_base,
                opts.freq_rope_scale,
                mul_mat_q,
                c_lora_adapter.as_ptr(),
                c_lora_base.as_ptr(),
                opts.perplexity,
            )
        };

        if ptr.is_null() {
            return Err(LlamaError::ModelLoad);
        }

        Ok(Model {
            ptr,
            embeddings: opts.embeddings,
        })
    }

    /// Builds a `llama_allocate_params` handle from `prompt` + `opts`,
    /// mirroring the marshalling shared by every generation call site in
    /// `llama.go` (`cLogitBias` + the inline antiprompt array + the fixed
    /// 40-argument call). `n_predict` is passed explicitly because each Go
    /// call site computes its own effective token budget when
    /// `opts.tokens == 0` (`TokenizeString` uses 4096, `Predict`/
    /// `PredictResult` use 99_999_999) before it ever reaches this shared
    /// shape.
    ///
    /// `pass_antiprompt` selects which of Go's two marshalling shapes to
    /// reproduce:
    /// - `true` — marshal `opts.stop_prompts` into the C `antiprompt` array,
    ///   matching Go's `Eval`/`TokenizeString` call sites (`llama.go` lines
    ///   178, 398), which have no Rust-side stop filter of their own.
    /// - `false` — pass `nil, 0` for the antiprompt array, matching Go's
    ///   `Predict`/`PredictResult` (`llama.go` lines 260, 324): stop
    ///   detection instead happens Rust-side via `stream::Filter`, so the C
    ///   antiprompt scan is redundant and skipped, mirroring Go exactly.
    fn allocate_params(
        &self,
        prompt: &str,
        opts: &PredictOptions,
        n_predict: i32,
        pass_antiprompt: bool,
    ) -> Result<ParamsGuard> {
        let c_prompt = CString::new(prompt).map_err(|_| LlamaError::Inference)?;
        let c_path_prompt_cache =
            CString::new(opts.path_prompt_cache.as_str()).map_err(|_| LlamaError::Inference)?;
        let c_main_gpu = CString::new(opts.main_gpu.as_str()).map_err(|_| LlamaError::Inference)?;
        let c_tensor_split =
            CString::new(opts.tensor_split.as_str()).map_err(|_| LlamaError::Inference)?;
        let c_grammar = CString::new(opts.grammar.as_str()).map_err(|_| LlamaError::Inference)?;
        let c_negative_prompt =
            CString::new(opts.negative_prompt.as_str()).map_err(|_| LlamaError::Inference)?;

        // Antiprompt array: `opts.stop_prompts` -> Vec<CString> (kept alive by
        // the guard) -> Vec<*const c_char> passed as `const char **`. When
        // `pass_antiprompt` is false (the streaming/predict path), Go passes
        // `nil, 0` instead (see the `allocate_params` doc comment above), so
        // the CStrings/pointers are still built (kept in the guard for
        // lifetime uniformity) but the pointer/count handed to the FFI are
        // forced to null/0.
        let antiprompt_cstrings: Vec<CString> = opts
            .stop_prompts
            .iter()
            .map(|s| CString::new(s.as_str()))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| LlamaError::Inference)?;
        let mut antiprompt_ptrs: Vec<*const c_char> =
            antiprompt_cstrings.iter().map(|c| c.as_ptr()).collect();
        let (antiprompt_arg, antiprompt_count): (*mut *const c_char, c_int) = if !pass_antiprompt
            || antiprompt_ptrs.is_empty()
        {
            (ptr::null_mut(), 0)
        } else {
            (antiprompt_ptrs.as_mut_ptr(), antiprompt_ptrs.len() as c_int)
        };

        // Logit bias: parse "<token>:<bias>,..."; a malformed spec is logged
        // and ignored (never aborts generation), matching Go's `cLogitBias`.
        let (logit_bias_tokens, logit_bias_values) = match logit_bias::parse(&opts.logit_bias) {
            Ok(entries) => {
                let toks: Vec<i32> = entries.iter().map(|e| e.token).collect();
                let vals: Vec<f32> = entries.iter().map(|e| e.bias).collect();
                (toks, vals)
            }
            Err(err) => {
                eprintln!(
                    "llama: ignoring malformed logit_bias {:?}: {}",
                    opts.logit_bias, err
                );
                (Vec::new(), Vec::new())
            }
        };
        let logit_bias_count = logit_bias_tokens.len() as c_int;
        let logit_bias_tokens_ptr = if logit_bias_tokens.is_empty() {
            ptr::null()
        } else {
            logit_bias_tokens.as_ptr()
        };
        let logit_bias_values_ptr = if logit_bias_values.is_empty() {
            ptr::null()
        } else {
            logit_bias_values.as_ptr()
        };

        // SAFETY: every pointer argument is backed by a CString/Vec owned by
        // this stack frame or moved into the returned ParamsGuard, so all of
        // them outlive this call. `llama_allocate_params` copies what it
        // needs synchronously and does not retain the input pointers.
        let params_ptr = unsafe {
            llama_allocate_params(
                c_prompt.as_ptr(),
                opts.seed,
                opts.threads,
                n_predict,
                opts.top_k,
                opts.top_p,
                opts.temperature,
                opts.penalty,
                opts.repeat,
                opts.ignore_eos,
                opts.f16_kv,
                opts.batch,
                opts.n_keep,
                antiprompt_arg,
                antiprompt_count,
                opts.tail_free_sampling_z,
                opts.typical_p,
                opts.frequency_penalty,
                opts.presence_penalty,
                opts.mirostat,
                opts.mirostat_eta,
                opts.mirostat_tau,
                opts.penalize_nl,
                c_path_prompt_cache.as_ptr(),
                opts.prompt_cache_all,
                opts.mlock,
                opts.mmap,
                c_main_gpu.as_ptr(),
                c_tensor_split.as_ptr(),
                opts.prompt_cache_ro,
                c_grammar.as_ptr(),
                opts.rope_freq_base,
                opts.rope_freq_scale,
                opts.negative_prompt_scale,
                c_negative_prompt.as_ptr(),
                opts.n_draft,
                opts.min_p,
                logit_bias_tokens_ptr,
                logit_bias_values_ptr,
                logit_bias_count,
            )
        };

        Ok(ParamsGuard {
            ptr: params_ptr,
            _prompt: c_prompt,
            _path_prompt_cache: c_path_prompt_cache,
            _main_gpu: c_main_gpu,
            _tensor_split: c_tensor_split,
            _grammar: c_grammar,
            _negative_prompt: c_negative_prompt,
            _antiprompt_cstrings: antiprompt_cstrings,
            _antiprompt_ptrs: antiprompt_ptrs,
            _logit_bias_tokens: logit_bias_tokens,
            _logit_bias_values: logit_bias_values,
        })
    }

    /// Tokenizes `text`, returning the token IDs. Mirrors Go's
    /// `TokenizeString` (`llama.go` lines 382-426): the output buffer is
    /// sized from `opts.tokens` (falling back to 4096 when `0`, same as
    /// Go), and the FFI's returned count is clamped to that buffer length.
    ///
    /// A negative return from `llama_tokenize_string` maps to
    /// `Err(LlamaError::Inference)`.
    pub fn tokenize(&self, text: &str, opts: &PredictOptions) -> Result<Vec<i32>> {
        let effective_tokens = if opts.tokens == 0 { 4096 } else { opts.tokens };
        // Matches Go's `TokenizeString` (`llama.go` line 398): the real
        // antiprompt list is not marshalled there either (it passes a nil
        // double-pointer), so `pass_antiprompt = false` here reproduces that
        // exactly (equivalent in effect to Go's fakeDblPtr/0 pair).
        let guard = self.allocate_params(text, opts, effective_tokens, false)?;

        let cap = effective_tokens.max(0) as usize;
        let mut buf: Vec<c_int> = vec![0; cap];

        // SAFETY: `buf` has `cap` c_int slots; llama_tokenize_string writes
        // at most `cap` of them (bounded by the `n_predict` it received via
        // `params_ptr`), matching binding.cpp's `llama_tokenize_string`.
        let n = unsafe {
            llama_tokenize_string(
                guard.ptr,
                self.ptr,
                buf.as_mut_ptr(),
            )
        };

        if n < 0 {
            return Err(LlamaError::Inference);
        }

        let len = (n as usize).min(buf.len());
        Ok(buf[..len].to_vec())
    }

    /// Generates a completion for `prompt` via `llama_predict_full`,
    /// implementing the resize-retry contract documented at `binding.h`
    /// lines 61-66: the FFI returns the FULL generated length; if that
    /// length is `>=` the buffer capacity, the buffer is grown to
    /// `length + 1` and the call is retried exactly once.
    ///
    /// Effective token budget mirrors Go's `Predict`/`PredictResult`
    /// (`llama.go` lines 229-230, 303-304): `opts.tokens == 0` becomes
    /// `99_999_999` (effectively unbounded).
    ///
    /// Maps a negative FFI return to `Err(LlamaError::Inference)`, and a
    /// failed buffer allocation to `Err(LlamaError::OutOfMemory)` (the
    /// `Vec` allocator aborts on OOM in practice, but the buffer-size
    /// overflow guard below returns `OutOfMemory` rather than attempting an
    /// unbounded allocation).
    pub fn predict(&self, prompt: &str, opts: &PredictOptions) -> Result<String> {
        let effective_tokens = if opts.tokens == 0 { 99_999_999 } else { opts.tokens };
        // Matches Go's `Predict`/`PredictResult` (`llama.go` lines 260, 324):
        // both pass `nil, 0` for the antiprompt array because stop-sequence
        // detection happens Go-side via the filtering sink, not in C. This
        // non-streaming `predict` has no Rust-side stop filter of its own
        // (it returns the model's full, unfiltered output), but antiprompt
        // parity with Go is kept regardless per the task's requirement.
        let guard = self.allocate_params(prompt, opts, effective_tokens, false)?;

        const INITIAL_SIZE: usize = 8192;
        let (full_len, buf) = self.predict_full_once(&guard, INITIAL_SIZE, opts.debug_mode)?;

        if full_len < INITIAL_SIZE {
            return Ok(decode_nul_terminated(&buf, full_len));
        }

        // Buffer was too small: grow to the reported full length and retry
        // exactly once, per binding.h's documented contract.
        let retry_size = full_len
            .checked_add(1)
            .ok_or(LlamaError::OutOfMemory)?;
        let (full_len, buf) = self.predict_full_once(&guard, retry_size, opts.debug_mode)?;

        if full_len >= retry_size {
            // Still didn't fit after the one permitted retry.
            return Err(LlamaError::Inference);
        }
        Ok(decode_nul_terminated(&buf, full_len))
    }

    /// Single `llama_predict_full` call into a freshly allocated
    /// `size`-byte buffer. Returns the FFI's reported FULL length alongside
    /// the buffer so the caller can decide whether to retry.
    fn predict_full_once(
        &self,
        guard: &ParamsGuard,
        size: usize,
        debug: bool,
    ) -> Result<(usize, Vec<u8>)> {
        if size == 0 || size > i32::MAX as usize {
            return Err(LlamaError::OutOfMemory);
        }
        let mut buf: Vec<u8> = vec![0u8; size];
        let mut n_tokens: c_int = 0;

        // SAFETY: `buf` has `size` bytes of capacity, which is also what is
        // passed as `result_size`; llama_predict_full never writes more than
        // `result_size - 1` bytes plus a NUL terminator (binding.cpp lines
        // 403-410).
        let full_len = unsafe {
            llama_predict_full(
                guard.ptr,
                self.ptr,
                buf.as_mut_ptr() as *mut c_char,
                size as c_int,
                &mut n_tokens,
                debug,
            )
        };

        if full_len < 0 {
            return Err(LlamaError::Inference);
        }
        Ok((full_len as usize, buf))
    }

    /// Generates a completion for `prompt` while streaming decoded pieces to
    /// `on_token` as they arrive. Mirrors Go's `Predict` +
    /// `SetTokenCallback`/`tokenCallback` combination (`llama.go` lines
    /// 207-282, 450-460):
    ///
    /// - Stop-sequence detection happens Rust-side via `stream::Filter`
    ///   (built from `opts.stop_prompts`), so `allocate_params` is called
    ///   with `pass_antiprompt = false` — matching Go's `nil, 0` antiprompt
    ///   array on this call site.
    /// - Each decoded piece the C shim reports (via the global `tokenCallback`
    ///   dispatch) is pushed through the `Filter`; text the filter deems
    ///   safe to emit is forwarded to `on_token` and appended to the
    ///   accumulated result. When the filter reports a stop match, `on_token`
    ///   is not called again and generation is signalled to stop.
    /// - `on_token` returning `false` (Go: "the predictor will return")
    ///   stops generation early, exactly like Go's callback contract.
    /// - The callback is registered/unregistered via `CallbackGuard`, which
    ///   unregisters on every exit path (including early return via `?`),
    ///   mirroring Go's `defer setCallback(l.state, prev)`.
    ///
    /// Returns the accumulated emitted text (post-filter, i.e. with any
    /// matched stop sequence and everything after it excluded), or
    /// `Err(LlamaError::Inference)` if the underlying FFI call fails.
    pub fn predict_stream(
        &self,
        prompt: &str,
        opts: &PredictOptions,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let effective_tokens = if opts.tokens == 0 { 99_999_999 } else { opts.tokens };
        let guard = self.allocate_params(prompt, opts, effective_tokens, false)?;

        let mut filter = Filter::new(opts.stop_prompts.clone());
        let mut accumulated = String::new();
        let mut stopped = false;

        // `dispatch` is what actually runs on the C callback thread (i.e.
        // synchronously inside `llama_predict_full` below): it pushes the
        // piece through `filter`, forwards any newly-safe-to-emit text to
        // `on_token`, and returns whether generation should continue.
        let mut dispatch = |piece: &str| -> bool {
            if stopped {
                return false;
            }
            let (emit, stop) = filter.push(piece);
            if !emit.is_empty() {
                accumulated.push_str(&emit);
                if !on_token(&emit) {
                    stopped = true;
                    return false;
                }
            }
            if stop {
                stopped = true;
                return false;
            }
            true
        };

        let trait_obj: &mut (dyn FnMut(&str) -> bool + 'static) =
            // SAFETY: this reference is transmuted-by-cast to a raw pointer
            // and registered below only for the duration of this function
            // call; `CallbackGuard` unregisters it (via `state`) before
            // `dispatch`'s stack frame (and hence `filter`/`accumulated`/
            // `stopped`, which it borrows) goes out of scope. The lifetime
            // erasure to `'static` is sound only because of that strict
            // scoping, enforced by the guard running in `Drop` even on an
            // early `?` return below.
            unsafe { std::mem::transmute(&mut dispatch as &mut dyn FnMut(&str) -> bool) };
        let cb_ptr: *mut (dyn FnMut(&str) -> bool + 'static) = trait_obj;

        // SAFETY: `cb_ptr` points at `dispatch`, which outlives this whole
        // function body (it is dropped only when the function returns); the
        // `_guard` below unregisters the entry no later than that, and
        // `self.ptr` is a live, exclusively-owned context pointer for the
        // lifetime of `self`.
        unsafe { register_callback(self.ptr, cb_ptr) };
        let _guard = CallbackGuard { state: self.ptr };

        const SCRATCH: usize = 32768;
        let mut buf: Vec<u8> = vec![0u8; SCRATCH];
        let mut n_tokens: c_int = 0;

        // SAFETY: `buf` has `SCRATCH` bytes of capacity, which is also what
        // is passed as `result_size`; the C result buffer's content is
        // ignored (the accumulated, filtered text comes from `dispatch`
        // instead), only the return code is checked. While this call runs,
        // every generated piece flows through the global `tokenCallback`,
        // which looks up and invokes `dispatch` via the registry entry
        // registered just above.
        let ret = unsafe {
            llama_predict_full(
                guard.ptr,
                self.ptr,
                buf.as_mut_ptr() as *mut c_char,
                SCRATCH as c_int,
                &mut n_tokens,
                opts.debug_mode,
            )
        };

        drop(_guard);

        if ret < 0 {
            return Err(LlamaError::Inference);
        }

        if !stopped {
            accumulated.push_str(&filter.flush());
        }

        Ok(accumulated)
    }

    /// Formats `(system, user)` using the model's embedded GGUF chat
    /// template. Mirrors Go's `ApplyChatTemplate` (`llama.go` lines
    /// 357-378): grows the result buffer and retries when the FFI reports a
    /// larger required size, returns `Ok(None)` when the model has no
    /// template (FFI returns `0`), and maps a negative return to
    /// `Err(LlamaError::Inference)`.
    pub fn apply_chat_template(&self, system: &str, user: &str) -> Result<Option<String>> {
        let c_system = CString::new(system).map_err(|_| LlamaError::Inference)?;
        let c_user = CString::new(user).map_err(|_| LlamaError::Inference)?;

        let mut size: usize = 8192;
        // Bounded retry loop: Go's version grows and retries unboundedly on
        // a too-small buffer; a generous fixed bound here prevents a
        // pathological/buggy FFI response from looping forever.
        const MAX_ATTEMPTS: u32 = 8;

        for _ in 0..MAX_ATTEMPTS {
            let mut buf: Vec<u8> = vec![0u8; size];

            // SAFETY: `buf` has `size` bytes of capacity, passed as
            // `result_size`; apply_chat_template never writes past that
            // capacity (binding.cpp lines 414-440).
            let n = unsafe {
                apply_chat_template(
                    self.ptr,
                    c_system.as_ptr(),
                    c_user.as_ptr(),
                    buf.as_mut_ptr() as *mut c_char,
                    size as c_int,
                )
            };

            if n < 0 {
                return Err(LlamaError::Inference);
            }
            if n == 0 {
                return Ok(None);
            }
            let n = n as usize;
            if n <= size {
                return Ok(Some(decode_nul_terminated(&buf, n)));
            }
            size = n.checked_add(1).ok_or(LlamaError::OutOfMemory)?;
        }

        Err(LlamaError::Inference)
    }

    /// Loads a multimodal projector (`mmproj` GGUF) at `mmproj_path` on top of
    /// this model, returning a [`VisionModel`] that can be used with
    /// [`Model::describe_image`].
    ///
    /// This is the ADDITIVE vision surface (the text side has no analogue in
    /// go-llama.cpp). The returned `VisionModel` internally references this
    /// `Model`'s `llama_model`/`llama_context`, so **this `Model` must outlive
    /// the returned `VisionModel`** and every `describe_image` call must pass
    /// the same `Model` (see [`VisionModel`]'s safety notes).
    ///
    /// Returns `Err(LlamaError::VisionLoad)` if the projector cannot be loaded,
    /// if the model has no vision support, or if `mmproj_path` contains an
    /// embedded NUL.
    pub fn load_mmproj(&self, mmproj_path: &str) -> Result<VisionModel> {
        let c_path = CString::new(mmproj_path).map_err(|_| LlamaError::VisionLoad)?;
        // SAFETY: `self.ptr` is a live shim `binding_state*` owned by this
        // `Model`; `c_path` outlives the call. `mtmd_load` copies what it
        // needs and does not retain the path pointer.
        let ctx = unsafe { mtmd_load(self.ptr, c_path.as_ptr()) };
        if ctx.is_null() {
            return Err(LlamaError::VisionLoad);
        }
        // SAFETY: `ctx` is a non-null `mtmd_context*` freshly returned by
        // `mtmd_load` and owned exclusively by the new `VisionModel`.
        Ok(unsafe { VisionModel::from_raw(ctx) })
    }

    /// Describes the image at `image_path` using `vision` (loaded via
    /// [`Model::load_mmproj`] from this same `Model`) and the text `prompt`.
    ///
    /// Runs the mtmd evaluate + sampling flow in the C shim (image encode →
    /// chunk eval into the llama context → greedy generation), returning the
    /// generated description. Implements the same resize-retry contract as
    /// [`Model::predict`]: the FFI returns the FULL length, and the buffer is
    /// grown to `length + 1` and the call retried exactly once if the initial
    /// buffer was too small.
    ///
    /// # Safety requirement
    /// `vision` must have been created from `self` via [`Model::load_mmproj`];
    /// passing a `VisionModel` from a different `Model` is undefined behavior.
    ///
    /// Maps a negative FFI return to `Err(LlamaError::Inference)`.
    pub fn describe_image(
        &self,
        vision: &VisionModel,
        image_path: &str,
        prompt: &str,
    ) -> Result<String> {
        let c_image = CString::new(image_path).map_err(|_| LlamaError::Inference)?;
        let c_prompt = CString::new(prompt).map_err(|_| LlamaError::Inference)?;

        const INITIAL_SIZE: usize = 8192;
        let (full_len, buf) =
            self.describe_once(vision, &c_image, &c_prompt, INITIAL_SIZE)?;
        if full_len < INITIAL_SIZE {
            return Ok(decode_nul_terminated(&buf, full_len));
        }

        let retry_size = full_len.checked_add(1).ok_or(LlamaError::OutOfMemory)?;
        let (full_len, buf) =
            self.describe_once(vision, &c_image, &c_prompt, retry_size)?;
        if full_len >= retry_size {
            return Err(LlamaError::Inference);
        }
        Ok(decode_nul_terminated(&buf, full_len))
    }

    /// Single `mtmd_describe` call into a freshly allocated `size`-byte buffer.
    /// Returns the FFI's reported FULL length alongside the buffer.
    fn describe_once(
        &self,
        vision: &VisionModel,
        c_image: &CStr,
        c_prompt: &CStr,
        size: usize,
    ) -> Result<(usize, Vec<u8>)> {
        if size == 0 || size > i32::MAX as usize {
            return Err(LlamaError::OutOfMemory);
        }
        let mut buf: Vec<u8> = vec![0u8; size];
        let mut n_tokens: c_int = 0;

        // SAFETY: `buf` has `size` bytes, passed as `result_size`;
        // `mtmd_describe` never writes past `result_size - 1` plus a NUL.
        // `vision.ctx` and `self.ptr` are both live and originate from this
        // same `Model` (per the documented safety requirement). The CStrs
        // outlive the call.
        let full_len = unsafe {
            mtmd_describe(
                vision.ctx,
                self.ptr,
                c_image.as_ptr(),
                c_prompt.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                size as c_int,
                &mut n_tokens,
            )
        };
        if full_len < 0 {
            return Err(LlamaError::Inference);
        }
        Ok((full_len as usize, buf))
    }

    /// Restores a previously saved context state. Not yet implemented on
    /// the C shim; mirrors Go's `LoadState` (`llama.go` lines 113-119).
    pub fn load_state(&mut self, _state: &str) -> Result<()> {
        Err(LlamaError::NotImplemented)
    }

    /// Writes the current context state to a file. Not yet implemented on
    /// the C shim; mirrors Go's `SaveState` (`llama.go` lines 121-127).
    pub fn save_state(&self, _dst: &str) -> Result<()> {
        Err(LlamaError::NotImplemented)
    }

    /// Returns the embedding vector for `text`. Mirrors Go's `Embeddings`
    /// (`llama.go` lines 141-151): `Err(LlamaError::EmbeddingsDisabled)` if
    /// the model was loaded without `ModelOptions.embeddings`, otherwise
    /// `Err(LlamaError::NotImplemented)` (the C shim is a stub).
    pub fn embeddings(&self, _text: &str, _opts: &PredictOptions) -> Result<Vec<f32>> {
        if !self.embeddings {
            return Err(LlamaError::EmbeddingsDisabled);
        }
        Err(LlamaError::NotImplemented)
    }

    /// Returns the embedding vectors for `tokens`. Mirrors Go's
    /// `TokenEmbeddings` (`llama.go` lines 129-139).
    pub fn token_embeddings(&self, _tokens: &[i32], _opts: &PredictOptions) -> Result<Vec<f32>> {
        if !self.embeddings {
            return Err(LlamaError::EmbeddingsDisabled);
        }
        Err(LlamaError::NotImplemented)
    }

    /// Generates text using `self` as the target model and `draft` as the
    /// draft model. Mirrors Go's `SpeculativeSampling` (`llama.go` lines
    /// 198-205); the C shim is a stub.
    pub fn speculative_sampling(
        &self,
        _draft: &Model,
        _text: &str,
        _opts: &PredictOptions,
    ) -> Result<String> {
        Err(LlamaError::NotImplemented)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by a successful `load_model` call
        // in `Model::load` and is owned exclusively by this `Model` (no
        // other code holds or frees it), so freeing it here exactly once is
        // sound.
        unsafe { llama_binding_free_model(self.ptr) };
    }
}

/// Decodes up to `len` bytes of `buf` as UTF-8 (lossily), stopping early at
/// an embedded NUL if one is present before `len` — the FFI always
/// NUL-terminates what it writes.
fn decode_nul_terminated(buf: &[u8], len: usize) -> String {
    let len = len.min(buf.len());
    let slice = &buf[..len];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(len);
    String::from_utf8_lossy(&slice[..end]).into_owned()
}

/// Owns every allocation backing a `llama_allocate_params` handle
/// (`params_ptr`) so they stay alive for as long as the handle is in use,
/// and frees the handle itself on drop via `llama_free_params`.
struct ParamsGuard {
    ptr: *mut c_void,
    _prompt: CString,
    _path_prompt_cache: CString,
    _main_gpu: CString,
    _tensor_split: CString,
    _grammar: CString,
    _negative_prompt: CString,
    _antiprompt_cstrings: Vec<CString>,
    _antiprompt_ptrs: Vec<*const c_char>,
    _logit_bias_tokens: Vec<i32>,
    _logit_bias_values: Vec<f32>,
}

impl Drop for ParamsGuard {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by a successful
        // `llama_allocate_params` call and is owned exclusively by this
        // guard, so freeing it here exactly once is sound.
        unsafe { llama_free_params(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads the model named by `LLAMA_TEST_MODEL`, or returns `None` to
    /// skip the caller's test when the env var is unset. All real-inference
    /// tests in this module are gated through this helper so they compile
    /// unconditionally but only exercise the FFI when a GGUF is available.
    fn test_model() -> Option<Model> {
        let path = std::env::var("LLAMA_TEST_MODEL").ok()?;
        Some(Model::load(&path, &ModelOptions::default()).expect("failed to load LLAMA_TEST_MODEL"))
    }

    #[test]
    fn loads_model_when_env_set() {
        let Some(_model) = test_model() else {
            eprintln!("skipping loads_model_when_env_set: LLAMA_TEST_MODEL not set");
            return;
        };
    }

    #[test]
    fn tokenize_yields_nonempty_tokens() {
        let Some(model) = test_model() else {
            eprintln!("skipping tokenize_yields_nonempty_tokens: LLAMA_TEST_MODEL not set");
            return;
        };
        let opts = PredictOptions::default();
        let toks = model
            .tokenize("The quick brown fox", &opts)
            .expect("tokenize failed");
        assert!(!toks.is_empty());
    }

    #[test]
    fn predict_yields_nonempty_text() {
        let Some(model) = test_model() else {
            eprintln!("skipping predict_yields_nonempty_text: LLAMA_TEST_MODEL not set");
            return;
        };
        let opts = PredictOptions {
            temperature: 0.0,
            seed: 1,
            tokens: 16,
            ..PredictOptions::default()
        };
        let out = model.predict("The capital of France is", &opts).expect("predict failed");
        assert!(!out.is_empty());
    }

    #[test]
    fn predict_stream_matches_predict_up_to_stop() {
        let Some(model) = test_model() else {
            eprintln!("skipping predict_stream_matches_predict_up_to_stop: LLAMA_TEST_MODEL not set");
            return;
        };
        let opts = PredictOptions {
            temperature: 0.0,
            seed: 1,
            tokens: 32,
            stop_prompts: vec!["\n".to_string()],
            ..PredictOptions::default()
        };

        let non_streaming = model
            .predict("The capital of France is", &opts)
            .expect("predict failed");

        let mut streamed = String::new();
        let result = model
            .predict_stream("The capital of France is", &opts, &mut |piece| {
                streamed.push_str(piece);
                true
            })
            .expect("predict_stream failed");

        assert_eq!(result, streamed);
        // The non-streaming path has no Rust-side stop filter, so it may run
        // longer; the streamed (filtered, stopped) text must be a prefix of
        // what the unfiltered model produced up to that point.
        assert!(
            non_streaming.starts_with(&streamed) || streamed.starts_with(&non_streaming),
            "streamed={streamed:?} non_streaming={non_streaming:?}"
        );
    }

    #[test]
    fn predict_stream_callback_stops_generation_early() {
        let Some(model) = test_model() else {
            eprintln!("skipping predict_stream_callback_stops_generation_early: LLAMA_TEST_MODEL not set");
            return;
        };
        let opts = PredictOptions {
            temperature: 0.0,
            seed: 1,
            tokens: 64,
            ..PredictOptions::default()
        };

        let mut count = 0usize;
        const N: usize = 3;
        let result = model
            .predict_stream("The capital of France is", &opts, &mut |_piece| {
                count += 1;
                count < N
            })
            .expect("predict_stream failed");

        assert!(count <= N);
        assert!(!result.is_empty());
    }

    /// Loads the vision GGUF named by `LLAMA_TEST_VISION_MODEL`, or returns
    /// `None` to skip. Separate from `test_model()` because a vision model +
    /// projector are only present when explicitly configured.
    fn test_vision_model() -> Option<Model> {
        let path = std::env::var("LLAMA_TEST_VISION_MODEL").ok()?;
        Some(
            Model::load(&path, &ModelOptions::default())
                .expect("failed to load LLAMA_TEST_VISION_MODEL"),
        )
    }

    #[test]
    fn describe_image_yields_nonempty_description() {
        let Some(model) = test_vision_model() else {
            eprintln!(
                "skipping describe_image_yields_nonempty_description: LLAMA_TEST_VISION_MODEL not set"
            );
            return;
        };
        let Ok(mmproj) = std::env::var("LLAMA_TEST_MMPROJ") else {
            eprintln!(
                "skipping describe_image_yields_nonempty_description: LLAMA_TEST_MMPROJ not set"
            );
            return;
        };
        // Default to the committed fixture; allow override via env.
        let image = std::env::var("LLAMA_TEST_IMAGE").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/red_circle.png").to_string()
        });

        let vision = model
            .load_mmproj(&mmproj)
            .expect("load_mmproj failed (mmproj load / no vision support)");
        let desc = model
            .describe_image(&vision, &image, "Describe this image in one sentence.")
            .expect("describe_image failed");
        eprintln!("describe_image => {desc:?}");
        assert!(!desc.trim().is_empty(), "description was empty");
    }

    #[test]
    fn embeddings_without_enable_is_disabled() {
        let Some(model) = test_model() else {
            eprintln!("skipping embeddings_without_enable_is_disabled: LLAMA_TEST_MODEL not set");
            return;
        };
        // test_model() loads with ModelOptions::default(), whose `embeddings`
        // field is `false` (see options.rs), so this must report disabled
        // rather than not-implemented.
        let opts = PredictOptions::default();
        assert_eq!(
            model.embeddings("hello", &opts),
            Err(LlamaError::EmbeddingsDisabled)
        );
    }
}
