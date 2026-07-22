//! Port of go-llama.cpp's `options.go`: `ModelOptions` and `PredictOptions`.
//!
//! Go used functional-option closures (`ModelOption`, `PredictOption`) to
//! build these structs incrementally. Idiomatic Rust prefers plain structs
//! with a `Default` impl plus struct-update syntax (`ModelOptions { seed: 1,
//! ..Default::default() }`) or simple builder methods, so no closure-based
//! option types are ported here.

/// Options controlling how a model is loaded.
///
/// Field-for-field port of Go's `ModelOptions` (`options.go` lines 3-22).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    pub context_size: i32,
    pub seed: i32,
    pub n_batch: i32,
    pub f16_memory: bool,
    pub mlock: bool,
    pub mmap: bool,
    pub low_vram: bool,
    pub embeddings: bool,
    pub numa: bool,
    pub n_gpu_layers: i32,
    pub main_gpu: String,
    pub tensor_split: String,
    pub freq_rope_base: f32,
    pub freq_rope_scale: f32,
    /// `nil` by default in Go (unset); `None` here.
    pub mul_mat_q: Option<bool>,
    pub lora_base: String,
    pub lora_adapter: String,
    pub perplexity: bool,
}

impl Default for ModelOptions {
    /// Matches Go's `DefaultModelOptions` (`options.go` lines 65-76).
    fn default() -> Self {
        Self {
            context_size: 512,
            seed: 0,
            n_batch: 512,
            f16_memory: false,
            mlock: false,
            mmap: true,
            low_vram: false,
            embeddings: false,
            numa: false,
            n_gpu_layers: 0,
            main_gpu: String::new(),
            tensor_split: String::new(),
            freq_rope_base: 10000.0,
            freq_rope_scale: 1.0,
            mul_mat_q: None,
            lora_base: String::new(),
            lora_adapter: String::new(),
            perplexity: false,
        }
    }
}

/// Options controlling a single prediction/generation call.
///
/// Field-for-field port of Go's `PredictOptions` (`options.go` lines 24-59).
#[derive(Clone)]
pub struct PredictOptions {
    pub seed: i32,
    pub threads: i32,
    pub tokens: i32,
    pub top_k: i32,
    pub repeat: i32,
    pub batch: i32,
    pub n_keep: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub penalty: f32,
    pub n_draft: i32,
    pub f16_kv: bool,
    pub debug_mode: bool,
    pub stop_prompts: Vec<String>,
    pub ignore_eos: bool,

    /// No-op upstream: tail-free sampling was removed from llama.cpp's
    /// sampler API and is no longer wired into the sampling chain. Kept as a
    /// field for source parity with Go's `SetTailFreeSamplingZ`.
    pub tail_free_sampling_z: f32,
    pub typical_p: f32,
    pub min_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_eta: f32,
    pub mirostat_tau: f32,
    /// No-op upstream: newline penalization was folded into the unified
    /// penalties sampler and is no longer wired as a standalone knob. Kept
    /// for source parity with Go's `SetPenalizeNL`.
    pub penalize_nl: bool,
    pub logit_bias: String,
    /// Go's `TokenCallback func(string) bool`.
    pub token_callback: Option<std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>>,

    pub path_prompt_cache: String,
    pub mlock: bool,
    pub mmap: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,
    pub grammar: String,
    pub main_gpu: String,
    pub tensor_split: String,

    // Rope parameters
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,

    // Negative prompt parameters
    pub negative_prompt_scale: f32,
    pub negative_prompt: String,
}

impl std::fmt::Debug for PredictOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictOptions")
            .field("seed", &self.seed)
            .field("threads", &self.threads)
            .field("tokens", &self.tokens)
            .field("top_k", &self.top_k)
            .field("repeat", &self.repeat)
            .field("batch", &self.batch)
            .field("n_keep", &self.n_keep)
            .field("top_p", &self.top_p)
            .field("temperature", &self.temperature)
            .field("penalty", &self.penalty)
            .field("n_draft", &self.n_draft)
            .field("f16_kv", &self.f16_kv)
            .field("debug_mode", &self.debug_mode)
            .field("stop_prompts", &self.stop_prompts)
            .field("ignore_eos", &self.ignore_eos)
            .field("tail_free_sampling_z", &self.tail_free_sampling_z)
            .field("typical_p", &self.typical_p)
            .field("min_p", &self.min_p)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("presence_penalty", &self.presence_penalty)
            .field("mirostat", &self.mirostat)
            .field("mirostat_eta", &self.mirostat_eta)
            .field("mirostat_tau", &self.mirostat_tau)
            .field("penalize_nl", &self.penalize_nl)
            .field("logit_bias", &self.logit_bias)
            .field("token_callback", &self.token_callback.is_some())
            .field("path_prompt_cache", &self.path_prompt_cache)
            .field("mlock", &self.mlock)
            .field("mmap", &self.mmap)
            .field("prompt_cache_all", &self.prompt_cache_all)
            .field("prompt_cache_ro", &self.prompt_cache_ro)
            .field("grammar", &self.grammar)
            .field("main_gpu", &self.main_gpu)
            .field("tensor_split", &self.tensor_split)
            .field("rope_freq_base", &self.rope_freq_base)
            .field("rope_freq_scale", &self.rope_freq_scale)
            .field("negative_prompt_scale", &self.negative_prompt_scale)
            .field("negative_prompt", &self.negative_prompt)
            .finish()
    }
}

impl Default for PredictOptions {
    /// Matches Go's `DefaultOptions` (`options.go` lines 78-99).
    fn default() -> Self {
        Self {
            seed: -1,
            threads: 4,
            tokens: 128,
            top_k: 40,
            repeat: 64,
            batch: 512,
            n_keep: 64,
            top_p: 0.95,
            temperature: 0.8,
            penalty: 1.1,
            n_draft: 0,
            f16_kv: false,
            debug_mode: false,
            stop_prompts: Vec::new(),
            ignore_eos: false,

            tail_free_sampling_z: 1.0,
            typical_p: 1.0,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0,
            mirostat_eta: 0.1,
            mirostat_tau: 5.0,
            penalize_nl: false,
            logit_bias: String::new(),
            token_callback: None,

            path_prompt_cache: String::new(),
            mlock: false,
            mmap: true,
            prompt_cache_all: false,
            prompt_cache_ro: false,
            grammar: String::new(),
            main_gpu: String::new(),
            tensor_split: String::new(),

            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,

            negative_prompt_scale: 0.0,
            negative_prompt: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_defaults_match_go() {
        let d = ModelOptions::default();
        assert_eq!(d.context_size, 512);
        assert_eq!(d.n_batch, 512);
        assert_eq!(d.mmap, true);
        assert_eq!(d.freq_rope_base, 10000.0);
        assert_eq!(d.freq_rope_scale, 1.0);
        assert_eq!(d.seed, 0);
        assert_eq!(d.f16_memory, false);
        assert_eq!(d.mlock, false);
        assert_eq!(d.low_vram, false);
        assert_eq!(d.embeddings, false);
        assert_eq!(d.numa, false);
        assert_eq!(d.n_gpu_layers, 0);
        assert_eq!(d.main_gpu, "");
        assert_eq!(d.tensor_split, "");
        assert_eq!(d.mul_mat_q, None);
        assert_eq!(d.lora_base, "");
        assert_eq!(d.lora_adapter, "");
        assert_eq!(d.perplexity, false);
    }

    #[test]
    fn predict_defaults_match_go() {
        let d = PredictOptions::default();
        assert_eq!(d.seed, -1);
        assert_eq!(d.threads, 4);
        assert_eq!(d.tokens, 128);
        assert_eq!(d.top_k, 40);
        assert_eq!(d.temperature, 0.8);
        assert_eq!(d.penalty, 1.1);
        assert_eq!(d.repeat, 64);
        assert_eq!(d.batch, 512);
        assert_eq!(d.n_keep, 64);
        assert_eq!(d.top_p, 0.95);
        assert_eq!(d.n_draft, 0);
        assert_eq!(d.f16_kv, false);
        assert_eq!(d.debug_mode, false);
        assert!(d.stop_prompts.is_empty());
        assert_eq!(d.ignore_eos, false);
        assert_eq!(d.tail_free_sampling_z, 1.0);
        assert_eq!(d.typical_p, 1.0);
        assert_eq!(d.min_p, 0.0);
        assert_eq!(d.frequency_penalty, 0.0);
        assert_eq!(d.presence_penalty, 0.0);
        assert_eq!(d.mirostat, 0);
        assert_eq!(d.mirostat_tau, 5.0);
        assert_eq!(d.mirostat_eta, 0.1);
        assert_eq!(d.penalize_nl, false);
        assert_eq!(d.logit_bias, "");
        assert!(d.token_callback.is_none());
        assert_eq!(d.path_prompt_cache, "");
        assert_eq!(d.mlock, false);
        assert_eq!(d.mmap, true);
        assert_eq!(d.prompt_cache_all, false);
        assert_eq!(d.prompt_cache_ro, false);
        assert_eq!(d.grammar, "");
        assert_eq!(d.main_gpu, "");
        assert_eq!(d.tensor_split, "");
        assert_eq!(d.rope_freq_base, 10000.0);
        assert_eq!(d.rope_freq_scale, 1.0);
        assert_eq!(d.negative_prompt_scale, 0.0);
        assert_eq!(d.negative_prompt, "");
    }
}
