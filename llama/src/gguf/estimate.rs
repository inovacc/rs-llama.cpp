//! Derived from `github.com/ollama/ollama` (MIT License) memory-estimation
//! logic. Ported from `gguf/estimate.go`.

use std::collections::HashMap;

use super::gguf::File;
use super::graph::llama_graph_size;
use super::GgufError;

/// Mirrors Ollama's non-Metal `DeviceInfo.MinimumMemory`.
pub const DEFAULT_MIN_RESERVE_BYTES: u64 = 457 << 20;

/// Parameterizes a layer-fit estimate.
#[derive(Debug, Clone, Default)]
pub struct EstimateOptions {
    /// Context length in tokens (0 => 2048).
    pub num_ctx: i64,
    /// Tokens per batch (0 => 512).
    pub batch_size: i64,
    /// Concurrent sequences (0 => 1).
    pub num_parallel: i64,
    /// `"f16"` (default), `"q8_0"`, `"q4_0"`, `"f32"`.
    pub kv_cache_type: String,
    /// Caller-provided budget in bytes (required, > 0).
    pub free_vram: u64,
    /// Extra reserve on top of the minimum reserve.
    pub overhead_bytes: u64,
    /// Minimum reserve floor; 0 => `DEFAULT_MIN_RESERVE_BYTES`.
    pub min_reserve_bytes: u64,
    /// Accepted; does not yet alter the formula.
    pub flash_attention: bool,
}

/// The result of a layer-fit computation.
#[derive(Debug, Clone, Default)]
pub struct Estimate {
    /// Recommended n_gpu_layers (repeating blocks offloaded).
    pub layers: i64,
    /// True if all blocks + the output layer fit.
    pub fully_offloaded: bool,
    /// Weights+kv+graph actually placed on the GPU.
    pub total_vram: u64,
    /// Offloaded block weights (+ output if `fully_offloaded`).
    pub weights: u64,
    /// KV bytes for the offloaded blocks.
    pub kv_cache: u64,
    /// Compute buffer (full- or partial-offload value).
    pub graph: u64,
    /// Weights+kv per block, index `0..block_count-1`.
    pub per_layer_bytes: Vec<u64>,
    /// True when a non-llama arch used the fallback formula.
    pub approximate: bool,
}

/// Returns the per-element KV-cache byte size for a cache type.
fn kv_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "q8_0" => 1.0,
        "q4_0" => 0.5,
        "f32" => 4.0,
        _ => 2.0, // "f16" and unknown
    }
}

/// Sums repeating `blk.<i>.*` tensor bytes per block and the non-repeating
/// output layer (`output[_norm]`, else tied `token_embd`).
fn group_layers(f: &File) -> (HashMap<i64, u64>, u64) {
    let mut blocks: HashMap<i64, u64> = HashMap::new();
    let mut output_w: u64 = 0;
    let mut token_embd: u64 = 0;

    for (_, ti) in f.tensor_infos() {
        let name = ti.name.as_str();

        if let Some(rest) = name.strip_prefix("blk.") {
            let Some((idx_str, _)) = rest.split_once('.') else {
                continue;
            };
            let Ok(idx) = idx_str.parse::<i64>() else {
                continue;
            };
            *blocks.entry(idx).or_insert(0) += ti.num_bytes() as u64;
        } else if name.starts_with("output_norm") || name.starts_with("output.") {
            output_w += ti.num_bytes() as u64;
        } else if name.starts_with("token_embd") {
            token_embd += ti.num_bytes() as u64;
        }
    }

    if output_w == 0 {
        output_w = token_embd; // tied embeddings
    }

    (blocks, output_w)
}

fn or_default(v: i64, def: i64) -> i64 {
    if v <= 0 {
        def
    } else {
        v
    }
}

/// Opens `path`, reads metadata + tensor sizes, computes how many transformer
/// blocks fit in `opts.free_vram`, and closes the file. Single-GPU,
/// Llama-family graph model; non-llama dense architectures get an
/// approximate estimate (`Estimate::approximate`). Recurrent/SSM
/// architectures are unsupported.
pub fn estimate_layers(path: &str, opts: &EstimateOptions) -> Result<Estimate, GgufError> {
    if opts.free_vram == 0 {
        return Err(GgufError::Unsupported("FreeVRAM budget required".into()));
    }

    let f = File::open(path)?;

    // key_value auto-prefixes the architecture for keys not starting with
    // "general."/"tokenizer.", so "attention.head_count" resolves to
    // "<arch>.attention.head_count".
    let arch = f.key_value("general.architecture").string();
    let block_count = f.key_value("block_count").uint() as i64;

    let embedding = f.key_value("embedding_length").uint();
    if block_count == 0 || embedding == 0 {
        return Err(GgufError::Unsupported(
            "missing block_count/embedding_length".into(),
        ));
    }

    let heads = f.key_value("attention.head_count").uint();
    if heads == 0 {
        return Err(GgufError::Unsupported(
            "recurrent or unsupported architecture (no attention heads)".into(),
        ));
    }

    let mut heads_kv = f.key_value("attention.head_count_kv").uint();
    if heads_kv == 0 {
        heads_kv = heads;
    }

    let mut key_len = f.key_value("attention.key_length").uint();
    if key_len == 0 {
        key_len = embedding / heads;
    }

    let mut val_len = f.key_value("attention.value_length").uint();
    if val_len == 0 {
        val_len = embedding / heads;
    }

    let mut vocab = f.key_value("vocab_size").uint();
    if vocab == 0 {
        vocab = f.key_value("tokenizer.ggml.tokens").strings().len() as u64;
    }

    let ctx = or_default(opts.num_ctx, 2048) as u64 * or_default(opts.num_parallel, 1) as u64;
    let batch = or_default(opts.batch_size, 512) as u64;

    let (blocks, output) = group_layers(&f);
    let kv_per_layer =
        (ctx * (key_len + val_len) * heads_kv) as f64 * kv_bytes_per_element(&opts.kv_cache_type);
    let kv_per_layer = kv_per_layer as u64;
    let (full, partial) = llama_graph_size(embedding, heads, key_len, heads_kv, ctx, batch, vocab);

    let block_count_usize = block_count as usize;
    let mut per_layer = vec![0u64; block_count_usize];
    for (i, slot) in per_layer.iter_mut().enumerate() {
        *slot = blocks.get(&(i as i64)).copied().unwrap_or(0) + kv_per_layer;
    }

    let min_reserve = if opts.min_reserve_bytes == 0 {
        DEFAULT_MIN_RESERVE_BYTES
    } else {
        opts.min_reserve_bytes
    };
    let reserve = min_reserve + opts.overhead_bytes;

    let mut est = Estimate {
        per_layer_bytes: per_layer.clone(),
        approximate: arch != "llama",
        ..Default::default()
    };

    // Fill repeating blocks largest-cost-first while they fit under the
    // partial-offload graph budget. Dense blocks are equal-sized, so order
    // only matters under heterogeneity; largest-first biases conservative.
    let mut order: Vec<usize> = (0..block_count_usize).collect();
    order.sort_by(|&a, &b| per_layer[b].cmp(&per_layer[a]));

    let avail = opts.free_vram as i64 - reserve as i64 - partial as i64;

    let mut weights_used: u64 = 0;
    let mut kv_used: u64 = 0;
    let mut n: i64 = 0;

    if avail > 0 {
        for &i in &order {
            if (weights_used + kv_used + per_layer[i]) as i64 <= avail {
                weights_used += blocks.get(&(i as i64)).copied().unwrap_or(0);
                kv_used += kv_per_layer;
                n += 1;
            } else {
                break;
            }
        }
    }

    est.layers = n;
    est.graph = partial;

    if n == block_count {
        // All blocks fit: try to also offload the output layer under the
        // (larger) full-offload graph budget.
        let remaining = opts.free_vram as i64
            - reserve as i64
            - full as i64
            - (weights_used + kv_used + output) as i64;
        if remaining >= 0 {
            est.fully_offloaded = true;
            est.graph = full;
            weights_used += output;
        }
    }

    est.weights = weights_used;
    est.kv_cache = kv_used;
    est.total_vram = est.weights + est.kv_cache + est.graph;

    Ok(est)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testutil::{sample_llama_model, write_gguf, KvPair, TestTensor};

    // Ported from estimate_test.go: TestGroupLayers
    #[test]
    fn test_group_layers() {
        const N_BLOCKS: usize = 3;
        let f = File::open(sample_llama_model(N_BLOCKS)).expect("open");
        let (blocks, output) = group_layers(&f);
        assert_eq!(blocks.len(), N_BLOCKS);
        for i in 0..N_BLOCKS as i64 {
            assert_eq!(blocks[&i], 4096); // [64,16] F32
        }
        assert_eq!(output, 4096); // output.weight [64,16] F32
    }

    // Ported from estimate_test.go: TestKVBytesPerElement
    #[test]
    fn test_kv_bytes_per_element() {
        assert_eq!(kv_bytes_per_element("f16"), 2.0);
        assert_eq!(kv_bytes_per_element(""), 2.0);
        assert_eq!(kv_bytes_per_element("q8_0"), 1.0);
        assert_eq!(kv_bytes_per_element("q4_0"), 0.5);
        assert_eq!(kv_bytes_per_element("f32"), 4.0);
    }

    // Ported from estimate_test.go: TestEstimateRequiresBudget
    #[test]
    fn test_estimate_requires_budget() {
        let path = sample_llama_model(4);
        let err = estimate_layers(
            path.to_str().unwrap(),
            &EstimateOptions {
                free_vram: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, GgufError::Unsupported(_)));
    }

    // Ported from estimate_test.go: fixtureGraph — computes the graph + KV +
    // per-block-weight the estimator will use for the sample_llama_model
    // fixture, so fit assertions don't hard-code the graph literal. Dims must
    // match sample_llama_model: embedding=64 heads=8 headsKV=8 key=val=8
    // vocab=128.
    fn fixture_graph(opts: &EstimateOptions) -> (u64, u64, u64) {
        let ctx = or_default(opts.num_ctx, 2048) as u64 * or_default(opts.num_parallel, 1) as u64;
        let batch = or_default(opts.batch_size, 512) as u64;
        let (_, partial) = llama_graph_size(64, 8, 8, 8, ctx, batch, 128);
        let kv_per_layer =
            (ctx * (8 + 8) * 8) as f64 * kv_bytes_per_element(&opts.kv_cache_type);
        let kv_per_layer = kv_per_layer as u64;
        let block_weight = 4096u64;
        (partial, kv_per_layer, block_weight)
    }

    // Ported from estimate_test.go: TestEstimateFullOffload
    #[test]
    fn test_estimate_full_offload() {
        let opts = EstimateOptions {
            free_vram: 1u64 << 40, // 1 TiB
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let est = estimate_layers(sample_llama_model(4).to_str().unwrap(), &opts).unwrap();
        assert_eq!(est.layers, 4);
        assert!(est.fully_offloaded);
        assert!(!est.approximate);
    }

    // Ported from estimate_test.go: TestEstimateNoFit
    #[test]
    fn test_estimate_no_fit() {
        let opts = EstimateOptions {
            free_vram: 1024,
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let est = estimate_layers(sample_llama_model(4).to_str().unwrap(), &opts).unwrap();
        assert_eq!(est.layers, 0);
        assert!(!est.fully_offloaded);
    }

    // Ported from estimate_test.go: TestEstimatePartialFit
    #[test]
    fn test_estimate_partial_fit() {
        let mut opts = EstimateOptions {
            num_ctx: 2048,
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let (partial, kv, w) = fixture_graph(&opts);
        let per_layer = kv + w;
        // reserve + graph + exactly 2 layers + slack
        opts.free_vram = 4096 + partial + 2 * per_layer + 8;

        let est = estimate_layers(sample_llama_model(4).to_str().unwrap(), &opts).unwrap();
        assert_eq!(est.layers, 2);
        assert!(!est.fully_offloaded);
        assert_eq!(est.kv_cache, 2 * kv);
    }

    // Ported from estimate_test.go: TestEstimateKVQuantHalvesF16
    #[test]
    fn test_estimate_kv_quant_halves_f16() {
        let base = EstimateOptions {
            num_ctx: 2048,
            min_reserve_bytes: 4096,
            free_vram: 1u64 << 40,
            ..Default::default()
        };
        let (_, kv_f16, _) = fixture_graph(&base);
        let mut q = base.clone();
        q.kv_cache_type = "q8_0".into();
        let (_, kv_q8, _) = fixture_graph(&q);
        assert_eq!(kv_q8, kv_f16 / 2);
    }

    // Ported from estimate_test.go: TestEstimateApproximateForUnknownArch
    #[test]
    fn test_estimate_approximate_for_unknown_arch() {
        let path = write_gguf(
            &[
                KvPair::str("general.architecture", "mistral"),
                KvPair::u32("mistral.block_count", 2),
                KvPair::u32("mistral.embedding_length", 64),
                KvPair::u32("mistral.attention.head_count", 8),
                KvPair::u32("mistral.attention.head_count_kv", 8),
                KvPair::u32("mistral.vocab_size", 128),
            ],
            &[
                TestTensor {
                    name: "blk.0.attn.weight".into(),
                    tensor_type: 0,
                    shape: vec![64, 16],
                    data: Vec::new(),
                },
                TestTensor {
                    name: "blk.1.attn.weight".into(),
                    tensor_type: 0,
                    shape: vec![64, 16],
                    data: Vec::new(),
                },
                TestTensor {
                    name: "output.weight".into(),
                    tensor_type: 0,
                    shape: vec![64, 16],
                    data: Vec::new(),
                },
            ],
        );

        let opts = EstimateOptions {
            free_vram: 1u64 << 40,
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let est = estimate_layers(path.to_str().unwrap(), &opts).unwrap();
        assert!(est.approximate);
        assert_eq!(est.layers, 2);
    }

    // Ported from estimate_test.go: TestEstimateRecurrentUnsupported
    #[test]
    fn test_estimate_recurrent_unsupported() {
        let path = write_gguf(
            &[
                KvPair::str("general.architecture", "mamba"),
                KvPair::u32("mamba.block_count", 2),
                KvPair::u32("mamba.embedding_length", 64),
            ],
            &[TestTensor {
                name: "blk.0.x.weight".into(),
                tensor_type: 0,
                shape: vec![64, 16],
                data: Vec::new(),
            }],
        );

        let opts = EstimateOptions {
            free_vram: 1u64 << 40,
            ..Default::default()
        };
        let err = estimate_layers(path.to_str().unwrap(), &opts).unwrap_err();
        assert!(matches!(err, GgufError::Unsupported(_)));
    }

    // Ported from estimate_test.go: TestEstimateOverheadReducesLayers
    #[test]
    fn test_estimate_overhead_reduces_layers() {
        let mut base = EstimateOptions {
            num_ctx: 2048,
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let (partial, kv, w) = fixture_graph(&base);
        let per_layer = kv + w;
        base.free_vram = 4096 + partial + 3 * per_layer + 8; // fits 3 with no overhead

        let no_overhead = estimate_layers(sample_llama_model(4).to_str().unwrap(), &base).unwrap();

        let mut with_overhead = base.clone();
        with_overhead.overhead_bytes = per_layer; // one layer's worth of extra reserve

        let got = estimate_layers(sample_llama_model(4).to_str().unwrap(), &with_overhead).unwrap();
        assert_eq!(got.layers, no_overhead.layers - 1);
    }

    // Ported from estimate_test.go: TestEstimateNumParallelDoublesKV
    #[test]
    fn test_estimate_num_parallel_doubles_kv() {
        let one = EstimateOptions {
            num_ctx: 2048,
            num_parallel: 1,
            min_reserve_bytes: 4096,
            free_vram: 1u64 << 40,
            ..Default::default()
        };
        let mut two = one.clone();
        two.num_parallel = 2;

        let est1 = estimate_layers(sample_llama_model(4).to_str().unwrap(), &one).unwrap();
        let est2 = estimate_layers(sample_llama_model(4).to_str().unwrap(), &two).unwrap();
        assert_eq!(est2.kv_cache, 2 * est1.kv_cache);
    }

    // Ported from estimate_test.go: TestEstimateRealModel — env-gated, skips
    // unless LLMARK_TEST_GGUF points at a real, readable .gguf file.
    #[test]
    fn test_estimate_real_model() {
        let path = match std::env::var("LLMARK_TEST_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("skipping: set LLMARK_TEST_GGUF to a .gguf file to run this test");
                return;
            }
        };

        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: LLMARK_TEST_GGUF not readable");
            return;
        }

        let opts = EstimateOptions {
            num_ctx: 4096,
            free_vram: 3500u64 << 20,
            ..Default::default()
        };
        let est = estimate_layers(&path, &opts).unwrap_or_else(|e| {
            panic!("estimate_layers({path}): {e}");
        });

        assert!(est.layers >= 0 && est.layers as usize <= est.per_layer_bytes.len());

        eprintln!(
            "layers={} fullyOffloaded={} weights={}MiB kv={}MiB graph={}MiB approx={}",
            est.layers,
            est.fully_offloaded,
            est.weights >> 20,
            est.kv_cache >> 20,
            est.graph >> 20,
            est.approximate
        );
    }

    // Offline coverage for a hand-derived expected layer count (no env dep).
    //
    // Fixture: sample_llama_model(4) — 4 blocks, each a [64,16] F32 tensor
    // (4096 bytes), embedding=64 heads=headsKV=8 key=val=8 vocab=128.
    //
    // With num_ctx=2048, num_parallel=1 (default), kv_cache_type="f16"
    // (default => 2 bytes/element):
    //   ctx = 2048
    //   kv_per_layer = ctx*(key_len+val_len)*heads_kv * 2
    //                = 2048*(8+8)*8 * 2 = 2048*128*2 = 524288 bytes
    //   per_layer = block_weight + kv_per_layer = 4096 + 524288 = 528384
    //
    // partial graph = llama_graph_size(64,8,8,8,2048,512,128).1 (batch
    // defaults to 512), computed via the ported formula directly rather than
    // re-derived by hand, to avoid duplicating graph.rs's own arithmetic.
    //
    // Budget chosen to fit exactly 3 of the 4 layers:
    //   free_vram = min_reserve(4096) + partial + 3*per_layer + 8 (slack)
    // => avail = free_vram - reserve - partial = 3*per_layer + 8, which fits
    // 3 whole layers (3*per_layer <= avail) but not a 4th
    // (4*per_layer > avail since per_layer > 8).
    #[test]
    fn test_estimate_layers_offline_expected_value() {
        let opts = EstimateOptions {
            num_ctx: 2048,
            min_reserve_bytes: 4096,
            ..Default::default()
        };
        let (partial, kv_per_layer, block_weight) = fixture_graph(&opts);
        let per_layer = kv_per_layer + block_weight;
        assert_eq!(kv_per_layer, 524_288);
        assert_eq!(per_layer, 528_384);

        let mut opts = opts;
        opts.free_vram = 4096 + partial + 3 * per_layer + 8;

        let est = estimate_layers(sample_llama_model(4).to_str().unwrap(), &opts).unwrap();
        assert_eq!(est.layers, 3, "expected exactly 3 of 4 layers to fit");
        assert!(!est.fully_offloaded);
        assert_eq!(est.kv_cache, 3 * kv_per_layer);
        assert_eq!(est.weights, 3 * block_weight);
    }
}
