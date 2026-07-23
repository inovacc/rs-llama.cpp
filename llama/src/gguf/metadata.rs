//! Derived from `github.com/ollama/ollama/fs/gguf` (MIT License).
//! Ported from `gguf/metadata.go`.

use std::collections::HashMap;
use std::path::Path;

use super::gguf::File;
use super::tensor::TensorType;
use super::GgufError;

/// Holds commonly-needed model facts, read without loading the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Info {
    pub architecture: String,
    pub name: String,
    /// `general.file_type` (quant scheme enum).
    pub file_type: u32,
    /// Human label derived from `file_type` (fallback: dominant tensor type).
    pub quantization: String,
    pub context_length: u64,
    pub embedding_length: u64,
    /// Number of transformer blocks (layers).
    pub block_count: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    /// Empty if the model embeds no chat template.
    pub chat_template: String,
    pub num_tensors: usize,
}

/// Opens `path`, reads common metadata, and closes the file.
/// Missing keys yield zero values, not errors — GGUF keys vary by architecture.
pub fn stat<P: AsRef<Path>>(path: P) -> Result<Info, GgufError> {
    let f = File::open(path)?;

    let mut info = Info {
        architecture: f.key_value("general.architecture").string(),
        name: f.key_value("general.name").string(),
        file_type: f.key_value("general.file_type").uint() as u32,
        context_length: f.key_value("context_length").uint(),
        embedding_length: f.key_value("embedding_length").uint(),
        block_count: f.key_value("block_count").uint(),
        head_count: f.key_value("attention.head_count").uint(),
        head_count_kv: f.key_value("attention.head_count_kv").uint(),
        chat_template: f.key_value("tokenizer.chat_template").string(),
        num_tensors: f.num_tensors(),
        quantization: String::new(),
    };
    info.quantization = quant_label(info.file_type, &f);

    Ok(info)
}

// NOTE: extend file_type_names as the llama_ftype enum grows.
/// Maps the well-known llama_ftype enum values to labels.
fn file_type_names() -> HashMap<u32, &'static str> {
    HashMap::from([
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (7, "Q8_0"),
        (8, "Q5_0"),
        (9, "Q5_1"),
        (10, "Q2_K"),
        (11, "Q3_K_S"),
        (12, "Q3_K_M"),
        (13, "Q3_K_L"),
        (14, "Q4_K_S"),
        (15, "Q4_K_M"),
        (16, "Q5_K_S"),
        (17, "Q5_K_M"),
        (18, "Q6_K"),
    ])
}

/// Returns a human quantization label for a `file_type` enum value.
/// Unknown enums fall back to the dominant tensor type, then to "ftype_N".
fn quant_label(ft: u32, f: &File) -> String {
    if let Some(name) = file_type_names().get(&ft) {
        return (*name).to_string();
    }

    let mut counts: HashMap<TensorType, i64> = HashMap::new();
    for (_, ti) in f.tensor_infos() {
        *counts.entry(ti.tensor_type).or_insert(0) += 1;
    }

    let mut best = TensorType(0);
    let mut best_n: i64 = -1;
    for (tt, n) in counts {
        if n > best_n {
            best = tt;
            best_n = n;
        }
    }

    if best_n <= 0 {
        return format!("ftype_{ft}");
    }

    best.to_string().to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testutil::{sample_model, write_gguf, KvPair, TestTensor};
    use std::env;

    // Ported from metadata_test.go: TestStatReadsCommonFields
    #[test]
    fn test_stat_reads_common_fields() {
        let path = sample_model();
        let info = stat(&path).expect("stat");

        assert_eq!(info.architecture, "llama");
        assert_eq!(info.name, "tiny-test");
        assert_eq!(info.context_length, 4096);
        assert_eq!(info.embedding_length, 3);
        assert_eq!(info.block_count, 2);
        assert_eq!(info.head_count, 2);
        assert_eq!(info.head_count_kv, 1);
        assert_eq!(info.chat_template, "{{ .Prompt }}");
        assert_eq!(info.num_tensors, 1);
        assert_eq!(info.quantization, "Q8_0");
    }

    // Ported from metadata_test.go: TestStatRealModel.
    // ENV-GATED: skipped (returns early) unless LLMARK_TEST_GGUF points to a
    // real, readable .gguf file. Stays green in CI without a model.
    #[test]
    fn test_stat_real_model() {
        let path = match env::var("LLMARK_TEST_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("skipping: set LLMARK_TEST_GGUF to a .gguf file to run this test");
                return;
            }
        };

        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: LLMARK_TEST_GGUF not readable: {path}");
            return;
        }

        let info = stat(&path).unwrap_or_else(|e| panic!("stat({path}): {e}"));

        assert!(!info.architecture.is_empty(), "Architecture empty — metadata parse likely failed");
        assert!(info.num_tensors != 0, "NumTensors = 0 — tensor block parse likely failed");

        eprintln!(
            "arch={} name={:?} ctx={} embd={} blocks={} quant={} tensors={}",
            info.architecture,
            info.name,
            info.context_length,
            info.embedding_length,
            info.block_count,
            info.quantization,
            info.num_tensors
        );
    }

    // Ported from metadata_test.go: TestStatQuantFallbackUppercase
    #[test]
    fn test_stat_quant_fallback_uppercase() {
        // file_type 99 is not in file_type_names, forcing the dominant-tensor fallback.
        let path = write_gguf(
            &[
                KvPair::str("general.architecture", "llama"),
                KvPair::u32("general.file_type", 99),
                KvPair::u32("llama.block_count", 1),
            ],
            &[
                TestTensor {
                    name: "blk.0.attn_q.weight".into(),
                    tensor_type: 2, // Q4_0
                    shape: vec![32],
                    data: vec![0u8; 18],
                },
                TestTensor {
                    name: "blk.0.attn_k.weight".into(),
                    tensor_type: 2, // Q4_0
                    shape: vec![32],
                    data: vec![0u8; 18],
                },
            ],
        );

        let info = stat(&path).expect("stat");

        assert_eq!(
            info.quantization, "Q4_0",
            "Quantization = {:?}, want Q4_0 (uppercase, from dominant tensor fallback)",
            info.quantization
        );
    }
}
