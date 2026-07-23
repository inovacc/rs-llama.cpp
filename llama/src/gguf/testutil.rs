//! Test-only GGUF writer fixture, ported from `gguf/writer_test.go`.
//!
//! Lets gguf parser tests construct valid GGUF bytes in memory, write them to
//! a temp file, and parse them back with `File::open` — no external model
//! files needed. `#[cfg(test)]`-gated; `pub(crate)` so later gguf dispatches
//! (metadata/graph/estimate/lazy tests) can reuse it.

use std::io::Write;
use std::path::PathBuf;

// GGUF metadata value type tags (subset emitted by fixtures); mirrors
// writer_test.go's wUint32..wUint64 constants.
const W_UINT32: u32 = 4;
const W_INT32: u32 = 5;
const W_FLOAT32: u32 = 6;
const W_BOOL: u32 = 7;
const W_STR: u32 = 8;
const W_ARRAY: u32 = 9;
const W_UINT64: u32 = 10;

/// A key-value pair to be written into a fixture GGUF file (Go's `kvPair`).
pub struct KvPair {
    pub key: String,
    pub val: KvVal,
}

/// The value shapes writer_test.go's `wValue` supports (Go's `any` there is
/// closed over these concrete cases in practice).
#[allow(dead_code)] // full set kept for reuse by later gguf dispatches' fixtures
pub enum KvVal {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
    /// Raw (possibly non-UTF-8) string bytes, for exercising lossless string
    /// parsing.
    StrBytes(Vec<u8>),
    StrArray(Vec<String>),
    I32Array(Vec<i32>),
}

impl KvPair {
    pub fn str(key: &str, val: &str) -> Self {
        KvPair {
            key: key.to_string(),
            val: KvVal::Str(val.to_string()),
        }
    }
    pub fn u32(key: &str, val: u32) -> Self {
        KvPair {
            key: key.to_string(),
            val: KvVal::U32(val),
        }
    }
}

/// A tensor descriptor + raw data to be written into a fixture GGUF file
/// (Go's `testTensor`).
pub struct TestTensor {
    pub name: String,
    /// TensorType value (0 = F32).
    pub tensor_type: u32,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Writes a GGUF-encoded length-prefixed string (Go's `writeStr`).
pub fn write_str(b: &mut Vec<u8>, s: &str) {
    write_str_bytes(b, s.as_bytes());
}

/// Writes a GGUF-encoded length-prefixed string from raw bytes (may be
/// invalid UTF-8), for exercising lossless string parsing.
pub fn write_str_bytes(b: &mut Vec<u8>, s: &[u8]) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s);
}

/// Writes a type-tagged GGUF metadata value (Go's `wValue`).
pub fn w_value(b: &mut Vec<u8>, v: &KvVal) {
    match v {
        KvVal::U32(x) => {
            b.extend_from_slice(&W_UINT32.to_le_bytes());
            b.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::I32(x) => {
            b.extend_from_slice(&W_INT32.to_le_bytes());
            b.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::U64(x) => {
            b.extend_from_slice(&W_UINT64.to_le_bytes());
            b.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::F32(x) => {
            b.extend_from_slice(&W_FLOAT32.to_le_bytes());
            b.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::Bool(x) => {
            b.extend_from_slice(&W_BOOL.to_le_bytes());
            b.push(if *x { 1 } else { 0 });
        }
        KvVal::Str(x) => {
            b.extend_from_slice(&W_STR.to_le_bytes());
            write_str(b, x);
        }
        KvVal::StrBytes(x) => {
            b.extend_from_slice(&W_STR.to_le_bytes());
            write_str_bytes(b, x);
        }
        KvVal::StrArray(xs) => {
            b.extend_from_slice(&W_ARRAY.to_le_bytes());
            b.extend_from_slice(&W_STR.to_le_bytes());
            b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for s in xs {
                write_str(b, s);
            }
        }
        KvVal::I32Array(xs) => {
            b.extend_from_slice(&W_ARRAY.to_le_bytes());
            b.extend_from_slice(&W_INT32.to_le_bytes());
            b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for e in xs {
                b.extend_from_slice(&e.to_le_bytes());
            }
        }
    }
}

fn align_up(n: u64, a: u64) -> u64 {
    (n + a - 1) / a * a
}

fn unique_temp_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);

    let dir = std::env::temp_dir().join(format!(
        "rs-llama-gguf-{tag}-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        seq
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Serializes a minimal valid GGUF v3 file and returns its path
/// (Go's `writeGGUF`).
pub fn write_gguf(kvs: &[KvPair], tensors: &[TestTensor]) -> PathBuf {
    const ALIGNMENT: u64 = 32;

    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor count

    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes()); // kv count
    for kv in kvs {
        write_str(&mut b, &kv.key);
        w_value(&mut b, &kv.val);
    }

    let mut offsets = vec![0u64; tensors.len()];
    let mut data_len: u64 = 0;
    for (i, t) in tensors.iter().enumerate() {
        data_len = align_up(data_len, ALIGNMENT);
        offsets[i] = data_len;
        data_len += t.data.len() as u64;
    }

    for (i, t) in tensors.iter().enumerate() {
        write_str(&mut b, &t.name);

        b.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for d in &t.shape {
            b.extend_from_slice(&d.to_le_bytes());
        }

        b.extend_from_slice(&t.tensor_type.to_le_bytes());
        b.extend_from_slice(&offsets[i].to_le_bytes());
    }

    let pad = (ALIGNMENT - (b.len() as u64) % ALIGNMENT) % ALIGNMENT;
    if pad > 0 {
        b.extend(std::iter::repeat(0u8).take(pad as usize));
    }

    let mut data = vec![0u8; data_len as usize];
    for (i, t) in tensors.iter().enumerate() {
        let start = offsets[i] as usize;
        data[start..start + t.data.len()].copy_from_slice(&t.data);
    }
    b.extend_from_slice(&data);

    let dir = unique_temp_dir("model");
    let path = dir.join("model.gguf");
    let mut f = std::fs::File::create(&path).expect("write fixture");
    f.write_all(&b).expect("write fixture bytes");

    path
}

/// Builds an n-block llama-arch GGUF fixture with known sizes, for
/// `estimate.rs` tests (Go's `sampleLlamaModel`).
///
/// Each block has one F32 weight tensor of shape `[64,16]` =>
/// `64*16*4 = 4096` bytes. The output tensor is `[64,16]` => 4096 bytes.
/// `token_embd` is present (tied). Metadata uses tiny dims so graph/KV are
/// hand-computable in `estimate.rs` tests:
/// `embedding=64 head_count=8 head_count_kv=8 key_length=8 value_length=8
/// vocab=128`.
///
/// Tensor data is left empty: `num_bytes()` comes from shape, and the
/// estimator never reads tensor data, so the file stays tiny.
pub fn sample_llama_model(n: usize) -> PathBuf {
    let kvs = vec![
        KvPair::str("general.architecture", "llama"),
        KvPair::str("general.name", "tiny-llama-fixture"),
        KvPair::u32("llama.block_count", n as u32),
        KvPair::u32("llama.embedding_length", 64),
        KvPair::u32("llama.attention.head_count", 8),
        KvPair::u32("llama.attention.head_count_kv", 8),
        KvPair::u32("llama.attention.key_length", 8),
        KvPair::u32("llama.attention.value_length", 8),
        KvPair::u32("llama.vocab_size", 128),
    ];

    let mut tensors = Vec::with_capacity(n + 2);
    for i in 0..n {
        tensors.push(TestTensor {
            name: format!("blk.{i}.attn.weight"),
            tensor_type: 0, // F32
            shape: vec![64, 16],
            data: Vec::new(),
        });
    }
    tensors.push(TestTensor {
        name: "token_embd.weight".into(),
        tensor_type: 0,
        shape: vec![128, 64],
        data: Vec::new(),
    });
    tensors.push(TestTensor {
        name: "output.weight".into(),
        tensor_type: 0,
        shape: vec![64, 16],
        data: Vec::new(),
    });

    write_gguf(&kvs, &tensors)
}

/// Builds a small llama-arch GGUF fixture used across gguf parser tests
/// (Go's `sampleModel`).
pub fn sample_model() -> PathBuf {
    write_gguf(
        &[
            KvPair::str("general.architecture", "llama"),
            KvPair::str("general.name", "tiny-test"),
            KvPair::u32("general.file_type", 7), // Q8_0
            KvPair::u32("llama.context_length", 4096),
            KvPair::u32("llama.embedding_length", 3),
            KvPair::u32("llama.block_count", 2),
            KvPair::u32("llama.attention.head_count", 2),
            KvPair::u32("llama.attention.head_count_kv", 1),
            KvPair::str("tokenizer.chat_template", "{{ .Prompt }}"),
        ],
        &[TestTensor {
            name: "token_embd.weight".into(),
            tensor_type: 0,
            shape: vec![2, 3],
            data: vec![0xAB; 24],
        }],
    )
}
