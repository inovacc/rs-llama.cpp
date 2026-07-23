//! GGUF model-file header parser (parser core only: keyvalue, tensor, reader, gguf).
//!
//! Derived from `github.com/ollama/ollama/fs/gguf` (MIT License), via
//! `github.com/dyammarcano/go-llama.cpp`'s `gguf` package. Ported to Rust,
//! faithful to the Go byte-level parsing (magic, version, counts, kv/tensor
//! entries, alignment/padding).
//!
//! Scope note: `lazy.go` is intentionally NOT ported as a separate module.
//! `metadata.go`, `graph.go`, and `estimate.go` are ported (see `metadata.rs`,
//! `graph.rs`, `estimate.rs`).
//! The lazy-iterator abstraction from `lazy.go` is not ported as a separate
//! generic type here: `File::open` parses key-values and tensor descriptors
//! eagerly (in file-byte order), which yields identical externally-observable
//! results (same values, counts, and tensor data offsets) without the
//! coroutine-style pull machinery.

pub mod estimate;
pub mod gguf;
pub mod graph;
pub mod keyvalue;
pub mod metadata;
pub mod reader;
pub mod tensor;

#[cfg(test)]
pub(crate) mod testutil;

pub use estimate::{Estimate, EstimateOptions, DEFAULT_MIN_RESERVE_BYTES};
pub use gguf::{File, TensorDataReader};
pub use graph::llama_graph_size;
pub use keyvalue::{GgufValue, KeyValue, Value};
pub use metadata::{stat, Info};
pub use tensor::{TensorInfo, TensorType};

use thiserror::Error;

/// Errors produced while parsing a GGUF file.
#[derive(Debug, Error)]
pub enum GgufError {
    /// Underlying I/O failure (including unexpected EOF while reading a
    /// fixed-size field).
    #[error("gguf: io error: {0}")]
    Io(#[from] std::io::Error),

    /// The file's magic bytes were not `GGUF`.
    #[error("gguf: unsupported: bad magic {0:?}")]
    BadMagic(Vec<u8>),

    /// The file's version field was below the minimum supported version (2).
    #[error("gguf: unsupported version {0}")]
    UnsupportedVersion(u32),

    /// A key-value or array element type tag was not recognized.
    #[error("gguf: unsupported type {0}")]
    UnsupportedType(u32),

    /// `File::tensor_reader` was asked for a tensor that does not exist (or
    /// has zero size).
    #[error("gguf: tensor {0} not found")]
    TensorNotFound(String),

    /// Mirrors Go's `ErrUnsupported` sentinel (`estimate.go`): a required
    /// input is missing, or the model's architecture/shape is not supported
    /// by the layer-fit estimator (e.g. a recurrent/SSM architecture with no
    /// attention heads).
    #[error("gguf: unsupported: {0}")]
    Unsupported(String),
}
