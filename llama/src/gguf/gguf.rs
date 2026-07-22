//! Ported from `gguf/gguf.go` (the `File`/`Open` type).
//!
//! # Eager vs. lazy parsing
//!
//! Go's `File` decodes key-values and tensor descriptors lazily via a
//! generic pull-iterator (`lazy.go`, intentionally excluded from this
//! dispatch — see `mod.rs`). Callers there always fully drain `keyValues`
//! before touching `tensors` (`TensorInfo`/`TensorInfos` call
//! `f.keyValues.rest()` first), which matches the actual on-disk layout:
//! header counts, then all key-value pairs, then all tensor descriptors,
//! then (aligned) tensor data.
//!
//! This Rust port reads key-values and tensor descriptors eagerly, in that
//! same file-byte order, during `File::open`. The externally observable
//! results (decoded values, `num_key_values`/`num_tensors`, tensor data
//! offsets, alignment padding) are identical to the Go behavior; only the
//! laziness of the Go implementation is not reproduced.

use std::io::Read;
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt};

use super::keyvalue::{GgufValue, KeyValue, Value};
use super::reader::BufferedReader;
use super::tensor::{TensorInfo, TensorType};
use super::GgufError;

// GGUF metadata value type tags (mirrors gguf.go's typeUint8..typeFloat64).
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

/// An open GGUF file: header, decoded key-value metadata, and tensor
/// descriptors. Tensor data itself is read on demand via `tensor_reader`.
#[derive(Debug)]
pub struct File {
    pub magic: [u8; 4],
    pub version: u32,

    path: PathBuf,
    key_values: Vec<KeyValue>,
    tensors: Vec<TensorInfo>,
    /// Absolute byte offset (from the start of the file) where the
    /// (alignment-padded) tensor data section begins.
    offset: u64,
}

/// Opens the GGUF file at `path`, validates the magic bytes and version, and
/// eagerly decodes all key-value pairs and tensor descriptors.
pub fn open<P: AsRef<Path>>(path: P) -> Result<File, GgufError> {
    let path = path.as_ref().to_path_buf();
    let raw = std::fs::File::open(&path)?;
    let mut reader = BufferedReader::new(raw, 32 << 10);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic(magic.to_vec()));
    }

    let version = reader.read_u32::<LittleEndian>()?;
    if version < 2 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut key_values = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        key_values.push(read_key_value(&mut reader)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor(&mut reader)?);
    }

    let alignment = {
        let a = key_value_lookup(&key_values, "general.alignment").int();
        if a != 0 {
            a as u64
        } else {
            32
        }
    };
    let raw_offset = reader.offset;
    let offset = raw_offset + (alignment - raw_offset % alignment) % alignment;

    Ok(File {
        magic,
        version,
        path,
        key_values,
        tensors,
        offset,
    })
}

impl File {
    /// Opens the GGUF file at `path` (see the module-level `open` function).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        open(path)
    }

    /// Looks up a metadata key-value pair by key. Keys without a
    /// `general.`/`tokenizer.` namespace prefix are qualified with the
    /// model's architecture (e.g. `context_length` -> `llama.context_length`).
    pub fn key_value(&self, key: &str) -> KeyValue {
        let full_key = if key.starts_with("general.") || key.starts_with("tokenizer.") {
            key.to_string()
        } else {
            format!("{}.{}", self.key_value("general.architecture").string(), key)
        };
        key_value_lookup(&self.key_values, &full_key)
    }

    /// Returns the total number of key-value pairs declared in the file.
    pub fn num_key_values(&self) -> usize {
        self.key_values.len()
    }

    /// Iterates over all key-value pairs in the file, in declaration order.
    pub fn key_values(&self) -> impl Iterator<Item = (usize, &KeyValue)> {
        self.key_values.iter().enumerate()
    }

    /// Looks up tensor metadata by name.
    pub fn tensor_info(&self, name: &str) -> TensorInfo {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns the total number of tensors declared in the file.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Iterates over all tensor descriptors in the file, in declaration order.
    pub fn tensor_infos(&self) -> impl Iterator<Item = (usize, &TensorInfo)> {
        self.tensors.iter().enumerate()
    }

    /// Returns the `TensorInfo` and the raw bytes for the named tensor's
    /// data. Errors if no tensor with that name (or a zero-byte one) exists.
    pub fn tensor_reader(&self, name: &str) -> Result<(TensorInfo, std::io::Cursor<Vec<u8>>), GgufError> {
        let t = self.tensor_info(name);
        if t.num_bytes() == 0 {
            return Err(GgufError::TensorNotFound(name.to_string()));
        }

        use std::io::{Seek, SeekFrom};
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.offset + t.offset))?;

        let mut buf = vec![0u8; t.num_bytes() as usize];
        f.read_exact(&mut buf)?;

        Ok((t, std::io::Cursor::new(buf)))
    }
}

fn key_value_lookup(key_values: &[KeyValue], key: &str) -> KeyValue {
    key_values
        .iter()
        .find(|kv| kv.key == key)
        .cloned()
        .unwrap_or_default()
}

fn read<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    Ok(reader.read_u64::<LittleEndian>()?)
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let n = read(reader)?;
    let mut bts = vec![0u8; n as usize];
    reader.read_exact(&mut bts)?;
    Ok(String::from_utf8_lossy(&bts).into_owned())
}

fn read_tensor<R: Read>(reader: &mut R) -> Result<TensorInfo, GgufError> {
    let name = read_string(reader)?;

    let dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(dims as usize);
    for _ in 0..dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    let type_ = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(TensorInfo {
        name,
        offset,
        shape,
        tensor_type: TensorType(type_),
    })
}

fn read_key_value<R: Read>(reader: &mut R) -> Result<KeyValue, GgufError> {
    let key = read_string(reader)?;
    let t = reader.read_u32::<LittleEndian>()?;
    let value = read_value(reader, t)?;

    Ok(KeyValue {
        key,
        value: Value::new(value),
    })
}

fn read_value<R: Read>(reader: &mut R, t: u32) -> Result<GgufValue, GgufError> {
    match t {
        TYPE_UINT8 => Ok(GgufValue::U8(reader.read_u8()?)),
        TYPE_INT8 => Ok(GgufValue::I8(reader.read_i8()?)),
        TYPE_UINT16 => Ok(GgufValue::U16(reader.read_u16::<LittleEndian>()?)),
        TYPE_INT16 => Ok(GgufValue::I16(reader.read_i16::<LittleEndian>()?)),
        TYPE_UINT32 => Ok(GgufValue::U32(reader.read_u32::<LittleEndian>()?)),
        TYPE_INT32 => Ok(GgufValue::I32(reader.read_i32::<LittleEndian>()?)),
        TYPE_UINT64 => Ok(GgufValue::U64(reader.read_u64::<LittleEndian>()?)),
        TYPE_INT64 => Ok(GgufValue::I64(reader.read_i64::<LittleEndian>()?)),
        TYPE_FLOAT32 => Ok(GgufValue::F32(reader.read_f32::<LittleEndian>()?)),
        TYPE_FLOAT64 => Ok(GgufValue::F64(reader.read_f64::<LittleEndian>()?)),
        TYPE_BOOL => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        TYPE_STRING => Ok(GgufValue::String(read_string(reader)?)),
        TYPE_ARRAY => read_array(reader),
        other => Err(GgufError::UnsupportedType(other)),
    }
}

fn read_array<R: Read>(reader: &mut R) -> Result<GgufValue, GgufError> {
    let t = reader.read_u32::<LittleEndian>()?;
    let n = read(reader)?;

    macro_rules! array_of {
        ($read_one:expr, $variant:ident) => {{
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push($read_one(reader)?);
            }
            Ok(GgufValue::$variant(v))
        }};
    }

    match t {
        TYPE_UINT8 => array_of!(|r: &mut R| -> Result<u8, GgufError> { Ok(r.read_u8()?) }, ArrayU8),
        TYPE_INT8 => array_of!(|r: &mut R| -> Result<i8, GgufError> { Ok(r.read_i8()?) }, ArrayI8),
        TYPE_UINT16 => array_of!(
            |r: &mut R| -> Result<u16, GgufError> { Ok(r.read_u16::<LittleEndian>()?) },
            ArrayU16
        ),
        TYPE_INT16 => array_of!(
            |r: &mut R| -> Result<i16, GgufError> { Ok(r.read_i16::<LittleEndian>()?) },
            ArrayI16
        ),
        TYPE_UINT32 => array_of!(
            |r: &mut R| -> Result<u32, GgufError> { Ok(r.read_u32::<LittleEndian>()?) },
            ArrayU32
        ),
        TYPE_INT32 => array_of!(
            |r: &mut R| -> Result<i32, GgufError> { Ok(r.read_i32::<LittleEndian>()?) },
            ArrayI32
        ),
        TYPE_UINT64 => array_of!(
            |r: &mut R| -> Result<u64, GgufError> { Ok(r.read_u64::<LittleEndian>()?) },
            ArrayU64
        ),
        TYPE_INT64 => array_of!(
            |r: &mut R| -> Result<i64, GgufError> { Ok(r.read_i64::<LittleEndian>()?) },
            ArrayI64
        ),
        TYPE_FLOAT32 => array_of!(
            |r: &mut R| -> Result<f32, GgufError> { Ok(r.read_f32::<LittleEndian>()?) },
            ArrayF32
        ),
        TYPE_FLOAT64 => array_of!(
            |r: &mut R| -> Result<f64, GgufError> { Ok(r.read_f64::<LittleEndian>()?) },
            ArrayF64
        ),
        TYPE_BOOL => array_of!(
            |r: &mut R| -> Result<bool, GgufError> { Ok(r.read_u8()? != 0) },
            ArrayBool
        ),
        TYPE_STRING => {
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push(read_string(reader)?);
            }
            Ok(GgufValue::ArrayString(v))
        }
        other => Err(GgufError::UnsupportedType(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testutil::{sample_model, write_gguf, KvPair, TestTensor};

    // Ported from gguf_test.go: TestOpenReadsHeaderAndKV
    #[test]
    fn test_open_reads_header_and_kv() {
        let path = sample_model();
        let f = File::open(&path).expect("Open");

        assert_eq!(f.num_tensors(), 1);
        assert_eq!(f.key_value("general.architecture").string(), "llama");
        assert_eq!(f.key_value("context_length").uint(), 4096);
        assert_eq!(f.key_value("block_count").uint(), 2);
    }

    // Ported from gguf_test.go: TestTensorInfoAndReader
    #[test]
    fn test_tensor_info_and_reader() {
        let path = sample_model();
        let f = File::open(&path).expect("Open");

        let ti = f.tensor_info("token_embd.weight");
        assert!(ti.valid(), "token_embd.weight not found");
        assert_eq!(ti.num_bytes(), 24);

        let (_, mut r) = f.tensor_reader("token_embd.weight").expect("TensorReader");
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut data).expect("ReadAll");

        assert_eq!(data.len(), 24);
        assert_eq!(data[0], 0xAB);
        assert_eq!(data[23], 0xAB);
    }

    // Ported from gguf_test.go: TestOpenRejectsBadMagic
    #[test]
    fn test_open_rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!(
            "rs-llama-gguf-badmagic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        std::fs::write(&path, b"NOPExxxxxxxxxxxx").unwrap();

        let err = File::open(&path).expect_err("Open(bad magic) should fail");
        assert!(
            matches!(err, GgufError::BadMagic(_)),
            "Open(bad magic) err = {err:?}, want BadMagic"
        );
    }

    // Ported from gguf_test.go: TestTensorReaderMultiOffset
    #[test]
    fn test_tensor_reader_multi_offset() {
        let path = write_gguf(
            &[KvPair::str("general.architecture", "llama")],
            &[
                TestTensor {
                    name: "a".into(),
                    tensor_type: 0,
                    shape: vec![2, 3],
                    data: vec![0x11; 24],
                },
                TestTensor {
                    name: "b".into(),
                    tensor_type: 0,
                    shape: vec![1, 4],
                    data: vec![0x22; 16],
                },
            ],
        );

        let f = File::open(&path).expect("Open");

        let bi = f.tensor_info("b");
        assert_eq!(bi.offset, 32);

        let (_, mut rb) = f.tensor_reader("b").expect("TensorReader(b)");
        let mut db = Vec::new();
        std::io::Read::read_to_end(&mut rb, &mut db).expect("ReadAll(b)");
        assert_eq!(db.len(), 16);
        assert_eq!(db[0], 0x22);
        assert_eq!(db[15], 0x22);

        let (_, mut ra) = f.tensor_reader("a").expect("TensorReader(a)");
        let mut da = Vec::new();
        std::io::Read::read_to_end(&mut ra, &mut da).expect("ReadAll(a)");
        assert_eq!(da.len(), 24);
        assert_eq!(da[0], 0x11);
        assert_eq!(da[23], 0x11);
    }
}
