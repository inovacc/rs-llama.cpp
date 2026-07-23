//! Ported from `gguf/keyvalue.go`.
//!
//! Go's `Value` wraps an `any` and uses `reflect` to coerce it to the
//! requested scalar/slice kind, returning the zero value on mismatch. Rust
//! has no direct equivalent of unconstrained `any` + reflection, so `Value`
//! is a closed enum (`GgufValue`) covering every GGUF metadata value shape;
//! the accessor methods perform the same "coerce within kind-family, else
//! zero value" behavior as the Go code.

/// A decoded GGUF metadata value (scalar or array).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    /// Raw string bytes, stored losslessly (GGUF strings are not guaranteed
    /// UTF-8 — vocab entries from byte-fallback BPE tokenizers commonly
    /// contain arbitrary bytes). Mirrors Go's `string`, which holds raw bytes
    /// without a UTF-8 validity guarantee. Use `Value::string`/`string_bytes`
    /// to obtain a UTF-8 view (lossy) only at the point of use.
    String(Vec<u8>),

    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF32(Vec<f32>),
    ArrayF64(Vec<f64>),
    ArrayBool(Vec<bool>),
    /// Raw string bytes for each array element (see `String` above).
    ArrayString(Vec<Vec<u8>>),
}

/// Wraps an arbitrary GGUF metadata value (mirrors Go's `Value`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Value(pub Option<GgufValue>);

impl Value {
    pub fn new(v: GgufValue) -> Self {
        Value(Some(v))
    }

    /// Returns Value as a signed integer. If it is not a signed integer, returns 0.
    pub fn int(&self) -> i64 {
        match &self.0 {
            Some(GgufValue::I8(v)) => *v as i64,
            Some(GgufValue::I16(v)) => *v as i64,
            Some(GgufValue::I32(v)) => *v as i64,
            Some(GgufValue::I64(v)) => *v,
            _ => 0,
        }
    }

    /// Returns Value as a signed integer slice. If it is not one, returns an empty Vec.
    pub fn ints(&self) -> Vec<i64> {
        match &self.0 {
            Some(GgufValue::ArrayI8(v)) => v.iter().map(|x| *x as i64).collect(),
            Some(GgufValue::ArrayI16(v)) => v.iter().map(|x| *x as i64).collect(),
            Some(GgufValue::ArrayI32(v)) => v.iter().map(|x| *x as i64).collect(),
            Some(GgufValue::ArrayI64(v)) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Converts an unsigned integer value to u64. If not unsigned, returns 0.
    pub fn uint(&self) -> u64 {
        match &self.0 {
            Some(GgufValue::U8(v)) => *v as u64,
            Some(GgufValue::U16(v)) => *v as u64,
            Some(GgufValue::U32(v)) => *v as u64,
            Some(GgufValue::U64(v)) => *v,
            _ => 0,
        }
    }

    /// Returns Value as an unsigned integer slice. If not one, returns an empty Vec.
    pub fn uints(&self) -> Vec<u64> {
        match &self.0 {
            Some(GgufValue::ArrayU8(v)) => v.iter().map(|x| *x as u64).collect(),
            Some(GgufValue::ArrayU16(v)) => v.iter().map(|x| *x as u64).collect(),
            Some(GgufValue::ArrayU32(v)) => v.iter().map(|x| *x as u64).collect(),
            Some(GgufValue::ArrayU64(v)) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Returns Value as a float. If it is not a float, returns 0.
    pub fn float(&self) -> f64 {
        match &self.0 {
            Some(GgufValue::F32(v)) => *v as f64,
            Some(GgufValue::F64(v)) => *v,
            _ => 0.0,
        }
    }

    /// Returns Value as a float slice. If not one, returns an empty Vec.
    pub fn floats(&self) -> Vec<f64> {
        match &self.0 {
            Some(GgufValue::ArrayF32(v)) => v.iter().map(|x| *x as f64).collect(),
            Some(GgufValue::ArrayF64(v)) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Returns Value as a bool. If it is not a bool, returns false.
    pub fn bool(&self) -> bool {
        match &self.0 {
            Some(GgufValue::Bool(v)) => *v,
            _ => false,
        }
    }

    /// Returns Value as a bool slice. If not one, returns an empty Vec.
    pub fn bools(&self) -> Vec<bool> {
        match &self.0 {
            Some(GgufValue::ArrayBool(v)) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Returns Value as the raw string bytes, losslessly. If it is not a
    /// string, returns an empty Vec. This is the lossless counterpart to
    /// `string()` — use it when non-UTF-8 bytes (e.g. byte-fallback BPE
    /// vocab entries) must be preserved exactly.
    pub fn string_bytes(&self) -> &[u8] {
        match &self.0 {
            Some(GgufValue::String(v)) => v,
            _ => &[],
        }
    }

    /// Returns Value as a string. If it is not a string, returns "".
    ///
    /// This decodes the raw bytes as UTF-8 lossily (invalid sequences become
    /// U+FFFD) purely as a display/consumption convenience at the point of
    /// use. Use `string_bytes()` when losslessness matters.
    pub fn string(&self) -> String {
        match &self.0 {
            Some(GgufValue::String(v)) => String::from_utf8_lossy(v).into_owned(),
            _ => String::new(),
        }
    }

    /// Returns Value as a slice of raw string byte vectors, losslessly. If
    /// not a string array, returns an empty Vec.
    pub fn strings_bytes(&self) -> &[Vec<u8>] {
        match &self.0 {
            Some(GgufValue::ArrayString(v)) => v,
            _ => &[],
        }
    }

    /// Returns Value as a string slice. If not one, returns an empty Vec.
    ///
    /// Each element is decoded as UTF-8 lossily; see `string()`'s doc. Use
    /// `strings_bytes()` when losslessness matters.
    pub fn strings(&self) -> Vec<String> {
        match &self.0 {
            Some(GgufValue::ArrayString(v)) => {
                v.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect()
            }
            _ => Vec::new(),
        }
    }
}

/// A single GGUF metadata key-value pair (Go's `KeyValue`, which embeds `Value`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeyValue {
    pub key: String,
    pub value: Value,
}

impl KeyValue {
    /// Reports whether the KeyValue has a non-empty key and a set value.
    pub fn valid(&self) -> bool {
        !self.key.is_empty() && self.value.0.is_some()
    }

    pub fn int(&self) -> i64 {
        self.value.int()
    }
    pub fn ints(&self) -> Vec<i64> {
        self.value.ints()
    }
    pub fn uint(&self) -> u64 {
        self.value.uint()
    }
    pub fn uints(&self) -> Vec<u64> {
        self.value.uints()
    }
    pub fn float(&self) -> f64 {
        self.value.float()
    }
    pub fn floats(&self) -> Vec<f64> {
        self.value.floats()
    }
    pub fn bool(&self) -> bool {
        self.value.bool()
    }
    pub fn bools(&self) -> Vec<bool> {
        self.value.bools()
    }
    pub fn string(&self) -> String {
        self.value.string()
    }
    pub fn string_bytes(&self) -> &[u8] {
        self.value.string_bytes()
    }
    pub fn strings(&self) -> Vec<String> {
        self.value.strings()
    }
    pub fn strings_bytes(&self) -> &[Vec<u8>] {
        self.value.strings_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from keyvalue_test.go: TestValueScalars
    #[test]
    fn test_value_scalars() {
        assert_eq!(Value::new(GgufValue::I32(7)).int(), 7);
        assert_eq!(Value::new(GgufValue::U16(9)).uint(), 9);
        assert_eq!(Value::new(GgufValue::F32(1.5)).float(), 1.5);
        assert!(Value::new(GgufValue::Bool(true)).bool());
        assert!(!Value::new(GgufValue::Bool(false)).bool());
        assert_eq!(Value::new(GgufValue::String(b"hi".to_vec())).string(), "hi");
        assert_eq!(Value::new(GgufValue::String(b"hi".to_vec())).int(), 0);
    }

    // Ported from keyvalue_test.go: TestValueSlices
    #[test]
    fn test_value_slices() {
        assert_eq!(
            Value::new(GgufValue::ArrayI32(vec![1, 2, 3])).ints(),
            vec![1i64, 2, 3]
        );
        assert_eq!(
            Value::new(GgufValue::ArrayString(vec![b"a".to_vec(), b"b".to_vec()])).strings(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    // New: lossless string bytes round-trip through the Value/KeyValue API.
    #[test]
    fn test_value_string_bytes_lossless() {
        let raw = vec![0xFFu8, 0xFE, b'x', 0x00, 0x80];
        let v = Value::new(GgufValue::String(raw.clone()));
        assert_eq!(v.string_bytes(), raw.as_slice());

        let arr = Value::new(GgufValue::ArrayString(vec![raw.clone()]));
        assert_eq!(arr.strings_bytes(), &[raw]);
    }
}
