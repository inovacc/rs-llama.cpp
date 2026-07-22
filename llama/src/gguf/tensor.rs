//! Ported from `gguf/tensor.go`.

/// Tensor name, shape, type, and data offset (relative to the tensor-data
/// section start).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub offset: u64,
    pub shape: Vec<u64>,
    pub tensor_type: TensorType,
}

impl TensorInfo {
    /// Reports whether the TensorInfo has a non-empty name and non-zero byte size.
    pub fn valid(&self) -> bool {
        !self.name.is_empty() && self.num_bytes() > 0
    }

    /// Returns the total number of scalar elements in the tensor.
    pub fn num_values(&self) -> i64 {
        let mut n: i64 = 1;
        for dim in &self.shape {
            n *= *dim as i64;
        }
        n
    }

    /// Returns the number of bytes in the tensor.
    pub fn num_bytes(&self) -> i64 {
        (self.num_values() as f64 * self.tensor_type.num_bytes()) as i64
    }
}

/// Identifies the quantization or numeric format of a tensor.
///
/// Ported as a newtype over `u32` (rather than a closed Rust `enum`) so that
/// unknown/unused discriminants read from a file (matching Go's `default:`
/// switch arms, which return 0 / "unknown") do not require a fallible
/// conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TensorType(pub u32);

#[allow(non_upper_case_globals)]
impl TensorType {
    pub const F32: TensorType = TensorType(0);
    pub const F16: TensorType = TensorType(1);
    pub const Q4_0: TensorType = TensorType(2);
    pub const Q4_1: TensorType = TensorType(3);
    // 4, 5: unexported in Go (Q4_2, Q4_3) — unused in gguf.
    pub const Q5_0: TensorType = TensorType(6);
    pub const Q5_1: TensorType = TensorType(7);
    pub const Q8_0: TensorType = TensorType(8);
    pub const Q8_1: TensorType = TensorType(9);
    pub const Q2_K: TensorType = TensorType(10);
    pub const Q3_K: TensorType = TensorType(11);
    pub const Q4_K: TensorType = TensorType(12);
    pub const Q5_K: TensorType = TensorType(13);
    pub const Q6_K: TensorType = TensorType(14);
    pub const Q8_K: TensorType = TensorType(15);
    // 16-23: unexported in Go (IQ2_XXS..IQ4_XS) — unquantizable by ollama.
    pub const I8: TensorType = TensorType(24);
    pub const I16: TensorType = TensorType(25);
    pub const I32: TensorType = TensorType(26);
    pub const I64: TensorType = TensorType(27);
    pub const F64: TensorType = TensorType(28);
    // 29: unexported in Go (IQ1_M) — unquantizable by ollama.
    pub const BF16: TensorType = TensorType(30);
    // 31-38: unexported in Go (Q4_0_4_4.. IQ4_NL_8_8) — unused/unquantizable.

    /// Returns bytes-per-element (may be fractional for block-quantized types).
    pub fn num_bytes(&self) -> f64 {
        self.type_size() as f64 / self.block_size() as f64
    }

    fn type_size(&self) -> i64 {
        let bs = self.block_size();
        match self.0 {
            0 => 4,                    // F32
            1 => 2,                    // F16
            2 => 2 + bs / 2,           // Q4_0
            3 => 2 + 2 + bs / 2,       // Q4_1
            6 => 2 + 4 + bs / 2,       // Q5_0
            7 => 2 + 2 + 4 + bs / 2,   // Q5_1
            8 => 2 + bs,               // Q8_0
            9 => 2 + 2 + bs,           // Q8_1
            10 => bs / 16 + bs / 4 + 2 + 2, // Q2_K
            11 => bs / 8 + bs / 4 + 12 + 2, // Q3_K
            12 => 2 + 2 + 12 + bs / 2, // Q4_K
            13 => 2 + 2 + 12 + bs / 8 + bs / 2, // Q5_K
            14 => bs / 2 + bs / 4 + bs / 16 + 2, // Q6_K
            15 => 4 + bs + 2 * bs / 16, // Q8_K
            16 => 2 + 2 * bs / 8,      // IQ2_XXS
            17 => 2 + 2 * bs / 8 + bs / 32, // IQ2_XS
            18 => 2 + bs / 4 + bs / 8, // IQ3_XXS
            19 => 2 + bs / 8 + bs / 16, // IQ1_S
            20 => 2 + bs / 2,          // IQ4_NL
            21 => 2 + bs / 4 + bs / 8 + bs / 32 + 4, // IQ3_S
            22 => 2 + bs / 4 + bs / 16, // IQ2_S
            23 => 2 + 2 + bs / 2 + bs / 64, // IQ4_XS
            24 => 1,                   // I8
            25 => 2,                   // I16
            26 => 4,                   // I32
            27 => 8,                   // I64
            28 => 8,                   // F64
            29 => bs / 8 + bs / 16 + bs / 32, // IQ1_M
            30 => 2,                   // BF16
            _ => 0,
        }
    }

    fn block_size(&self) -> i64 {
        match self.0 {
            0 | 1 | 24 | 25 | 26 | 27 | 28 | 30 => 1, // F32,F16,I8,I16,I32,I64,F64,BF16
            2 | 3 | 6 | 7 | 8 | 9 | 20 => 32, // Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q8_1,IQ4_NL
            _ => 256,
        }
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self.0 {
            0 => "f32",
            1 => "f16",
            2 => "q4_0",
            3 => "q4_1",
            4 => "q4_2",
            5 => "q4_3",
            6 => "q5_0",
            7 => "q5_1",
            8 => "q8_0",
            9 => "q8_1",
            10 => "q2_k",
            11 => "q3_k",
            12 => "q4_k",
            13 => "q5_k",
            14 => "q6_k",
            15 => "q8_k",
            16 => "iq2_xxs",
            17 => "iq2_xs",
            18 => "iq3_xxs",
            19 => "iq1_s",
            20 => "iq4_nl",
            21 => "iq3_s",
            22 => "iq2_s",
            23 => "iq4_xs",
            24 => "i8",
            25 => "i16",
            26 => "i32",
            27 => "i64",
            28 => "f64",
            29 => "iq1_m",
            30 => "bf16",
            31 => "q4_0_4_4",
            32 => "q4_0_4_8",
            33 => "q4_0_8_8",
            34 => "tq1_0",
            35 => "tq2_0",
            36 => "iq4_nl_4_4",
            37 => "iq4_nl_4_8",
            38 => "iq4_nl_8_8",
            _ => "unknown",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from keyvalue_test.go: TestTensorNumBytes
    #[test]
    fn test_tensor_num_bytes() {
        let ti = TensorInfo {
            name: "x".into(),
            shape: vec![2, 3],
            tensor_type: TensorType::F32,
            ..Default::default()
        };
        assert_eq!(ti.num_values(), 6);
        assert_eq!(ti.num_bytes(), 24);

        let q = TensorInfo {
            name: "y".into(),
            shape: vec![32],
            tensor_type: TensorType::Q4_0,
            ..Default::default()
        };
        assert_eq!(q.num_bytes(), 18);
    }
}
