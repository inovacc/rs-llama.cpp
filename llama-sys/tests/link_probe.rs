//! Executable link proof for llama-sys.
//!
//! Building this test binary forces a FULL native link: the Rust test exe must
//! resolve the C shim symbols (`load_model`, ...) and, transitively, every
//! llama.cpp symbol the shim references, against the static archives produced by
//! `build.rs`. It never runs inference — no model is needed for a link check.
//!
//! The shim CALLS `tokenCallback` during generation, so the test binary must
//! provide a definition for the exe to link. In real use the higher-level
//! `llama` crate supplies this via `#[no_mangle]`; here we stub it solely to
//! satisfy the standalone link.

use std::os::raw::{c_char, c_void};

#[no_mangle]
pub extern "C" fn tokenCallback(_state: *mut c_void, _token: *mut c_char) -> u8 {
    1
}

#[test]
fn link_resolves() {
    // Reference the FFI so the linker keeps the shim + llama.cpp archives.
    let probe = llama_sys::_link_probe();
    assert!(probe != 0, "FFI symbol addresses should be non-null");
}
