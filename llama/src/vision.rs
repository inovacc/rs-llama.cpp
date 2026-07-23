//! Local image description via llama.cpp's `mtmd` (multimodal) library.
//!
//! This is the one ADDITIVE surface of the port: the text side is a faithful
//! 1:1 port of go-llama.cpp, which has no vision support. [`VisionModel`] wraps
//! an `mtmd_context` initialized from a multimodal projector (`mmproj`) GGUF,
//! and [`crate::model::Model::describe_image`] runs the mtmd evaluate +
//! sampling flow (mirroring `mtmd-cli.cpp`) against the model's existing llama
//! context.

use std::os::raw::c_void;

use llama_sys::mtmd_free_ctx;

/// An initialized mtmd (vision) context, owning the `mtmd_context*` returned by
/// the C shim's `mtmd_load`.
///
/// # Safety / lifetime requirement
///
/// A `VisionModel` is created from a [`crate::model::Model`] via
/// [`crate::model::Model::load_mmproj`] and internally references that model's
/// `llama_model`/`llama_context`. The originating `Model` **must outlive** the
/// `VisionModel` (and every `describe_image` call must pass the same `Model`
/// the projector was loaded from). Dropping the `Model` first and then using
/// the `VisionModel` is undefined behavior. This is not enforced by a lifetime
/// parameter to keep the API ergonomic (matching the raw shim contract); it is
/// the caller's responsibility, exactly as documented here.
pub struct VisionModel {
    pub(crate) ctx: *mut c_void,
}

impl VisionModel {
    /// Constructs a `VisionModel` from a raw, non-null `mtmd_context*`.
    ///
    /// # Safety
    /// `ctx` must be a valid pointer returned by the shim's `mtmd_load`, not
    /// yet freed, and owned exclusively by the returned `VisionModel`.
    pub(crate) unsafe fn from_raw(ctx: *mut c_void) -> Self {
        VisionModel { ctx }
    }
}

impl Drop for VisionModel {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was returned by a successful `mtmd_load` and is
        // owned exclusively by this `VisionModel`, so freeing it exactly once
        // here is sound. `mtmd_free_ctx` tolerates (and ignores) NULL.
        unsafe { mtmd_free_ctx(self.ctx) };
    }
}
