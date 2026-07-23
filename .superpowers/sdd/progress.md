# rs-llama.cpp port — progress ledger

Plan: D:\capture_describer\docs\superpowers\plans\2026-07-22-rs-llama-cpp-port.md
Sequencing: pure-Rust-first (submodule/native build deferred to M2)

Task 0.1: complete (commit f80d912, cargo build clean)
Task 1.1: complete (commit b4916b8, messages match errors.go verbatim, review clean)
  Minor (final review): Cargo.lock committed despite being in .gitignore.
Task 1.2: complete (commit 373102a, 12/12 Go cases ported, parse mirrors logitbias.go, review clean)
Task 1.3: complete (commit af9eeb8, all 4 Go test fns ported verbatim, byte-index semantics correct, review clean)
Task 1.4: complete (commit 7d004ac, 9/9 Go cases, review clean). Cumulative: 26 llama tests green.
Task 1.5: complete (commit 0bcfded, 7/7 sink cases; Sink wraps Filter; no header in sink.go). Controller did not re-read sink.go (only unit not cross-checked from context) — trusted ported tests.
Task 1.6: complete (commit 45b7507). Full field+default parity verified vs options.go (18 ModelOptions + 38 PredictOptions fields).
  Minor (M3.4 heads-up): token_callback is Arc<dyn Fn> not FnMut — streaming task must adapt.
Task 1.7a: complete (gguf core: keyvalue/tensor/reader/gguf + test writer helper). Commits 32fd6da,0ac2a54,5f5a30f; reviewed (parity ✅, all 39 tensor codes verified); fixed 4961bf9 (lossless Vec<u8> strings + lazy TensorDataReader + bool clarity). 44 tests green.
  API: llama::gguf::{GgufError,File,KeyValue,Value,GgufValue,TensorInfo,TensorType,TensorDataReader}; string accessors .string()/.string_bytes(). lazy.go folded into eager File::open (observationally identical).
Task 1.7b: complete (commits 6ce00b9 metadata.rs, 26b1d1c graph.rs). Info/stat + graph ported; offline metadata test via testutil; LLMARK_TEST_GGUF real test skips. 48 tests green.
Task 1.7c: complete (commit 7240727, estimate.go + 13 tests incl offline). Added GgufError::Unsupported. 61 tests green.

=== MILESTONE 1 COMPLETE (all pure-Rust ports) — 61 tests green, 14 commits ===
Minor findings for final whole-branch review:
  - Cargo.lock committed despite .gitignore listing it.
  - clippy type_complexity warnings in stream/sink.rs:24 (Option<Box<dyn FnMut>>) — extract a type alias to satisfy "clippy clean" constraint.
  - token_callback in options.rs is Arc<dyn Fn> (not FnMut) — M3.4 streaming must adapt.
Next: Milestone 2 (llama-sys) — needs llama.cpp submodule clone @178a6c4 + from-source MinGW build. Env-gated model tests (M3/M4) need real GGUF + vision model/mmproj files.

=== MILESTONE 2 COMPLETE (llama-sys native build + FFI) ===
Commits 28c8d85 (submodule@178a6c4 + cmake/ninja CPU build + shim), 6644df5 (raw FFI + link probe), d9e12d7 (.cargo/config.toml pin gnu target).
- Build: cmake+Ninja, MinGW gcc/g++, static; libs linked order llama>ggml>ggml-cpu>ggml-base + stdc++,pthread,m,advapi32. ggml archives copied to libNAME.a in OUT_DIR. build_target("llama") to skip common/httplib.
- FFI verified vs binding.h (all 15 fns, correct sigs incl 41-arg llama_allocate_params). tokenCallback declared-not-defined (llama crate defines it).
- WORKSPACE MUST BUILD gnu: pinned via .cargo/config.toml build.target=x86_64-pc-windows-gnu. 61 llama tests still green.
Next: M3 safe Model wrapper (env-gated on LLAMA_TEST_MODEL — user provides path to verify). M4 vision (LLAMA_TEST_VISION_MODEL+MMPROJ).

Task 3a (M3 core): complete (commit d0bd58d). Model::{load,tokenize,predict(resize-retry),apply_chat_template} + NotImplemented stubs. 65 tests (env-gated skip). Model is !Send+!Sync.
  CARRY-FORWARD for streaming (3.4): (a) model.rs has a PLACEHOLDER #[no_mangle] tokenCallback returning 1 — streaming must REPLACE it (dup symbol = link fail); (b) shared params helper marshals stop_prompts into C antiprompt, but Go's Predict passes nil,0 and stop-detects via the streaming sink/Filter — streaming should null antiprompt in predict path + use stream::Filter for parity.
  CLEANUP PENDING (clippy-clean constraint): options.rs 10x bool_assert_comparison + 1x type_complexity(token_callback); stream/sink.rs 2x type_complexity; gguf/mod.rs module_inception; gguf/testutil.rs manual_div_ceil, manual_repeat_n. Do a focused clippy pass.
NOT-YET-VERIFIED: all M3 inference paths compile-only until user supplies LLAMA_TEST_MODEL.

Task 3.4 (streaming): complete (commit 0a5060c, 67 tests, env-gated skip). Real tokenCallback registry (Mutex<HashMap<usize,_>> keyed by state ptr) replaced the stub; predict_stream + stream::Filter stop-detection; predict now passes nil,0 antiprompt (Go Predict parity); allocate_params gained pass_antiprompt flag (true branch = Go Eval, currently no caller until eval() ported).
  FINAL-REVIEW MUST-CHECK: predict_stream uses transmute for callback lifetime-erasure — soundness depends on CallbackGuard unregistering before closure frame drops (implementer verified all paths incl FFI-error early return). Highest-risk unsafe in the port.
