# rs-llama.cpp
<!-- rev:003 (RFC 3339) 2026-07-23T02:56:16Z -->

Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp) — a
faithful port of [go-llama.cpp](https://github.com/dyammarcano/go-llama.cpp)'s
public surface, plus one additive capability: local image description via
llama.cpp's `mtmd` (multimodal) library.

## What this is

- **Faithful port**: `Model::load`, `tokenize`, `predict`, `predict_stream`,
  and `apply_chat_template` mirror go-llama.cpp's `New`, `TokenizeString`,
  `Predict`/`PredictResult`, `SetTokenCallback`, and `ApplyChatTemplate`
  field-for-field and behavior-for-behavior. Surfaces go-llama.cpp declared
  but never wired up (`LoadState`, `SaveState`, `Embeddings`,
  `TokenEmbeddings`, `SpeculativeSampling`) are ported as typed
  `Err(LlamaError::NotImplemented)` stubs rather than silently omitted.
- **Additive vision**: `Model::load_mmproj` + `Model::describe_image` load a
  multimodal projector (`mmproj` GGUF) and run the mtmd encode → eval →
  greedy-generate flow to describe an image with a text prompt. go-llama.cpp
  has no equivalent — this is new surface built directly on llama.cpp's
  `tools/mtmd`.
- **Pure-Rust GGUF reader**: `llama::gguf` parses GGUF file headers
  (metadata, tensor descriptors, layer-fit estimation) without loading a
  model or linking against llama.cpp at all.

## Workspace layout

| Crate | Purpose |
|---|---|
| [`llama-sys`](llama-sys) | Raw FFI to llama.cpp. Builds llama.cpp from source (CMake + Ninja) as part of `cargo build`, and compiles a small C++17 shim (`llama-sys/csrc/binding.{h,cpp}`) that adapts llama.cpp's C++ API to a flat C ABI `llama-sys` binds against (`llama-sys/build.rs`, `llama-sys/src`). |
| [`llama`](llama) | Safe Rust API over `llama-sys`: `Model` (load/tokenize/predict/stream/chat-template/vision) plus the pure-Rust `gguf` metadata reader (no `llama-sys` dependency). |

## Build prerequisites

`llama-sys` compiles llama.cpp **from source** on every build, so you need:

- A C++17 toolchain (see the Windows note below)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)
- The `llama.cpp` git submodule, checked out:
  ```sh
  git submodule update --init
  ```

### Windows

The llama.cpp build and the C shim are compiled with **MinGW gcc/g++** (GNU
ABI) — an MSVC-ABI Rust toolchain cannot link the resulting objects. The
workspace pins the GNU target in [`.cargo/config.toml`](.cargo/config.toml):

```toml
[build]
target = "x86_64-pc-windows-gnu"
```

so `cargo build` / `cargo test` work without an explicit `--target` flag, as
long as you have:
- `rustup target add x86_64-pc-windows-gnu`
- A MinGW `gcc`/`g++` on `PATH` (e.g. via MSYS2 or the MinGW toolchain rustup
  installs alongside the `-gnu` target)

### GPU backends

The from-source build currently targets **CPU only** (`GGML_CUDA=OFF`,
`GGML_VULKAN=OFF`, `GGML_NATIVE=OFF` in `llama-sys/build.rs`) and that is what
has been built and verified against llama.cpp release **b10091** (text:
load/tokenize/predict/stream/chat; vision: `describe_image` via mtmd). CUDA and
Vulkan backends are future work, not
yet wired up or verified — see `llama-sys/build.rs` if you want to
experiment.

## Usage

### Load a model and generate text

```rust
use llama::{Model, ModelOptions, PredictOptions};

let model = Model::load("/path/to/model.gguf", &ModelOptions::default())?;

let opts = PredictOptions {
    temperature: 0.0,
    seed: 1,
    tokens: 64,
    ..PredictOptions::default()
};

let text = model.predict("The capital of France is", &opts)?;
println!("{text}");
```

### Tokenize

```rust
let tokens = model.tokenize("The quick brown fox", &PredictOptions::default())?;
```

### Streaming generation

```rust
let mut streamed = String::new();
model.predict_stream("The capital of France is", &opts, &mut |piece| {
    streamed.push_str(piece);
    print!("{piece}");
    true // return false to stop generation early
})?;
```

### Chat template

```rust
if let Some(prompt) = model.apply_chat_template("You are a helpful assistant.", "Hello!")? {
    let reply = model.predict(&prompt, &opts)?;
}
```

### Describe an image (mtmd vision — additive)

```rust
let vision = model.load_mmproj("/path/to/mmproj.gguf")?;
let description = model.describe_image(&vision, "/path/to/image.png", "Describe this image in one sentence.")?;
println!("{description}");
```

`vision` must always be used with the same `Model` it was created from — see
the safety notes on [`VisionModel`](llama/src/vision.rs) and
[`Model::describe_image`](llama/src/model.rs).

### Pure-Rust GGUF metadata (no model load, no llama.cpp)

```rust
use llama::gguf;

let info = gguf::stat("/path/to/model.gguf")?;
println!("{} ({} layers, {})", info.architecture, info.block_count, info.quantization);

let estimate = gguf::estimate_layers(
    "/path/to/model.gguf",
    &gguf::EstimateOptions {
        free_vram: 8 << 30, // 8 GiB budget
        ..Default::default()
    },
)?;
println!("recommended n_gpu_layers = {}", estimate.layers);
```

`gguf::stat` and `gguf::estimate_layers` are ports of go-llama.cpp's `gguf`
package (itself derived from `github.com/ollama/ollama/fs/gguf`, MIT
licensed) — see [`llama/src/gguf`](llama/src/gguf) for the full module.

## Testing

```sh
cargo test --workspace
```

Pure-Rust unit tests (options, streaming filter, GGUF parsing, etc.) always
run. Tests that need a real model or an actual llama.cpp/mtmd context are
gated behind environment variables and are skipped (not failed) when unset:

| Env var | Gates |
|---|---|
| `LLAMA_TEST_MODEL` | Text-model tests: load, tokenize, predict, streaming |
| `LLAMA_TEST_VISION_MODEL` | Vision-capable base model for `describe_image` |
| `LLAMA_TEST_MMPROJ` | Multimodal projector (`mmproj`) GGUF for `describe_image` |
| `LLAMA_TEST_IMAGE` | Optional; defaults to the committed fixture at `llama/tests/fixtures/red_circle.png` |

## License

[BSD 3-Clause](LICENSE), copyright inovacc.

Ported from [go-llama.cpp](https://github.com/dyammarcano/go-llama.cpp). The
`llama::gguf` module is further derived from `github.com/ollama/ollama/fs/gguf`
(MIT licensed) via go-llama.cpp's `gguf` package — see the doc comments in
[`llama/src/gguf`](llama/src/gguf) for provenance.
