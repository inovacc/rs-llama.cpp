use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_src = manifest.parent().unwrap().join("llama.cpp");
    assert!(
        llama_src.join("CMakeLists.txt").exists(),
        "llama.cpp submodule not found at {} — run `git submodule update --init`",
        llama_src.display()
    );

    // ---- 1. Build llama.cpp (CPU only) via CMake + Ninja + MinGW gcc/g++. ----
    let dst = cmake::Config::new(&llama_src)
        .generator("Ninja")
        .define("CMAKE_C_COMPILER", "gcc")
        .define("CMAKE_CXX_COMPILER", "g++")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_CUDA", "OFF")
        .define("GGML_VULKAN", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_CURL", "OFF")
        // Build ONLY the llama static lib (which transitively builds ggml,
        // ggml-base, ggml-cpu). Avoids the common/app/server targets that need
        // generated headers (build-info.h) we don't use.
        .build_target("llama")
        .build();

    // ---- 2. Collect the static archives the build produced. ----
    // llama.cpp emits `libllama.a` but the ggml archives as bare `ggml.a` /
    // `ggml-base.a` / `ggml-cpu.a` (no `lib` prefix), which GNU `-lNAME` cannot
    // find. Normalize every needed archive into OUT_DIR as `libNAME.a`.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut found: std::collections::HashMap<String, PathBuf> = std::collections::HashMap::new();
    collect_archives(&dst.join("build"), &mut found);

    // GNU ld is single-pass: dependents before dependencies.
    // shim -> llama -> ggml -> ggml-cpu -> ggml-base.
    let ordered = ["llama", "ggml", "ggml-cpu", "ggml-base"];
    for name in ordered {
        let src = found.get(name).unwrap_or_else(|| {
            let keys: Vec<_> = found.keys().collect();
            panic!("expected static lib '{name}' not produced by llama.cpp build; found: {keys:?}")
        });
        let dstlib = out_dir.join(format!("lib{name}.a"));
        std::fs::copy(src, &dstlib).unwrap();
        println!("cargo:rustc-link-lib=static={name}");
    }
    // Any other ggml backends that slipped in (e.g. ggml-blas) — after base.
    for (name, src) in &found {
        if name.starts_with("ggml") && !ordered.contains(&name.as_str()) {
            std::fs::copy(src, out_dir.join(format!("lib{name}.a"))).unwrap();
            println!("cargo:rustc-link-lib=static={name}");
        }
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // ---- 3. Compile the C++ shim against the llama.cpp headers. ----
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(manifest.join("csrc"))
        .file(manifest.join("csrc/binding.cpp"))
        .compile("llama_binding");

    // ---- 4. C++ runtime + platform libs required by the MinGW static build. ----
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    // ggml-cpu queries the Windows registry (CPU feature detection) on Windows.
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        println!("cargo:rustc-link-lib=advapi32");
    }

    // ---- 5. Rebuild triggers. ----
    println!("cargo:rerun-if-changed=csrc/binding.cpp");
    println!("cargo:rerun-if-changed=csrc/binding.h");
    println!("cargo:rerun-if-changed=build.rs");
}

fn collect_archives(dir: &Path, found: &mut std::collections::HashMap<String, PathBuf>) {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_archives(&path, found);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            // Accept `libNAME.a`, bare `NAME.a`, or `NAME.lib`.
            if let Some(stem) = name.strip_suffix(".a").or_else(|| name.strip_suffix(".lib")) {
                let stem = stem.strip_prefix("lib").unwrap_or(stem);
                found.entry(stem.to_string()).or_insert(path.clone());
            }
        }
    }
}
