Set-Location D:\new_page\rs-llama.cpp
git add llama/src/gguf llama/src/lib.rs llama/Cargo.toml Cargo.lock .scripts

git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(gguf): port keyvalue.go (parity with keyvalue_test.go)" -- llama/src/gguf/keyvalue.rs

git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(gguf): port reader/tensor core + test GGUF writer" -- llama/src/gguf/reader.rs llama/src/gguf/tensor.rs llama/src/gguf/testutil.rs

git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(gguf): port gguf.go (parity with gguf_test.go)" -- llama/src/gguf/gguf.rs llama/src/gguf/mod.rs llama/src/lib.rs llama/Cargo.toml Cargo.lock .scripts

git log --oneline -5
