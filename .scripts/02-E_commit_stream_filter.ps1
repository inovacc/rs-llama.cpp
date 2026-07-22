Set-Location D:\new_page\rs-llama.cpp
git add llama/src/stream/filter.rs llama/src/stream/mod.rs
git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(llama): port incremental stream Filter (parity with filter_test.go)"
git rev-parse HEAD
