Set-Location D:\new_page\rs-llama.cpp
git add llama/src/stream llama/src/lib.rs
git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(llama): port stream stop helpers (parity with stop_test.go)"
