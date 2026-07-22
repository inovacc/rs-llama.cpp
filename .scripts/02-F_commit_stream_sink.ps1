cd D:\new_page\rs-llama.cpp
git add llama/src/stream/sink.rs llama/src/stream/mod.rs
git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(llama): port stream sink (parity with sink_test.go)"
