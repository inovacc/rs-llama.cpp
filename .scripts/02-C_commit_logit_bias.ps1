Set-Location D:\new_page\rs-llama.cpp
git add llama/src/logit_bias.rs llama/src/lib.rs
git -c user.name=rs-llama -c user.email=rs-llama@local commit -m "feat(llama): port logit_bias parser (parity with logitbias_test.go)"
git rev-parse HEAD
