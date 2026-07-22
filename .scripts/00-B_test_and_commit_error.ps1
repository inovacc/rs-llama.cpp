#!/usr/bin/env pwsh
# Task 1.1: Port errors.go → sentinel error type
# This script tests the error module and commits the changes.

Set-Location "D:\new_page\rs-llama.cpp"

# Step 1: Run tests for the error module
Write-Host "Running tests for error module..." -ForegroundColor Cyan
cargo test -p llama error:: --lib
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Tests passed!" -ForegroundColor Green

# Step 2: Commit the changes
Write-Host "Committing changes..." -ForegroundColor Cyan
git add llama/src/error.rs llama/src/lib.rs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git add failed!" -ForegroundColor Red
    exit 1
}

git commit -m "feat(llama): port sentinel errors from errors.go"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git commit failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Commit successful!" -ForegroundColor Green

# Get the commit hash
$commitHash = git rev-parse HEAD
Write-Host "Commit hash: $commitHash" -ForegroundColor Green
