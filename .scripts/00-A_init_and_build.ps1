# Initialize git, build, and commit the workspace
Set-Location "D:\new_page\rs-llama.cpp"

# Initialize git
Write-Output "Initializing git repository..."
git init

# Add all files
Write-Output "Adding files to git..."
git add -A

# Commit
Write-Output "Committing files..."
git commit -m "chore: scaffold rs-llama.cpp workspace"

# Capture commit hash
$commitHash = git rev-parse HEAD
Write-Output "Commit hash: $commitHash"

# Build
Write-Output "Building workspace..."
cargo build 2>&1

Write-Output "Build complete"
