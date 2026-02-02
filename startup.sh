#!/bin/bash

set -e

echo "========================================="
echo "vLLM MI300X Development Environment"
echo "Starting setup..."
echo "========================================="

VLLM_PATH="/workspace/vllm"

git config --global user.name "Mohamed Ghayad"
git config --global user.email "mohamed.sayed.ghayad@gmail.com"

if [ -d "$VLLM_PATH" ]; then
    echo "Updating vLLM from GitHub..."
    cd "$VLLM_PATH"
    git stash
    git pull origin main
    git stash pop || true
else
    echo "Cloning vLLM from GitHub..."
    cd /workspace
    git clone https://github.com/MohamedSayedFathy/vllm_AMD_MI300X_Optimized-.git vllm
    cd "$VLLM_PATH"
fi

echo "Installing vLLM in editable mode..."
pip install -e . --no-build-isolation

echo "✓ Setup Complete!"
echo "Location: $VLLM_PATH"

exec /bin/bash