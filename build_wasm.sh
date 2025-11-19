#!/bin/bash

# Build WebAssembly module for Athena

echo "Building Athena for WebAssembly..."

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Please install it with:"
    echo "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build the WASM module
wasm-pack build --target web --out-dir pkg --features wasm

# Copy example files
cp examples/wasm/index.html pkg/

echo "Build complete! To test:"
echo "1. cd pkg"
echo "2. python -m http.server 8000"
echo "3. Open http://localhost:8000 in your browser"