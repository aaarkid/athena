#!/bin/bash

echo "=== Athena Setup Verification Script ==="
echo

# Check Rust version
echo "1. Checking Rust installation:"
rustc --version
cargo --version
echo

# Check if we're in WSL
echo "2. Checking environment:"
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "   ✓ Running in WSL2"
    echo "   ⚠ Note: GPU acceleration will use mock backend in WSL2"
else
    echo "   ✓ Running in native Linux/macOS"
fi
echo

# Check OpenCL availability
echo "3. Checking OpenCL:"
if command -v clinfo &> /dev/null; then
    echo "   ✓ clinfo found"
    clinfo 2>/dev/null | grep -E "(Platform|Device)" | head -10 || echo "   ⚠ No OpenCL platforms found"
else
    echo "   ⚠ clinfo not installed (optional)"
fi

# Check for Intel GPU
echo "   Checking for Intel GPU:"
lspci 2>/dev/null | grep -i "vga\|display" | grep -i intel || echo "   ⚠ No Intel GPU detected via lspci"
echo

# Build tests
echo "4. Building Athena:"
echo "   Building without GPU..."
cargo build --release 2>&1 | grep -E "(Finished|error)" || echo "   ✗ Build failed"

echo "   Building with GPU..."
cargo build --release --features gpu 2>&1 | grep -E "(Finished|error)" || echo "   ✗ Build failed"
echo

# Run tests
echo "5. Running tests:"
echo "   Running basic tests..."
cargo test --release 2>&1 | grep "test result" | tail -1

echo "   Running GPU tests..."
cargo test --release --features gpu gpu_tests 2>&1 | grep "test result" | grep gpu_tests | tail -1
echo

# Run examples
echo "6. Running examples:"
echo "   Running grid navigation example..."
timeout 5 cargo run --release --example grid_navigation 2>&1 | head -20

echo
echo "   Running GPU acceleration example..."
cargo run --release --example gpu_acceleration --features gpu 2>&1 | grep -E "(Intel Arc GPU|Mock|Device:|Speedup:|Summary)" | head -20
echo

# Check features
echo "7. Checking implemented features:"
echo -n "   ✓ Neural Networks: "; ls src/network.rs &>/dev/null && echo "Yes" || echo "No"
echo -n "   ✓ RL Algorithms: "; ls src/algorithms/ &>/dev/null && echo "Yes" || echo "No"  
echo -n "   ✓ GPU Support: "; ls src/gpu/ &>/dev/null && echo "Yes" || echo "No"
echo -n "   ✓ LSTM/GRU: "; ls src/layers/lstm.rs &>/dev/null && echo "Yes" || echo "No"
echo -n "   ✓ Embeddings: "; ls src/layers/embedding.rs &>/dev/null && echo "Yes" || echo "No"
echo -n "   ✓ Tensorboard: "; ls src/tensorboard/ &>/dev/null && echo "Yes" || echo "No"
echo -n "   ✓ Property Tests: "; ls tests/property_tests.rs &>/dev/null && echo "Yes" || echo "No"
echo

# Performance check
echo "8. Quick performance check:"
cargo run --release --example gpu_acceleration --features gpu 2>&1 | grep -A5 "Batch Size Performance" | tail -6
echo

echo "=== Verification Complete ==="
echo
echo "Next steps:"
echo "1. If running on native Linux/Windows with Intel Arc GPU:"
echo "   - Install Intel Compute Runtime for OpenCL"
echo "   - Install clinfo: sudo apt-get install clinfo"
echo "   - Check GPU with: clinfo | grep 'Device Name'"
echo
echo "2. To start implementing advanced features:"
echo "   - Review ADVANCED_FEATURES_PLAN.md"
echo "   - Start with multi-head attention: src/layers/attention.rs"
echo "   - Implement AdamW optimizer: src/optimizer/adamw.rs"
echo
echo "3. For development in WSL2:"
echo "   - GPU will use mock backend (this is normal)"
echo "   - For real GPU testing, use native OS"
echo "   - All features work correctly with mock backend"