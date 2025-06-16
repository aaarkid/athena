# Windows Setup Guide for Athena

## Prerequisites

### 1. Rust Installation
Ensure you have Rust installed with MSVC toolchain:
```powershell
# Install Rust from https://rustup.rs/
# Make sure to install Visual Studio Build Tools or Visual Studio Community
```

### 2. OpenCL Setup (for GPU support)

#### Option A: Intel GPU (Including Intel Arc)
1. Download Intel GPU drivers from: https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html
2. The drivers include OpenCL runtime

#### Option B: NVIDIA GPU
1. Install NVIDIA drivers (includes OpenCL)
2. Optionally install CUDA Toolkit for development files

#### Option C: AMD GPU
1. Install AMD drivers with OpenCL support
2. Download AMD APP SDK if needed

#### Option D: CPU-only OpenCL
1. Download Intel CPU Runtime for OpenCL from: https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html

## Building Athena

### Without GPU Support
```powershell
cd C:\DEV\athena
cargo build --release
```

### With GPU Support

#### Method 1: Automatic (if OpenCL is in standard locations)
```powershell
cargo build --release --features gpu
```

#### Method 2: Manual OpenCL.lib Location
If you get "cannot open input file 'OpenCL.lib'" error:

1. **Find OpenCL.lib on your system:**
   ```powershell
   # Common locations:
   # NVIDIA: C:\Windows\System32\OpenCL.dll
   # Intel: C:\Windows\System32\OpenCL.dll
   # Or in GPU SDK folders
   
   # Search for OpenCL.lib:
   Get-ChildItem -Path C:\ -Filter OpenCL.lib -Recurse -ErrorAction SilentlyContinue
   ```

2. **Create a lib file from OpenCL.dll (if only DLL exists):**
   ```powershell
   # Open "x64 Native Tools Command Prompt for VS 2022"
   cd C:\Windows\System32
   dumpbin /exports OpenCL.dll > OpenCL.exports
   # Create a .def file manually from the exports
   lib /def:OpenCL.def /out:OpenCL.lib /machine:x64
   ```

3. **Set environment variable:**
   ```powershell
   $env:OPENCL_LIB_DIR = "C:\Path\To\OpenCL\lib"
   ```

4. **Update build.rs (if needed):**
   Add your specific OpenCL path to the `possible_paths` vector in build.rs

## Troubleshooting

### Error: "cannot open input file 'OpenCL.lib'"

**Solution 1: Install OpenCL SDK**
- Intel: https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/choose-download.html
- NVIDIA: Comes with CUDA Toolkit
- AMD: AMD APP SDK

**Solution 2: Use Pre-built OpenCL.lib**
1. Download from: https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
2. Extract to a folder (e.g., `C:\OpenCL-SDK`)
3. Add to build.rs or set environment variable

**Solution 3: Disable GPU support**
```powershell
# Build without GPU feature
cargo build --release
```

### Error: "Platform::list: Error retrieving platform list"

This means OpenCL runtime is not installed. Install appropriate GPU drivers:
- Intel Arc: Latest Intel Graphics Driver
- NVIDIA: Latest Game Ready or Studio Driver
- AMD: Latest Adrenalin Driver

### Verifying OpenCL Installation

1. **Check if OpenCL.dll exists:**
   ```powershell
   Test-Path C:\Windows\System32\OpenCL.dll
   ```

2. **Use GPU-Z or similar tools to verify OpenCL support**

3. **Run clinfo (if available):**
   ```powershell
   # Install clinfo via vcpkg or download pre-built
   clinfo
   ```

## Running Examples

### Basic Tests
```powershell
# Without GPU
cargo test --release

# With GPU
cargo test --release --features gpu
```

### Benchmarks
```powershell
# Simple benchmark
cargo run --release --example simple_benchmark --features gpu

# GPU acceleration example
cargo run --release --example gpu_acceleration --features gpu
```

## Testing Your Setup

After installation, run one of the test scripts:

```batch
# Full test suite (may fail on comprehensive_benchmark example)
test_windows_build.bat

# Minimal test suite (recommended for initial testing)
test_windows_minimal.bat
```

The minimal test will:
1. Build without GPU support
2. Build with GPU mock (no OpenCL required)
3. Run core library tests
4. Run simple benchmark

The full test additionally includes:
- Build with full GPU support (if OpenCL is installed)
- All example builds (some may have issues)

## Expected Output

With working GPU support:
```
=== Intel Arc GPU Acceleration Example ===

Initializing GPU layer...
GPU Backend initialized on: IntelGpu

GPU Device Info:
Device: Intel(R) Arc(TM) A770 Graphics
Vendor: Intel(R) Corporation
Version: OpenCL 3.0 NEO
Compute Units: 512
Max Work Group Size: 1024
Global Memory: 16225 MB
```

Without GPU (fallback to mock):
```
Failed to initialize real GPU: OpenCL not available...
Using mock GPU backend for demonstration.
```

## Performance Tips

1. **Use Release builds:** Debug builds are significantly slower
2. **Batch operations:** GPU performs better with larger batches
3. **Layer sizes:** GPU acceleration is most effective for layers > 256 units
4. **Memory:** Ensure sufficient GPU memory for large models

## Building Python Bindings (Optional)

```powershell
# Install Python development headers
# Ensure python.exe is in PATH

# Build Python bindings
maturin build --release --features python

# Install locally
pip install target/wheels/athena-*.whl
```

## Next Steps

1. Run verification: `cargo run --release --example simple_benchmark --features gpu`
2. Check GPU is detected correctly
3. Try the grid navigation example: `cargo run --release --example grid_navigation`
4. Read ADVANCED_FEATURES_PLAN.md for future development