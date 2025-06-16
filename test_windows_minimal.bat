@echo off
echo === Testing Athena Build on Windows (Minimal) ===
echo.

echo 1. Testing build without GPU...
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo Build without GPU failed!
    exit /b 1
)
echo Build without GPU: SUCCESS

echo.
echo 2. Testing build with GPU mock (no OpenCL required)...
cargo build --release --features gpu-mock
if %ERRORLEVEL% NEQ 0 (
    echo Build with GPU mock failed!
    exit /b 1
)
echo Build with GPU mock: SUCCESS

echo.
echo 3. Running minimal tests...
cargo test --release --lib
if %ERRORLEVEL% NEQ 0 (
    echo Tests failed!
    exit /b 1
)
echo Tests: SUCCESS

echo.
echo 4. Running simple benchmark...
cargo run --release --example simple_benchmark --features gpu-mock
if %ERRORLEVEL% NEQ 0 (
    echo Benchmark failed!
    exit /b 1
)

echo.
echo === All tests completed successfully! ===
echo.
echo Next steps:
echo - If you need real GPU support, install OpenCL SDK for your GPU
echo - See WINDOWS_SETUP.md for detailed instructions
echo - Use --features gpu-mock for development without OpenCL