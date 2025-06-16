//! Simple GPU Test Example
//! 
//! This example tests basic GPU functionality to ensure it's working correctly

use athena::gpu::{GpuBackend, ComputeBackend};
use ndarray::Array2;
use std::sync::Arc;

fn main() {
    println!("=== Simple GPU Test ===\n");
    
    #[cfg(feature = "gpu")]
    {
        match test_gpu() {
            Ok(()) => println!("\nGPU test completed successfully!"),
            Err(e) => println!("\nGPU test failed: {}", e),
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features gpu");
}

#[cfg(feature = "gpu")]
fn test_gpu() -> Result<(), String> {
    // Initialize GPU backend
    let gpu_backend = Arc::new(GpuBackend::new()?);
    
    // Print GPU info
    println!("GPU Device Info:");
    println!("{}\n", gpu_backend.device_info()?);
    
    // Test 1: Small matrix multiplication
    println!("Test 1: Small matrix multiplication (128x128)");
    let a = Array2::from_shape_fn((128, 128), |_| rand::random::<f32>());
    let b = Array2::from_shape_fn((128, 128), |_| rand::random::<f32>());
    
    let result = gpu_backend.matmul(a.view(), b.view())?;
    println!("✓ Matrix multiplication successful, result shape: {:?}", result.shape());
    
    // Test 2: Matrix-vector multiplication
    println!("\nTest 2: Matrix-vector multiplication");
    let matrix = Array2::from_shape_fn((256, 128), |_| rand::random::<f32>());
    let vector = Array2::from_shape_fn((128, 1), |_| rand::random::<f32>());
    
    let result = gpu_backend.matmul(matrix.view(), vector.view())?;
    println!("✓ Matrix-vector multiplication successful, result shape: {:?}", result.shape());
    
    // Test 3: Batch operations
    println!("\nTest 3: Batch matrix multiplication");
    let batch_a = Array2::from_shape_fn((512, 256), |_| rand::random::<f32>());
    let batch_b = Array2::from_shape_fn((256, 128), |_| rand::random::<f32>());
    
    let result = gpu_backend.matmul(batch_a.view(), batch_b.view())?;
    println!("✓ Batch multiplication successful, result shape: {:?}", result.shape());
    
    // Test 4: Quick performance test
    println!("\nTest 4: Quick performance comparison");
    use std::time::Instant;
    
    let size = 512;
    let a = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
    let b = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
    
    // CPU timing
    let start = Instant::now();
    let _cpu_result = a.dot(&b);
    let cpu_time = start.elapsed().as_secs_f32() * 1000.0;
    
    // GPU timing (with warmup)
    let _ = gpu_backend.matmul(a.view(), b.view())?;
    let start = Instant::now();
    let _gpu_result = gpu_backend.matmul(a.view(), b.view())?;
    let gpu_time = start.elapsed().as_secs_f32() * 1000.0;
    
    println!("CPU time: {:.2}ms", cpu_time);
    println!("GPU time: {:.2}ms", gpu_time);
    println!("Speedup: {:.2}x", cpu_time / gpu_time);
    
    Ok(())
}