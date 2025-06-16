//! Simple GPU acceleration example that directly uses the GPU backend

use ndarray::Array2;
use std::time::Instant;

#[cfg(feature = "gpu")]
use athena::gpu::{GpuBackend, ComputeBackend};

fn main() {
    println!("=== Simple GPU Acceleration Test ===\n");
    
    #[cfg(feature = "gpu")]
    {
        match test_gpu_operations() {
            Ok(()) => println!("\nGPU tests completed successfully!"),
            Err(e) => println!("\nGPU test failed: {}", e),
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features gpu");
}

#[cfg(feature = "gpu")]
fn test_gpu_operations() -> Result<(), String> {
    // Initialize GPU backend
    let gpu_backend = GpuBackend::new()?;
    
    // Get device info
    println!("GPU Device Info:");
    println!("{}\n", gpu_backend.device_info()?);
    
    // Test different matrix sizes
    let test_sizes = vec![
        (128, 128, 128),    // Small
        (512, 512, 512),    // Medium
        (1024, 1024, 1024), // Large
        (2048, 2048, 2048), // Very large
    ];
    
    println!("Matrix Multiplication Performance:");
    println!("{:<20} {:<15} {:<15} {:<10}", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    for (m, n, k) in test_sizes {
        // Create random matrices
        let a = Array2::from_shape_fn((m, k), |_| rand::random::<f32>());
        let b = Array2::from_shape_fn((k, n), |_| rand::random::<f32>());
        
        // CPU benchmark
        let cpu_time = benchmark_cpu_matmul(&a, &b);
        
        // GPU benchmark
        let gpu_time = benchmark_gpu_matmul(&gpu_backend, &a, &b)?;
        
        let speedup = cpu_time / gpu_time;
        println!("{:<20} {:<15.2} {:<15.2} {:<10.2}x", 
            format!("{}x{}x{}", m, n, k),
            cpu_time,
            gpu_time,
            speedup
        );
    }
    
    // Test batch operations
    println!("\n\nBatch Operation Performance (1024x1024 matrices):");
    println!("{:<20} {:<15} {:<15} {:<10}", "Batch Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    let matrix_size = 1024;
    for batch_size in &[1, 4, 8, 16, 32] {
        let total_cpu_time = batch_benchmark_cpu(matrix_size, *batch_size);
        let total_gpu_time = batch_benchmark_gpu(&gpu_backend, matrix_size, *batch_size)?;
        
        let speedup = total_cpu_time / total_gpu_time;
        println!("{:<20} {:<15.2} {:<15.2} {:<10.2}x", 
            batch_size,
            total_cpu_time,
            total_gpu_time,
            speedup
        );
    }
    
    // Memory transfer analysis
    println!("\n\nMemory Transfer Analysis (2048x2048):");
    let size = 2048;
    let a = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
    let b = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
    
    // First run (includes transfer)
    let start = Instant::now();
    let _ = gpu_backend.matmul(a.view(), b.view())?;
    let first_run = start.elapsed().as_secs_f32() * 1000.0;
    
    // Subsequent runs (data already on GPU in ideal case)
    let mut subsequent_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = gpu_backend.matmul(a.view(), b.view())?;
        subsequent_times.push(start.elapsed().as_secs_f32() * 1000.0);
    }
    
    let avg_subsequent = subsequent_times.iter().sum::<f32>() / subsequent_times.len() as f32;
    
    println!("First run (with transfer): {:.2}ms", first_run);
    println!("Average subsequent runs: {:.2}ms", avg_subsequent);
    println!("Estimated transfer overhead: {:.2}ms", first_run - avg_subsequent);
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn benchmark_cpu_matmul(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let start = Instant::now();
    let _ = a.dot(b);
    start.elapsed().as_secs_f32() * 1000.0
}

#[cfg(feature = "gpu")]
fn benchmark_gpu_matmul(gpu: &GpuBackend, a: &Array2<f32>, b: &Array2<f32>) -> Result<f32, String> {
    // Warmup
    let _ = gpu.matmul(a.view(), b.view())?;
    
    let start = Instant::now();
    let _ = gpu.matmul(a.view(), b.view())?;
    Ok(start.elapsed().as_secs_f32() * 1000.0)
}

#[cfg(feature = "gpu")]
fn batch_benchmark_cpu(size: usize, batch_count: usize) -> f32 {
    let matrices_a: Vec<_> = (0..batch_count)
        .map(|_| Array2::from_shape_fn((size, size), |_| rand::random::<f32>()))
        .collect();
    let matrices_b: Vec<_> = (0..batch_count)
        .map(|_| Array2::from_shape_fn((size, size), |_| rand::random::<f32>()))
        .collect();
    
    let start = Instant::now();
    for i in 0..batch_count {
        let _ = matrices_a[i].dot(&matrices_b[i]);
    }
    start.elapsed().as_secs_f32() * 1000.0
}

#[cfg(feature = "gpu")]
fn batch_benchmark_gpu(gpu: &GpuBackend, size: usize, batch_count: usize) -> Result<f32, String> {
    let matrices_a: Vec<_> = (0..batch_count)
        .map(|_| Array2::from_shape_fn((size, size), |_| rand::random::<f32>()))
        .collect();
    let matrices_b: Vec<_> = (0..batch_count)
        .map(|_| Array2::from_shape_fn((size, size), |_| rand::random::<f32>()))
        .collect();
    
    // Warmup
    let _ = gpu.matmul(matrices_a[0].view(), matrices_b[0].view())?;
    
    let start = Instant::now();
    for i in 0..batch_count {
        let _ = gpu.matmul(matrices_a[i].view(), matrices_b[i].view())?;
    }
    Ok(start.elapsed().as_secs_f32() * 1000.0)
}