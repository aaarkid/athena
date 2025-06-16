//! GPU Benchmark Example
//! 
//! This example benchmarks GPU acceleration for matrix operations and neural networks

use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::{Array2};
use std::time::Instant;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use athena::gpu::{GpuBackend, ComputeBackend, GpuOptimizedNetwork};

fn main() {
    println!("=== Athena GPU Benchmark ===\n");
    
    #[cfg(feature = "gpu")]
    {
        match run_benchmarks() {
            Ok(()) => println!("\nGPU benchmarks completed successfully!"),
            Err(e) => println!("\nGPU benchmark failed: {}", e),
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features gpu");
}

#[cfg(feature = "gpu")]
fn run_benchmarks() -> Result<(), String> {
    // Initialize GPU backend
    let gpu_backend = Arc::new(GpuBackend::new()?);
    
    // Print GPU info
    println!("GPU Device Info:");
    println!("{}\n", gpu_backend.device_info()?);
    
    // Test 1: Raw matrix multiplication performance
    println!("=== Test 1: Matrix Multiplication Performance ===");
    benchmark_matmul(gpu_backend.clone())?;
    
    // Test 2: Neural network comparison
    println!("\n=== Test 2: Neural Network Performance ===");
    benchmark_neural_networks()?;
    
    // Test 3: GPU-optimized network
    println!("\n=== Test 3: GPU-Optimized Network (data stays on GPU) ===");
    benchmark_gpu_optimized_network(gpu_backend)?;
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn benchmark_matmul(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    println!("{:<20} {:<15} {:<15} {:<10}", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    for size in &[256, 512, 1024, 2048, 4096] {
        let a = Array2::from_shape_fn((*size, *size), |_| rand::random::<f32>());
        let b = Array2::from_shape_fn((*size, *size), |_| rand::random::<f32>());
        
        // CPU benchmark
        let start = Instant::now();
        let _ = a.dot(&b);
        let cpu_time = start.elapsed().as_secs_f32() * 1000.0;
        
        // GPU benchmark (with warmup)
        let _ = gpu_backend.matmul(a.view(), b.view())?;
        let start = Instant::now();
        let _ = gpu_backend.matmul(a.view(), b.view())?;
        let gpu_time = start.elapsed().as_secs_f32() * 1000.0;
        
        let speedup = cpu_time / gpu_time;
        println!("{:<20} {:<15.2} {:<15.2} {:<10.2}x", 
            format!("{}x{}", size, size),
            cpu_time,
            gpu_time,
            speedup
        );
    }
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn benchmark_neural_networks() -> Result<(), String> {
    let sizes = vec![512, 1024, 1024, 512, 256];
    let activations = vec![Activation::Relu, Activation::Relu, Activation::Relu, Activation::Linear];
    let batch_size = 256;
    let iterations = 50;
    
    println!("Network architecture: {:?}", sizes);
    println!("Batch size: {}, Iterations: {}\n", batch_size, iterations);
    
    // Create test data
    let input_data = Array2::from_shape_fn((batch_size, sizes[0]), |_| rand::random::<f32>());
    
    // Build CPU network
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut cpu_network = NeuralNetwork::new(&sizes, &activations, optimizer);
    
    // Warmup
    for _ in 0..5 {
        let _ = cpu_network.forward_batch(input_data.view());
    }
    
    // Benchmark CPU
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cpu_network.forward_batch(input_data.view());
    }
    let cpu_time = start.elapsed().as_secs_f32() * 1000.0;
    
    println!("CPU Network: {:.2}ms total, {:.2}ms per iteration", 
        cpu_time, cpu_time / iterations as f32);
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn benchmark_gpu_optimized_network(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    let sizes = vec![1024, 2048, 2048, 1024, 512];
    let activations = vec![Activation::Relu, Activation::Relu, Activation::Relu, Activation::Linear];
    let batch_sizes = vec![64, 128, 256, 512, 1024];
    let iterations = 100;
    
    println!("Network architecture: {:?}", sizes);
    println!("Testing different batch sizes...\n");
    
    println!("{:<15} {:<20} {:<20}", "Batch Size", "Time/Iteration (ms)", "Throughput (samples/s)");
    println!("{:-<55}", "");
    
    for &batch_size in &batch_sizes {
        // Create test data
        let input_data = Array2::from_shape_fn((batch_size, sizes[0]), |_| rand::random::<f32>());
        
        // Create GPU-optimized network
        let gpu_network = GpuOptimizedNetwork::new(&sizes, &activations, gpu_backend.clone(), batch_size)?;
        
        // Benchmark
        let avg_time = gpu_network.benchmark(input_data.view(), iterations)?;
        let throughput = (batch_size as f32 * 1000.0) / avg_time;
        
        println!("{:<15} {:<20.2} {:<20.0}", 
            batch_size,
            avg_time,
            throughput
        );
    }
    
    // Compare with single large batch
    println!("\n=== Large Batch Performance ===");
    let large_batch_size = 2048;
    let input_data = Array2::from_shape_fn((large_batch_size, sizes[0]), |_| rand::random::<f32>());
    
    // Create network with larger capacity
    let gpu_network = GpuOptimizedNetwork::new(&sizes, &activations, gpu_backend, large_batch_size)?;
    
    // Measure time for single forward pass
    let start = Instant::now();
    let _ = gpu_network.forward_gpu(input_data.view())?;
    let single_pass = start.elapsed().as_secs_f32() * 1000.0;
    
    println!("Single forward pass (batch={}): {:.2}ms", large_batch_size, single_pass);
    println!("Throughput: {:.0} samples/second", (large_batch_size as f32 * 1000.0) / single_pass);
    
    Ok(())
}

// Additional benchmarking utilities
#[cfg(feature = "gpu")]
fn measure_memory_transfer_overhead(gpu_backend: &GpuBackend, size: usize) -> Result<f32, String> {
    let data = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
    
    // First run includes transfer
    let start = Instant::now();
    let _ = gpu_backend.matmul(data.view(), data.view())?;
    let first_time = start.elapsed().as_secs_f32() * 1000.0;
    
    // Subsequent runs (ideally cached)
    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = gpu_backend.matmul(data.view(), data.view())?;
        times.push(start.elapsed().as_secs_f32() * 1000.0);
    }
    
    let avg_time = times.iter().sum::<f32>() / times.len() as f32;
    Ok(first_time - avg_time)
}