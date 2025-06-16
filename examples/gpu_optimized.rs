//! GPU-Optimized Neural Network Example
//! 
//! This example demonstrates:
//! - Direct GPU acceleration for matrix operations
//! - GPU-optimized network that keeps data on GPU between layers
//! - Performance comparison between CPU and GPU
//! - Scaling behavior with different network and batch sizes

use athena::network::NeuralNetwork;
use athena::activations::Activation;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::{Array2};
use std::time::Instant;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use athena::gpu::{GpuBackend, ComputeBackend, GpuOptimizedNetwork};

fn main() {
    println!("=== GPU-Optimized Neural Network Example ===\n");
    
    #[cfg(feature = "gpu")]
    {
        match run_gpu_benchmarks() {
            Ok(()) => println!("\nAll GPU benchmarks completed successfully!"),
            Err(e) => println!("\nGPU benchmark failed: {}", e),
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features gpu to enable GPU acceleration.");
}

#[cfg(feature = "gpu")]
fn run_gpu_benchmarks() -> Result<(), String> {
    // Initialize GPU backend
    let gpu_backend = Arc::new(GpuBackend::new()?);
    
    // Print GPU info
    println!("GPU Device Info:");
    println!("{}\n", gpu_backend.device_info()?);
    
    // Test 1: Raw GPU matrix multiplication performance
    println!("=== Test 1: Matrix Multiplication Performance ===");
    test_matmul_performance(gpu_backend.clone())?;
    
    // Test 2: GPU-optimized network (data stays on GPU)
    println!("\n=== Test 2: GPU-Optimized Network Performance ===");
    test_gpu_optimized_network(gpu_backend.clone())?;
    
    // Test 3: Scaling analysis
    println!("\n=== Test 3: Performance Scaling Analysis ===");
    scaling_analysis(gpu_backend.clone())?;
    
    // Test 4: Memory transfer overhead
    println!("\n=== Test 4: Memory Transfer Analysis ===");
    memory_transfer_analysis(gpu_backend)?;
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn test_matmul_performance(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    println!("Comparing CPU vs GPU matrix multiplication:\n");
    println!("{:<20} {:<15} {:<15} {:<10}", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    for size in &[512, 1024, 2048, 4096] {
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
fn test_gpu_optimized_network(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    let layer_sizes = vec![1024, 2048, 2048, 1024, 512];
    let activations = vec![Activation::Relu, Activation::Relu, Activation::Relu, Activation::Linear];
    let batch_size = 512;
    let iterations = 100;
    
    println!("Network architecture: {:?}", layer_sizes);
    println!("Batch size: {}, Iterations: {}\n", batch_size, iterations);
    
    // Create test data
    let input_data = Array2::from_shape_fn((batch_size, layer_sizes[0]), |_| rand::random::<f32>());
    
    // Create GPU-optimized network
    let gpu_network = GpuOptimizedNetwork::new(&layer_sizes, &activations, gpu_backend.clone(), batch_size)?;
    
    // Benchmark GPU-optimized network
    let avg_gpu_time = gpu_network.benchmark(input_data.view(), iterations)?;
    println!("GPU-Optimized Network: {:.2}ms per iteration", avg_gpu_time);
    
    // Compare with CPU network
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut cpu_network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
    
    // Warmup CPU
    for _ in 0..5 {
        let _ = cpu_network.forward_batch(input_data.view());
    }
    
    // Benchmark CPU
    let start = Instant::now();
    for _ in 0..10 { // Fewer iterations for CPU
        let _ = cpu_network.forward_batch(input_data.view());
    }
    let cpu_time = start.elapsed().as_secs_f32() * 1000.0 / 10.0;
    
    println!("CPU Network: {:.2}ms per iteration", cpu_time);
    println!("\n🚀 GPU Speedup: {:.2}x", cpu_time / avg_gpu_time);
    
    // Test throughput
    let throughput_gpu = (batch_size as f32 * 1000.0) / avg_gpu_time;
    let throughput_cpu = (batch_size as f32 * 1000.0) / cpu_time;
    
    println!("\nThroughput:");
    println!("GPU: {:.0} samples/second", throughput_gpu);
    println!("CPU: {:.0} samples/second", throughput_cpu);
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn scaling_analysis(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    // Test different layer sizes
    println!("Layer Size Scaling (batch_size=256, 5-layer network):\n");
    println!("{:<20} {:<15} {:<15} {:<10}", "Layer Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    for size in &[256, 512, 1024, 2048] {
        let layer_sizes = vec![*size; 6]; // 5-layer network
        let activations = vec![Activation::Relu; 5];
        let batch_size = 256;
        let input_data = Array2::from_shape_fn((batch_size, *size), |_| rand::random::<f32>());
        
        // CPU network
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut cpu_network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
        
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cpu_network.forward_batch(input_data.view());
        }
        let cpu_time = start.elapsed().as_secs_f32() * 100.0; // Convert to ms
        
        // GPU network
        let gpu_network = GpuOptimizedNetwork::new(&layer_sizes, &activations, gpu_backend.clone(), batch_size)?;
        let gpu_time = gpu_network.benchmark(input_data.view(), 10)?;
        
        let speedup = cpu_time / gpu_time;
        println!("{:<20} {:<15.2} {:<15.2} {:<10.2}x", 
            format!("{} neurons", size),
            cpu_time,
            gpu_time,
            speedup
        );
    }
    
    // Test different batch sizes
    println!("\n\nBatch Size Scaling (2048x2048x2048 network):");
    println!("{:<20} {:<15} {:<15} {:<10}", "Batch Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");
    
    let layer_sizes = vec![2048, 2048, 2048];
    let activations = vec![Activation::Relu, Activation::Linear];
    
    for batch_size in &[32, 64, 128, 256, 512, 1024] {
        let input_data = Array2::from_shape_fn((*batch_size, 2048), |_| rand::random::<f32>());
        
        // CPU network
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut cpu_network = NeuralNetwork::new(&layer_sizes, &activations, optimizer);
        
        let start = Instant::now();
        for _ in 0..5 {
            let _ = cpu_network.forward_batch(input_data.view());
        }
        let cpu_time = start.elapsed().as_secs_f32() * 200.0; // Convert to ms
        
        // GPU network
        let gpu_network = GpuOptimizedNetwork::new(&layer_sizes, &activations, gpu_backend.clone(), *batch_size)?;
        let gpu_time = gpu_network.benchmark(input_data.view(), 5)?;
        
        let speedup = cpu_time / gpu_time;
        println!("{:<20} {:<15.2} {:<15.2} {:<10.2}x", 
            batch_size,
            cpu_time,
            gpu_time,
            speedup
        );
    }
    
    Ok(())
}

#[cfg(feature = "gpu")]
fn memory_transfer_analysis(gpu_backend: Arc<GpuBackend>) -> Result<(), String> {
    println!("Analyzing memory transfer overhead...\n");
    
    let sizes = vec![1024, 2048, 4096];
    
    println!("{:<20} {:<20} {:<20} {:<20}", "Matrix Size", "First Run (ms)", "Avg Subsequent (ms)", "Transfer Overhead");
    println!("{:-<80}", "");
    
    for size in sizes {
        let a = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
        let b = Array2::from_shape_fn((size, size), |_| rand::random::<f32>());
        
        // First run (includes transfer)
        let start = Instant::now();
        let _ = gpu_backend.matmul(a.view(), b.view())?;
        let first_run = start.elapsed().as_secs_f32() * 1000.0;
        
        // Subsequent runs
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = gpu_backend.matmul(a.view(), b.view())?;
            times.push(start.elapsed().as_secs_f32() * 1000.0);
        }
        
        let avg_subsequent = times.iter().sum::<f32>() / times.len() as f32;
        let overhead = first_run - avg_subsequent;
        
        println!("{:<20} {:<20.2} {:<20.2} {:<20.2}", 
            format!("{}x{}", size, size),
            first_run,
            avg_subsequent,
            overhead
        );
    }
    
    println!("\nNote: Memory transfer overhead is the difference between first run and subsequent runs.");
    println!("In a real application, data would ideally stay on GPU between operations.");
    
    Ok(())
}