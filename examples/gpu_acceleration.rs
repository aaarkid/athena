#[cfg(feature = "gpu")]
use athena::layers::{GpuDenseLayer, DenseLayer, LayerTrait};
#[cfg(feature = "gpu")]
use athena::activations::Activation;
#[cfg(feature = "gpu")]
use ndarray::Array2;
#[cfg(feature = "gpu")]
use ndarray_rand::RandomExt;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU support not enabled. Run with: cargo run --example gpu_acceleration --features gpu");
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Intel Arc GPU Acceleration Example ===\n");
    
    // Create GPU layer
    println!("Initializing GPU layer...");
    let mut gpu_layer = match GpuDenseLayer::new(1024, 512, Activation::Relu) {
        Ok(layer) => layer,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}", e);
            eprintln!("Make sure you have an Intel Arc GPU and OpenCL drivers installed.");
            return Ok(());
        }
    };
    
    // Print device info
    println!("\nGPU Device Info:");
    println!("{}", gpu_layer.device_info()?);
    
    // Create CPU layer for comparison
    let mut cpu_layer = DenseLayer::new(1024, 512, Activation::Relu);
    
    // Generate test data
    let batch_size = 128;
    let input_size = 1024;
    let test_input = Array2::random((batch_size, input_size), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    
    // Benchmark single forward pass
    println!("\n=== Single Forward Pass Benchmark ===");
    let single_input = test_input.row(0);
    
    // CPU timing
    let start = Instant::now();
    let cpu_result = cpu_layer.forward(single_input);
    let cpu_time = start.elapsed();
    
    // GPU timing
    let start = Instant::now();
    let gpu_result = gpu_layer.forward(single_input);
    let gpu_time = start.elapsed();
    
    println!("CPU Time: {:?}", cpu_time);
    println!("GPU Time: {:?}", gpu_time);
    println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    // Verify results are similar
    let diff: f32 = (&cpu_result - &gpu_result).mapv(f32::abs).sum();
    println!("Result difference (should be small): {:.6}", diff);
    
    // Benchmark batch forward pass
    println!("\n=== Batch Forward Pass Benchmark ===");
    
    // Warm up
    for _ in 0..5 {
        let _ = cpu_layer.forward_batch(test_input.view());
        let _ = gpu_layer.forward_batch(test_input.view());
    }
    
    // CPU timing
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cpu_layer.forward_batch(test_input.view());
    }
    let cpu_time = start.elapsed();
    
    // GPU timing
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gpu_layer.forward_batch(test_input.view());
    }
    let gpu_time = start.elapsed();
    
    println!("CPU Time ({} iterations): {:?}", iterations, cpu_time);
    println!("GPU Time ({} iterations): {:?}", iterations, gpu_time);
    println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    // Test different batch sizes
    println!("\n=== Batch Size Performance ===");
    let batch_sizes = vec![1, 8, 32, 64, 128, 256, 512];
    
    for &batch_size in &batch_sizes {
        let test_batch = Array2::random((batch_size, input_size), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
        
        // CPU
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cpu_layer.forward_batch(test_batch.view());
        }
        let cpu_time = start.elapsed() / 10;
        
        // GPU
        let start = Instant::now();
        for _ in 0..10 {
            let _ = gpu_layer.forward_batch(test_batch.view());
        }
        let gpu_time = start.elapsed() / 10;
        
        println!("Batch size {}: CPU {:?}, GPU {:?}, Speedup: {:.2}x", 
                 batch_size, cpu_time, gpu_time, 
                 cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }
    
    // Memory transfer overhead test
    println!("\n=== Memory Transfer Analysis ===");
    let large_batch = Array2::random((1024, 1024), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    
    let start = Instant::now();
    let _ = gpu_layer.forward_batch(large_batch.view());
    let first_run = start.elapsed();
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = gpu_layer.forward_batch(large_batch.view());
    }
    let subsequent_avg = start.elapsed() / 10;
    
    println!("First run (includes memory transfer): {:?}", first_run);
    println!("Subsequent runs average: {:?}", subsequent_avg);
    println!("Estimated memory transfer overhead: {:?}", first_run - subsequent_avg);
    
    println!("\n=== Summary ===");
    println!("GPU acceleration is most effective for:");
    println!("- Large batch sizes (>32)");
    println!("- Large layer dimensions (>256)");
    println!("- Multiple sequential operations");
    println!("\nFor small operations, CPU may be faster due to memory transfer overhead.");
    
    Ok(())
}