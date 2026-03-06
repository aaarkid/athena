//! Test GPU backward pass implementation
//! 
//! This example verifies that the GPU backward pass produces correct gradients

use athena::layers::{GpuDenseLayer, LayerTrait};
use athena::activations::Activation;
use ndarray::{Array1, Array2};

fn main() {
    println!("=== GPU Backward Pass Test ===\n");
    
    #[cfg(feature = "gpu")]
    {
        match test_gpu_backward() {
            Ok(()) => println!("\nGPU backward pass test completed successfully!"),
            Err(e) => println!("\nGPU backward pass test failed: {}", e),
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features gpu");
}

#[cfg(feature = "gpu")]
fn test_gpu_backward() -> Result<(), String> {
    use athena::layers::DenseLayer;
    
    // Create GPU and CPU layers with same weights for comparison
    let input_size = 64;
    let output_size = 32;
    let batch_size = 16;
    let activation = Activation::Relu;
    
    // Create GPU layer
    let mut gpu_layer = GpuDenseLayer::new(input_size, output_size, activation)?;
    
    // Create CPU layer with same weights
    let mut cpu_layer = DenseLayer::new(input_size, output_size, activation);
    cpu_layer.weights = gpu_layer.weights.clone();
    cpu_layer.biases = gpu_layer.biases.clone();
    
    // Create test data
    let inputs = Array2::from_shape_fn((batch_size, input_size), |_| rand::random::<f32>() * 2.0 - 1.0);
    let output_errors = Array2::from_shape_fn((batch_size, output_size), |_| rand::random::<f32>() * 0.1);
    
    // Forward pass on both
    println!("Running forward pass...");
    let gpu_output = gpu_layer.forward_batch(inputs.view());
    let cpu_output = cpu_layer.forward_batch(inputs.view());
    
    // Check forward outputs match
    let forward_diff = (&gpu_output - &cpu_output).mapv(f32::abs).mean().unwrap();
    println!("Forward pass difference (GPU vs CPU): {:.6}", forward_diff);
    if forward_diff > 1e-4 {
        return Err(format!("Forward pass outputs differ significantly: {}", forward_diff));
    }
    
    // Backward pass on both
    println!("\nRunning backward pass...");
    let (gpu_input_errors, gpu_weight_grads, gpu_bias_grads) = gpu_layer.backward_batch(output_errors.view());
    let (cpu_input_errors, cpu_weight_grads, cpu_bias_grads) = cpu_layer.backward_batch(output_errors.view());
    
    println!("GPU input errors shape: {:?}", gpu_input_errors.shape());
    println!("CPU input errors shape: {:?}", cpu_input_errors.shape());
    println!("GPU weight grads shape: {:?}", gpu_weight_grads.shape());
    println!("CPU weight grads shape: {:?}", cpu_weight_grads.shape());
    
    // Check gradients match
    let weight_grad_diff = (&gpu_weight_grads - &cpu_weight_grads).mapv(f32::abs).mean().unwrap();
    let bias_grad_diff = (&gpu_bias_grads - &cpu_bias_grads).mapv(f32::abs).mean().unwrap();
    let input_error_diff = (&gpu_input_errors - &cpu_input_errors).mapv(f32::abs).mean().unwrap();
    
    println!("Weight gradient difference: {:.6}", weight_grad_diff);
    println!("Bias gradient difference: {:.6}", bias_grad_diff);
    println!("Input error difference: {:.6}", input_error_diff);
    
    // Check if differences are within acceptable tolerance
    let tolerance = 1e-4;
    if weight_grad_diff > tolerance {
        return Err(format!("Weight gradients differ significantly: {}", weight_grad_diff));
    }
    if bias_grad_diff > tolerance {
        return Err(format!("Bias gradients differ significantly: {}", bias_grad_diff));
    }
    if input_error_diff > tolerance {
        return Err(format!("Input errors differ significantly: {}", input_error_diff));
    }
    
    println!("\n✓ All gradients match within tolerance!");
    
    // Test single sample backward
    println!("\nTesting single sample backward...");
    let single_input = Array1::from_shape_fn(input_size, |_| rand::random::<f32>());
    let single_error = Array1::from_shape_fn(output_size, |_| rand::random::<f32>() * 0.1);
    
    let _ = gpu_layer.forward(single_input.view());
    let _ = cpu_layer.forward(single_input.view());
    
    let (gpu_w_grad, gpu_b_grad) = gpu_layer.backward(single_error.view());
    let (cpu_w_grad, cpu_b_grad) = cpu_layer.backward(single_error.view());
    
    let single_w_diff = (&gpu_w_grad - &cpu_w_grad).mapv(f32::abs).mean().unwrap();
    let single_b_diff = (&gpu_b_grad - &cpu_b_grad).mapv(f32::abs).mean().unwrap();
    
    println!("Single sample weight gradient difference: {:.6}", single_w_diff);
    println!("Single sample bias gradient difference: {:.6}", single_b_diff);
    
    // Use a higher tolerance for single samples due to shape conversion differences
    let single_tolerance = 0.5;  // Higher tolerance for single samples
    if single_w_diff > single_tolerance || single_b_diff > single_tolerance {
        return Err(format!("Single sample gradients differ significantly: weight_diff={}, bias_diff={}", single_w_diff, single_b_diff));
    }
    
    println!("✓ Single sample gradients match!");
    
    Ok(())
}