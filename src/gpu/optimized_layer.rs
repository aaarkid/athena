//! Optimized GPU implementation that minimizes memory transfers

use ndarray::{Array2, ArrayView2};
use ocl::{Buffer, Kernel};
use crate::activations::Activation;
use crate::gpu::backend::GpuBackend;
use std::sync::Arc;

/// GPU-optimized network that keeps intermediate results on GPU
pub struct GpuOptimizedNetwork {
    layers: Vec<GpuLayerData>,
    gpu_backend: Arc<GpuBackend>,
    // Pre-allocated buffers for activations
    activation_buffers: Vec<Buffer<f32>>,
    max_batch_size: usize,
}

struct GpuLayerData {
    weights_buffer: Buffer<f32>,
    bias_buffer: Buffer<f32>,
    input_size: usize,
    output_size: usize,
    activation: Activation,
}

impl GpuOptimizedNetwork {
    /// Create a new GPU-optimized network
    pub fn new(
        layer_sizes: &[usize],
        activations: &[Activation],
        gpu_backend: Arc<GpuBackend>,
        max_batch_size: usize,
    ) -> Result<Self, String> {
        if layer_sizes.len() < 2 {
            return Err("Need at least 2 layer sizes".to_string());
        }
        if layer_sizes.len() - 1 != activations.len() {
            return Err("Number of activations must be one less than layer sizes".to_string());
        }
        
        let queue = &gpu_backend.queue;
        let mut layers = Vec::new();
        let mut activation_buffers = Vec::new();
        
        // Create layers and upload weights to GPU
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            let activation = activations[i];
            
            // Initialize weights
            let scale = match activation {
                Activation::Relu => (2.0 / input_size as f32).sqrt(),
                _ => (1.0 / input_size as f32).sqrt(),
            };
            
            let weights: Vec<f32> = (0..input_size * output_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            
            // Create GPU buffers
            let weights_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_WRITE | ocl::flags::MEM_COPY_HOST_PTR)
                .len(input_size * output_size)
                .copy_host_slice(&weights)
                .build()
                .map_err(|e| format!("Failed to create weights buffer: {}", e))?;
            
            let bias = vec![0.0f32; output_size];
            let bias_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_WRITE | ocl::flags::MEM_COPY_HOST_PTR)
                .len(output_size)
                .copy_host_slice(&bias)
                .build()
                .map_err(|e| format!("Failed to create bias buffer: {}", e))?;
            
            layers.push(GpuLayerData {
                weights_buffer,
                bias_buffer,
                input_size,
                output_size,
                activation,
            });
            
            // Pre-allocate activation buffer
            let activation_buffer = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_WRITE)
                .len(max_batch_size * output_size)
                .build()
                .map_err(|e| format!("Failed to create activation buffer: {}", e))?;
            
            activation_buffers.push(activation_buffer);
        }
        
        Ok(Self {
            layers,
            gpu_backend,
            activation_buffers,
            max_batch_size,
        })
    }
    
    /// Forward pass that keeps all intermediate results on GPU
    pub fn forward_gpu(&self, input: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let batch_size = input.shape()[0];
        if batch_size > self.max_batch_size {
            return Err(format!("Batch size {} exceeds maximum {}", batch_size, self.max_batch_size));
        }
        
        let queue = &self.gpu_backend.queue;
        let program = &self.gpu_backend.program;
        
        // Upload input to GPU
        let mut current_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .len(input.len())
            .copy_host_slice(input.as_slice().ok_or("Failed to convert input array to slice")?)
            .build()
            .map_err(|e| format!("Failed to create input buffer: {}", e))?;
        
        let mut current_size = (batch_size, self.layers[0].input_size);
        
        // Process each layer
        for (i, layer) in self.layers.iter().enumerate() {
            let output_buffer = &self.activation_buffers[i];
            
            // Select appropriate kernel based on activation
            let kernel_name = match layer.activation {
                Activation::Relu => "matmul_bias_relu",
                Activation::Sigmoid => "matmul_bias_sigmoid",
                Activation::Tanh => "matmul_bias_tanh",
                _ => "matmul_bias",
            };
            
            // Build and execute fused kernel
            let kernel = Kernel::builder()
                .program(program)
                .name(kernel_name)
                .queue(queue.clone())
                .arg(&current_buffer)
                .arg(&layer.weights_buffer)
                .arg(&layer.bias_buffer)
                .arg(output_buffer)
                .arg(batch_size as i32)
                .arg(layer.output_size as i32)
                .arg(layer.input_size as i32)
                .build()
                .map_err(|e| format!("Failed to create kernel: {}", e))?;
            
            unsafe {
                kernel
                    .cmd()
                    .global_work_size([batch_size, layer.output_size])
                    .enq()
                    .map_err(|e| format!("Failed to execute kernel: {}", e))?;
            }
            
            // For activations not handled by fused kernels, apply separately
            if !matches!(layer.activation, Activation::Relu | Activation::Sigmoid | Activation::Tanh | Activation::Linear) {
                self.apply_activation_gpu(output_buffer, batch_size * layer.output_size, layer.activation)?;
            }
            
            // Update current buffer and size for next layer
            current_buffer = output_buffer.clone();
            current_size = (batch_size, layer.output_size);
        }
        
        // Read final result back to CPU
        let mut result = vec![0.0f32; current_size.0 * current_size.1];
        current_buffer.read(&mut result).enq()
            .map_err(|e| format!("Failed to read result: {}", e))?;
        
        Ok(Array2::from_shape_vec(current_size, result)
            .map_err(|e| format!("Failed to reshape result: {}", e))?)
    }
    
    /// Apply activation function on GPU (for non-fused activations)
    fn apply_activation_gpu(&self, buffer: &Buffer<f32>, size: usize, activation: Activation) -> Result<(), String> {
        match activation {
            Activation::LeakyRelu { alpha } => {
                // Use a custom kernel for LeakyReLU
                let kernel = Kernel::builder()
                    .program(&self.gpu_backend.program)
                    .name("leaky_relu")
                    .queue(self.gpu_backend.queue.clone())
                    .arg(buffer)
                    .arg(buffer) // In-place
                    .arg(size as i32)
                    .arg(alpha) // Leak factor
                    .build()
                    .map_err(|e| format!("Failed to create LeakyReLU kernel: {}", e))?;
                
                unsafe {
                    kernel
                        .cmd()
                        .global_work_size(size)
                        .enq()
                        .map_err(|e| format!("Failed to execute LeakyReLU kernel: {}", e))?;
                }
            }
            _ => {
                // For other activations, we'd need to implement more kernels
                // or fall back to CPU
            }
        }
        Ok(())
    }
    
    /// Benchmark forward pass performance
    pub fn benchmark(&self, input: ArrayView2<f32>, iterations: usize) -> Result<f32, String> {
        use std::time::Instant;
        
        // Warmup
        for _ in 0..5 {
            let _ = self.forward_gpu(input)?;
        }
        
        // Actual benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.forward_gpu(input)?;
        }
        
        Ok(start.elapsed().as_secs_f32() * 1000.0 / iterations as f32)
    }
}

/// Additional optimized kernels to add to kernels.cl
pub const ADDITIONAL_KERNELS: &str = r#"
// Leaky ReLU activation
__kernel void leaky_relu(
    __global const float* input,
    __global float* output,
    const int size,
    const float leak_factor
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : leak_factor * val;
    }
}

// GELU activation approximation
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// Optimized matrix multiplication with tiling
#define TILE_SIZE 16

__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into local memory
        if (row < M && t * TILE_SIZE + tx < K) {
            Asub[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            Asub[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && col < N) {
            Bsub[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bsub[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[ty][k] * Bsub[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;