// OpenCL kernels for neural network operations

// Matrix multiplication kernel
// C[i,j] = sum(A[i,k] * B[k,j])
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,  // rows of A
    const int N,  // cols of B
    const int K   // cols of A, rows of B
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise addition
__kernel void element_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Element-wise multiplication
__kernel void element_multiply(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// ReLU activation
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = fmax(0.0f, input[idx]);
    }
}

// ReLU derivative
__kernel void relu_derivative(
    __global const float* input,
    __global const float* grad_output,
    __global float* grad_input,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// Sigmoid activation
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

// Tanh activation
__kernel void tanh_activation(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = tanh(input[idx]);
    }
}

// Bias addition (broadcasting)
__kernel void add_bias(
    __global float* output,
    __global const float* bias,
    const int batch_size,
    const int output_size
) {
    int batch = get_global_id(0);
    int idx = get_global_id(1);
    
    if (batch < batch_size && idx < output_size) {
        output[batch * output_size + idx] += bias[idx];
    }
}

// Matrix transpose
__kernel void transpose(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Batch matrix multiplication
// C[b,i,j] = sum(A[b,i,k] * B[b,k,j])
__kernel void batch_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int batch_size,
    const int M,
    const int N,
    const int K
) {
    int batch = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);
    
    if (batch < batch_size && row < M && col < N) {
        float sum = 0.0f;
        int a_offset = batch * M * K;
        int b_offset = batch * K * N;
        
        for (int k = 0; k < K; k++) {
            sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
        }
        C[batch * M * N + row * N + col] = sum;
    }
}

// Sum reduction (for loss calculation)
__kernel void sum_reduction(
    __global const float* input,
    __global float* partial_sums,
    __local float* local_sums,
    const int size
) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    
    // Load data to local memory
    if (global_id < size) {
        local_sums[local_id] = input[global_id];
    } else {
        local_sums[local_id] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce in local memory
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sums[local_id] += local_sums[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work group
    if (local_id == 0) {
        partial_sums[get_group_id(0)] = local_sums[0];
    }
}

// ===== Optimized Fused Kernels =====

// Fused matmul + bias + ReLU
__kernel void matmul_bias_relu(
    __global const float* input,    // [batch_size, input_size]
    __global const float* weights,  // [output_size, input_size]
    __global const float* bias,     // [output_size]
    __global float* output,         // [batch_size, output_size]
    const int batch_size,
    const int output_size,
    const int input_size
) {
    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        // Compute dot product
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[out_idx * input_size + i];
        }
        
        // Apply ReLU activation
        output[batch_idx * output_size + out_idx] = fmax(0.0f, sum);
    }
}

// Fused matmul + bias + Sigmoid
__kernel void matmul_bias_sigmoid(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int batch_size,
    const int output_size,
    const int input_size
) {
    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[out_idx * input_size + i];
        }
        
        // Apply sigmoid activation
        output[batch_idx * output_size + out_idx] = 1.0f / (1.0f + exp(-sum));
    }
}

// Fused matmul + bias + Tanh
__kernel void matmul_bias_tanh(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int batch_size,
    const int output_size,
    const int input_size
) {
    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[out_idx * input_size + i];
        }
        
        // Apply tanh activation
        output[batch_idx * output_size + out_idx] = tanh(sum);
    }
}

// Fused matmul + bias (no activation)
__kernel void matmul_bias(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int batch_size,
    const int output_size,
    const int input_size
) {
    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[out_idx * input_size + i];
        }
        
        output[batch_idx * output_size + out_idx] = sum;
    }
}

// ===== Additional Optimized Kernels =====

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