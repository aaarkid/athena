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