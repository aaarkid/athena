use ocl::{Buffer, Queue};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

/// GPU memory pool for efficient memory management
pub struct GpuMemoryPool {
    queue: Queue,
    buffers: HashMap<(usize, usize), Vec<Buffer<f32>>>,
    allocated_count: usize,
    reused_count: usize,
}

impl GpuMemoryPool {
    /// Create a new memory pool
    pub fn new(queue: Queue) -> Self {
        Self {
            queue,
            buffers: HashMap::new(),
            allocated_count: 0,
            reused_count: 0,
        }
    }
    
    /// Get a buffer of the specified size, reusing if possible
    pub fn get_buffer(&mut self, rows: usize, cols: usize) -> Result<Buffer<f32>, String> {
        let size = rows * cols;
        let key = (rows, cols);
        
        // Check if we have a free buffer of this size
        if let Some(buffer_list) = self.buffers.get_mut(&key) {
            if let Some(buffer) = buffer_list.pop() {
                self.reused_count += 1;
                return Ok(buffer);
            }
        }
        
        // Allocate new buffer
        self.allocated_count += 1;
        Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(size)
            .build()
            .map_err(|e| format!("Failed to allocate GPU buffer: {}", e))
    }
    
    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: Buffer<f32>, rows: usize, cols: usize) {
        let key = (rows, cols);
        self.buffers.entry(key).or_insert_with(Vec::new).push(buffer);
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.allocated_count, self.reused_count)
    }
    
    /// Clear all buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.allocated_count = 0;
        self.reused_count = 0;
    }
}

/// GPU array wrapper for automatic memory management
pub struct GpuArray2 {
    buffer: Buffer<f32>,
    shape: (usize, usize),
}

impl GpuArray2 {
    /// Create from CPU array
    pub fn from_array(array: ArrayView2<f32>, queue: &Queue) -> Result<Self, String> {
        let shape = array.dim();
        let buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(shape.0 * shape.1)
            .copy_host_slice(array.as_slice().unwrap())
            .build()
            .map_err(|e| e.to_string())?;
        
        Ok(Self { buffer, shape })
    }
    
    /// Copy to CPU array
    pub fn to_array(&self) -> Result<Array2<f32>, String> {
        let mut data = vec![0.0f32; self.shape.0 * self.shape.1];
        self.buffer.read(&mut data).enq().map_err(|e| e.to_string())?;
        Ok(Array2::from_shape_vec(self.shape, data).unwrap())
    }
    
    /// Get shape
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    /// Get buffer reference
    pub fn buffer(&self) -> &Buffer<f32> {
        &self.buffer
    }
}

/// Pinned memory for faster CPU-GPU transfers
pub struct PinnedMemory {
    data: Vec<f32>,
    size: usize,
}

impl PinnedMemory {
    /// Allocate pinned memory
    pub fn new(size: usize) -> Self {
        // Note: In a real implementation, this would use OpenCL's
        // CL_MEM_ALLOC_HOST_PTR flag for actual pinned memory
        Self {
            data: vec![0.0f32; size],
            size,
        }
    }
    
    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    /// Get slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }
}