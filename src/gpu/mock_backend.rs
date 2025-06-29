use ndarray::{Array2, ArrayView2};
use std::time::Duration;
use std::thread;
use super::constants::*;

#[cfg(feature = "gpu")]
use super::{ComputeBackend, DeviceType};

// Define traits locally when gpu feature is not enabled
#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Cpu,
    IntelGpu,
    NvidiaGpu,
    AmdGpu,
}

#[cfg(not(feature = "gpu"))]
pub trait ComputeBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    fn add(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    fn multiply(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    fn relu(&self, input: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    fn device_type(&self) -> DeviceType;
}

/// Mock GPU backend for demonstration when real GPU is not available
pub struct MockGpuBackend {
    device_type: DeviceType,
    simulate_delay: bool,
}

impl MockGpuBackend {
    pub fn new() -> Self {
        eprintln!("Warning: Using mock GPU backend. Real GPU acceleration not available.");
        eprintln!("This is common in WSL2 environments. For real GPU support, use native Linux or Windows.");
        Self {
            device_type: DeviceType::IntelGpu,
            simulate_delay: true,
        }
    }
    
    pub fn device_info(&self) -> Result<String, String> {
        Ok(format!(
            "Device: Mock Intel Arc GPU (Simulated)\n\
             Vendor: Intel Corporation (Mock)\n\
             Version: OpenCL 3.0 (Mock)\n\
             Compute Units: {} (Mock)\n\
             Max Work Group Size: {} (Mock)\n\
             Global Memory: {} MB (Mock)",
            MOCK_GPU_COMPUTE_UNITS,
            MOCK_GPU_MAX_WORK_GROUP_SIZE,
            MOCK_GPU_GLOBAL_MEMORY_MB
        ))
    }
    
    fn simulate_gpu_delay(&self, size: usize) {
        if self.simulate_delay {
            // Simulate GPU computation time (much faster than CPU for large operations)
            let delay_us = (size as f64).sqrt() as u64 / GPU_SIMULATION_DELAY_DIVISOR;
            thread::sleep(Duration::from_micros(delay_us.min(MAX_GPU_SIMULATION_DELAY_US)));
        }
    }
}

impl ComputeBackend for MockGpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            return Err(format!("Dimension mismatch: ({}, {}) x ({}, {})", m, k, k2, n));
        }
        
        self.simulate_gpu_delay(m * n * k);
        
        // Use CPU for actual computation
        Ok(a.dot(&b))
    }
    
    fn add(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        if a.dim() != b.dim() {
            return Err("Dimension mismatch for addition".to_string());
        }
        
        self.simulate_gpu_delay(a.len());
        Ok(&a + &b)
    }
    
    fn multiply(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        if a.dim() != b.dim() {
            return Err("Dimension mismatch for multiplication".to_string());
        }
        
        self.simulate_gpu_delay(a.len());
        Ok(&a * &b)
    }
    
    fn relu(&self, input: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        self.simulate_gpu_delay(input.len());
        Ok(input.mapv(|x| x.max(0.0)))
    }
    
    fn device_type(&self) -> DeviceType {
        self.device_type
    }
}