use ndarray::{Array2, ArrayView2};
use super::{ComputeBackend, DeviceType};
use std::time::Duration;
use std::thread;

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
             Compute Units: 96 (Mock)\n\
             Max Work Group Size: 1024 (Mock)\n\
             Global Memory: 16384 MB (Mock)"
        ))
    }
    
    fn simulate_gpu_delay(&self, size: usize) {
        if self.simulate_delay {
            // Simulate GPU computation time (much faster than CPU for large operations)
            let delay_us = (size as f64).sqrt() as u64 / 10;
            thread::sleep(Duration::from_micros(delay_us.min(1000)));
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