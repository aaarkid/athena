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
        eprintln!("Note: this is the mock GPU backend. Every operation runs on the CPU.");
        eprintln!("Its timings are CPU timings and device_info below is fabricated.");
        Self {
            device_type: DeviceType::IntelGpu,
            // Off by default: an artificial sleep makes every measurement taken
            // against this backend meaningless. Turn it on only to exercise the
            // code path that waits.
            simulate_delay: false,
        }
    }

    /// Insert an artificial delay in every operation, scaled to the problem size.
    ///
    /// There is no reason to turn this on outside a test of the delay itself: it does
    /// not model any real device, and it makes timings taken against this backend
    /// describe nothing.
    pub fn set_simulate_delay(&mut self, simulate: bool) {
        self.simulate_delay = simulate;
    }

    /// A fabricated device string.
    ///
    /// The compute-unit, work-group and memory figures are constants in this file, not
    /// anything queried from hardware. Nothing here describes the machine it runs on.
    pub fn device_info(&self) -> Result<String, String> {
        Ok(format!(
            "Device: none. This is the mock backend; all work runs on the CPU.\n\
             The figures below are constants in src/gpu/mock_backend.rs, not hardware.\n\
             Compute Units: {}\n\
             Max Work Group Size: {}\n\
             Global Memory: {} MB",
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