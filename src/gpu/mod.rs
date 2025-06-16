//! GPU acceleration support for Athena
//! 
//! This module provides GPU compute backends for accelerating neural network operations.
//! Currently supports Intel Arc GPUs via OpenCL.

#[cfg(feature = "gpu")]
pub mod backend;

#[cfg(feature = "gpu")]
pub mod kernels;

#[cfg(feature = "gpu")]
pub mod memory;

#[cfg(feature = "gpu")]
pub mod mock_backend;

#[cfg(feature = "gpu")]
pub use backend::{GpuBackend, ComputeBackend, DeviceType};

#[cfg(feature = "gpu")]
pub use mock_backend::MockGpuBackend;

#[cfg(not(feature = "gpu"))]
pub struct GpuBackend;

#[cfg(not(feature = "gpu"))]
impl GpuBackend {
    pub fn new() -> Result<Self, String> {
        Err("GPU support not compiled. Enable with --features gpu".to_string())
    }
}