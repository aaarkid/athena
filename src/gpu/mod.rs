//! GPU acceleration support for Athena
//! 
//! This module provides GPU compute backends for accelerating neural network operations.
//! Currently supports Intel Arc GPUs via OpenCL.

pub mod constants;

#[cfg(feature = "gpu")]
pub mod backend;

#[cfg(feature = "gpu")]
pub mod kernels;

#[cfg(feature = "gpu")]
pub mod memory;

#[cfg(feature = "gpu")]
pub mod layers;

#[cfg(feature = "gpu")]
pub mod optimized_layer;

#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
pub mod mock_backend;

#[cfg(feature = "gpu")]
pub use backend::{GpuBackend, ComputeBackend, DeviceType};

#[cfg(feature = "gpu")]
pub use layers::GpuDenseLayer;

#[cfg(feature = "gpu")]
pub use optimized_layer::{GpuOptimizedNetwork, ADDITIONAL_KERNELS};

#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
pub use mock_backend::MockGpuBackend;

// Re-export types for gpu-mock feature
#[cfg(all(feature = "gpu-mock", not(feature = "gpu")))]
pub use mock_backend::{DeviceType, ComputeBackend};

// Provide GpuBackend that always returns mock when gpu-mock is used without gpu
#[cfg(all(feature = "gpu-mock", not(feature = "gpu")))]
pub struct GpuBackend;

#[cfg(all(feature = "gpu-mock", not(feature = "gpu")))]
impl GpuBackend {
    pub fn new() -> Result<MockGpuBackend, String> {
        Ok(MockGpuBackend::new())
    }
}

#[cfg(all(not(feature = "gpu"), not(feature = "gpu-mock")))]
pub struct GpuBackend;

#[cfg(all(not(feature = "gpu"), not(feature = "gpu-mock")))]
impl GpuBackend {
    pub fn new() -> Result<Self, String> {
        Err("GPU support not compiled. Enable with --features gpu".to_string())
    }
}