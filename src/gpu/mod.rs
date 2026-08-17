//! GPU acceleration support for Athena
//!
//! Compute backends for the parts of a network that have kernels. Work goes through
//! OpenCL: the backend picks an Intel Arc device first, then falls back to NVIDIA, AMD,
//! or whatever else the platform reports.
//!
//! Matrix multiplication, elementwise addition and multiplication, and the activation
//! functions have kernels. Everything else stays on the CPU, so a network is not moved
//! wholesale to the device.
//!
//! # The two features
//!
//! - `gpu` compiles the OpenCL backend and needs OpenCL drivers installed.
//! - `gpu-mock` compiles the same API with no OpenCL dependency. **Every operation runs
//!   on the CPU.** `device_type` reports `IntelGpu` and `device_info` returns a
//!   fabricated device string with invented compute-unit and memory figures. It exists
//!   so the GPU API compiles and can be tested for shape on a machine with no SDK; its
//!   timings are CPU timings and mean nothing about any device.
//!
//! `--all-features` does not link without OpenCL installed.

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