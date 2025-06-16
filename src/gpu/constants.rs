//! Constants for GPU operations

/// Default tile size for tiled matrix multiplication
pub const DEFAULT_TILE_SIZE: usize = 16;

/// Maximum GPU simulation delay in microseconds (for mock backend)
pub const MAX_GPU_SIMULATION_DELAY_US: u64 = 1000;

/// GPU simulation delay divisor (for mock backend)
pub const GPU_SIMULATION_DELAY_DIVISOR: u64 = 10;

/// Default maximum buffers to keep per size in memory pool
pub const DEFAULT_MAX_BUFFERS_PER_SIZE: usize = 10;

/// Mock GPU specifications
pub const MOCK_GPU_COMPUTE_UNITS: u32 = 96;
pub const MOCK_GPU_MAX_WORK_GROUP_SIZE: usize = 1024;
pub const MOCK_GPU_GLOBAL_MEMORY_MB: u64 = 16384;

/// OpenCL memory flags combinations
pub const GPU_MEM_READ_ONLY: u32 = 1 << 2; // CL_MEM_READ_ONLY
pub const GPU_MEM_WRITE_ONLY: u32 = 1 << 1; // CL_MEM_WRITE_ONLY
pub const GPU_MEM_READ_WRITE: u32 = 1 << 0; // CL_MEM_READ_WRITE

/// Performance thresholds
pub const MIN_SIZE_FOR_GPU_BENEFIT: usize = 128; // Minimum matrix size where GPU shows benefit