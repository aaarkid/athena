use ocl::{Platform, Device, Context, Queue, Program, Kernel, Buffer};
use ocl::DeviceType as OclDeviceType;
use ndarray::{Array2, ArrayView2};
use std::fmt;

/// Supported device types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Cpu,
    IntelGpu,
    NvidiaGpu,
    AmdGpu,
}

/// Trait for compute backends
pub trait ComputeBackend {
    /// Matrix multiplication
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    
    /// Element-wise addition
    fn add(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    
    /// Element-wise multiplication
    fn multiply(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    
    /// Apply activation function
    fn relu(&self, input: ArrayView2<f32>) -> Result<Array2<f32>, String>;
    
    /// Get device type
    fn device_type(&self) -> DeviceType;
}

/// GPU backend using OpenCL
pub struct GpuBackend {
    pub queue: Queue,
    device: Device,
    device_type: DeviceType,
    pub program: Program,
}

impl GpuBackend {
    /// Create a new GPU backend, preferring Intel Arc if available
    pub fn new() -> Result<Self, String> {
        // Try to initialize real GPU
        match Self::try_new_real() {
            Ok(backend) => Ok(backend),
            Err(e) => {
                eprintln!("Failed to initialize real GPU: {}", e);
                Err(e)
            }
        }
    }
    
    /// Try to create a real GPU backend
    fn try_new_real() -> Result<Self, String> {
        // Find Intel GPU first, then fall back to other GPUs
        let (platform, device, device_type) = Self::find_best_device()?;
        
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .map_err(|e| format!("Failed to create context: {}", e))?;
        
        let queue = Queue::new(&context, device, None)
            .map_err(|e| format!("Failed to create queue: {}", e))?;
        
        // Build kernels
        let program_src = include_str!("kernels.cl");
        let program = Program::builder()
            .source(program_src)
            .devices(device)
            .build(&context)
            .map_err(|e| format!("Failed to build program: {}", e))?;
        
        println!("GPU Backend initialized on: {:?}", device_type);
        
        Ok(Self {
            queue,
            device,
            device_type,
            program,
        })
    }
    
    /// Find the best available device
    fn find_best_device() -> Result<(Platform, Device, DeviceType), String> {
        // Handle platform list errors gracefully
        let platforms = match std::panic::catch_unwind(|| Platform::list()) {
            Ok(platforms) => platforms,
            Err(_) => {
                return Err("OpenCL not available. This is common in WSL2. Use native Linux or Windows for GPU support.".to_string());
            }
        };
        
        if platforms.is_empty() {
            return Err("No OpenCL platforms found. Please install OpenCL drivers for your GPU. In WSL2, GPU support may be limited.".to_string());
        }
        
        // Priority: Intel GPU > NVIDIA GPU > AMD GPU > Any GPU
        for platform in &platforms {
            let devices = Device::list_all(platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            
            // Check for Intel GPU
            for device in &devices {
                let device_type = device.info(ocl::enums::DeviceInfo::Type).map_err(|e| e.to_string())?;
                if let ocl::enums::DeviceInfoResult::Type(dtype) = device_type {
                    if dtype == OclDeviceType::GPU {
                        let vendor = device.vendor().map_err(|e| e.to_string())?;
                        if vendor.contains("Intel") {
                            return Ok((*platform, *device, DeviceType::IntelGpu));
                        }
                    }
                }
            }
        }
        
        // Check for NVIDIA GPU
        for platform in &platforms {
            let devices = Device::list_all(platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            
            for device in &devices {
                let device_type = device.info(ocl::enums::DeviceInfo::Type).map_err(|e| e.to_string())?;
                if let ocl::enums::DeviceInfoResult::Type(dtype) = device_type {
                    if dtype == OclDeviceType::GPU {
                        let vendor = device.vendor().map_err(|e| e.to_string())?;
                        if vendor.contains("NVIDIA") {
                            return Ok((*platform, *device, DeviceType::NvidiaGpu));
                        }
                    }
                }
            }
        }
        
        // Check for AMD GPU
        for platform in &platforms {
            let devices = Device::list_all(platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            
            for device in &devices {
                let device_type = device.info(ocl::enums::DeviceInfo::Type).map_err(|e| e.to_string())?;
                if let ocl::enums::DeviceInfoResult::Type(dtype) = device_type {
                    if dtype == OclDeviceType::GPU {
                        let vendor = device.vendor().map_err(|e| e.to_string())?;
                        if vendor.contains("AMD") || vendor.contains("Advanced Micro Devices") {
                            return Ok((*platform, *device, DeviceType::AmdGpu));
                        }
                    }
                }
            }
        }
        
        // Fall back to any GPU
        for platform in &platforms {
            let devices = Device::list_all(platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            
            for device in &devices {
                let device_type = device.info(ocl::enums::DeviceInfo::Type).map_err(|e| e.to_string())?;
                if let ocl::enums::DeviceInfoResult::Type(dtype) = device_type {
                    if dtype == OclDeviceType::GPU {
                        return Ok((*platform, *device, DeviceType::IntelGpu)); // Generic GPU
                    }
                }
            }
        }
        
        // Last resort: Try CPU device
        for platform in &platforms {
            let devices = Device::list_all(platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            
            for device in &devices {
                let device_type = device.info(ocl::enums::DeviceInfo::Type).map_err(|e| e.to_string())?;
                if let ocl::enums::DeviceInfoResult::Type(dtype) = device_type {
                    if dtype == OclDeviceType::CPU {
                        eprintln!("Warning: No GPU found, using CPU OpenCL device");
                        return Ok((*platform, *device, DeviceType::Cpu));
                    }
                }
            }
        }
        
        Err("No OpenCL device found (neither GPU nor CPU)".to_string())
    }
    
    /// Get device info
    pub fn device_info(&self) -> Result<String, String> {
        let name = self.device.name().map_err(|e| e.to_string())?;
        let vendor = self.device.vendor().map_err(|e| e.to_string())?;
        let version = self.device.version().map_err(|e| e.to_string())?;
        let max_compute_units = match self.device.info(ocl::enums::DeviceInfo::MaxComputeUnits).map_err(|e| e.to_string())? {
            ocl::enums::DeviceInfoResult::MaxComputeUnits(units) => units,
            _ => return Err("Failed to get max compute units".to_string()),
        };
        let max_work_group_size = self.device.max_wg_size().map_err(|e| e.to_string())?;
        let global_mem_size: u64 = match self.device.info(ocl::enums::DeviceInfo::GlobalMemSize).map_err(|e| e.to_string())? {
            ocl::enums::DeviceInfoResult::GlobalMemSize(size) => size,
            _ => return Err("Failed to get global memory size".to_string()),
        };
        
        Ok(format!(
            "Device: {}\nVendor: {}\nVersion: {}\nCompute Units: {}\nMax Work Group Size: {}\nGlobal Memory: {} MB",
            name, vendor, version, max_compute_units, max_work_group_size, global_mem_size / (1024 * 1024)
        ))
    }
}

impl ComputeBackend for GpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            return Err(format!("Dimension mismatch: ({}, {}) x ({}, {})", m, k, k2, n));
        }
        
        // Create buffers
        let a_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(m * k)
            .copy_host_slice(a.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let b_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(k * n)
            .copy_host_slice(b.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let c_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(m * n)
            .build()
            .map_err(|e| e.to_string())?;
        
        // Create kernel with arguments
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("matmul")
            .queue(self.queue.clone())
            .arg(&a_buffer)
            .arg(&b_buffer)
            .arg(&c_buffer)
            .arg(m as i32)
            .arg(n as i32)
            .arg(k as i32)
            .build()
            .map_err(|e| format!("Failed to create matmul kernel: {}", e))?;
        
        // Execute kernel
        unsafe {
            kernel
                .cmd()
                .global_work_size([m, n])
                .enq()
                .map_err(|e| e.to_string())?;
        }
        
        // Read result
        let mut result = vec![0.0f32; m * n];
        c_buffer.read(&mut result).enq().map_err(|e| e.to_string())?;
        
        Array2::from_shape_vec((m, n), result)
            .map_err(|e| format!("Failed to create result array: {}", e))
    }
    
    fn add(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let shape = a.dim();
        if shape != b.dim() {
            return Err("Dimension mismatch for addition".to_string());
        }
        
        let size = shape.0 * shape.1;
        
        // Create buffers
        let a_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(a.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let b_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(b.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let c_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(size)
            .build()
            .map_err(|e| e.to_string())?;
        
        // Create kernel with arguments
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("element_add")
            .queue(self.queue.clone())
            .arg(&a_buffer)
            .arg(&b_buffer)
            .arg(&c_buffer)
            .arg(size as i32)
            .build()
            .map_err(|e| format!("Failed to create add kernel: {}", e))?;
        
        // Execute kernel
        unsafe {
            kernel
                .cmd()
                .global_work_size(size)
                .enq()
                .map_err(|e| e.to_string())?;
        }
        
        // Read result
        let mut result = vec![0.0f32; size];
        c_buffer.read(&mut result).enq().map_err(|e| e.to_string())?;
        
        Array2::from_shape_vec(shape, result)
            .map_err(|e| format!("Failed to create result array: {}", e))
    }
    
    fn multiply(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let shape = a.dim();
        if shape != b.dim() {
            return Err("Dimension mismatch for multiplication".to_string());
        }
        
        let size = shape.0 * shape.1;
        
        // Create buffers
        let a_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(a.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let b_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(b.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let c_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(size)
            .build()
            .map_err(|e| e.to_string())?;
        
        // Create kernel with arguments
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("element_multiply")
            .queue(self.queue.clone())
            .arg(&a_buffer)
            .arg(&b_buffer)
            .arg(&c_buffer)
            .arg(size as i32)
            .build()
            .map_err(|e| format!("Failed to create multiply kernel: {}", e))?;
        
        // Execute kernel
        unsafe {
            kernel
                .cmd()
                .global_work_size(size)
                .enq()
                .map_err(|e| e.to_string())?;
        }
        
        // Read result
        let mut result = vec![0.0f32; size];
        c_buffer.read(&mut result).enq().map_err(|e| e.to_string())?;
        
        Array2::from_shape_vec(shape, result)
            .map_err(|e| format!("Failed to create result array: {}", e))
    }
    
    fn relu(&self, input: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        let shape = input.dim();
        let size = shape.0 * shape.1;
        
        // Create buffers
        let input_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(input.as_slice().ok_or("Failed to convert array to slice")?)
            .build()
            .map_err(|e| e.to_string())?;
        
        let output_buffer = Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(size)
            .build()
            .map_err(|e| e.to_string())?;
        
        // Create kernel with arguments
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("relu")
            .queue(self.queue.clone())
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(size as i32)
            .build()
            .map_err(|e| format!("Failed to create relu kernel: {}", e))?;
        
        // Execute kernel
        unsafe {
            kernel
                .cmd()
                .global_work_size(size)
                .enq()
                .map_err(|e| e.to_string())?;
        }
        
        // Read result
        let mut result = vec![0.0f32; size];
        output_buffer.read(&mut result).enq().map_err(|e| e.to_string())?;
        
        Array2::from_shape_vec(shape, result)
            .map_err(|e| format!("Failed to create result array: {}", e))
    }
    
    fn device_type(&self) -> DeviceType {
        self.device_type
    }
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU Backend ({:?})", self.device_type)
    }
}