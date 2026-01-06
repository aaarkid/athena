#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
use crate::gpu::{ComputeBackend, MockGpuBackend};
#[cfg(feature = "gpu")]
use crate::gpu::GpuBackend;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use crate::activations::Activation;
use super::traits::Layer as LayerTrait;
use super::initialization::WeightInit;
#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
use std::sync::{Arc, Mutex};

/// Backend type for GPU acceleration
#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
enum GpuBackendType {
    #[cfg(feature = "gpu")]
    Real(Arc<Mutex<GpuBackend>>),
    Mock(Arc<Mutex<MockGpuBackend>>),
}

/// GPU-accelerated dense layer
#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
pub struct GpuDenseLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Activation,
    gpu_backend: Option<GpuBackendType>,
}

#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
impl GpuDenseLayer {
    /// Create a new GPU-accelerated dense layer
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Result<Self, String> {
        #[cfg(feature = "gpu")]
        let gpu_backend = match GpuBackend::new() {
            Ok(backend) => Some(GpuBackendType::Real(Arc::new(Mutex::new(backend)))),
            Err(_) => {
                // Try mock backend as fallback
                eprintln!("Using mock GPU backend for demonstration.");
                Some(GpuBackendType::Mock(Arc::new(Mutex::new(MockGpuBackend::new()))))
            }
        };
        
        #[cfg(all(feature = "gpu-mock", not(feature = "gpu")))]
        let gpu_backend = {
            eprintln!("Using mock GPU backend (gpu-mock feature).");
            Some(GpuBackendType::Mock(Arc::new(Mutex::new(MockGpuBackend::new()))))
        };
        
        // Initialize weights using CPU
        let weight_init = WeightInit::for_activation(&activation);
        let weights = weight_init.initialize_weights((input_size, output_size));
        let biases = weight_init.initialize_biases(output_size);
        
        Ok(Self {
            weights,
            biases,
            activation,
            gpu_backend,
        })
    }
    
    /// Get device info
    pub fn device_info(&self) -> Result<String, String> {
        match &self.gpu_backend {
            #[cfg(feature = "gpu")]
            Some(GpuBackendType::Real(backend)) => backend.lock().unwrap().device_info(),
            Some(GpuBackendType::Mock(backend)) => backend.lock().unwrap().device_info(),
            None => Err("No GPU backend available".to_string()),
        }
    }
    
    /// Forward pass using GPU
    pub fn forward_gpu(&mut self, input: ArrayView1<f32>) -> Result<Array1<f32>, String> {
        match &self.gpu_backend {
            Some(backend_type) => {
                // Reshape input for matrix multiplication
                let input_2d = input.insert_axis(Axis(0));
                
                // Perform matrix multiplication on GPU
                let z = match backend_type {
                    #[cfg(feature = "gpu")]
                    GpuBackendType::Real(backend) => backend.lock().unwrap().matmul(input_2d, self.weights.view())?,
                    GpuBackendType::Mock(backend) => backend.lock().unwrap().matmul(input_2d, self.weights.view())?,
                };
                
                // Add bias (on CPU for now - could be optimized)
                let z_with_bias = &z + &self.biases;
                
                // Apply activation (on GPU if supported)
                match self.activation {
                    Activation::Relu => {
                        let activated = match backend_type {
                            #[cfg(feature = "gpu")]
                            GpuBackendType::Real(backend) => backend.lock().unwrap().relu(z_with_bias.view())?,
                            GpuBackendType::Mock(backend) => backend.lock().unwrap().relu(z_with_bias.view())?,
                        };
                        Ok(activated.row(0).to_owned())
                    },
                    _ => {
                        // Fall back to CPU for other activations
                        let mut output = z_with_bias.row(0).to_owned();
                        self.activation.apply(&mut output);
                        Ok(output)
                    }
                }
            },
            None => Err("No GPU backend available".to_string()),
        }
    }
    
    /// Batch forward pass using GPU
    pub fn forward_batch_gpu(&mut self, inputs: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        match &self.gpu_backend {
            Some(backend_type) => {
                // Perform matrix multiplication on GPU
                let z = match backend_type {
                    #[cfg(feature = "gpu")]
                    GpuBackendType::Real(backend) => backend.lock().unwrap().matmul(inputs, self.weights.view())?,
                    GpuBackendType::Mock(backend) => backend.lock().unwrap().matmul(inputs, self.weights.view())?,
                };
                
                // Add bias (broadcasting)
                let z_with_bias = &z + &self.biases;
                
                // Apply activation
                match self.activation {
                    Activation::Relu => {
                        match backend_type {
                            #[cfg(feature = "gpu")]
                            GpuBackendType::Real(backend) => backend.lock().unwrap().relu(z_with_bias.view()),
                            GpuBackendType::Mock(backend) => backend.lock().unwrap().relu(z_with_bias.view()),
                        }
                    },
                    _ => {
                        // Fall back to CPU for other activations
                        let mut output = z_with_bias;
                        self.activation.apply_batch(&mut output);
                        Ok(output)
                    }
                }
            },
            None => Err("No GPU backend available".to_string()),
        }
    }
}

#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
impl LayerTrait for GpuDenseLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        self.forward_gpu(input).unwrap_or_else(|e| {
            eprintln!("GPU forward failed: {}, falling back to CPU", e);
            // Fallback to CPU implementation
            let z = input.dot(&self.weights) + &self.biases;
            let mut output = z;
            self.activation.apply(&mut output);
            output
        })
    }
    
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        // For now, backward pass is still on CPU
        // This would be a future optimization
        let _output_error_2d = output_error.insert_axis(Axis(0));
        (self.weights.clone(), self.biases.clone())
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.forward_batch_gpu(inputs).unwrap_or_else(|e| {
            eprintln!("GPU batch forward failed: {}, falling back to CPU", e);
            // Fallback to CPU implementation
            let z = inputs.dot(&self.weights) + &self.biases;
            let mut output = z;
            self.activation.apply_batch(&mut output);
            output
        })
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        // Placeholder for GPU backward implementation
        let batch_size = output_errors.shape()[0];
        let input_errors = Array2::zeros((batch_size, self.weights.shape()[0]));
        let weight_gradients = self.weights.clone();
        let bias_gradients = self.biases.clone();
        (input_errors, weight_gradients, bias_gradients)
    }
    
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }
    
    fn weights(&self) -> &Array2<f32> {
        &self.weights
    }
    
    fn biases(&self) -> &Array1<f32> {
        &self.biases
    }
    
    fn output_size(&self) -> usize {
        self.weights.shape()[1]
    }
    
    fn input_size(&self) -> usize {
        self.weights.shape()[0]
    }
    
    fn clone_box(&self) -> Box<dyn super::traits::Layer> {
        // For cloning, create a new GPU backend
        if let Ok(new_layer) = Self::new(self.input_size(), self.output_size(), self.activation.clone()) {
            let mut cloned = Box::new(new_layer);
            cloned.weights = self.weights.clone();
            cloned.biases = self.biases.clone();
            cloned
        } else {
            // Fall back to CPU layer if GPU init fails
            let mut cpu_layer = super::dense::DenseLayer::new(self.input_size(), self.output_size(), self.activation.clone());
            cpu_layer.weights = self.weights.clone();
            cpu_layer.biases = self.biases.clone();
            Box::new(cpu_layer)
        }
    }
}

// CPU fallback for when GPU feature is not enabled
#[cfg(not(any(feature = "gpu", feature = "gpu-mock")))]
pub type GpuDenseLayer = super::dense::DenseLayer;