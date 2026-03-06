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
    // Cache for backward pass
    pre_activation_output: Option<Array2<f32>>,
    inputs: Option<Array2<f32>>,
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
            pre_activation_output: None,
            inputs: None,
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
                let mut z_with_bias = z;
                for mut row in z_with_bias.rows_mut() {
                    row += &self.biases;
                }
                
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
    
    /// Batch forward pass using GPU (returns pre-activation output)
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
                let mut z_with_bias = z;
                for mut row in z_with_bias.rows_mut() {
                    row += &self.biases;
                }
                Ok(z_with_bias)
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
        let output_error = output_error.insert_axis(Axis(0));
        let (_adjusted_error, weight_gradients, bias_gradients) = self.backward_batch(output_error.view());
        (weight_gradients, bias_gradients)
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // Cache inputs for backward pass
        self.inputs = Some(inputs.to_owned());
        
        let z = self.forward_batch_gpu(inputs).unwrap_or_else(|e| {
            eprintln!("GPU batch forward failed: {}, falling back to CPU", e);
            // Fallback to CPU implementation
            inputs.dot(&self.weights) + &self.biases
        });
        
        // Cache pre-activation output
        self.pre_activation_output = Some(z.clone());
        
        // Apply activation
        let mut output = z;
        self.activation.apply_batch(&mut output);
        output
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let pre_activation_output = self.pre_activation_output.as_ref()
            .expect("No pre-activation output stored. forward_batch() must be called before backward_batch()");
        let inputs = self.inputs.as_ref()
            .expect("No inputs stored. forward_batch() must be called before backward_batch()");
        
        // Compute activation derivative
        let activation_deriv = self.activation.derivative_batch(pre_activation_output.view());
        let adjusted_error = output_errors.to_owned() * &activation_deriv;
        
        // Try to use GPU for gradient computation
        match &self.gpu_backend {
            Some(backend_type) => {
                // Weight gradients: inputs^T × adjusted_error
                let weight_gradients = match backend_type {
                    #[cfg(feature = "gpu")]
                    GpuBackendType::Real(backend) => {
                        backend.lock().unwrap().matmul(inputs.t(), adjusted_error.view())
                            .unwrap_or_else(|_| inputs.t().dot(&adjusted_error))
                    },
                    GpuBackendType::Mock(backend) => {
                        backend.lock().unwrap().matmul(inputs.t(), adjusted_error.view())
                            .unwrap_or_else(|_| inputs.t().dot(&adjusted_error))
                    },
                };
                
                // Bias gradients: sum across batch dimension
                let bias_gradients = adjusted_error.sum_axis(Axis(0));
                
                // Return adjusted_error as the first element (to match DenseLayer behavior)
                (adjusted_error, weight_gradients, bias_gradients)
            },
            None => {
                // CPU fallback
                let weight_gradients = inputs.t().dot(&adjusted_error);
                let bias_gradients = adjusted_error.sum_axis(Axis(0));
                (adjusted_error, weight_gradients, bias_gradients)
            }
        }
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
            cloned.pre_activation_output = self.pre_activation_output.clone();
            cloned.inputs = self.inputs.clone();
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