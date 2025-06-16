//! GPU-accelerated layer implementations

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::layers::traits::Layer;
use crate::activations::Activation;
use crate::gpu::backend::{GpuBackend, ComputeBackend};
use std::sync::Arc;

/// GPU-accelerated dense layer
#[derive(Clone)]
pub struct GpuDenseLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
    gpu_backend: Arc<GpuBackend>,
    // Cache for backward pass
    last_input: Option<Array2<f32>>,
    last_output_pre_activation: Option<Array2<f32>>,
    last_output: Option<Array2<f32>>,
}

impl GpuDenseLayer {
    /// Create a new GPU-accelerated dense layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        gpu_backend: Arc<GpuBackend>,
    ) -> Result<Self, String> {
        // Initialize weights with Xavier/He initialization
        let scale = match activation {
            Activation::Relu => (2.0 / input_size as f32).sqrt(), // He initialization for ReLU
            _ => (1.0 / input_size as f32).sqrt(), // Xavier for others
        };
        
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });
        
        let biases = Array1::zeros(output_size);
        
        Ok(Self {
            weights,
            biases,
            activation,
            gpu_backend,
            last_input: None,
            last_output_pre_activation: None,
            last_output: None,
        })
    }
}

impl Layer for GpuDenseLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        // Convert to batch format and process
        let input_2d = input.insert_axis(ndarray::Axis(0));
        let output_2d = self.forward_batch(input_2d.view());
        output_2d.row(0).to_owned()
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // Cache input for backward pass
        self.last_input = Some(inputs.to_owned());
        
        // Perform matrix multiplication on GPU
        let z = match self.gpu_backend.matmul(inputs, self.weights.view()) {
            Ok(mut output) => {
                // Add bias (on CPU for now, could be fused into GPU kernel)
                for mut row in output.rows_mut() {
                    row += &self.biases;
                }
                output
            }
            Err(_) => {
                // Fallback to CPU computation
                inputs.dot(&self.weights) + &self.biases
            }
        };
        
        // Cache pre-activation values
        self.last_output_pre_activation = Some(z.clone());
        
        // Apply activation
        let output = match self.activation {
            Activation::Relu => {
                // Try GPU ReLU if available
                match self.gpu_backend.relu(z.view()) {
                    Ok(activated) => activated,
                    Err(_) => z.mapv(|x| x.max(0.0)),
                }
            }
            Activation::Sigmoid => {
                z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            }
            Activation::Tanh => {
                z.mapv(|x| x.tanh())
            }
            Activation::Linear => z,
            _ => {
                // For other activations, apply on CPU
                self.apply_activation(&z)
            }
        };
        
        // Cache output for backward pass
        self.last_output = Some(output.clone());
        output
    }
    
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        // Convert to batch format
        let error_2d = output_error.insert_axis(ndarray::Axis(0));
        let (_input_errors, weight_gradients, bias_gradients) = self.backward_batch(error_2d.view());
        
        // Return gradients for single sample
        (weight_gradients, bias_gradients)
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let last_input = self.last_input.as_ref()
            .expect("No cached input found. Did you call forward_batch first?");
        let last_output = self.last_output.as_ref()
            .expect("No cached output found. Did you call forward_batch first?");
        
        // Compute activation derivative
        let activation_derivative = match self.activation {
            Activation::Relu => {
                last_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::Sigmoid => {
                last_output * &(1.0 - last_output)
            }
            Activation::Tanh => {
                1.0 - last_output * last_output
            }
            Activation::Linear => {
                Array2::ones(output_errors.dim())
            }
            _ => {
                self.compute_activation_derivative(last_output)
            }
        };
        
        // Element-wise multiply errors by activation derivative
        let adjusted_errors = &output_errors * &activation_derivative;
        
        // Compute weight gradients: input^T × adjusted_errors
        let weight_gradients = match self.gpu_backend.matmul(
            last_input.t(),
            adjusted_errors.view()
        ) {
            Ok(grads) => grads,
            Err(_) => last_input.t().dot(&adjusted_errors),
        };
        
        // Compute bias gradients: sum across batch dimension
        let bias_gradients = adjusted_errors.sum_axis(ndarray::Axis(0));
        
        // Propagate error: adjusted_errors × weights^T
        let input_errors = match self.gpu_backend.matmul(
            adjusted_errors.view(),
            self.weights.t()
        ) {
            Ok(errors) => errors,
            Err(_) => adjusted_errors.dot(&self.weights.t()),
        };
        
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
        self.weights.ncols()
    }
    
    fn input_size(&self) -> usize {
        self.weights.nrows()
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

impl GpuDenseLayer {
    /// Apply activation function (CPU fallback)
    fn apply_activation(&self, z: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::Relu => z.mapv(|x| x.max(0.0)),
            Activation::Sigmoid => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Tanh => z.mapv(|x| x.tanh()),
            Activation::Linear => z.clone(),
            Activation::LeakyRelu { alpha } => z.mapv(|x| if x > 0.0 { x } else { alpha * x }),
            Activation::Gelu => {
                // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                z.mapv(|x| {
                    let inner = 0.7978845608 * (x + 0.044715 * x.powi(3));
                    0.5 * x * (1.0 + inner.tanh())
                })
            }
            Activation::Elu { alpha } => {
                z.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }
        }
    }
    
    /// Compute activation derivative
    fn compute_activation_derivative(&self, output: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::LeakyRelu { alpha } => {
                output.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
            }
            Activation::Gelu => {
                // Derivative of GELU is more complex
                // Using approximation
                self.last_output_pre_activation.as_ref()
                    .map(|z| {
                        z.mapv(|x| {
                            let inner = 0.7978845608 * (x + 0.044715 * x.powi(3));
                            let tanh_inner = inner.tanh();
                            0.5 * (1.0 + tanh_inner) + 
                            0.5 * x * (1.0 - tanh_inner.powi(2)) * 
                            0.7978845608 * (1.0 + 0.134145 * x.powi(2))
                        })
                    })
                    .unwrap_or_else(|| Array2::ones(output.dim()))
            }
            _ => Array2::ones(output.dim()),
        }
    }
}