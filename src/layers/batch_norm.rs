use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Serialize, Deserialize};
use super::traits::Layer as LayerTrait;

/// Batch Normalization Layer
/// 
/// Normalizes the inputs across the batch dimension to have mean 0 and variance 1,
/// then scales and shifts using learnable parameters gamma and beta.
#[derive(Serialize, Deserialize, Clone)]
pub struct BatchNormLayer {
    /// Scale parameter (gamma)
    pub gamma: Array1<f32>,
    
    /// Shift parameter (beta)
    pub beta: Array1<f32>,
    
    /// Running mean for inference
    pub running_mean: Array1<f32>,
    
    /// Running variance for inference
    pub running_var: Array1<f32>,
    
    /// Weight given to the current batch when updating the running statistics.
    ///
    /// `running = running * (1 - momentum) + batch * momentum`, so a larger value tracks
    /// the current batch more closely. 0.1 is the usual choice. This is the same sense
    /// as PyTorch's `momentum`, and the opposite of a momentum term in an optimizer.
    pub momentum: f32,
    
    /// Small constant for numerical stability
    pub epsilon: f32,
    
    /// Whether we're in training mode
    pub training: bool,
    
    /// Cached values for backward pass
    // Written by the forward pass for the backward pass to read, and skipped on the
    // wire: they are as large as the batch that wrote them.
    #[serde(skip)]
    cached_normalized: Option<Array2<f32>>,
    #[serde(skip)]
    cached_std: Option<Array1<f32>>,
    #[serde(skip)]
    cached_mean: Option<Array1<f32>>,
    #[serde(skip)]
    cached_inputs: Option<Array2<f32>>,

    /// Whether the last forward pass used batch statistics.
    ///
    /// Forward falls back to the running statistics for a batch of one even in training
    /// mode, since a single sample has no variance. Backward has to take the same branch
    /// or it reads caches that pass never wrote.
    #[serde(skip)]
    cached_used_batch_stats: bool,
}

impl BatchNormLayer {
    /// Create a new batch normalization layer
    pub fn new(num_features: usize, momentum: f32, epsilon: f32) -> Self {
        BatchNormLayer {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum,
            epsilon,
            training: true,
            cached_used_batch_stats: false,
            cached_normalized: None,
            cached_std: None,
            cached_mean: None,
            cached_inputs: None,
        }
    }
    
    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    
    /// Forward pass for batch normalization
    fn batch_norm_forward(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let batch_size = inputs.shape()[0];
        let num_features = inputs.shape()[1];
        
        let use_batch_stats = self.training && batch_size > 1;
        self.cached_used_batch_stats = use_batch_stats;

        if use_batch_stats {
            // Training mode: use batch statistics
            let mean = inputs.mean_axis(Axis(0))
                .unwrap_or_else(|| Array1::zeros(num_features));
            let var = inputs.var_axis(Axis(0), 0.0);
            let std = var.mapv(|v| (v + self.epsilon).sqrt());
            
            // Normalize
            let mut normalized = Array2::zeros(inputs.dim());
            for i in 0..batch_size {
                for j in 0..num_features {
                    normalized[[i, j]] = (inputs[[i, j]] - mean[j]) / std[j];
                }
            }
            
            // Update running statistics
            self.running_mean = &self.running_mean * (1.0 - self.momentum) + &mean * self.momentum;
            self.running_var = &self.running_var * (1.0 - self.momentum) + &var * self.momentum;
            
            // Cache for backward
            self.cached_normalized = Some(normalized.clone());
            self.cached_std = Some(std);
            self.cached_mean = Some(mean);
            self.cached_inputs = Some(inputs.to_owned());
            
            // Scale and shift
            let mut output = Array2::zeros(inputs.dim());
            for i in 0..batch_size {
                for j in 0..num_features {
                    output[[i, j]] = self.gamma[j] * normalized[[i, j]] + self.beta[j];
                }
            }
            
            output
        } else {
            // Inference mode: use running statistics
            let std = self.running_var.mapv(|v| (v + self.epsilon).sqrt());
            self.cached_inputs = Some(inputs.to_owned());
            
            let mut output = Array2::zeros(inputs.dim());
            for i in 0..batch_size {
                for j in 0..num_features {
                    let normalized = (inputs[[i, j]] - self.running_mean[j]) / std[j];
                    output[[i, j]] = self.gamma[j] * normalized + self.beta[j];
                }
            }
            
            output
        }
    }
    
    /// Backward pass for batch normalization
    fn batch_norm_backward(&self, grad_output: ArrayView2<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        // Take the same branch the last forward pass took, not whatever `training` says
        // now: a batch of one uses the running statistics even in training mode, and the
        // caches this branch needs were never written in that case.
        if !self.cached_used_batch_stats {
            let batch_size = grad_output.shape()[0];
            let num_features = grad_output.shape()[1];
            let std = self.running_var.mapv(|v| (v + self.epsilon).sqrt());

            let mut grad_input = Array2::zeros(grad_output.dim());
            let mut grad_gamma = Array1::zeros(num_features);

            for i in 0..batch_size {
                for j in 0..num_features {
                    grad_input[[i, j]] = grad_output[[i, j]] * self.gamma[j] / std[j];

                    // gamma multiplies the normalized value, so its gradient carries that
                    // value. Using grad_output alone makes it a copy of grad_beta.
                    let inputs = self.cached_inputs.as_ref();
                    let normalized = match inputs {
                        Some(x) => (x[[i, j]] - self.running_mean[j]) / std[j],
                        None => 0.0,
                    };
                    grad_gamma[j] += grad_output[[i, j]] * normalized;
                }
            }

            let grad_beta = grad_output.sum_axis(Axis(0));

            return (grad_input, grad_gamma, grad_beta);
        }

        // Training mode: full backpropagation.
        //
        // With x_hat the normalized input and g = grad_output * gamma,
        //   dx = (N * g - sum(g) - x_hat * sum(g * x_hat)) / (N * std)
        // The mean and variance paths are both folded into that expression.
        let normalized = self.cached_normalized.as_ref()
            .expect("No cached normalized values. Forward must be called before backward.");
        let std = self.cached_std.as_ref()
            .expect("No cached std values.");

        let rows = grad_output.shape()[0];
        let num_features = grad_output.shape()[1];
        let batch_size = rows as f32;

        // Gradients w.r.t. gamma and beta
        let grad_gamma = (grad_output.to_owned() * normalized).sum_axis(Axis(0));
        let grad_beta = grad_output.sum_axis(Axis(0));

        // Gradient w.r.t. the normalized inputs
        let mut grad_normalized = Array2::zeros(grad_output.dim());
        for i in 0..rows {
            for j in 0..num_features {
                grad_normalized[[i, j]] = grad_output[[i, j]] * self.gamma[j];
            }
        }

        let sum_grad = grad_normalized.sum_axis(Axis(0));
        let sum_grad_times_normalized = (&grad_normalized * normalized).sum_axis(Axis(0));

        let mut grad_input = Array2::zeros(grad_output.dim());
        for i in 0..rows {
            for j in 0..num_features {
                grad_input[[i, j]] = (batch_size * grad_normalized[[i, j]]
                    - sum_grad[j]
                    - normalized[[i, j]] * sum_grad_times_normalized[j])
                    / (batch_size * std[j]);
            }
        }

        (grad_input, grad_gamma, grad_beta)
    }
}

impl LayerTrait for BatchNormLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0));
        let output = self.forward_batch(input.view());
        output.index_axis(Axis(0), 0).to_owned()
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.batch_norm_forward(inputs)
    }
    
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        let output_error = output_error.insert_axis(Axis(0));
        let (_grad_input, _grad_gamma, grad_beta) = self.backward_batch(output_error.view());
        
        // For single input, BatchNorm returns dummy weight gradients
        // since parameters are updated differently
        (Array2::zeros((1, self.gamma.len())), grad_beta)
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let (grad_input, grad_gamma, grad_beta) = self.batch_norm_backward(output_errors);
        
        // Return gamma gradients as "weight" gradients
        (grad_input, grad_gamma.insert_axis(Axis(0)), grad_beta)
    }
    
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        panic!("BatchNorm doesn't have traditional weights. Use gamma/beta accessors.");
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.beta
    }
    
    fn weights(&self) -> &Array2<f32> {
        panic!("BatchNorm doesn't have traditional weights. Use gamma/beta accessors.");
    }
    
    fn biases(&self) -> &Array1<f32> {
        &self.beta
    }
    
    fn output_size(&self) -> usize {
        self.gamma.len()
    }
    
    fn input_size(&self) -> usize {
        self.gamma.len()
    }
    
    fn clone_box(&self) -> Box<dyn LayerTrait> {
        Box::new(self.clone())
    }
}