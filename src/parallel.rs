//! Parallel computation utilities using ndarray's parallel features
//! 
//! This module provides parallelized versions of common operations
//! to leverage multi-core processors for improved performance.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis};
use ndarray::parallel::prelude::*;
use crate::network::NeuralNetwork;
use crate::layers::{Layer, LayerTrait};
use rand::seq::SliceRandom;

/// Parallel batch forward pass for neural networks
pub struct ParallelNetwork {
    network: NeuralNetwork,
}

impl ParallelNetwork {
    /// Create a parallel network from a regular network
    pub fn from_network(network: &NeuralNetwork, _num_threads: usize) -> Self {
        ParallelNetwork {
            network: network.clone(),
        }
    }
    
    /// Parallel forward pass for a batch of inputs
    pub fn forward_batch_parallel(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let batch_size = inputs.nrows();
        let output_size = self.network.layers.last().unwrap().biases.len();
        
        // For truly parallel processing, we need to clone the network for each thread
        // as forward pass currently requires mutable access
        let outputs: Vec<Array1<f32>> = inputs.axis_iter(Axis(0))
            .into_par_iter()
            .map(|input| {
                // Clone network for thread-local processing
                let mut local_network = self.network.clone();
                local_network.forward(input)
            })
            .collect();
        
        // Combine results
        let mut result = Array2::zeros((batch_size, output_size));
        for (i, output) in outputs.into_iter().enumerate() {
            result.row_mut(i).assign(&output);
        }
        
        result
    }
    
}

/// Parallel matrix multiplication using ndarray's built-in parallel features
pub fn parallel_matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    // ndarray's dot product is already optimized and can use BLAS
    a.dot(&b)
}

/// Parallel convolution operations
pub struct ParallelConv2D;

impl ParallelConv2D {
    /// Parallel 2D convolution using ndarray's parallel iterators
    pub fn convolve2d_parallel(
        input: ArrayView4<f32>,
        kernels: ArrayView4<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Array4<f32> {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (out_channels, kernel_in_channels, kernel_height, kernel_width) = kernels.dim();
        assert_eq!(in_channels, kernel_in_channels);
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        
        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
        
        // Use ndarray parallel features for batch processing
        ndarray::Zip::from(output.axis_iter_mut(Axis(0)))
            .and(input.axis_iter(Axis(0)))
            .par_for_each(|mut out_batch, in_batch| {
                for oc in 0..out_channels {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let h_start = oh * stride.0;
                            let w_start = ow * stride.1;
                            
                            let mut sum = 0.0;
                            for ic in 0..in_channels {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let h = h_start + kh;
                                        let w = w_start + kw;
                                        
                                        if h >= padding.0 && h < in_height + padding.0 &&
                                           w >= padding.1 && w < in_width + padding.1 {
                                            let h_idx = h - padding.0;
                                            let w_idx = w - padding.1;
                                            if h_idx < in_height && w_idx < in_width {
                                                sum += in_batch[[ic, h_idx, w_idx]] * kernels[[oc, ic, kh, kw]];
                                            }
                                        }
                                    }
                                }
                            }
                            out_batch[[oc, oh, ow]] = sum;
                        }
                    }
                }
            });
        
        output
    }
}

/// Parallel gradient computation
pub struct ParallelGradients;

impl ParallelGradients {
    /// Compute gradients for a batch in parallel
    pub fn compute_batch_gradients(
        network: &NeuralNetwork,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let batch_size = inputs.nrows();
        
        // Compute gradients for each sample in parallel
        let gradients: Vec<(Vec<Array2<f32>>, Vec<Array1<f32>>)> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let input = inputs.row(i);
                let target = targets.row(i);
                
                // Forward pass (clone network for thread safety)
                let mut local_network = network.clone();
                let mut activations = vec![input.to_owned()];
                let mut current = input.to_owned();
                
                for layer in &mut local_network.layers {
                    current = layer.forward(current.view());
                    activations.push(current.clone());
                }
                
                // Compute loss gradient
                let output = &activations[activations.len() - 1];
                let error = output - &target;
                
                // Backward pass
                compute_single_gradients(&local_network.layers, &activations, error.view())
            })
            .collect();
        
        // Average gradients
        average_gradients(gradients, network.layers.len())
    }
}

/// Compute gradients for a single sample
fn compute_single_gradients(
    layers: &[Layer],
    activations: &[Array1<f32>],
    error: ArrayView1<f32>,
) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
    let mut weight_grads = Vec::new();
    let mut bias_grads = Vec::new();
    let mut delta = error.to_owned();
    
    // Backward pass through layers
    for (i, layer) in layers.iter().enumerate().rev() {
        let input = &activations[i];
        
        // Apply activation derivative
        let pre_activation = layer.forward_pre_activation(input.view());
        let activation_deriv = layer.activation.derivative(&pre_activation);
        delta = delta * activation_deriv;
        
        // Compute gradients
        let w_grad = delta.clone().insert_axis(Axis(1)) * input.clone().insert_axis(Axis(0));
        let b_grad = delta.clone();
        
        weight_grads.push(w_grad);
        bias_grads.push(b_grad);
        
        // Propagate error to previous layer
        if i > 0 {
            delta = layer.weights.t().dot(&delta);
        }
    }
    
    weight_grads.reverse();
    bias_grads.reverse();
    
    (weight_grads, bias_grads)
}

/// Average gradients from multiple samples
fn average_gradients(
    gradients: Vec<(Vec<Array2<f32>>, Vec<Array1<f32>>)>,
    num_layers: usize,
) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
    let batch_size = gradients.len() as f32;
    
    let mut avg_weight_grads = Vec::with_capacity(num_layers);
    let mut avg_bias_grads = Vec::with_capacity(num_layers);
    
    for layer_idx in 0..num_layers {
        // Sum weight gradients for this layer
        let mut w_sum = gradients[0].0[layer_idx].clone();
        for i in 1..gradients.len() {
            w_sum = w_sum + &gradients[i].0[layer_idx];
        }
        avg_weight_grads.push(w_sum / batch_size);
        
        // Sum bias gradients for this layer
        let mut b_sum = gradients[0].1[layer_idx].clone();
        for i in 1..gradients.len() {
            b_sum = b_sum + &gradients[i].1[layer_idx];
        }
        avg_bias_grads.push(b_sum / batch_size);
    }
    
    (avg_weight_grads, avg_bias_grads)
}

/// Parallel experience replay sampling
pub struct ParallelReplayBuffer<T: Send + Sync> {
    pub buffer: Vec<T>,
    capacity: usize,
    position: usize,
}

impl<T: Send + Sync + Clone> ParallelReplayBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        ParallelReplayBuffer {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }
    
    pub fn add(&mut self, experience: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
            self.position = (self.position + 1) % self.capacity;
        }
    }
    
    /// Sample batch in parallel
    pub fn sample_parallel(&self, batch_size: usize) -> Vec<T> {
        let mut rng = rand::thread_rng();
        
        // Generate random indices
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(batch_size);
        
        // Fetch experiences in parallel
        indices.into_par_iter()
            .map(|idx| self.buffer[idx].clone())
            .collect()
    }
}

/// Parallel data augmentation
pub struct ParallelAugmentation;

impl ParallelAugmentation {
    /// Apply random augmentations to a batch of images in parallel
    pub fn augment_batch(images: ArrayView4<f32>) -> Array4<f32> {
        let mut result = images.to_owned();
        
        // Process each image in parallel
        ndarray::Zip::from(result.axis_iter_mut(Axis(0)))
            .and(images.axis_iter(Axis(0)))
            .par_for_each(|mut output_image, input_image| {
                let augmented = Self::augment_single(input_image);
                output_image.assign(&augmented);
            });
        
        result
    }
    
    fn augment_single(image: ArrayView3<f32>) -> Array3<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut result = image.to_owned();
        
        // Random horizontal flip
        if rng.gen_bool(0.5) {
            let (channels, height, width) = result.dim();
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width / 2 {
                        let temp = result[[c, h, w]];
                        result[[c, h, w]] = result[[c, h, width - 1 - w]];
                        result[[c, h, width - 1 - w]] = temp;
                    }
                }
            }
        }
        
        // Random brightness adjustment
        if rng.gen_bool(0.3) {
            let factor = rng.gen_range(0.8..1.2);
            result.mapv_inplace(|x| (x * factor).clamp(0.0, 1.0));
        }
        
        result
    }
}

// Extension to Layer for parallel operations
impl Layer {
    /// Forward pass without activation (for gradient computation)
    pub fn forward_pre_activation(&self, input: ArrayView1<f32>) -> Array1<f32> {
        self.weights.dot(&input) + &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{OptimizerWrapper, SGD};
    use crate::activations::Activation;
    
    #[test]
    fn test_parallel_matmul() {
        let a = Array2::from_shape_vec((100, 50), (0..5000).map(|i| i as f32).collect()).unwrap();
        let b = Array2::from_shape_vec((50, 30), (0..1500).map(|i| i as f32).collect()).unwrap();
        
        let result = parallel_matmul(a.view(), b.view());
        assert_eq!(result.dim(), (100, 30));
    }
    
    #[test]
    fn test_parallel_network() {
        let network = NeuralNetwork::new(
            &[10, 20, 10],
            &[Activation::Relu, Activation::Sigmoid],
            OptimizerWrapper::SGD(SGD::new())
        );
        
        let mut parallel_net = ParallelNetwork::from_network(&network, 4);
        
        let batch = Array2::ones((8, 10));
        let output = parallel_net.forward_batch_parallel(batch.view());
        
        assert_eq!(output.dim(), (8, 10));
    }
    
    #[test]
    fn test_parallel_replay_buffer() {
        let mut buffer = ParallelReplayBuffer::new(1000);
        
        for i in 0..1000 {
            buffer.add(i);
        }
        
        let batch = buffer.sample_parallel(32);
        assert_eq!(batch.len(), 32);
    }
}