//! Convolutional layers for processing spatial data
//!
//! This module provides 1D and 2D convolutional layers commonly used in
//! computer vision and signal processing tasks.

use ndarray::{Array1, Array3, Array4, ArrayView3, ArrayView4, s, Axis};
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use crate::activations::Activation;
use crate::layers::initialization::WeightInit;
use rand_distr::Normal;

/// 2D Convolutional Layer
/// 
/// Applies 2D convolution over an input signal composed of several input planes.
/// Commonly used in image processing and computer vision tasks.
#[derive(Serialize, Deserialize, Clone)]
pub struct Conv2DLayer {
    /// Convolution kernels/filters [out_channels, in_channels, kernel_height, kernel_width]
    pub kernels: Array4<f32>,
    
    /// Bias terms for each output channel
    pub biases: Array1<f32>,
    
    /// Activation function
    pub activation: Activation,
    
    /// Stride for convolution
    pub stride: (usize, usize),
    
    /// Padding for convolution
    pub padding: (usize, usize),
    
    /// Number of input channels
    pub in_channels: usize,
    
    /// Number of output channels (filters)
    pub out_channels: usize,
    
    /// Kernel size
    pub kernel_size: (usize, usize),
    
    /// Cached input for backward pass
    #[serde(skip)]
    cached_input: Option<Array4<f32>>,
    
    /// Cached pre-activation output
    #[serde(skip)]
    cached_pre_activation: Option<Array4<f32>>,
}

impl Conv2DLayer {
    /// Create a new 2D convolutional layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        activation: Activation,
    ) -> Self {
        let weight_init = WeightInit::for_activation(&activation);
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let fan_out = out_channels * kernel_size.0 * kernel_size.1;
        
        let kernels = weight_init.initialize_conv_weights(
            (out_channels, in_channels, kernel_size.0, kernel_size.1),
            fan_in,
            fan_out,
        );
        
        let biases = Array1::zeros(out_channels);
        
        Conv2DLayer {
            kernels,
            biases,
            activation,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_size,
            cached_input: None,
            cached_pre_activation: None,
        }
    }
    
    /// Perform 2D convolution
    fn convolve2d(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, _, in_height, in_width) = input.dim();
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        
        let mut output = Array4::zeros((batch_size, self.out_channels, out_height, out_width));
        
        // Apply padding if needed
        let padded_input = if self.padding.0 > 0 || self.padding.1 > 0 {
            self.pad_input(input)
        } else {
            input.clone()
        };
        
        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;
                        
                        let mut sum = 0.0;
                        
                        // Convolve with kernel
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    sum += padded_input[[b, ic, h_start + kh, w_start + kw]]
                                         * self.kernels[[oc, ic, kh, kw]];
                                }
                            }
                        }
                        
                        output[[b, oc, oh, ow]] = sum + self.biases[oc];
                    }
                }
            }
        }
        
        output
    }
    
    /// Pad input with zeros
    fn pad_input(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, channels, height, width) = input.dim();
        let padded_height = height + 2 * self.padding.0;
        let padded_width = width + 2 * self.padding.1;
        
        let mut padded = Array4::zeros((batch_size, channels, padded_height, padded_width));
        
        // Copy input to center of padded array
        let h_slice = s![.., .., self.padding.0..self.padding.0 + height, self.padding.1..self.padding.1 + width];
        padded.slice_mut(h_slice).assign(input);
        
        padded
    }
    
    /// Forward pass for batch of images [batch, channels, height, width]
    pub fn forward_batch(&mut self, input: ArrayView4<f32>) -> Array4<f32> {
        // Store input for backward pass
        self.cached_input = Some(input.to_owned());
        
        // Convolution
        let conv_output = self.convolve2d(&input.to_owned());
        
        // Store pre-activation for backward pass
        self.cached_pre_activation = Some(conv_output.clone());
        
        // Apply activation
        let mut output = conv_output;
        for mut batch in output.axis_iter_mut(Axis(0)) {
            for mut channel in batch.axis_iter_mut(Axis(0)) {
                let flat_channel = channel.as_slice_mut().unwrap();
                let mut flat_array = Array1::from_shape_vec(flat_channel.len(), flat_channel.to_vec()).unwrap();
                self.activation.apply(&mut flat_array);
                channel.assign(&flat_array.into_shape(channel.dim()).unwrap());
            }
        }
        
        output
    }
    
    /// Backward pass
    pub fn backward_batch(&mut self, output_gradient: ArrayView4<f32>) -> (Array4<f32>, Array4<f32>, Array1<f32>) {
        let input = self.cached_input.as_ref().expect("Forward pass must be called before backward");
        let pre_activation = self.cached_pre_activation.as_ref().expect("Forward pass must be called before backward");
        
        // Apply activation derivative
        let mut grad = output_gradient.to_owned();
        for (mut grad_batch, pre_batch) in grad.axis_iter_mut(Axis(0)).zip(pre_activation.axis_iter(Axis(0))) {
            for (mut grad_channel, pre_channel) in grad_batch.axis_iter_mut(Axis(0)).zip(pre_batch.axis_iter(Axis(0))) {
                let pre_flat = pre_channel.as_slice().unwrap();
                let pre_array = Array1::from_shape_vec(pre_flat.len(), pre_flat.to_vec()).unwrap();
                let deriv = self.activation.derivative(&pre_array);
                
                grad_channel.zip_mut_with(&deriv.into_shape(grad_channel.dim()).unwrap(), |g, d| *g *= d);
            }
        }
        
        // Compute gradients
        let kernel_gradients = self.compute_kernel_gradients(input, &grad);
        let bias_gradients = self.compute_bias_gradients(&grad);
        let input_gradients = self.compute_input_gradients(input, &grad);
        
        (input_gradients, kernel_gradients, bias_gradients)
    }
    
    /// Compute gradients for kernels
    fn compute_kernel_gradients(&self, input: &Array4<f32>, grad_output: &Array4<f32>) -> Array4<f32> {
        let mut kernel_grads = Array4::zeros(self.kernels.dim());
        let padded_input = if self.padding.0 > 0 || self.padding.1 > 0 {
            self.pad_input(input)
        } else {
            input.clone()
        };
        
        let (batch_size, _, out_height, out_width) = grad_output.dim();
        
        for oc in 0..self.out_channels {
            for ic in 0..self.in_channels {
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let mut sum = 0.0;
                        
                        for b in 0..batch_size {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    let h_idx = oh * self.stride.0 + kh;
                                    let w_idx = ow * self.stride.1 + kw;
                                    
                                    sum += padded_input[[b, ic, h_idx, w_idx]] 
                                         * grad_output[[b, oc, oh, ow]];
                                }
                            }
                        }
                        
                        kernel_grads[[oc, ic, kh, kw]] = sum;
                    }
                }
            }
        }
        
        kernel_grads
    }
    
    /// Compute gradients for biases
    fn compute_bias_gradients(&self, grad_output: &Array4<f32>) -> Array1<f32> {
        let mut bias_grads = Array1::zeros(self.out_channels);
        
        for oc in 0..self.out_channels {
            bias_grads[oc] = grad_output.slice(s![.., oc, .., ..]).sum();
        }
        
        bias_grads
    }
    
    /// Compute gradients for input (transpose convolution)
    fn compute_input_gradients(&self, input: &Array4<f32>, grad_output: &Array4<f32>) -> Array4<f32> {
        let (batch_size, _, in_height, in_width) = input.dim();
        let (_, _, out_height, out_width) = grad_output.dim();

        // The gradient has to match the input the forward pass saw, not the smallest
        // input that could have produced this output. The forward pass floor-divides,
        // so with a stride that does not divide the input evenly the last rows and
        // columns take part in no output position and get a zero gradient.
        let padded_height = in_height + 2 * self.padding.0;
        let padded_width = in_width + 2 * self.padding.1;

        let mut grad_input_padded = Array4::zeros((batch_size, self.in_channels, padded_height, padded_width));
        
        // Transpose convolution
        for b in 0..batch_size {
            for ic in 0..self.in_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        for oc in 0..self.out_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    let h_idx = oh * self.stride.0 + kh;
                                    let w_idx = ow * self.stride.1 + kw;
                                    
                                    grad_input_padded[[b, ic, h_idx, w_idx]] +=
                                        grad_output[[b, oc, oh, ow]] * self.kernels[[oc, ic, kh, kw]];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Remove padding from gradient
        if self.padding.0 > 0 || self.padding.1 > 0 {
            let h_end = self.padding.0 + in_height;
            let w_end = self.padding.1 + in_width;
            grad_input_padded.slice(s![.., .., self.padding.0..h_end, self.padding.1..w_end]).to_owned()
        } else {
            grad_input_padded
        }
    }
}

/// 1D Convolutional Layer
/// 
/// Applies 1D convolution over an input signal composed of several input planes.
/// Commonly used for time series and sequence processing.
#[derive(Serialize, Deserialize, Clone)]
pub struct Conv1DLayer {
    /// Convolution kernels/filters [out_channels, in_channels, kernel_size]
    pub kernels: Array3<f32>,
    
    /// Bias terms for each output channel
    pub biases: Array1<f32>,
    
    /// Activation function
    pub activation: Activation,
    
    /// Stride for convolution
    pub stride: usize,
    
    /// Padding for convolution
    pub padding: usize,
    
    /// Number of input channels
    pub in_channels: usize,
    
    /// Number of output channels (filters)
    pub out_channels: usize,
    
    /// Kernel size
    pub kernel_size: usize,
    
    /// Cached input for backward pass
    #[serde(skip)]
    cached_input: Option<Array3<f32>>,
    
    /// Cached pre-activation output
    #[serde(skip)]
    cached_pre_activation: Option<Array3<f32>>,
}

impl Conv1DLayer {
    /// Create a new 1D convolutional layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
    ) -> Self {
        let weight_init = WeightInit::for_activation(&activation);
        let fan_in = in_channels * kernel_size;
        let fan_out = out_channels * kernel_size;
        
        let kernels = weight_init.initialize_conv1d_weights(
            (out_channels, in_channels, kernel_size),
            fan_in,
            fan_out,
        );
        
        let biases = Array1::zeros(out_channels);
        
        Conv1DLayer {
            kernels,
            biases,
            activation,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_size,
            cached_input: None,
            cached_pre_activation: None,
        }
    }
    
    /// Forward pass for batch of sequences [batch, channels, length]
    pub fn forward_batch(&mut self, input: ArrayView3<f32>) -> Array3<f32> {
        // Store input for backward pass
        self.cached_input = Some(input.to_owned());
        
        let (batch_size, _, in_length) = input.dim();
        
        // Calculate output length
        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        
        let mut output = Array3::zeros((batch_size, self.out_channels, out_length));
        
        // Apply padding if needed
        let padded_input = if self.padding > 0 {
            self.pad_input_1d(&input.to_owned())
        } else {
            input.to_owned()
        };
        
        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let l_start = ol * self.stride;
                    
                    let mut sum = 0.0;
                    
                    // Convolve with kernel
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            sum += padded_input[[b, ic, l_start + k]] * self.kernels[[oc, ic, k]];
                        }
                    }
                    
                    output[[b, oc, ol]] = sum + self.biases[oc];
                }
            }
        }
        
        // Store pre-activation for backward pass
        self.cached_pre_activation = Some(output.clone());
        
        // Apply activation
        for mut batch in output.axis_iter_mut(Axis(0)) {
            for mut channel in batch.axis_iter_mut(Axis(0)) {
                let mut channel_1d = channel.to_owned();
                self.activation.apply(&mut channel_1d);
                channel.assign(&channel_1d);
            }
        }
        
        output
    }
    
    /// Backward pass for a batch of sequences.
    ///
    /// Returns the gradient with respect to the input, the kernels and the biases, in
    /// that order, matching `Conv2DLayer::backward_batch`.
    pub fn backward_batch(&mut self, output_gradient: ArrayView3<f32>) -> (Array3<f32>, Array3<f32>, Array1<f32>) {
        let input = self.cached_input.as_ref()
            .expect("Forward pass must be called before backward");
        let pre_activation = self.cached_pre_activation.as_ref()
            .expect("Forward pass must be called before backward");

        // Through the activation first
        let mut grad = output_gradient.to_owned();
        for (mut grad_batch, pre_batch) in grad.axis_iter_mut(Axis(0)).zip(pre_activation.axis_iter(Axis(0))) {
            for (mut grad_channel, pre_channel) in grad_batch.axis_iter_mut(Axis(0)).zip(pre_batch.axis_iter(Axis(0))) {
                let deriv = self.activation.derivative(&pre_channel.to_owned());
                grad_channel.zip_mut_with(&deriv, |g, d| *g *= d);
            }
        }

        let (batch_size, _, in_length) = input.dim();
        let out_length = grad.shape()[2];

        let padded_input = if self.padding > 0 {
            self.pad_input_1d(input)
        } else {
            input.clone()
        };

        let mut kernel_grads = Array3::zeros(self.kernels.dim());
        let mut bias_grads = Array1::zeros(self.out_channels);
        // Sized from the input the forward pass saw. Deriving it from the output would
        // drop the tail positions whenever the stride does not divide the input evenly.
        let mut grad_input_padded = Array3::zeros((batch_size, self.in_channels, in_length + 2 * self.padding));

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let g = grad[[b, oc, ol]];
                    bias_grads[oc] += g;

                    let l_start = ol * self.stride;
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            kernel_grads[[oc, ic, k]] += padded_input[[b, ic, l_start + k]] * g;
                            grad_input_padded[[b, ic, l_start + k]] += self.kernels[[oc, ic, k]] * g;
                        }
                    }
                }
            }
        }

        let grad_input = if self.padding > 0 {
            grad_input_padded
                .slice(s![.., .., self.padding..self.padding + in_length])
                .to_owned()
        } else {
            grad_input_padded
        };

        (grad_input, kernel_grads, bias_grads)
    }

    /// Pad 1D input with zeros
    fn pad_input_1d(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, channels, length) = input.dim();
        let padded_length = length + 2 * self.padding;
        
        let mut padded = Array3::zeros((batch_size, channels, padded_length));
        
        // Copy input to center of padded array
        padded.slice_mut(s![.., .., self.padding..self.padding + length]).assign(input);
        
        padded
    }
}

// Extension to WeightInit for convolutional layers
impl WeightInit {
    /// Initialize weights for 2D convolutional layer
    pub fn initialize_conv_weights(&self, shape: (usize, usize, usize, usize), fan_in: usize, fan_out: usize) -> Array4<f32> {
        match self {
            WeightInit::XavierUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                Array4::random(shape, rand_distr::Uniform::new(-limit, limit))
            }
            
            WeightInit::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let dist = Normal::new(0.0, std).unwrap_or(Normal::new(0.0, 0.01).expect("valid normal"));
                Array4::random(shape, dist)
            }
            
            WeightInit::HeUniform => {
                let limit = (6.0 / fan_in as f32).sqrt();
                Array4::random(shape, rand_distr::Uniform::new(-limit, limit))
            }
            
            WeightInit::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                let dist = Normal::new(0.0, std).unwrap_or(Normal::new(0.0, 0.01).expect("valid normal"));
                Array4::random(shape, dist)
            }
            
            _ => Array4::random(shape, Normal::new(0.0, 0.01).expect("valid normal"))
        }
    }
    
    /// Initialize weights for 1D convolutional layer
    pub fn initialize_conv1d_weights(&self, shape: (usize, usize, usize), fan_in: usize, fan_out: usize) -> Array3<f32> {
        match self {
            WeightInit::XavierUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                Array3::random(shape, rand_distr::Uniform::new(-limit, limit))
            }
            
            WeightInit::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let dist = Normal::new(0.0, std).unwrap_or(Normal::new(0.0, 0.01).expect("valid normal"));
                Array3::random(shape, dist)
            }
            
            WeightInit::HeUniform => {
                let limit = (6.0 / fan_in as f32).sqrt();
                Array3::random(shape, rand_distr::Uniform::new(-limit, limit))
            }
            
            WeightInit::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                let dist = Normal::new(0.0, std).unwrap_or(Normal::new(0.0, 0.01).expect("valid normal"));
                Array3::random(shape, dist)
            }
            
            _ => Array3::random(shape, Normal::new(0.0, 0.01).expect("valid normal"))
        }
    }
}

/// Builder for Conv2D layers
pub struct Conv2DLayerBuilder {
    in_channels: Option<usize>,
    out_channels: Option<usize>,
    kernel_size: Option<(usize, usize)>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    activation: Option<Activation>,
}

impl Conv2DLayerBuilder {
    pub fn new() -> Self {
        Conv2DLayerBuilder {
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            stride: Some((1, 1)),
            padding: Some((0, 0)),
            activation: Some(Activation::Relu),
        }
    }
    
    pub fn in_channels(mut self, channels: usize) -> Self {
        self.in_channels = Some(channels);
        self
    }
    
    pub fn out_channels(mut self, channels: usize) -> Self {
        self.out_channels = Some(channels);
        self
    }
    
    pub fn kernel_size(mut self, size: (usize, usize)) -> Self {
        self.kernel_size = Some(size);
        self
    }
    
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = Some(stride);
        self
    }
    
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = Some(padding);
        self
    }
    
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }
    
    pub fn build(self) -> Result<Conv2DLayer, crate::error::AthenaError> {
        let in_channels = self.in_channels
            .ok_or_else(|| crate::error::AthenaError::InvalidParameter { 
                name: "in_channels".to_string(), 
                reason: "in_channels is required for Conv2DLayer".to_string() 
            })?;
        let out_channels = self.out_channels
            .ok_or_else(|| crate::error::AthenaError::InvalidParameter { 
                name: "out_channels".to_string(), 
                reason: "out_channels is required for Conv2DLayer".to_string() 
            })?;
        let kernel_size = self.kernel_size
            .ok_or_else(|| crate::error::AthenaError::InvalidParameter { 
                name: "kernel_size".to_string(), 
                reason: "kernel_size is required for Conv2DLayer".to_string() 
            })?;
        
        Ok(Conv2DLayer::new(
            in_channels,
            out_channels,
            kernel_size,
            self.stride.unwrap_or((1, 1)),
            self.padding.unwrap_or((0, 0)),
            self.activation.unwrap_or(Activation::Relu),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conv2d_forward() {
        let mut layer = Conv2DLayer::new(
            1,  // in_channels
            2,  // out_channels
            (3, 3),  // kernel_size
            (1, 1),  // stride
            (1, 1),  // padding
            Activation::Relu,
        );
        
        // Create a simple 4x4 input image
        let input = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ]
        ).unwrap();
        
        let output = layer.forward_batch(input.view());
        
        // With padding=1, output should be same size as input
        assert_eq!(output.dim(), (1, 2, 4, 4));
    }
    
    #[test]
    fn test_conv1d_forward() {
        let mut layer = Conv1DLayer::new(
            1,  // in_channels
            2,  // out_channels
            3,  // kernel_size
            1,  // stride
            1,  // padding
            Activation::Relu,
        );
        
        // Create a simple sequence
        let input = Array3::from_shape_vec(
            (1, 1, 10),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ).unwrap();
        
        let output = layer.forward_batch(input.view());
        
        // With padding=1, output length should be same as input
        assert_eq!(output.dim(), (1, 2, 10));
    }
    
    #[test]
    fn test_conv2d_builder() {
        let layer = Conv2DLayerBuilder::new()
            .in_channels(3)
            .out_channels(64)
            .kernel_size((3, 3))
            .stride((1, 1))
            .padding((1, 1))
            .activation(Activation::Relu)
            .build()
            .unwrap();
        
        assert_eq!(layer.in_channels, 3);
        assert_eq!(layer.out_channels, 64);
        assert_eq!(layer.kernel_size, (3, 3));
    }

    #[test]
    fn conv2d_input_gradient_matches_the_input_shape() {
        // Strides that do not divide the input evenly are where the geometry used to
        // disagree with the forward pass
        for &(stride, padding) in &[(1usize, 0usize), (2, 0), (2, 1), (3, 1)] {
            let mut layer = Conv2DLayer::new(
                1,
                2,
                (3, 3),
                (stride, stride),
                (padding, padding),
                Activation::Relu,
            );

            let input = Array4::from_shape_fn((1, 1, 6, 6), |(_, _, h, w)| {
                ((h * 6 + w) as f32 * 0.17).sin()
            });
            let output = layer.forward_batch(input.view());

            let output_gradient = Array4::from_shape_fn(output.raw_dim(), |(_, c, h, w)| {
                ((c + h * 2 + w * 3) as f32 * 0.31).cos()
            });
            let (input_gradient, kernel_gradient, bias_gradient) =
                layer.backward_batch(output_gradient.view());

            assert_eq!(
                input_gradient.dim(),
                input.dim(),
                "stride {} padding {}: input gradient shape",
                stride,
                padding
            );
            assert_eq!(kernel_gradient.dim(), layer.kernels.dim());
            assert_eq!(bias_gradient.len(), layer.out_channels);
        }
    }

    #[test]
    fn conv2d_backward_matches_finite_differences() {
        let mut layer = Conv2DLayer::new(1, 2, (3, 3), (2, 2), (0, 0), Activation::Linear);

        let input = Array4::from_shape_fn((1, 1, 6, 6), |(_, _, h, w)| {
            ((h * 6 + w) as f32 * 0.23).sin()
        });
        let output = layer.forward_batch(input.view());
        let target = Array4::from_shape_fn(output.raw_dim(), |(_, c, h, w)| {
            ((c * 3 + h + w) as f32 * 0.41).cos() * 0.5
        });

        let output_gradient = &output - &target;
        let (input_gradient, kernel_gradient, _) = layer.backward_batch(output_gradient.view());

        let loss = |layer: &mut Conv2DLayer, input: &Array4<f32>| -> f32 {
            let out = layer.forward_batch(input.view());
            0.5 * (&out - &target).mapv(|v| v * v).sum()
        };

        // Large epsilon on purpose: the loss is an f32, so the difference of two losses
        // is quantized at about loss * 1e-7 and a smaller step is all noise
        let eps = 1e-2;
        let tolerance = 5e-2;

        for &(oc, ic, kh, kw) in &[(0usize, 0usize, 0usize, 0usize), (1, 0, 2, 1)] {
            let original = layer.kernels[[oc, ic, kh, kw]];

            layer.kernels[[oc, ic, kh, kw]] = original + eps;
            let plus = loss(&mut layer, &input);
            layer.kernels[[oc, ic, kh, kw]] = original - eps;
            let minus = loss(&mut layer, &input);
            layer.kernels[[oc, ic, kh, kw]] = original;

            let numerical = (plus - minus) / (2.0 * eps);
            let analytic = kernel_gradient[[oc, ic, kh, kw]];
            let scale = (analytic.abs() + numerical.abs()).max(1e-3);
            assert!(
                (analytic - numerical).abs() / scale < tolerance,
                "kernel [{}, {}, {}, {}]: analytic {} vs numerical {}",
                oc, ic, kh, kw, analytic, numerical
            );
        }

        // The last row and column take part in no output position at stride 2 over 6
        // rows with a 3x3 kernel, so their gradient must be exactly zero
        assert_eq!(input_gradient[[0, 0, 5, 5]], 0.0);

        for &(h, w) in &[(0usize, 0usize), (2, 3)] {
            let mut perturbed = input.clone();
            perturbed[[0, 0, h, w]] = input[[0, 0, h, w]] + eps;
            let plus = loss(&mut layer, &perturbed);
            perturbed[[0, 0, h, w]] = input[[0, 0, h, w]] - eps;
            let minus = loss(&mut layer, &perturbed);

            let numerical = (plus - minus) / (2.0 * eps);
            let analytic = input_gradient[[0, 0, h, w]];
            let scale = (analytic.abs() + numerical.abs()).max(1e-3);
            assert!(
                (analytic - numerical).abs() / scale < tolerance,
                "input [{}, {}]: analytic {} vs numerical {}",
                h, w, analytic, numerical
            );
        }
    }
}
