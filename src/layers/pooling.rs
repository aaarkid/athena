//! Pooling layers for downsampling spatial data
//!
//! This module provides max pooling and average pooling layers for 1D and 2D data.

use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, s};
use serde::{Serialize, Deserialize};

/// 2D Max Pooling Layer
/// 
/// Applies max pooling over spatial dimensions to reduce size and extract dominant features.
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxPool2DLayer {
    /// Pooling window size
    pub pool_size: (usize, usize),
    
    /// Stride for pooling
    pub stride: (usize, usize),
    
    /// Cached indices for backward pass
    #[serde(skip)]
    cached_indices: Option<Array4<(usize, usize)>>,
    
    /// Cached input shape for backward pass
    #[serde(skip)]
    cached_input_shape: Option<(usize, usize, usize, usize)>,
}

impl MaxPool2DLayer {
    /// Create a new 2D max pooling layer
    pub fn new(pool_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        let stride = stride.unwrap_or(pool_size);
        
        MaxPool2DLayer {
            pool_size,
            stride,
            cached_indices: None,
            cached_input_shape: None,
        }
    }
    
    /// Forward pass for batch of images [batch, channels, height, width]
    pub fn forward_batch(&mut self, input: ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, channels, in_height, in_width) = input.dim();
        self.cached_input_shape = Some((batch_size, channels, in_height, in_width));
        
        // Calculate output dimensions
        let out_height = (in_height - self.pool_size.0) / self.stride.0 + 1;
        let out_width = (in_width - self.pool_size.1) / self.stride.1 + 1;
        
        let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
        let mut indices = Array4::from_elem((batch_size, channels, out_height, out_width), (0, 0));
        
        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;
                        let h_end = h_start + self.pool_size.0;
                        let w_end = w_start + self.pool_size.1;
                        
                        // Find max in pooling window
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_h = h_start;
                        let mut max_w = w_start;
                        
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let val = input[[b, c, h, w]];
                                if val > max_val {
                                    max_val = val;
                                    max_h = h;
                                    max_w = w;
                                }
                            }
                        }
                        
                        output[[b, c, oh, ow]] = max_val;
                        indices[[b, c, oh, ow]] = (max_h, max_w);
                    }
                }
            }
        }
        
        self.cached_indices = Some(indices);
        output
    }
    
    /// Backward pass
    pub fn backward_batch(&self, output_gradient: ArrayView4<f32>) -> Array4<f32> {
        let indices = self.cached_indices.as_ref().expect("Forward pass must be called before backward");
        let input_shape = self.cached_input_shape.expect("Forward pass must be called before backward");
        
        let mut input_gradient = Array4::zeros(input_shape);
        let (batch_size, channels, out_height, out_width) = output_gradient.dim();
        
        // Propagate gradients to max locations
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let (h, w) = indices[[b, c, oh, ow]];
                        input_gradient[[b, c, h, w]] += output_gradient[[b, c, oh, ow]];
                    }
                }
            }
        }
        
        input_gradient
    }
}

/// 2D Average Pooling Layer
/// 
/// Applies average pooling over spatial dimensions to reduce size and smooth features.
#[derive(Serialize, Deserialize, Clone)]
pub struct AvgPool2DLayer {
    /// Pooling window size
    pub pool_size: (usize, usize),
    
    /// Stride for pooling
    pub stride: (usize, usize),
    
    /// Cached input shape for backward pass
    #[serde(skip)]
    cached_input_shape: Option<(usize, usize, usize, usize)>,
}

impl AvgPool2DLayer {
    /// Create a new 2D average pooling layer
    pub fn new(pool_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        let stride = stride.unwrap_or(pool_size);
        
        AvgPool2DLayer {
            pool_size,
            stride,
            cached_input_shape: None,
        }
    }
    
    /// Forward pass for batch of images [batch, channels, height, width]
    pub fn forward_batch(&mut self, input: ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, channels, in_height, in_width) = input.dim();
        self.cached_input_shape = Some((batch_size, channels, in_height, in_width));
        
        // Calculate output dimensions
        let out_height = (in_height - self.pool_size.0) / self.stride.0 + 1;
        let out_width = (in_width - self.pool_size.1) / self.stride.1 + 1;
        
        let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
        let pool_area = (self.pool_size.0 * self.pool_size.1) as f32;
        
        // Perform average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;
                        let h_end = h_start + self.pool_size.0;
                        let w_end = w_start + self.pool_size.1;
                        
                        // Calculate average in pooling window
                        let mut sum = 0.0;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                sum += input[[b, c, h, w]];
                            }
                        }
                        
                        output[[b, c, oh, ow]] = sum / pool_area;
                    }
                }
            }
        }
        
        output
    }
    
    /// Backward pass
    pub fn backward_batch(&self, output_gradient: ArrayView4<f32>) -> Array4<f32> {
        let input_shape = self.cached_input_shape.expect("Forward pass must be called before backward");
        let mut input_gradient = Array4::zeros(input_shape);
        
        let (batch_size, channels, out_height, out_width) = output_gradient.dim();
        let pool_area = (self.pool_size.0 * self.pool_size.1) as f32;
        
        // Distribute gradients evenly across pooling windows
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;
                        let h_end = h_start + self.pool_size.0;
                        let w_end = w_start + self.pool_size.1;
                        
                        let grad_val = output_gradient[[b, c, oh, ow]] / pool_area;
                        
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                input_gradient[[b, c, h, w]] += grad_val;
                            }
                        }
                    }
                }
            }
        }
        
        input_gradient
    }
}

/// Global Average Pooling Layer
/// 
/// Reduces each channel to a single value by averaging across all spatial dimensions.
/// Commonly used before the final classification layer in CNNs.
#[derive(Serialize, Deserialize, Clone)]
pub struct GlobalAvgPoolLayer;

impl GlobalAvgPoolLayer {
    /// Create a new global average pooling layer
    pub fn new() -> Self {
        GlobalAvgPoolLayer
    }
    
    /// Forward pass for batch of images [batch, channels, height, width]
    /// Returns [batch, channels]
    pub fn forward_batch(&self, input: ArrayView4<f32>) -> Array2<f32> {
        let (batch_size, channels, height, width) = input.dim();
        let mut output = Array2::zeros((batch_size, channels));
        
        let spatial_size = (height * width) as f32;
        
        for b in 0..batch_size {
            for c in 0..channels {
                let channel_sum = input.slice(s![b, c, .., ..]).sum();
                output[[b, c]] = channel_sum / spatial_size;
            }
        }
        
        output
    }
    
    /// Backward pass
    pub fn backward_batch(&self, output_gradient: ArrayView2<f32>, input_shape: (usize, usize, usize, usize)) -> Array4<f32> {
        let (batch_size, channels, height, width) = input_shape;
        let mut input_gradient = Array4::zeros(input_shape);
        
        let spatial_size = (height * width) as f32;
        
        for b in 0..batch_size {
            for c in 0..channels {
                let grad_val = output_gradient[[b, c]] / spatial_size;
                input_gradient.slice_mut(s![b, c, .., ..]).fill(grad_val);
            }
        }
        
        input_gradient
    }
}

/// 1D Max Pooling Layer
#[derive(Serialize, Deserialize, Clone)]
pub struct MaxPool1DLayer {
    /// Pooling window size
    pub pool_size: usize,
    
    /// Stride for pooling
    pub stride: usize,
    
    /// Cached indices for backward pass
    #[serde(skip)]
    cached_indices: Option<Array3<usize>>,
    
    /// Cached input shape for backward pass
    #[serde(skip)]
    cached_input_shape: Option<(usize, usize, usize)>,
}

impl MaxPool1DLayer {
    /// Create a new 1D max pooling layer
    pub fn new(pool_size: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(pool_size);
        
        MaxPool1DLayer {
            pool_size,
            stride,
            cached_indices: None,
            cached_input_shape: None,
        }
    }
    
    /// Forward pass for batch of sequences [batch, channels, length]
    pub fn forward_batch(&mut self, input: ArrayView3<f32>) -> Array3<f32> {
        let (batch_size, channels, in_length) = input.dim();
        self.cached_input_shape = Some((batch_size, channels, in_length));
        
        // Calculate output length
        let out_length = (in_length - self.pool_size) / self.stride + 1;
        
        let mut output = Array3::zeros((batch_size, channels, out_length));
        let mut indices = Array3::zeros((batch_size, channels, out_length));
        
        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for ol in 0..out_length {
                    let l_start = ol * self.stride;
                    let l_end = l_start + self.pool_size;
                    
                    // Find max in pooling window
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = l_start;
                    
                    for l in l_start..l_end {
                        let val = input[[b, c, l]];
                        if val > max_val {
                            max_val = val;
                            max_idx = l;
                        }
                    }
                    
                    output[[b, c, ol]] = max_val;
                    indices[[b, c, ol]] = max_idx;
                }
            }
        }
        
        self.cached_indices = Some(indices);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_maxpool2d() {
        let mut layer = MaxPool2DLayer::new((2, 2), None);
        
        // Create a simple 4x4 input
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
        
        // Output should be 2x2
        assert_eq!(output.dim(), (1, 1, 2, 2));
        
        // Check max values
        assert_eq!(output[[0, 0, 0, 0]], 6.0);
        assert_eq!(output[[0, 0, 0, 1]], 8.0);
        assert_eq!(output[[0, 0, 1, 0]], 14.0);
        assert_eq!(output[[0, 0, 1, 1]], 16.0);
    }
    
    #[test]
    fn test_avgpool2d() {
        let mut layer = AvgPool2DLayer::new((2, 2), None);
        
        // Create a simple 4x4 input
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
        
        // Output should be 2x2
        assert_eq!(output.dim(), (1, 1, 2, 2));
        
        // Check average values
        assert_eq!(output[[0, 0, 0, 0]], 3.5);  // (1+2+5+6)/4
        assert_eq!(output[[0, 0, 0, 1]], 5.5);  // (3+4+7+8)/4
    }
    
    #[test]
    fn test_global_avgpool() {
        let layer = GlobalAvgPoolLayer::new();
        
        // Create a 2x2 input with 2 channels
        let input = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0, 2.0, 3.0, 4.0,  // Channel 0
                5.0, 6.0, 7.0, 8.0,  // Channel 1
            ]
        ).unwrap();
        
        let output = layer.forward_batch(input.view());
        
        // Output should be [1, 2]
        assert_eq!(output.dim(), (1, 2));
        assert_eq!(output[[0, 0]], 2.5);  // (1+2+3+4)/4
        assert_eq!(output[[0, 1]], 6.5);  // (5+6+7+8)/4
    }
}