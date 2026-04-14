use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::Rng;
use serde::{Serialize, Deserialize};
use super::traits::Layer as LayerTrait;

/// Dropout Layer
/// 
/// Randomly sets input units to 0 with probability p during training,
/// which helps prevent overfitting.
#[derive(Serialize, Deserialize, Clone)]
pub struct DropoutLayer {
    /// Dropout probability (probability of dropping a unit)
    pub dropout_rate: f32,
    
    /// Whether we're in training mode
    pub training: bool,
    
    /// Size of the layer
    size: usize,
    
    /// Cached mask for backward pass
    cached_mask: Option<Array2<f32>>,
}

impl DropoutLayer {
    /// Create a new dropout layer
    pub fn new(size: usize, dropout_rate: f32) -> Self {
        assert!((0.0..1.0).contains(&dropout_rate), 
                "Dropout rate must be in [0, 1)");
        
        DropoutLayer {
            dropout_rate,
            training: true,
            size,
            cached_mask: None,
        }
    }
    
    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if !training {
            self.cached_mask = None;
        }
    }
    
    /// Apply dropout to inputs
    fn apply_dropout(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        if !self.training || self.dropout_rate == 0.0 {
            // No dropout during inference or if rate is 0
            return inputs.to_owned();
        }
        
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.dropout_rate);
        
        // Generate dropout mask
        let mut mask = Array2::zeros(inputs.dim());
        for i in 0..mask.shape()[0] {
            for j in 0..mask.shape()[1] {
                if rng.gen::<f32>() > self.dropout_rate {
                    mask[[i, j]] = scale;
                }
            }
        }
        
        // Cache mask for backward pass
        self.cached_mask = Some(mask.clone());
        
        // Apply mask
        inputs.to_owned() * &mask
    }
    
    /// Backward pass for dropout
    fn dropout_backward(&self, grad_output: ArrayView2<f32>) -> Array2<f32> {
        if !self.training || self.dropout_rate == 0.0 {
            return grad_output.to_owned();
        }
        
        if let Some(ref mask) = self.cached_mask {
            grad_output.to_owned() * mask
        } else {
            // This shouldn't happen if forward was called properly
            grad_output.to_owned()
        }
    }
}

impl LayerTrait for DropoutLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0));
        let output = self.forward_batch(input.view());
        output.index_axis(Axis(0), 0).to_owned()
    }
    
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.apply_dropout(inputs)
    }
    
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        let output_error = output_error.insert_axis(Axis(0));
        let _grad_input = self.dropout_backward(output_error.view());
        
        // Dropout has no learnable parameters
        (Array2::zeros((1, 1)), Array1::zeros(1))
    }
    
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let grad_input = self.dropout_backward(output_errors);
        
        // Dropout has no learnable parameters
        (grad_input, Array2::zeros((1, 1)), Array1::zeros(1))
    }
    
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        panic!("Dropout layer has no weights");
    }
    
    fn biases_mut(&mut self) -> &mut Array1<f32> {
        panic!("Dropout layer has no biases");
    }
    
    fn weights(&self) -> &Array2<f32> {
        panic!("Dropout layer has no weights");
    }
    
    fn biases(&self) -> &Array1<f32> {
        panic!("Dropout layer has no biases");
    }
    
    fn output_size(&self) -> usize {
        self.size
    }
    
    fn input_size(&self) -> usize {
        self.size
    }
    
    fn clone_box(&self) -> Box<dyn LayerTrait> {
        Box::new(self.clone())
    }
}