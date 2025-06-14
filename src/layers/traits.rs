use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Trait defining the interface for neural network layers
pub trait Layer: Send + Sync {
    /// Perform forward propagation for a single input
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32>;
    
    /// Perform forward propagation for a batch of inputs
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32>;
    
    /// Perform backward propagation for a single output error
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>);
    
    /// Perform backward propagation for a batch of output errors
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>);
    
    /// Get mutable reference to weights
    fn weights_mut(&mut self) -> &mut Array2<f32>;
    
    /// Get mutable reference to biases
    fn biases_mut(&mut self) -> &mut Array1<f32>;
    
    /// Get reference to weights
    fn weights(&self) -> &Array2<f32>;
    
    /// Get reference to biases  
    fn biases(&self) -> &Array1<f32>;
    
    /// Get the output size of the layer
    fn output_size(&self) -> usize;
    
    /// Get the input size of the layer
    fn input_size(&self) -> usize;
    
    /// Clone the layer into a boxed trait object
    fn clone_box(&self) -> Box<dyn Layer>;
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}