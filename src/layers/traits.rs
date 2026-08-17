use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Trait defining the interface for neural network layers.
///
/// The trait carries one weight matrix and one bias vector per layer. Layers whose
/// parameters do not fit that shape are not implemented through it: LSTM and GRU expose
/// `forward_sequence` and `backward_sequence` instead, and the convolutional and pooling
/// layers work on 3D and 4D arrays through their own inherent methods.
///
/// # What backward returns
///
/// Both backward methods return gradients, never updated parameters, and they take
/// `&self`: applying them is the caller's job, usually `NeuralNetwork`'s.
///
/// Every implementation expects the matching forward pass to have run first on the same
/// inputs, since backward reads the values it cached. Calling backward without that
/// panics rather than returning something wrong.
///
/// Gradients are summed over the batch, not averaged.
/// `NeuralNetwork::backward_batch` divides by the batch size before handing them to the
/// optimizer, so a caller driving layers directly has to do the same or its effective
/// learning rate scales with batch size.
pub trait Layer: Send + Sync {
    /// Perform forward propagation for a single input
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32>;
    
    /// Perform forward propagation for a batch of inputs
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32>;

    /// Forward propagation that caches nothing, writing the result into `out`.
    ///
    /// Takes `&self`, so a layer behind an `Arc` can serve many callers at once, and
    /// reuses `out` when it already has the right shape, so a per-frame call allocates
    /// nothing. `backward_batch` will not work after this: it reads the caches that
    /// `forward_batch` writes and this method deliberately does not.
    ///
    /// The default clones the layer and runs `forward_batch` on the copy, which is
    /// correct but allocates. Implementations that can write straight into `out` should
    /// override it.
    fn forward_batch_into(&self, inputs: ArrayView2<f32>, out: &mut Array2<f32>) {
        let mut scratch = self.clone_box();
        *out = scratch.forward_batch(inputs);
    }

    /// Single-input form of `forward_batch_into`.
    ///
    /// Kept separate because a one-row matrix product is measurably slower than a
    /// matrix-vector product on the same numbers, and a game asks for one action at a
    /// time. The default routes through the batch version.
    fn forward_into(&self, input: ArrayView1<f32>, out: &mut Array1<f32>) {
        let mut batch = Array2::zeros((0, 0));
        self.forward_batch_into(input.insert_axis(ndarray::Axis(0)), &mut batch);
        let width = batch.shape()[1];
        *out = batch
            .into_shape((width,))
            .expect("forward_batch_into returned more than one row");
    }

    /// Gradients for a single output error.
    ///
    /// Returns `(weight_gradients, bias_gradients)`. Note that this one does not return
    /// the gradient with respect to the input, so it cannot be chained; `backward_batch`
    /// with a one-row batch can.
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>);
    
    /// Gradients for a batch of output errors.
    ///
    /// Returns `(input_gradients, weight_gradients, bias_gradients)` in that order. The
    /// first element is what the previous layer receives; note it is the error already
    /// multiplied by this layer's activation derivative, but *not* yet by its weights,
    /// so a caller chaining layers by hand still has to apply
    /// `adjusted_error.dot(&weights.t())`.
    ///
    /// `input_gradients` has the shape of this layer's input, `weight_gradients` the
    /// shape of its weights, and `bias_gradients` the length of its output.
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