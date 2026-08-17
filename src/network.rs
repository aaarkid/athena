//! # Neural Network Module
//! 
//! This module provides the core neural network implementation for Athena.
//! It supports feedforward networks with arbitrary architectures, various
//! activation functions, and different optimization algorithms.
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use athena::network::NeuralNetwork;
//! use athena::activations::Activation;
//! use athena::optimizer::{OptimizerWrapper, SGD};
//! use ndarray::array;
//! 
//! // Create a simple network: 2 inputs -> 3 hidden -> 1 output
//! let network = NeuralNetwork::new(
//!     &[2, 3, 1],
//!     &[Activation::Relu, Activation::Sigmoid],
//!     OptimizerWrapper::SGD(SGD::new())
//! );
//! ```
//! 
//! ## Features
//! 
//! - **Flexible Architecture**: Support for any number of layers
//! - **Batch Processing**: Efficient forward and backward passes for batches
//! - **Serialization**: Save and load trained models
//! - **Optimizer Integration**: Works with any optimizer implementing the Optimizer trait

use ndarray::{Array1, Array2, ArrayView1, Axis, ArrayView2};
use serde::{Serialize, Deserialize};
use std::fs;
use std::io::{Read, Write};
use bincode::{serialize, deserialize};

use crate::optimizer::{Optimizer, OptimizerWrapper};
use crate::layers::{Layer, LayerTrait};
use crate::activations::Activation;

/// Scratch space for `NeuralNetwork::predict_into`.
///
/// Two buffers the network alternates between as it walks the layers. They grow to the
/// widest layer on the first call and are reused after that, so a game loop holding one
/// of these allocates nothing per frame. One instance serves one call at a time: give
/// each thread its own.
#[derive(Clone, Default, Debug)]
pub struct InferenceBuffers {
    front: Array2<f32>,
    back: Array2<f32>,
    front_row: Array1<f32>,
    back_row: Array1<f32>,
}

impl InferenceBuffers {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A Neural Network consisting of multiple layers, an optimizer, and methods for training
/// and making predictions.
#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub optimizer: OptimizerWrapper,
}

impl NeuralNetwork {
    /// Create a new neural network with the given layer sizes, activations, and optimizer.
    /// This function constructs a new neural network by creating layers with the specified sizes
    /// and activation functions. The optimizer is used for updating the weights and biases during training.
    pub fn new(layer_sizes: &[usize], activations: &[Activation], optimizer: OptimizerWrapper) -> Self {
        assert_eq!(layer_sizes.len() - 1, activations.len());
    
        let layers = layer_sizes
            .windows(2)
            .zip(activations.iter())
            .map(|(window, &activation)| {
                let input_size = window[0];
                let output_size = window[1];
                Layer::new(input_size, output_size, activation)
            })
            .collect::<Vec<_>>();
    
        NeuralNetwork { layers, optimizer }
    }
    
    /// Create an empty neural network
    pub fn new_empty() -> Self {
        NeuralNetwork {
            layers: vec![],
            optimizer: OptimizerWrapper::SGD(crate::optimizer::SGD::new()),
        }
    }

    pub fn with_layers(mut self, layers: Vec<Layer>) -> Self {
        self.layers = layers;
        self
    }

    /// Perform a forward pass for a single input vector.
    /// This function computes the output of the neural network by successively applying each layer's
    /// forward function to the input vector.
    pub fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let output = self.forward_batch(input.view());
        let output_shape = output.shape()[1];
        output.into_shape((output_shape,)).expect("Failed to reshape output")
    }

    /// Width of the input this network accepts.
    pub fn input_size(&self) -> usize {
        self.layers.first().map(|l| l.weights.shape()[0]).unwrap_or(0)
    }

    /// Width of the output this network produces.
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.weights.shape()[1]).unwrap_or(0)
    }

    /// Forward pass that reports a wrong input width instead of panicking inside ndarray.
    ///
    /// `forward` multiplies straight into the first layer's weights, so a state vector of
    /// the wrong length aborts the process. Anything reading input from a game or a file
    /// should come through here.
    pub fn try_forward(&mut self, input: ArrayView1<f32>) -> crate::error::Result<Array1<f32>> {
        self.check_input_width(input.len())?;
        Ok(self.forward(input))
    }

    /// Batch form of `try_forward`.
    pub fn try_forward_batch(&mut self, inputs: ArrayView2<f32>) -> crate::error::Result<Array2<f32>> {
        self.check_input_width(inputs.shape()[1])?;
        Ok(self.forward_batch(inputs))
    }

    fn check_input_width(&self, width: usize) -> crate::error::Result<()> {
        if self.layers.is_empty() {
            return Err(crate::error::AthenaError::TrainingError(
                "Network has no layers".to_string(),
            ));
        }

        let expected = self.input_size();
        if width != expected {
            return Err(crate::error::AthenaError::dimension_mismatch(
                format!("input width {}", expected),
                format!("input width {}", width),
            ));
        }

        Ok(())
    }

    /// Perform a forward pass for a batch of input vectors.
    /// This function computes the output of the neural network for each input vector in the batch
    /// by successively applying each layer's forward_batch function.
    pub fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // Feed the input straight into the first layer rather than copying it first
        let mut layers = self.layers.iter_mut();
        let mut current_output = match layers.next() {
            Some(first) => first.forward_batch(inputs),
            None => return inputs.to_owned(),
        };
        for layer in layers {
            current_output = layer.forward_batch(current_output.view());
        }
        current_output
    }

    /// Forward pass that caches nothing and takes `&self`.
    ///
    /// `forward` stores each layer's input and pre-activation output so `backward_batch`
    /// can read them, which costs allocations on every call and forces `&mut self`. This
    /// one does neither, so an `Arc<NeuralNetwork>` can serve every entity in a game at
    /// once. It produces the same numbers as `forward`; what it does not do is leave the
    /// network ready for a backward pass.
    pub fn predict(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut buffers = InferenceBuffers::new();
        // The borrow of the returned view ends here, leaving the result in `front_row`
        let _ = self.predict_into(input, &mut buffers);
        std::mem::take(&mut buffers.front_row)
    }

    /// Batch form of `predict`.
    pub fn predict_batch(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let mut buffers = InferenceBuffers::new();
        // The borrow of the returned view ends here, leaving the result in `front`
        let _ = self.predict_batch_into(inputs, &mut buffers);
        std::mem::take(&mut buffers.front)
    }

    /// `predict` writing into caller-owned buffers, so a per-frame call allocates nothing
    /// once the buffers have been sized by the first call.
    ///
    /// The returned view borrows `buffers`, so read or copy it before the next call.
    pub fn predict_into<'b>(
        &self,
        input: ArrayView1<f32>,
        buffers: &'b mut InferenceBuffers,
    ) -> ArrayView1<'b, f32> {
        let front = &mut buffers.front_row;
        let back = &mut buffers.back_row;

        let mut layers = self.layers.iter();
        match layers.next() {
            Some(first) => first.forward_into(input, front),
            None => {
                *front = input.to_owned();
                return front.view();
            }
        }

        for layer in layers {
            layer.forward_into(front.view(), back);
            std::mem::swap(front, back);
        }

        front.view()
    }

    /// Batch form of `predict_into`.
    pub fn predict_batch_into<'b>(
        &self,
        inputs: ArrayView2<f32>,
        buffers: &'b mut InferenceBuffers,
    ) -> ArrayView2<'b, f32> {
        let front = &mut buffers.front;
        let back = &mut buffers.back;

        let mut layers = self.layers.iter();
        match layers.next() {
            Some(first) => first.forward_batch_into(inputs, front),
            None => {
                *front = inputs.to_owned();
                return front.view();
            }
        }

        // Ping-pong between the two buffers so no layer reads what it is writing
        for layer in layers {
            layer.forward_batch_into(front.view(), back);
            std::mem::swap(front, back);
        }

        front.view()
    }

    /// `predict` that reports a wrong input width instead of panicking inside ndarray.
    pub fn try_predict(&self, input: ArrayView1<f32>) -> crate::error::Result<Array1<f32>> {
        self.check_input_width(input.len())?;
        Ok(self.predict(input))
    }

    /// Batch form of `try_predict`.
    pub fn try_predict_batch(&self, inputs: ArrayView2<f32>) -> crate::error::Result<Array2<f32>> {
        self.check_input_width(inputs.shape()[1])?;
        Ok(self.predict_batch(inputs))
    }

    /// Compute gradients for the neural network's weights and biases for a batch of input vectors.
    /// This function calculates the gradients of the weights and biases for each input vector in the batch
    /// with respect to the target outputs using backpropagation.
    /// Parameter gradients for a batch, averaged over the batch.
    ///
    /// The per-layer `backward_batch` sums over the batch, so this divides by the batch
    /// size. Without that the effective learning rate scales with batch size: the same
    /// learning rate that is stable at batch 32 takes steps 32 times larger at batch
    /// 1024. Every framework averages here, so a learning rate carried over from one
    /// behaves as expected.
    pub fn backward_batch(&mut self, output_errors: ArrayView2<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        let mut gradients: Vec<(Array2<f32>, Array1<f32>)> = Vec::new();
        let mut current_error = output_errors.to_owned();

        let batch_size = output_errors.shape()[0].max(1) as f32;
        let scale = 1.0 / batch_size;

        let length = self.layers.len();
        for i in (0..length).rev() {
            let layer = &mut self.layers[i];
            let (adjusted_error, weight_gradients, bias_gradients) = layer.backward_batch(current_error.view());
            gradients.push((weight_gradients * scale, bias_gradients * scale));
        
            if i != 0 {
                current_error = adjusted_error.dot(&layer.weights.t());
            }
        }
    
        gradients.reverse();
        gradients
    }

    /// Gradient of the loss with respect to the network's own input.
    ///
    /// Needed when a network sits downstream of another one, as a critic does in
    /// an actor-critic method: the actor's gradient has to travel back through
    /// the critic to reach the action. `forward_batch` must have been called with
    /// the same inputs first, since the backward pass reads the cached
    /// pre-activations.
    pub fn input_gradient_batch(&mut self, output_errors: ArrayView2<f32>) -> Array2<f32> {
        let mut current_error = output_errors.to_owned();

        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            let (adjusted_error, _, _) = layer.backward_batch(current_error.view());
            current_error = adjusted_error.dot(&layer.weights.t());
        }

        current_error
    }

    /// Apply an output-error signal directly, instead of deriving it from targets.
    ///
    /// `train_minibatch` assumes a squared error against targets. When the error
    /// comes from somewhere else, pass it here.
    pub fn train_with_output_errors(
        &mut self,
        inputs: ArrayView2<f32>,
        output_errors: ArrayView2<f32>,
        learning_rate: f32,
    ) {
        let _ = self.forward_batch(inputs);
        self.apply_output_errors(output_errors, learning_rate);
    }

    /// Backpropagate an error signal through the caches the last forward pass wrote,
    /// and let the optimizer apply the result.
    ///
    /// Carries the same contract as `input_gradient_batch`: `forward_batch` must have run
    /// on the inputs this error belongs to, immediately before, since the backward pass
    /// reads the cached pre-activations. Calling it without that panics.
    ///
    /// This is what a caller already holding the outputs should use. `train_minibatch`
    /// and the rest are this method with a forward pass in front.
    pub fn apply_output_errors(&mut self, output_errors: ArrayView2<f32>, learning_rate: f32) {
        let gradients = self.backward_batch(output_errors);
        self.apply_gradients(gradients, learning_rate);
    }

    /// `apply_output_errors` with the global gradient norm capped at `max_norm`.
    ///
    /// Returns the gradient norm before clipping.
    pub fn apply_output_errors_clipped(
        &mut self,
        output_errors: ArrayView2<f32>,
        learning_rate: f32,
        max_norm: f32,
    ) -> f32 {
        let mut gradients = self.backward_batch(output_errors);
        let norm = Self::clip_gradient_norm(&mut gradients, max_norm);
        self.apply_gradients(gradients, learning_rate);
        norm
    }

    /// Scale a set of gradients so their combined L2 norm is at most `max_norm`.
    ///
    /// Applied across all layers at once, which is what "global gradient norm" means; a
    /// per-layer clip would change the direction of the update, not just its length.
    /// Returns the norm before clipping, which is worth logging when training diverges.
    fn clip_gradient_norm(
        gradients: &mut [(Array2<f32>, Array1<f32>)],
        max_norm: f32,
    ) -> f32 {
        let squared: f32 = gradients
            .iter()
            .map(|(w, b)| w.iter().chain(b.iter()).map(|v| v * v).sum::<f32>())
            .sum();
        let norm = squared.sqrt();

        if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
            let scale = max_norm / norm;
            for (w, b) in gradients.iter_mut() {
                *w *= scale;
                *b *= scale;
            }
        }

        norm
    }

    fn apply_gradients(&mut self, gradients: Vec<(Array2<f32>, Array1<f32>)>, learning_rate: f32) {
        for (idx, (layer, (weight_gradients, bias_gradients))) in
            self.layers.iter_mut().zip(gradients).enumerate()
        {
            self.optimizer.update_weights(idx, &mut layer.weights, &weight_gradients, learning_rate);
            self.optimizer.update_biases(idx, &mut layer.biases, &bias_gradients, learning_rate);
        }
    }

    /// `train_minibatch` with the global gradient norm capped at `max_norm`.
    ///
    /// Returns the gradient norm before clipping.
    pub fn train_minibatch_clipped(
        &mut self,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        learning_rate: f32,
        max_norm: f32,
    ) -> f32 {
        let outputs = self.forward_batch(inputs);
        let output_errors = &outputs - &targets;
        self.apply_output_errors_clipped(output_errors.view(), learning_rate, max_norm)
    }

    /// `train_policy_gradient` with the global gradient norm capped at `max_norm`.
    ///
    /// Returns the gradient norm before clipping.
    pub fn train_policy_gradient_clipped(
        &mut self,
        inputs: ArrayView2<f32>,
        output_gradients: ArrayView2<f32>,
        learning_rate: f32,
        max_norm: f32,
    ) -> f32 {
        let _ = self.forward_batch(inputs);
        self.apply_output_errors_clipped(output_gradients, learning_rate, max_norm)
    }

    /// Train the neural network for a batch of input vectors and target outputs.
    /// This function updates the weights and biases of the neural network using the gradients computed
    /// by the backward_batch function and the optimizer.
    pub fn train_minibatch(
        &mut self,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        learning_rate: f32,
    ) {
        let outputs = self.forward_batch(inputs);
        let output_errors = &outputs - &targets;
        self.apply_output_errors(output_errors.view(), learning_rate);
    }

    /// Train using policy gradient method.
    ///
    /// Unlike train_minibatch which computes MSE loss gradients, this method takes
    /// the output gradient directly (e.g., advantage-weighted log-probability gradient)
    /// and backpropagates it through the network.
    ///
    /// # Arguments
    /// * `inputs` - Batch of input states
    /// * `output_gradients` - The gradient with respect to the network outputs
    ///   (e.g., for policy gradient: advantage * ∇log π(a|s))
    /// * `learning_rate` - Learning rate for the optimizer
    pub fn train_policy_gradient(
        &mut self,
        inputs: ArrayView2<f32>,
        output_gradients: ArrayView2<f32>,
        learning_rate: f32,
    ) {
        // Forward pass to cache activations, then backpropagate the given gradient
        // directly instead of computing output - target like train_minibatch
        let _ = self.forward_batch(inputs);
        self.apply_output_errors(output_gradients, learning_rate);
    }

    /// Save the neural network's state to a file.
    /// This function serializes the neural network, including its layers and optimizer, and writes
    /// the serialized data to a file at the specified path.
    pub fn save(&self, path: &str) -> crate::error::Result<()> {
        let serialized = serialize(self)?;
        let mut file = fs::File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Load a neural network from a file.
    /// This function reads the serialized data from a file at the specified path, deserializes it,
    /// and constructs a new neural network with the loaded state.
    pub fn load(path: &str) -> crate::error::Result<Self> {
        let mut file = fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let deserialized: Self = deserialize(&buffer)?;
        Ok(deserialized)
    }
}