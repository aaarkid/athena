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
        let mut current_output = inputs.to_owned();
        for layer in &mut self.layers {
            current_output = layer.forward_batch(current_output.view());
        }
        current_output
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
        let outputs = self.forward_batch(inputs);
        // train_minibatch computes outputs - targets, so this makes that difference
        // exactly the error that was asked for
        let targets = &outputs - &output_errors.to_owned();
        self.train_minibatch(inputs, targets.view(), learning_rate);
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
        let gradients = self.backward_batch(output_errors.view());
    
        for (idx, (layer, (weight_gradients, bias_gradients))) in self.layers.iter_mut().zip(gradients).enumerate() {
            self.optimizer.update_weights(idx, &mut layer.weights, &weight_gradients, learning_rate);
            self.optimizer.update_biases(idx, &mut layer.biases, &bias_gradients, learning_rate);
        }
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
        // Forward pass to cache activations
        let _ = self.forward_batch(inputs);

        // Backward pass using the provided gradients directly
        // (instead of computing output - target like in train_minibatch)
        let gradients = self.backward_batch(output_gradients);

        // Apply gradients using optimizer
        for (idx, (layer, (weight_gradients, bias_gradients))) in self.layers.iter_mut().zip(gradients).enumerate() {
            self.optimizer.update_weights(idx, &mut layer.weights, &weight_gradients, learning_rate);
            self.optimizer.update_biases(idx, &mut layer.biases, &bias_gradients, learning_rate);
        }
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