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
    #[allow(dead_code)]
    fn backward_batch(&mut self, output_errors: ArrayView2<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        let mut gradients: Vec<(Array2<f32>, Array1<f32>)> = Vec::new();
        let mut current_error = output_errors.to_owned();
    
        let length = self.layers.len();
        for i in (0..length).rev() {
            let layer = &mut self.layers[i];
            let (adjusted_error, weight_gradients, bias_gradients) = layer.backward_batch(current_error.view());
            gradients.push((weight_gradients, bias_gradients));
        
            if i != 0 {
                current_error = adjusted_error.dot(&layer.weights.t());
            }
        }
    
        gradients.reverse();
        gradients
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