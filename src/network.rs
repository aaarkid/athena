use ndarray::{Array1, Array2, ArrayView1, Axis, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use std::fs;
use std::io::{Read, Write};
use bincode::{serialize, deserialize};

use crate::optimizer::{Optimizer, OptimizerWrapper};

/// A Layer in a neural network, consisting of weights, biases, and an activation function.
/// This struct represents a fully connected layer within a neural network.
/// It contains the weights and biases for the layer as well as the activation function to be applied.
#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Activation,
    pre_activation_output: Option<Array2<f32>>,
    inputs: Option<Array2<f32>>,
}

impl Layer {
    /// Create a new layer with the given input size, output size, and activation function.
    /// The weights are initialized with random values from a uniform distribution
    /// between -0.1 and 0.1. The biases are initialized with zeros.
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let weights = Array2::random((input_size, output_size), Uniform::new(-0.1, 0.1));
        let biases = Array1::zeros(output_size);
        Layer {
            weights,
            biases,
            activation,
            pre_activation_output: None,
            inputs: None,
        }
    }

    pub fn with_weights(mut self, weights: Array2<f32>) -> Self {
        assert_eq!(weights.dim(), (self.weights.dim().0, self.weights.dim().1));
        self.weights = weights;
        self
    }

    pub fn with_biases(mut self, biases: Array1<f32>) -> Self {
        assert_eq!(biases.dim(), self.biases.dim());
        self.biases = biases;
        self
    }

    /// Perform a forward pass for a single input vector.
    /// This function computes the output of the layer by applying the weights, biases, and activation function
    /// to the given input vector.
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let output = self.forward_minibatch(input.view());
        let shape = output.shape()[1];
        output.into_shape((shape,)).unwrap() // Remove the batch dimension
    }

    /// Perform a forward pass for a batch of input vectors.
    /// This function computes the output of the layer for each input vector in the batch
    /// by applying the weights, biases, and activation function.
    fn forward_minibatch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.inputs = Some(inputs.to_owned()); // Store the inputs
        let mut outputs = inputs.dot(&self.weights) + &self.biases.to_owned().insert_axis(Axis(0));
        self.pre_activation_output = Some(outputs.clone());
        self.activation.apply_minibatch(&mut outputs);
        outputs
    }

    /// Compute gradients for the layer's weights and biases for a single input vector.
    /// This function calculates the gradients of the weights and biases with respect to the output error
    /// using the chain rule and the derivative of the activation function.
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        let output_error = output_error.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let (_adjusted_error, weight_gradients, bias_gradients) = self.backward_minibatch(output_error.view());
        let shape = bias_gradients.shape()[1];
        (weight_gradients, bias_gradients.into_shape((shape,)).unwrap()) // Remove the batch dimension
    }

    /// Compute gradients for the layer's weights and biases for a batch of input vectors.
    /// This function calculates the gradients of the weights and biases for each input vector in the batch
    /// with respect to the output errors using the chain rule and the derivative of the activation function.
    fn backward_minibatch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let pre_activation_output = self.pre_activation_output.as_ref().expect("No pre-activation output stored. forward_minibatch() must be called before backward_minibatch()");
        let inputs = self.inputs.as_ref().expect("No inputs stored. forward_minibatch() must be called before backward_minibatch()");
        let activation_deriv = self.activation.derivative_minibatch(pre_activation_output.view());
        let adjusted_error = output_errors.to_owned() * &activation_deriv;
        let weight_gradients = inputs.t().dot(&adjusted_error);
        let bias_gradients = adjusted_error.sum_axis(Axis(0));
        (adjusted_error, weight_gradients, bias_gradients)
    }
}

/// An enumeration of the possible activation functions that can be used in a neural network layer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Linear,
}

impl Activation {
    /// Apply the activation function to an input array in-place.
    /// This function computes the output for each element in the input array by applying the
    /// activation function. The result is stored in-place in the input array.
    fn apply(&self, input: &mut Array1<f32>) {
        match self {
            Activation::Relu => {
                input.mapv_inplace(|v| v.max(0.0));
            }
            Activation::Linear => {}
        }
    }
    
    /// Apply the activation function to a batch of input arrays in-place.
    /// This function computes the output for each element in the input batch by applying the
    /// activation function. The result is stored in-place in the input batch.
    fn apply_minibatch(&self, inputs: &mut Array2<f32>) {
        match self {
            Activation::Relu => {
                inputs.mapv_inplace(|v| v.max(0.0));
            }
            Activation::Linear => {}
        }
    }

    /// Compute the derivative of the activation function for an input array.
    /// This function calculates the derivative of the activation function for each element
    /// in the input array and returns a new array with the results.
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::Relu => {
                input.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::Linear => {
                // Derivative of linear activation is always 1
                Array1::ones(input.len())
            }
        }
    }

    /// Compute the derivative of the activation function for a batch of input arrays.
    /// This function calculates the derivative of the activation function for each element
    /// in the input batch and returns a new array with the results.
    fn derivative_minibatch(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        match self {
            Activation::Relu => {
                inputs.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::Linear => {
                // Derivative of linear activation is always 1
                Array2::ones(inputs.dim())
            }
        }
    }
}

/// A Neural Network consisting of multiple layers, an optimizer, and methods for training
/// and making predictions.
#[derive(Serialize, Deserialize)]
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

    pub fn with_layers(mut self, layers: Vec<Layer>) -> Self {
        self.layers = layers;
        self
    }

    /// Perform a forward pass for a single input vector.
    /// This function computes the output of the neural network by successively applying each layer's
    /// forward function to the input vector.
    pub fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let output = self.forward_minibatch(input.view());
        let output_shape = output.shape()[1];
        output.into_shape((output_shape,)).unwrap() // Remove the batch dimension
    }

    /// Perform a forward pass for a batch of input vectors.
    /// This function computes the output of the neural network for each input vector in the batch
    /// by successively applying each layer's forward_minibatch function.
    fn forward_minibatch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        // println!("NeuralNetwork::forward_minibatch inputs shape: {:?}, expected: {:?}", inputs.shape(), (inputs.shape()[0], self.layers[0].weights.shape()[0]));
        let mut current_output = inputs.to_owned();
        for layer in &mut self.layers {
            current_output = layer.forward_minibatch(current_output.view());
        }
        current_output
    }

    /// Compute gradients for the neural network's weights and biases for a single input vector.
    /// This function calculates the gradients of the weights and biases for the given input vector
    /// with respect to the target output using backpropagation.
    fn backward(&mut self, output_error: ArrayView1<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        let output_error = output_error.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let gradients = self.backward_minibatch(output_error.view());
        // Remove the batch dimension from gradients
        gradients
            .into_iter()
            .map(|(wg, bg)| {
                let shape = bg.shape()[0];
                (wg, bg.into_shape((shape,)).unwrap())
            })
            .collect()
    }

    /// Compute gradients for the neural network's weights and biases for a batch of input vectors.
    /// This function calculates the gradients of the weights and biases for each input vector in the batch
    /// with respect to the target outputs using backpropagation.
    fn backward_minibatch(&mut self, output_errors: ArrayView2<f32>) -> Vec<(Array2<f32>, Array1<f32>)> {
        // println!("NeuralNetwork::backward_minibatch output_errors shape: {:?}, expected: {:?}", output_errors.shape(), (output_errors.shape()[0], self.layers.last().unwrap().weights.shape()[1]));
        let mut gradients: Vec<(Array2<f32>, Array1<f32>)> = Vec::new();
        let mut current_error = output_errors.to_owned();
    
        let length = self.layers.len();
        for i in (0..length).rev() {
            let layer = &mut self.layers[i];
            let (adjusted_error, weight_gradients, bias_gradients) = layer.backward_minibatch(current_error.view());
            // println!("NeuralNetwork::backward_minibatch weight_gradients shape: {:?}", weight_gradients.shape());
            // println!("NeuralNetwork::backward_minibatch bias_gradients shape: {:?}", bias_gradients.shape());
            gradients.push((weight_gradients, bias_gradients));
        
            // println!("NeuralNetwork::backward_minibatch layer.weights shape: {:?}", layer.weights.shape());
            // println!("NeuralNetwork::backward_minibatch current_error shape: {:?}", current_error.shape());
        
            if i != 0 {
                // println!("NeuralNetwork::backward_minibatch current_layer.weights shape: {:?}", layer.weights.shape());
                current_error = adjusted_error.dot(&layer.weights.t());
            }
            
        }
    
        gradients.reverse();
        gradients
    }

    /// Train the neural network for a single input vector and target output.
    /// This function updates the weights and biases of the neural network using the gradients computed
    /// by the backward function and the optimizer.
    fn train(&mut self, input: ArrayView1<f32>, target: ArrayView1<f32>, learning_rate: f32) {
        let input = input.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        let target = target.insert_axis(Axis(0)); // Treat single instance as a minibatch of size 1
        self.train_minibatch(input.view(), target.view(), learning_rate);
    }

    /// Train the neural network for a batch of input vectors and target outputs.
    /// This function updates the weights and biases of the neural network using the gradients computed
    /// by the backward_minibatch function and the optimizer.
    pub fn train_minibatch(
        &mut self,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        learning_rate: f32,
    ) {
        // println!("NeuralNetwork::train_minibatch inputs shape: {:?}, expected: {:?}", inputs.shape(), (inputs.shape()[0], self.layers[0].weights.shape()[0]));
        let outputs = self.forward_minibatch(inputs);
        // println!("NeuralNetwork::train_minibatch outputs shape: {:?}, expected: {:?}", outputs.shape(), self.layers.last().unwrap().weights.shape());        
        let output_errors = &outputs - &targets;
        // println!("NeuralNetwork::train_minibatch output_errors shape: {:?}, expected: {:?}", output_errors.shape(), self.layers.last().unwrap().weights.shape());
        let gradients = self.backward_minibatch(output_errors.view());
    
        for (layer, (weight_gradients, bias_gradients)) in self.layers.iter_mut().zip(gradients) {
            self.optimizer.update_weights(&mut layer.weights, &weight_gradients, learning_rate);
            self.optimizer.update_biases(&mut layer.biases, &bias_gradients, learning_rate);
        }
    }
    
    /// Save the neural network's state to a file.
    /// This function serializes the neural network, including its layers and optimizer, and writes
    /// the serialized data to a file at the specified path.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serialize(self)?;
        let mut file = fs::File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Load a neural network from a file.
    /// This function reads the serialized data from a file at the specified path, deserializes it,
    /// and constructs a new neural network with the loaded state.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let deserialized: Self = deserialize(&buffer)?;
        Ok(deserialized)
    }
}

/// A macro to create a new `Layer`.
///
/// # Examples
///
/// ```
/// use athena::network::{Layer, Activation};
/// use athena::create_layer;
/// let layer = create_layer!(4, 32, Activation::Relu);
/// ```
///
/// This will create a new `Layer` with an input size of 4, an output size of 32, and uses the ReLU activation function.
#[macro_export]
macro_rules! create_layer {
    ($input_size:expr, $output_size:expr, $activation:expr) => {
        Layer::new($input_size, $output_size, $activation)
    };
}

/// A macro to create a new `NeuralNetwork`.
///
/// # Examples
///
/// ```
/// use athena::optimizer::{OptimizerWrapper, SGD};
/// use athena::create_network;
/// use athena::network::{Activation, NeuralNetwork, Layer};
/// let optimizer = OptimizerWrapper::SGD(SGD::new());
/// let network = create_network!(optimizer,
///     (4, 32, Activation::Relu), 
///     (32, 2, Activation::Linear)
/// );
/// ```
///
/// This will create a new `NeuralNetwork` with two layers: the first layer has an input size of 4, an 
/// output size of 32, and uses the ReLU activation function; the second layer has an input size of 32 
/// (matching the output size of the first layer), an output size of 2, and uses the Linear activation function.
#[macro_export]
macro_rules! create_network {
    ($optimizer:expr, $( ($input_size:expr, $output_size:expr, $activation:expr) ),* ) => {
        {
            let layer_sizes = vec![$( ($input_size, $output_size) ),*];
            let activations = vec![$( $activation ),*];
            let layers = layer_sizes.iter().zip(&activations)
                .map(|(&(input_size, output_size), &activation)| Layer::new(input_size, output_size, activation))
                .collect::<Vec<_>>();
            NeuralNetwork { layers, optimizer: $optimizer }
        }
    }
}
