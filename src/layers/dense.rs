use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use crate::activations::Activation;
use super::traits::Layer as LayerTrait;

/// A fully connected (dense) layer in a neural network
#[derive(Serialize, Deserialize, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Activation,
    pre_activation_output: Option<Array2<f32>>,
    inputs: Option<Array2<f32>>,
}

impl DenseLayer {
    /// Create a new dense layer with the given input size, output size, and activation function.
    /// The weights are initialized with random values from a uniform distribution
    /// between -0.1 and 0.1. The biases are initialized with zeros.
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let weights = Array2::random((input_size, output_size), Uniform::new(-0.1, 0.1));
        let biases = Array1::zeros(output_size);
        DenseLayer {
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
}

impl LayerTrait for DenseLayer {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let input = input.insert_axis(Axis(0));
        let output = self.forward_batch(input.view());
        let shape = output.shape()[1];
        output.into_shape((shape,)).expect("Failed to reshape output")
    }

    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.inputs = Some(inputs.to_owned());
        let mut outputs = inputs.dot(&self.weights) + &self.biases.to_owned().insert_axis(Axis(0));
        self.pre_activation_output = Some(outputs.clone());
        self.activation.apply_batch(&mut outputs);
        outputs
    }

    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        let output_error = output_error.insert_axis(Axis(0));
        let (_adjusted_error, weight_gradients, bias_gradients) = self.backward_batch(output_error.view());
        let shape = bias_gradients.shape()[1];
        (weight_gradients, bias_gradients.into_shape((shape,)).expect("Failed to reshape bias gradients"))
    }

    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let pre_activation_output = self.pre_activation_output.as_ref()
            .expect("No pre-activation output stored. forward_batch() must be called before backward_batch()");
        let inputs = self.inputs.as_ref()
            .expect("No inputs stored. forward_batch() must be called before backward_batch()");
        
        let activation_deriv = self.activation.derivative_batch(pre_activation_output.view());
        let adjusted_error = output_errors.to_owned() * &activation_deriv;
        let weight_gradients = inputs.t().dot(&adjusted_error);
        let bias_gradients = adjusted_error.sum_axis(Axis(0));
        
        (adjusted_error, weight_gradients, bias_gradients)
    }

    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn biases(&self) -> &Array1<f32> {
        &self.biases
    }

    fn output_size(&self) -> usize {
        self.weights.shape()[1]
    }

    fn input_size(&self) -> usize {
        self.weights.shape()[0]
    }

    fn clone_box(&self) -> Box<dyn LayerTrait> {
        Box::new(self.clone())
    }
}

/// Backward compatibility alias
pub type Layer = DenseLayer;

impl Layer {
    /// Create a vector of layers from layer sizes (for backward compatibility)
    pub fn to_vector(layer_sizes: &[usize]) -> Vec<Layer> {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            let activation = if i == layer_sizes.len() - 2 {
                Activation::Linear
            } else {
                Activation::Relu
            };
            layers.push(Layer::new(input_size, output_size, activation));
        }
        layers
    }
    
    /// Backward compatibility methods
    #[inline]
    pub fn forward_minibatch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.forward_batch(inputs)
    }
    
    #[inline]
    pub fn backward_minibatch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        self.backward_batch(output_errors)
    }
}