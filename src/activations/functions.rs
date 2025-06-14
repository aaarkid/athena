use ndarray::{Array1, Array2, ArrayView2};
use serde::{Serialize, Deserialize};

/// An enumeration of the possible activation functions that can be used in a neural network layer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub enum Activation {
    #[default]
    Relu,
    Linear,
    Sigmoid,
    Tanh,
    LeakyRelu { alpha: f32 },
    Elu { alpha: f32 },
    Gelu,
}

impl Activation {
    /// Apply the activation function to an input array in-place.
    pub fn apply(&self, input: &mut Array1<f32>) {
        match self {
            Activation::Relu => {
                input.mapv_inplace(|v| v.max(0.0));
            }
            Activation::Linear => {}
            Activation::Sigmoid => {
                input.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
            }
            Activation::Tanh => {
                input.mapv_inplace(|v| v.tanh());
            }
            Activation::LeakyRelu { alpha } => {
                let a = *alpha;
                input.mapv_inplace(|v| if v > 0.0 { v } else { a * v });
            }
            Activation::Elu { alpha } => {
                let a = *alpha;
                input.mapv_inplace(|v| if v > 0.0 { v } else { a * (v.exp() - 1.0) });
            }
            Activation::Gelu => {
                use super::gelu::Gelu;
                Gelu::apply(input);
            }
        }
    }
    
    /// Apply the activation function to a batch of input arrays in-place.
    pub fn apply_batch(&self, inputs: &mut Array2<f32>) {
        match self {
            Activation::Relu => {
                inputs.mapv_inplace(|v| v.max(0.0));
            }
            Activation::Linear => {}
            Activation::Sigmoid => {
                inputs.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
            }
            Activation::Tanh => {
                inputs.mapv_inplace(|v| v.tanh());
            }
            Activation::LeakyRelu { alpha } => {
                let a = *alpha;
                inputs.mapv_inplace(|v| if v > 0.0 { v } else { a * v });
            }
            Activation::Elu { alpha } => {
                let a = *alpha;
                inputs.mapv_inplace(|v| if v > 0.0 { v } else { a * (v.exp() - 1.0) });
            }
            Activation::Gelu => {
                use super::gelu::Gelu;
                Gelu::apply_batch(inputs);
            }
        }
    }

    /// Compute the derivative of the activation function for an input array.
    pub fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::Relu => {
                input.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::Linear => {
                Array1::ones(input.len())
            }
            Activation::Sigmoid => {
                input.mapv(|v| {
                    let sigmoid = 1.0 / (1.0 + (-v).exp());
                    sigmoid * (1.0 - sigmoid)
                })
            }
            Activation::Tanh => {
                input.mapv(|v| {
                    let tanh_v = v.tanh();
                    1.0 - tanh_v * tanh_v
                })
            }
            Activation::LeakyRelu { alpha } => {
                let a = *alpha;
                input.mapv(|v| if v > 0.0 { 1.0 } else { a })
            }
            Activation::Elu { alpha } => {
                let a = *alpha;
                input.mapv(|v| if v > 0.0 { 1.0 } else { a * v.exp() })
            }
            Activation::Gelu => {
                use super::gelu::Gelu;
                Gelu::derivative(input)
            }
        }
    }

    /// Compute the derivative of the activation function for a batch of input arrays.
    pub fn derivative_batch(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        match self {
            Activation::Relu => {
                inputs.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::Linear => {
                Array2::ones(inputs.dim())
            }
            Activation::Sigmoid => {
                inputs.mapv(|v| {
                    let sigmoid = 1.0 / (1.0 + (-v).exp());
                    sigmoid * (1.0 - sigmoid)
                })
            }
            Activation::Tanh => {
                inputs.mapv(|v| {
                    let tanh_v = v.tanh();
                    1.0 - tanh_v * tanh_v
                })
            }
            Activation::LeakyRelu { alpha } => {
                let a = *alpha;
                inputs.mapv(|v| if v > 0.0 { 1.0 } else { a })
            }
            Activation::Elu { alpha } => {
                let a = *alpha;
                inputs.mapv(|v| if v > 0.0 { 1.0 } else { a * v.exp() })
            }
            Activation::Gelu => {
                use super::gelu::Gelu;
                Gelu::derivative_batch(inputs)
            }
        }
    }
}

/// Backward compatibility: renamed methods
impl Activation {
    #[inline]
    pub fn apply_minibatch(&self, inputs: &mut Array2<f32>) {
        self.apply_batch(inputs)
    }
    
    #[inline]
    pub fn derivative_minibatch(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.derivative_batch(inputs)
    }
}