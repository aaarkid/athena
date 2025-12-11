//! # Activation Functions Module
//! 
//! This module provides a collection of activation functions commonly used in neural networks.
//! Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.
//! 
//! ## Available Activations
//! 
//! - **ReLU** (Rectified Linear Unit): `max(0, x)` - The most popular activation
//! - **Sigmoid**: `1 / (1 + e^(-x))` - Outputs between 0 and 1
//! - **Tanh**: Hyperbolic tangent - Outputs between -1 and 1  
//! - **Linear**: Identity function - No transformation
//! - **LeakyReLU**: ReLU with small negative slope - Prevents dead neurons
//! - **ELU** (Exponential Linear Unit): Smooth alternative to ReLU
//! - **GELU** (Gaussian Error Linear Unit): Used in transformers
//! 
//! ## Usage Example
//! 
//! ```rust,no_run
//! use athena::activations::Activation;
//! use ndarray::array;
//! 
//! // Create different activation functions
//! let relu = Activation::Relu;
//! let leaky_relu = Activation::LeakyRelu { alpha: 0.01 };
//! let gelu = Activation::Gelu;
//! 
//! // Apply to data
//! let mut data = array![1.0, -0.5, 0.0, 2.0];
//! relu.apply(&mut data);
//! ```
//! 
//! ## Choosing an Activation Function
//! 
//! - **Hidden Layers**: ReLU is usually the best default choice
//! - **Output Layer**: 
//!   - Binary classification: Sigmoid
//!   - Multi-class classification: Softmax (applied externally)
//!   - Regression: Linear
//! - **Deep Networks**: Consider LeakyReLU or ELU to avoid vanishing gradients
//! - **Transformer Models**: GELU has shown good results
//! 
//! ## Performance Considerations
//! 
//! - ReLU is the fastest (simple max operation)
//! - Sigmoid and Tanh require exponential calculations
//! - GELU is computationally expensive but often worth it for accuracy

pub mod functions;
pub mod gelu;

pub use functions::Activation;