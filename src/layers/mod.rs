//! # Neural Network Layers Module
//! 
//! This module provides various layer types for building neural networks.
//! Each layer type implements the `LayerTrait` for forward and backward propagation.
//! 
//! ## Available Layers
//! 
//! ### Core Layers
//! 
//! - **DenseLayer** (Fully Connected)
//!   - Most common layer type
//!   - Linear transformation: `y = Wx + b`
//!   - Followed by activation function
//!   - Supports any input/output dimensions
//! 
//! ### Regularization Layers
//! 
//! - **BatchNormLayer**
//!   - Normalizes inputs to have zero mean and unit variance
//!   - Accelerates training and improves stability
//!   - Reduces internal covariate shift
//!   - Includes learnable scale and shift parameters
//! 
//! - **DropoutLayer**
//!   - Randomly zeros elements during training
//!   - Prevents overfitting by reducing co-adaptation
//!   - Automatically disabled during inference
//!   - Common rates: 0.2-0.5
//! 
//! ## Weight Initialization
//! 
//! Proper weight initialization is crucial for training success:
//! 
//! - **Xavier/Glorot**: Good for sigmoid/tanh activations
//! - **He/Kaiming**: Better for ReLU and variants
//! - **Uniform/Normal**: Basic distributions with custom ranges
//! 
//! ## Usage Example
//! 
//! ```rust,no_run
//! use athena::layers::{Layer, BatchNormLayer, DropoutLayer};
//! use athena::activations::Activation;
//! use athena::layers::WeightInit;
//! 
//! // Create a dense layer with He initialization
//! let dense = Layer::new_with_init(128, 64, Activation::Relu, WeightInit::HeUniform);
//! 
//! // Create regularization layers
//! let batch_norm = BatchNormLayer::new(64, 0.9, 1e-5);  // num_features, momentum, epsilon
//! let dropout = DropoutLayer::new(64, 0.3);  // size, dropout_rate
//! ```
//! 
//! ## Layer Composition
//! 
//! Typical patterns for deep networks:
//! 
//! 1. **Standard Block**: Dense → BatchNorm → Activation → Dropout
//! 2. **ResNet Block**: Add skip connections around blocks
//! 3. **Wide Networks**: Increase layer width for more capacity
//! 4. **Deep Networks**: Stack many layers with careful initialization
//! 
//! ## Performance Tips
//! 
//! - BatchNorm often allows higher learning rates
//! - Dropout is typically not used with BatchNorm
//! - Place Dropout after activation functions
//! - Initialize biases to zero, weights based on activation

pub mod traits;
pub mod dense;
pub mod batch_norm;
pub mod dropout;
pub mod initialization;
pub mod conv;
pub mod pooling;
pub mod lstm;
pub mod gru;
pub mod embedding;
#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
pub mod gpu_dense;
#[cfg(feature = "action-masking")]
pub mod masked;

pub use traits::Layer as LayerTrait;
pub use dense::{DenseLayer, Layer};
pub use batch_norm::BatchNormLayer;
pub use dropout::DropoutLayer;
pub use initialization::WeightInit;
pub use conv::{Conv1DLayer, Conv2DLayer, Conv2DLayerBuilder};
pub use pooling::{MaxPool1DLayer, MaxPool2DLayer, AvgPool2DLayer, GlobalAvgPoolLayer};
pub use lstm::{LSTMLayer, LSTMGradients};
pub use gru::{GRULayer, GRUGradients};
pub use embedding::EmbeddingLayer;
#[cfg(any(feature = "gpu", feature = "gpu-mock"))]
pub use gpu_dense::GpuDenseLayer;
#[cfg(feature = "action-masking")]
pub use masked::{MaskedLayer, MaskedSoftmax};