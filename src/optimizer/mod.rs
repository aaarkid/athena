//! # Optimization Algorithms Module
//! 
//! This module provides various optimization algorithms for training neural networks.
//! Optimizers are responsible for updating network parameters based on computed gradients.
//! 
//! ## Available Optimizers
//! 
//! - **SGD (Stochastic Gradient Descent)**
//!   - Simplest optimizer: `w = w - lr * gradient`
//!   - Fast but can get stuck in local minima
//!   - Good for convex problems
//! 
//! - **Adam (Adaptive Moment Estimation)**
//!   - Combines momentum with adaptive learning rates
//!   - Currently most popular optimizer
//!   - Works well out-of-the-box for most problems
//!   - Maintains per-parameter learning rates
//! 
//! - **RMSProp (Root Mean Square Propagation)**
//!   - Addresses diminishing learning rates in AdaGrad
//!   - Good for non-stationary objectives
//!   - Often used in RNNs
//! 
//! ## Additional Features
//! 
//! - **Gradient Clipping**: Prevents exploding gradients
//! - **Learning Rate Scheduling**: Adaptive learning rates over time
//! 
//! ## Usage Example
//! 
//! ```rust,no_run
//! use athena::optimizer::{OptimizerWrapper, SGD};
//! use athena::network::NeuralNetwork;
//! use athena::activations::Activation;
//! 
//! // Create SGD optimizer
//! let optimizer = OptimizerWrapper::SGD(SGD::new());
//! 
//! // Create network with optimizer
//! let network = NeuralNetwork::new(
//!     &[784, 128, 10],
//!     &[Activation::Relu, Activation::Linear],
//!     optimizer
//! );
//! ```
//! 
//! ## Choosing an Optimizer
//! 
//! | Optimizer | Best For | Learning Rate | Memory Usage |
//! |-----------|----------|---------------|--------------|
//! | SGD | Simple problems, fine-tuning | 0.01-0.1 | Low |
//! | Adam | General purpose, default choice | 0.0001-0.001 | High |
//! | RMSProp | RNNs, non-stationary problems | 0.001-0.01 | Medium |
//! 
//! ## Implementation Notes
//! 
//! - All optimizers maintain per-layer state for correct updates
//! - The `layer_idx` parameter ensures state isolation between layers
//! - Learning rates passed to update methods are multiplied with optimizer-specific rates
//! - Serialization preserves optimizer state for training resumption

pub mod gradient_clipper;
pub mod lr_scheduler;

use ndarray::{Array, Array2, Array1, Dimension, Zip};
use crate::layers::Layer;

/// One Adam update over a whole parameter array, in a single pass and without allocating.
///
/// The moment estimates, the bias correction and the step are all applied per element, so
/// the temporaries the expression form would build never exist.
fn adam_step<D: Dimension>(
    parameters: &mut Array<f32, D>,
    m: &mut Array<f32, D>,
    v: &mut Array<f32, D>,
    gradients: &Array<f32, D>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: i32,
    learning_rate: f32,
) {
    // powi once per call rather than once per element
    let correction1 = 1.0 - beta1.powi(t);
    let correction2 = 1.0 - beta2.powi(t);

    Zip::from(parameters)
        .and(m)
        .and(v)
        .and(gradients)
        .for_each(|parameter, m, v, &gradient| {
            *m = beta1 * *m + (1.0 - beta1) * gradient;
            *v = beta2 * *v + (1.0 - beta2) * gradient * gradient;

            let m_hat = *m / correction1;
            let v_hat = *v / correction2;
            *parameter -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        });
}

/// One RMSProp update over a whole parameter array, in a single pass and without allocating.
fn rmsprop_step<D: Dimension>(
    parameters: &mut Array<f32, D>,
    v: &mut Array<f32, D>,
    gradients: &Array<f32, D>,
    beta: f32,
    epsilon: f32,
    learning_rate: f32,
) {
    Zip::from(parameters)
        .and(v)
        .and(gradients)
        .for_each(|parameter, v, &gradient| {
            *v = beta * *v + (1.0 - beta) * gradient * gradient;
            *parameter -= learning_rate * gradient / (v.sqrt() + epsilon);
        });
}

pub use gradient_clipper::GradientClipper;
pub use lr_scheduler::LearningRateScheduler;

pub trait Optimizer {
    fn update_weights(&mut self, layer_idx: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>, learning_rate: f32);
    fn update_biases(&mut self, layer_idx: usize, biases: &mut Array1<f32>, gradients: &Array1<f32>, learning_rate: f32);
}

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OptimizerWrapper {
    SGD(SGD),
    Adam(Adam),
    RMSProp(RMSProp),
}

impl Optimizer for OptimizerWrapper {
    fn update_weights(&mut self, layer_idx: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>, learning_rate: f32) {
        match self {
            OptimizerWrapper::SGD(optimizer) => optimizer.update_weights(layer_idx, weights, gradients, learning_rate),
            OptimizerWrapper::Adam(optimizer) => optimizer.update_weights(layer_idx, weights, gradients, learning_rate),
            OptimizerWrapper::RMSProp(optimizer) => optimizer.update_weights(layer_idx, weights, gradients, learning_rate),
        }
    }

    fn update_biases(&mut self, layer_idx: usize, biases: &mut Array1<f32>, gradients: &Array1<f32>, learning_rate: f32) {
        match self {
            OptimizerWrapper::SGD(optimizer) => optimizer.update_biases(layer_idx, biases, gradients, learning_rate),
            OptimizerWrapper::Adam(optimizer) => optimizer.update_biases(layer_idx, biases, gradients, learning_rate),
            OptimizerWrapper::RMSProp(optimizer) => optimizer.update_biases(layer_idx, biases, gradients, learning_rate),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SGD;

impl SGD {
    pub fn new() -> SGD {
        SGD
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for SGD {
    fn update_weights(&mut self, _layer_idx: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>, learning_rate: f32) {
        weights.zip_mut_with(gradients, |w, &g| *w -= learning_rate * g);
    }

    fn update_biases(&mut self, _layer_idx: usize, biases: &mut Array1<f32>, gradients: &Array1<f32>, learning_rate: f32) {
        biases.zip_mut_with(gradients, |b, &g| *b -= learning_rate * g);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    m_weights: Vec<Array2<f32>>,
    v_weights: Vec<Array2<f32>>,
    m_biases: Vec<Array1<f32>>,
    v_biases: Vec<Array1<f32>>,
    pub t: usize,
    layer_count: usize,
    update_count: usize,
}

impl Adam {
    pub fn new(layers: &[Layer], beta1: f32, beta2: f32, epsilon: f32) -> Self {
        let m_weights = layers
            .iter()
            .map(|layer| Array2::<f32>::zeros(layer.weights.dim()))
            .collect();
        let v_weights = layers
            .iter()
            .map(|layer| Array2::<f32>::zeros(layer.weights.dim()))
            .collect();
        let m_biases = layers
            .iter()
            .map(|layer| Array1::<f32>::zeros(layer.biases.dim()))
            .collect();
        let v_biases = layers
            .iter()
            .map(|layer| Array1::<f32>::zeros(layer.biases.dim()))
            .collect();

        let layer_count = layers.len();
        
        Adam {
            beta1,
            beta2,
            epsilon,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            t: 1,
            layer_count,
            update_count: 0,
        }
    }

    pub fn default(layers: &[Layer]) -> Self {
        Self::new(layers, 0.9, 0.999, 1e-8)
    }
}

impl Optimizer for Adam {
    fn update_weights(&mut self, layer_idx: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>, learning_rate: f32) {
        // Lazy initialization: if momentum vectors don't exist for this layer, create them
        while self.m_weights.len() <= layer_idx {
            self.m_weights.push(Array2::<f32>::zeros((1, 1)));
            self.v_weights.push(Array2::<f32>::zeros((1, 1)));
            self.layer_count = self.m_weights.len();
        }

        // Resize if shape doesn't match
        if self.m_weights[layer_idx].dim() != gradients.dim() {
            self.m_weights[layer_idx] = Array2::<f32>::zeros(gradients.dim());
            self.v_weights[layer_idx] = Array2::<f32>::zeros(gradients.dim());
        }

        adam_step(
            weights,
            &mut self.m_weights[layer_idx],
            &mut self.v_weights[layer_idx],
            gradients,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.t as i32,
            learning_rate,
        );


        // Track updates
        self.update_count += 1;
        if self.update_count >= self.layer_count * 2 {
            self.t += 1;
            self.update_count = 0;
        }
    }

    fn update_biases(&mut self, layer_idx: usize, biases: &mut Array1<f32>, gradients: &Array1<f32>, learning_rate: f32) {
        // Lazy initialization: if momentum vectors don't exist for this layer, create them
        while self.m_biases.len() <= layer_idx {
            self.m_biases.push(Array1::<f32>::zeros(1));
            self.v_biases.push(Array1::<f32>::zeros(1));
        }

        // Resize if shape doesn't match
        if self.m_biases[layer_idx].len() != gradients.len() {
            self.m_biases[layer_idx] = Array1::<f32>::zeros(gradients.len());
            self.v_biases[layer_idx] = Array1::<f32>::zeros(gradients.len());
        }

        adam_step(
            biases,
            &mut self.m_biases[layer_idx],
            &mut self.v_biases[layer_idx],
            gradients,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.t as i32,
            learning_rate,
        );


        // Track updates
        self.update_count += 1;
        if self.update_count >= self.layer_count * 2 {
            self.t += 1;
            self.update_count = 0;
        }
    }
}


/// RMSProp optimizer
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RMSProp {
    pub beta: f32,
    pub epsilon: f32,
    v_weights: Vec<Array2<f32>>,
    v_biases: Vec<Array1<f32>>,
}

impl RMSProp {
    pub fn new(layers: &[Layer], beta: f32, epsilon: f32) -> Self {
        let v_weights = layers
            .iter()
            .map(|layer| Array2::<f32>::zeros(layer.weights.dim()))
            .collect();
        let v_biases = layers
            .iter()
            .map(|layer| Array1::<f32>::zeros(layer.biases.dim()))
            .collect();
            
        RMSProp {
            beta,
            epsilon,
            v_weights,
            v_biases,
        }
    }
    
    pub fn default(layers: &[Layer]) -> Self {
        Self::new(layers, 0.9, 1e-8)
    }
}

impl Optimizer for RMSProp {
    fn update_weights(&mut self, layer_idx: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>, learning_rate: f32) {
        // Lazy initialization
        while self.v_weights.len() <= layer_idx {
            self.v_weights.push(Array2::<f32>::zeros((1, 1)));
        }
        if self.v_weights[layer_idx].dim() != gradients.dim() {
            self.v_weights[layer_idx] = Array2::<f32>::zeros(gradients.dim());
        }

        rmsprop_step(
            weights,
            &mut self.v_weights[layer_idx],
            gradients,
            self.beta,
            self.epsilon,
            learning_rate,
        );
    }

    fn update_biases(&mut self, layer_idx: usize, biases: &mut Array1<f32>, gradients: &Array1<f32>, learning_rate: f32) {
        // Lazy initialization
        while self.v_biases.len() <= layer_idx {
            self.v_biases.push(Array1::<f32>::zeros(1));
        }
        if self.v_biases[layer_idx].len() != gradients.len() {
            self.v_biases[layer_idx] = Array1::<f32>::zeros(gradients.len());
        }

        rmsprop_step(
            biases,
            &mut self.v_biases[layer_idx],
            gradients,
            self.beta,
            self.epsilon,
            learning_rate,
        );
    }
}
