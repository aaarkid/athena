//! # Athena
//!
//! A deep learning library for Rust, with a focus on reinforcement learning.
//! It covers network construction and training, the common RL algorithms, and
//! deployment through Python bindings and WebAssembly.
//!
//! ## Getting started
//!
//! - [Tutorials](tutorials) - guides and worked examples
//! - [Getting Started Guide](tutorials::getting_started) - your first agent
//! - [Examples](https://github.com/aaarkid/athena/tree/master/examples) - runnable code samples
//!
//! ## Core concepts
//!
//! - [Neural Networks](network) - layer stacks and training
//! - [RL Agents](agent) - DQN and traits for custom agents
//! - [Algorithms](algorithms) - A2C, PPO, SAC, TD3
//! - [Optimizers](optimizer) - SGD, Adam, RMSProp
//!
//! ## Advanced topics
//!
//! - [Advanced Tutorial](tutorials::advanced) - custom layers and techniques
//! - [Performance Guide](tutorials::performance) - optimization tips
//! - [GPU Acceleration](gpu) - OpenCL backend, Intel Arc and NVIDIA
//! - [Best Practices](tutorials::best_practices) - recommended patterns
//! - [Algorithm Selection](tutorials::algorithms) - choosing an algorithm
//!
//! ## Features
//!
//! - Neural networks with several layer types and activations
//! - RL algorithms: DQN, A2C, PPO, SAC, TD3
//! - Optimizers: SGD, Adam, RMSProp, each with per-layer state
//! - Replay buffers, with optional prioritization
//! - Native Rust, Python bindings and WebAssembly targets
//!
//! ## Quick Start
//! 
//! ```rust,no_run
//! use athena::network::NeuralNetwork;
//! use athena::agent::DqnAgent;
//! use athena::activations::Activation;
//! use athena::optimizer::{OptimizerWrapper, Adam};
//! use athena::replay_buffer::ReplayBuffer;
//! 
//! // Create a neural network
//! let layer_sizes = &[4, 128, 128, 2];
//! let optimizer = OptimizerWrapper::SGD(athena::optimizer::SGD::new());
//! 
//! // Create a DQN agent
//! let agent = DqnAgent::new(layer_sizes, 0.1, optimizer, 1000, true);
//! 
//! // Create a replay buffer
//! let mut buffer = ReplayBuffer::new(10000);
//! ```
//! 
//! ## Module Organization
//! 
//! - [`activations`] - Activation functions (ReLU, Sigmoid, Tanh, etc.)
//! - [`agent`] - RL agents (DQN and traits for custom agents)
//! - [`algorithms`] - Advanced RL algorithms (A2C, PPO, SAC, TD3)
//! - [`builders`] - Builder patterns for convenient object construction
//! - [`debug`] - Debugging utilities for network inspection
//! - [`error`] - Error types and result handling
//! - [`export`] - Writing a trained network out to disk
//! - [`layers`] - Neural network layers (Dense, BatchNorm, Dropout)
//! - [`loss`] - Loss functions for training
//! - [`metrics`] - Training metrics and tracking
//! - [`network`] - Core neural network implementation
//! - [`optimizer`] - Optimization algorithms
//! - [`replay_buffer`] - Experience replay for RL
//! - [`types`] - Generic type definitions for states and actions
//! - [`visualization`] - Tools for visualizing networks and training

#[macro_use]
pub mod macros;

pub mod activations;
pub mod agent;
pub mod algorithms;
pub mod builders;
pub mod debug;
pub mod error;
pub mod export;
pub mod layers; 
pub mod loss;
pub mod metrics;
pub mod network;
pub mod optimizer;
pub mod replay_buffer;
pub mod types;
pub mod visualization;
pub mod memory_optimization;
pub mod parallel;
pub mod tensorboard;
pub mod gpu;
pub mod tutorials;
#[cfg(feature = "belief-states")]
pub mod belief;
#[cfg(feature = "multi-agent")]
pub mod multi_agent;

#[cfg(feature = "python")]
pub mod bindings;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(test)]
mod tests;