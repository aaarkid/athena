//! # Athena - High-Performance Deep Reinforcement Learning Library
//! 
//! Athena is a Rust-based reinforcement learning library designed for high performance,
//! modularity, and ease of use. It provides a comprehensive set of tools for building
//! and training deep neural networks, implementing various RL algorithms, and deploying
//! models across different platforms.
//! 
//! ## 📚 Documentation Hub
//! 
//! ### 🚀 Getting Started
//! - **[Tutorials](tutorials)** - Comprehensive guides and examples
//! - **[Getting Started Guide](tutorials::getting_started)** - Your first agent
//! - **[Examples](https://github.com/yourusername/athena/tree/main/examples)** - Working code samples
//! 
//! ### 🎯 Core Concepts
//! - **[Neural Networks](network)** - Flexible architecture
//! - **[RL Agents](agent)** - DQN and other agents
//! - **[Algorithms](algorithms)** - PPO, SAC, TD3, and more
//! - **[Optimizers](optimizer)** - Training optimization
//! 
//! ### 🔧 Advanced Topics
//! - **[Advanced Tutorial](tutorials::advanced)** - Custom layers and techniques
//! - **[Performance Guide](tutorials::performance)** - Optimization tips
//! - **[GPU Acceleration](gpu)** - Intel Arc and NVIDIA support
//! - **[Best Practices](tutorials::best_practices)** - Recommended patterns
//! 
//! ### 📖 Reference Guides
//! - **[Algorithm Selection](tutorials::algorithms)** - Choose the right algorithm
//! - **[API Reference](#modules)** - Complete module documentation
//! 
//! ## Key Features
//! 
//! - **Neural Networks**: Flexible architecture with various layer types and activations
//! - **RL Algorithms**: DQN, A2C, PPO, SAC, TD3, and more
//! - **Optimizers**: SGD, Adam, RMSProp with proper per-layer state management
//! - **Memory Efficiency**: Replay buffers with prioritization support
//! - **Cross-Platform**: Native Rust, Python bindings, and WebAssembly support
//! - **Type Safety**: Compile-time guarantees with Rust's type system
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
//! - [`export`] - Model export functionality (ONNX)
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

#[cfg(feature = "python")]
pub mod bindings;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(test)]
mod tests;