//! # Athena
//!
//! A deep learning library for Rust, with a focus on reinforcement learning.
//! It covers network construction and training, the common RL algorithms, and
//! deployment through Python bindings and WebAssembly.
//!
//! ## Getting started
//!
//! - [Quickstart](docs::quickstart) - the whole path: act, learn, save, reload
//! - [Conventions](docs::conventions) - shapes, weight orientation, what can be stacked
//! - [Examples](https://github.com/aaarkid/athena/tree/master/examples) - runnable code samples
//!
//! ## Core concepts
//!
//! - [Neural Networks](network) - dense layer stacks and training
//! - [Recurrent Networks](recurrent) - the only way to train an LSTM or GRU
//! - [RL Agents](agent) - DQN and traits for custom agents
//! - [Algorithms](algorithms) - A2C, PPO, SAC, TD3
//! - [Optimizers](optimizer) - SGD, Adam, RMSProp
//! - [Replay Buffers](replay_buffer) - uniform and prioritized
//!
//! ## Guides
//!
//! Every code sample in these is compiled by `cargo test`.
//!
//! - [Getting Started](docs::getting_started) - the basics at more length
//! - [Algorithms Guide](docs::algorithms_guide) - the five algorithms and how each is called
//! - [Best Practices](docs::best_practices) - what to check when an agent will not learn
//! - [Advanced Tutorial](docs::advanced) - writing a layer, multi-agent, partial observability
//! - [GPU Acceleration](gpu) - OpenCL backend, and what the mock does not do
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
//! ```rust
//! use athena::agent::DqnAgent;
//! use athena::optimizer::{Adam, OptimizerWrapper};
//! use athena::replay_buffer::ReplayBuffer;
//!
//! // Two observations in, four actions out, epsilon 1.0 to start
//! let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
//! let mut agent = DqnAgent::new(&[2, 64, 64, 4], 1.0, optimizer, 200, true);
//! let mut buffer = ReplayBuffer::new(20_000);
//!
//! let state = ndarray::array![0.0, 0.0];
//! let action = agent.act(state.view()).unwrap();
//! assert!(action < 4);
//! ```
//!
//! See [Quickstart](docs::quickstart) for the rest: the replay loop, epsilon decay,
//! evaluation and saving.
//!
//! ## Module Organization
//!
//! Core:
//!
//! - [`activations`] - Relu, Sigmoid, Tanh, Linear, LeakyRelu, Elu, Gelu
//! - [`layers`] - dense, conv, pooling, batch norm, dropout, LSTM, GRU, embedding
//! - [`network`] - `NeuralNetwork`, a stack of dense layers, plus training and inference
//! - [`recurrent`] - `RecurrentNetwork`, an LSTM or GRU cell with a dense head
//! - [`optimizer`] - SGD, Adam, RMSProp, gradient clipping, learning rate schedules
//! - [`loss`] - MSE, Huber, cross entropy
//!
//! Reinforcement learning:
//!
//! - [`agent`] - `DqnAgent` and the `RLAgent` traits
//! - [`algorithms`] - A2C, PPO, SAC, TD3
//! - [`replay_buffer`] - `Experience`, `ReplayBuffer`, `PrioritizedReplayBuffer`
//! - [`types`] - `State` and `Action` traits, `ActionSpace`
//!
//! Supporting:
//!
//! - [`builders`] - builder patterns for networks and agents
//! - [`debug`] - network inspection and numerical checks
//! - [`error`] - `AthenaError` and `Result`
//! - [`export`] - writing a network out as text or JSON
//! - [`gpu`] - OpenCL backend, and a mock that runs on the CPU
//! - [`memory_optimization`] - allocation-conscious training helpers
//! - [`metrics`] - training metrics and tracking
//! - [`parallel`] - rayon-backed batch helpers
//! - [`rng`] - seedable generators, so a run reproduces
//! - [`serialization`] - the on-disk file format
//! - [`tensorboard`] - event file writer
//! - [`tutorials`] - the guides, as module documentation
//! - [`visualization`] - text plots of training history
//!
//! Behind features: `belief` (`belief-states`), `multi_agent` (`multi-agent`),
//! `bindings` (`python`), `wasm` (`wasm`).

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
pub mod recurrent;
pub mod rng;
pub mod replay_buffer;
pub mod serialization;

/// The documents under `docs/`, compiled so their code samples cannot drift.
pub mod docs {
    #[doc = include_str!("../README.md")]
    pub mod readme {}

    #[doc = include_str!("../docs/quickstart.md")]
    pub mod quickstart {}

    #[doc = include_str!("../docs/conventions.md")]
    pub mod conventions {}

    #[doc = include_str!("../docs/algorithms_guide.md")]
    pub mod algorithms_guide {}

    #[doc = include_str!("../docs/tutorial_getting_started.md")]
    pub mod getting_started {}

    #[doc = include_str!("../docs/tutorial_advanced.md")]
    pub mod advanced {}

    #[doc = include_str!("../docs/best_practices.md")]
    pub mod best_practices {}
}
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