//! # Advanced RL Algorithms Module
//! 
//! This module implements state-of-the-art reinforcement learning algorithms beyond
//! basic DQN. Each algorithm is designed for different types of problems and offers
//! unique advantages.
//! 
//! ## Available Algorithms
//! 
//! ### Policy Gradient Methods
//! 
//! - **A2C (Advantage Actor-Critic)**
//!   - Combines value-based and policy-based methods
//!   - Uses advantage estimation to reduce variance
//!   - Good for discrete and continuous action spaces
//!   - Synchronous updates for stability
//! 
//! - **PPO (Proximal Policy Optimization)**
//!   - Currently most popular algorithm for continuous control
//!   - Clips policy updates to prevent instability
//!   - Excellent sample efficiency
//!   - Used in OpenAI's ChatGPT training
//! 
//! ### Off-Policy Methods
//! 
//! - **SAC (Soft Actor-Critic)**
//!   - Maximizes both reward and entropy
//!   - Excellent exploration properties
//!   - State-of-the-art for continuous control
//!   - Automatic temperature tuning
//! 
//! - **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**
//!   - Addresses overestimation in actor-critic methods
//!   - Uses twin Q-networks and delayed policy updates
//!   - Great for robotic control tasks
//!   - More stable than DDPG
//! 
//! ## Choosing an Algorithm
//! 
//! | Algorithm | Best For | Action Space | Sample Efficiency | Stability |
//! |-----------|----------|--------------|-------------------|-----------|
//! | DQN | Simple discrete tasks | Discrete | High | High |
//! | A2C | Fast learning | Both | Medium | Medium |
//! | PPO | General purpose | Both | High | High |
//! | SAC | Exploration-heavy tasks | Continuous | High | High |
//! | TD3 | Precise control | Continuous | High | Very High |
//! 
//! ## Example Usage
//! 
//! ```rust,no_run
//! use athena::algorithms::{PPOAgent, PPOBuilder};
//! use athena::optimizer::{OptimizerWrapper, Adam};
//! 
//! // Create a PPO agent for continuous control
//! let optimizer = OptimizerWrapper::Adam(Adam::new(3e-4, 0.9, 0.999, 1e-8));
//! 
//! let agent = PPOBuilder::new()
//!     .input_dim(24)      // Observation space
//!     .action_dim(4)      // Action space
//!     .hidden_dims(vec![256, 256])
//!     .optimizer(optimizer)
//!     .clip_epsilon(0.2)
//!     .value_coeff(0.5)
//!     .entropy_coeff(0.01)
//!     .build()
//!     .unwrap();
//! ```
//! 
//! ## Implementation Details
//! 
//! All algorithms in this module:
//! - Support both CPU and GPU training (with appropriate features)
//! - Include builder patterns for easy configuration
//! - Have comprehensive unit tests
//! - Follow the same interface patterns for consistency
//! - Support serialization for model saving/loading

pub mod a2c;
pub mod ppo;
pub mod sac;
pub mod td3;

pub use a2c::A2CAgent;
pub use ppo::PPOAgent;
pub use sac::SACAgent;
pub use td3::TD3Agent;