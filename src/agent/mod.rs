//! # Reinforcement Learning Agents Module
//! 
//! This module provides reinforcement learning agents and traits for building custom agents.
//! The primary agent implementation is DQN (Deep Q-Network), with traits to support
//! different types of RL algorithms.
//! 
//! ## Core Concepts
//! 
//! - **Agent**: An entity that interacts with an environment by taking actions
//! - **Q-Learning**: Learning the value of state-action pairs
//! - **Exploration vs Exploitation**: Balancing between trying new actions and using known good ones
//! - **Experience Replay**: Storing and reusing past experiences for stable learning
//! 
//! ## Available Agents
//! 
//! - **DqnAgent**: Deep Q-Network with experience replay and target network
//!   - Supports Double DQN for reduced overestimation
//!   - Epsilon-greedy exploration strategy
//!   - Batch training from replay buffer
//! 
//! ## Example Usage
//! 
//! ```rust,no_run
//! use athena::agent::DqnAgent;
//! use athena::optimizer::{OptimizerWrapper, Adam};
//! use athena::replay_buffer::ReplayBuffer;
//! use ndarray::array;
//! 
//! // Create a DQN agent for CartPole (4 inputs, 2 actions)
//! let layer_sizes = &[4, 128, 128, 2];
//! let optimizer = OptimizerWrapper::Adam(Adam::new(0.001, 0.9, 0.999, 1e-8));
//! let agent = DqnAgent::new(layer_sizes, 0.1, optimizer, 1000, true);
//! 
//! // Create replay buffer
//! let mut buffer = ReplayBuffer::new(10000);
//! 
//! // Training loop
//! let state = array![0.1, 0.2, -0.3, 0.4];
//! let action = agent.act(state.view()).unwrap();
//! ```
//! 
//! ## Generic Agent Traits
//! 
//! The module provides traits for building custom agents:
//! - `RLAgent`: Base trait for all RL agents
//! - `ValueBasedAgent`: For agents that learn value functions
//! - `PolicyBasedAgent`: For agents that learn policies directly
//! - `ActorCriticAgent`: For agents with separate actor and critic networks
//! 
//! ## Best Practices
//! 
//! 1. **Hyperparameters**: Start with small learning rates (1e-4 to 1e-3)
//! 2. **Network Architecture**: Two hidden layers of 128-256 units often work well
//! 3. **Exploration**: Decay epsilon from 1.0 to 0.01 over training
//! 4. **Replay Buffer**: Use at least 10,000 experiences for stability
//! 5. **Target Network**: Update every 100-1000 steps for stability

pub mod traits;

// Re-export the DqnAgent from the implementation
mod dqn;
pub use dqn::{DqnAgent, DqnAgentBuilder};