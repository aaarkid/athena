//! # Multi-Agent Reinforcement Learning Module
//! 
//! This module provides infrastructure for multi-agent reinforcement learning,
//! including environments, training algorithms, and coordination mechanisms.
//! 
//! ## Core Concepts
//! 
//! - **Multi-Agent Environment**: Environments with multiple agents acting simultaneously or sequentially
//! - **Self-Play**: Training agents by playing against copies of themselves
//! - **Population Training**: Maintaining a diverse population of agents
//! - **Communication**: Agents sharing information during episodes
//! 
//! ## Features
//! 
//! - Turn-based and simultaneous action environments
//! - Self-play training with various sampling strategies
//! - Agent pools with ELO ratings
//! - Communication channels between agents
//! - Tournament and league play systems

pub mod environment;
pub mod trainer;

#[cfg(feature = "multi-agent")]
pub mod communication;

pub use environment::{MultiAgentEnvironment, MultiAgentTransition, TurnBasedWrapper};
pub use trainer::{SelfPlayTrainer, SamplingStrategy, TrainingMetrics};

#[cfg(feature = "multi-agent")]
pub use communication::{CommunicationChannel, CommunicatingAgent, BroadcastChannel};