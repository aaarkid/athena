#[macro_use]
pub mod macros;

pub mod activations;
pub mod agent;
pub mod agent_v2;
pub mod debug;
pub mod error;
pub mod layers; 
pub mod loss;
pub mod metrics;
pub mod network;
pub mod optimizer;
pub mod replay_buffer;
pub mod replay_buffer_v2;
pub mod visualization;

// Re-export commonly used types for backward compatibility
pub use activations::Activation;
pub use layers::{Layer, DenseLayer};
pub use network::NeuralNetwork;

#[cfg(test)]
mod tests;