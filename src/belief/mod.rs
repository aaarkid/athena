//! # Belief States Module
//! 
//! This module provides belief state representations for partially observable environments.
//! Belief states maintain a probability distribution over possible hidden states,
//! allowing agents to reason under uncertainty.
//! 
//! ## Core Concepts
//! 
//! - **Belief State**: A probability distribution over possible world states
//! - **Particle Filter**: Approximate belief representation using samples
//! - **History-Based Belief**: Using observation/action history as belief
//! - **Belief Update**: Bayesian update based on actions and observations
//! 
//! ## Available Components
//! 
//! - `BeliefState` trait: Core interface for belief representations
//! - `HistoryBelief`: Maintains a fixed-size history window
//! - `ParticleFilter`: Monte Carlo approximation of belief
//! - `BeliefAgent`: Wrapper that adds belief tracking to any agent

use ndarray::Array1;
use serde::{Serialize, Deserialize};

/// Core trait for belief state representations
pub trait BeliefState: Send + Sync {
    type Observation;
    type State;
    
    /// Update belief based on action and observation
    fn update(&mut self, action: usize, observation: &Self::Observation);
    
    /// Sample a concrete state from belief distribution
    fn sample(&self) -> Self::State;
    
    /// Get belief as feature vector for neural network
    fn to_feature_vector(&self) -> Array1<f32>;
    
    /// Reset belief to initial distribution
    fn reset(&mut self);
    
    /// Get entropy of belief distribution (uncertainty measure)
    fn entropy(&self) -> f32;
}

/// Belief state that tracks history
#[derive(Clone, Serialize, Deserialize)]
pub struct HistoryBelief {
    max_history: usize,
    action_history: Vec<usize>,
    observation_history: Vec<Array1<f32>>,
    embedding_size: usize,
}

impl HistoryBelief {
    pub fn new(max_history: usize, embedding_size: usize) -> Self {
        Self {
            max_history,
            action_history: Vec::with_capacity(max_history),
            observation_history: Vec::with_capacity(max_history),
            embedding_size,
        }
    }
    
    /// Convert history to fixed-size embedding
    fn embed_history(&self) -> Array1<f32> {
        let mut embedding = Array1::zeros(self.embedding_size);
        let history_len = self.action_history.len();
        
        if history_len == 0 {
            return embedding;
        }
        
        // Simple embedding: concatenate recent observations and one-hot actions
        let obs_dim = self.observation_history[0].len();
        let action_dim = 10; // Assume max 10 actions
        let step_size = obs_dim + action_dim;
        
        for (i, (action, obs)) in self.action_history.iter()
            .zip(self.observation_history.iter())
            .rev()
            .take(self.embedding_size / step_size)
            .enumerate()
        {
            let offset = i * step_size;
            
            // Copy observation
            for (j, &val) in obs.iter().enumerate() {
                if offset + j < self.embedding_size {
                    embedding[offset + j] = val;
                }
            }
            
            // One-hot encode action
            let action_offset = offset + obs_dim;
            if action_offset + *action < self.embedding_size {
                embedding[action_offset + *action] = 1.0;
            }
        }
        
        embedding
    }
}

impl BeliefState for HistoryBelief {
    type Observation = Array1<f32>;
    type State = Array1<f32>;
    
    fn update(&mut self, action: usize, observation: &Self::Observation) {
        self.action_history.push(action);
        self.observation_history.push(observation.clone());
        
        // Keep only recent history
        if self.action_history.len() > self.max_history {
            self.action_history.remove(0);
            self.observation_history.remove(0);
        }
    }
    
    fn sample(&self) -> Self::State {
        // For history-based belief, return the embedding
        self.to_feature_vector()
    }
    
    fn to_feature_vector(&self) -> Array1<f32> {
        self.embed_history()
    }
    
    fn reset(&mut self) {
        self.action_history.clear();
        self.observation_history.clear();
    }
    
    fn entropy(&self) -> f32 {
        // History-based belief doesn't have a natural entropy
        // Return a measure based on history length
        1.0 - (self.action_history.len() as f32 / self.max_history as f32)
    }
}

#[cfg(feature = "belief-states")]
pub mod particle_filter;
#[cfg(feature = "belief-states")]
pub mod belief_agent;

#[cfg(feature = "belief-states")]
pub use particle_filter::ParticleFilter;
#[cfg(feature = "belief-states")]
pub use belief_agent::BeliefAgent;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_history_belief() {
        let mut belief = HistoryBelief::new(5, 30);
        
        // Update with some observations
        let obs1 = array![1.0, 2.0, 3.0];
        let obs2 = array![4.0, 5.0, 6.0];
        
        belief.update(0, &obs1);
        belief.update(1, &obs2);
        
        // Check embedding
        let embedding = belief.to_feature_vector();
        assert_eq!(embedding.len(), 30);
        
        // Check history limit
        for i in 0..10 {
            belief.update(i % 3, &array![i as f32, (i + 1) as f32, (i + 2) as f32]);
        }
        assert_eq!(belief.action_history.len(), 5);
    }
    
    #[test]
    fn test_belief_reset() {
        let mut belief = HistoryBelief::new(5, 30);
        
        belief.update(0, &array![1.0, 2.0]);
        belief.update(1, &array![3.0, 4.0]);
        
        assert!(!belief.action_history.is_empty());
        
        belief.reset();
        
        assert!(belief.action_history.is_empty());
        assert!(belief.observation_history.is_empty());
    }
}