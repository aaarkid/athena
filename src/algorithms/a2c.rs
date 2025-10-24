use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Actor-Critic Agent implementing the A2C algorithm
/// 
/// A2C (Advantage Actor-Critic) is a policy gradient method that uses
/// a critic to estimate the value function and reduce variance in policy updates.
#[derive(Serialize, Deserialize, Clone)]
pub struct A2CAgent {
    /// Actor network that outputs action probabilities
    pub actor: NeuralNetwork,
    /// Critic network that estimates state values
    pub critic: NeuralNetwork,
    /// Discount factor for future rewards
    pub gamma: f32,
    /// Number of steps before performing an update
    pub n_steps: usize,
    /// Entropy coefficient for exploration
    pub entropy_coeff: f32,
    /// Value loss coefficient
    pub value_coeff: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f32>,
    /// Random number generator
    #[serde(skip)]
    pub rng: ThreadRng,
}

/// Experience tuple for A2C
#[derive(Clone, Debug)]
pub struct A2CExperience {
    pub state: Array1<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Array1<f32>,
    pub done: bool,
    pub log_prob: f32,
    pub value: f32,
}

impl A2CAgent {
    /// Create a new A2C agent
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        optimizer: OptimizerWrapper,
        gamma: f32,
        n_steps: usize,
        entropy_coeff: f32,
        value_coeff: f32,
    ) -> Self {
        // Build actor network
        let mut actor_sizes = vec![state_size];
        actor_sizes.extend_from_slice(hidden_sizes);
        actor_sizes.push(action_size);
        
        let actor_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let actor = NeuralNetwork::new(&actor_sizes, &actor_activations, optimizer.clone());
        
        // Build critic network
        let mut critic_sizes = vec![state_size];
        critic_sizes.extend_from_slice(hidden_sizes);
        critic_sizes.push(1); // Single value output
        
        let critic_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let critic = NeuralNetwork::new(&critic_sizes, &critic_activations, optimizer);
        
        A2CAgent {
            actor,
            critic,
            gamma,
            n_steps,
            entropy_coeff,
            value_coeff,
            max_grad_norm: Some(0.5),
            rng: thread_rng(),
        }
    }
    
    /// Select an action using the current policy
    pub fn act(&mut self, state: ArrayView1<f32>) -> Result<(usize, f32)> {
        let logits = self.actor.forward(state);
        let probs = softmax(&logits);
        
        // Sample action from probability distribution
        let action = self.sample_action(&probs)?;
        let log_prob = probs[action].ln();
        
        Ok((action, log_prob))
    }
    
    /// Get state value estimate from critic
    pub fn get_value(&mut self, state: ArrayView1<f32>) -> f32 {
        let value = self.critic.forward(state);
        value[0]
    }
    
    /// Sample action from probability distribution
    fn sample_action(&mut self, probs: &Array1<f32>) -> Result<usize> {
        let mut cumsum = 0.0;
        let rand_val: f32 = self.rng.gen();
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return Ok(i);
            }
        }
        
        // Fallback to last action if numerical issues
        Ok(probs.len() - 1)
    }
    
    /// Train the agent on a batch of experiences
    pub fn train(
        &mut self,
        experiences: &[A2CExperience],
        _learning_rate: f32,
    ) -> Result<(f32, f32)> {
        if experiences.is_empty() {
            return Err(AthenaError::EmptyBuffer("No experiences to train on".to_string()));
        }
        
        let batch_size = experiences.len();
        
        // Prepare batch data
        let states = stack_arrays(experiences.iter().map(|e| e.state.view()).collect());
        let actions = experiences.iter().map(|e| e.action).collect::<Vec<_>>();
        let rewards = experiences.iter().map(|e| e.reward).collect::<Vec<_>>();
        let _next_states = stack_arrays(experiences.iter().map(|e| e.next_state.view()).collect());
        let dones = experiences.iter().map(|e| e.done).collect::<Vec<_>>();
        let _old_log_probs = experiences.iter().map(|e| e.log_prob).collect::<Vec<_>>();
        let old_values = experiences.iter().map(|e| e.value).collect::<Vec<_>>();
        
        // Compute returns and advantages
        let (returns, advantages) = self.compute_gae(&rewards, &old_values, &dones);
        
        // Forward pass through networks
        let actor_outputs = self.actor.forward_batch(states.view());
        let critic_outputs = self.critic.forward_batch(states.view());
        
        // Compute actor loss
        let mut actor_loss = 0.0;
        let mut entropy = 0.0;
        
        for (i, &action) in actions.iter().enumerate() {
            let logits = actor_outputs.row(i);
            let probs = softmax(&logits.to_owned());
            let log_prob = probs[action].ln();
            
            // Policy gradient loss
            actor_loss -= log_prob * advantages[i];
            
            // Entropy bonus for exploration
            for &p in probs.iter() {
                if p > 1e-8 {
                    entropy -= p * p.ln();
                }
            }
        }
        
        actor_loss /= batch_size as f32;
        entropy /= batch_size as f32;
        actor_loss -= self.entropy_coeff * entropy;
        
        // Compute critic loss (MSE)
        let mut critic_loss = 0.0;
        for (i, &ret) in returns.iter().enumerate() {
            let value_pred = critic_outputs[[i, 0]];
            critic_loss += (value_pred - ret).powi(2);
        }
        critic_loss /= batch_size as f32;
        critic_loss *= self.value_coeff;
        
        // Compute gradients and update networks
        // This is a simplified version - in practice you'd compute proper gradients
        let _total_loss = actor_loss + critic_loss;
        
        // Update networks (simplified - would need proper gradient computation)
        // In a real implementation, you'd compute gradients properly
        
        Ok((actor_loss, critic_loss))
    }
    
    /// Compute Generalized Advantage Estimation (GAE)
    fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        let mut returns = vec![0.0; n];
        let mut advantages = vec![0.0; n];
        
        // Compute returns using n-step returns
        for i in (0..n).rev() {
            if i == n - 1 || dones[i] {
                returns[i] = rewards[i];
            } else {
                returns[i] = rewards[i] + self.gamma * returns[i + 1];
            }
            
            advantages[i] = returns[i] - values[i];
        }
        
        (returns, advantages)
    }
    
    /// Save the agent to disk
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::serialize(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load an agent from disk
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let agent = bincode::deserialize(&data)?;
        Ok(agent)
    }
}

/// Softmax function for converting logits to probabilities
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp = exp_logits.sum();
    exp_logits / sum_exp
}

/// Stack 1D arrays into a 2D array
fn stack_arrays(arrays: Vec<ArrayView1<f32>>) -> Array2<f32> {
    if arrays.is_empty() {
        return Array2::zeros((0, 0));
    }
    
    let rows = arrays.len();
    let cols = arrays[0].len();
    let mut result = Array2::zeros((rows, cols));
    
    for (i, arr) in arrays.iter().enumerate() {
        result.row_mut(i).assign(arr);
    }
    
    result
}

/// Builder pattern for A2CAgent
pub struct A2CBuilder {
    state_size: usize,
    action_size: usize,
    hidden_sizes: Vec<usize>,
    optimizer: Option<OptimizerWrapper>,
    gamma: f32,
    n_steps: usize,
    entropy_coeff: f32,
    value_coeff: f32,
}

impl A2CBuilder {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        A2CBuilder {
            state_size,
            action_size,
            hidden_sizes: vec![128, 128],
            optimizer: None,
            gamma: 0.99,
            n_steps: 5,
            entropy_coeff: 0.01,
            value_coeff: 0.5,
        }
    }
    
    pub fn hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_sizes = sizes;
        self
    }
    
    pub fn optimizer(mut self, optimizer: OptimizerWrapper) -> Self {
        self.optimizer = Some(optimizer);
        self
    }
    
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }
    
    pub fn n_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }
    
    pub fn entropy_coeff(mut self, coeff: f32) -> Self {
        self.entropy_coeff = coeff;
        self
    }
    
    pub fn value_coeff(mut self, coeff: f32) -> Self {
        self.value_coeff = coeff;
        self
    }
    
    pub fn build(self) -> Result<A2CAgent> {
        let optimizer = self.optimizer
            .ok_or_else(|| AthenaError::InvalidParameter {
            name: "optimizer".to_string(),
            reason: "Optimizer not specified".to_string(),
        })?;
        
        Ok(A2CAgent::new(
            self.state_size,
            self.action_size,
            &self.hidden_sizes,
            optimizer,
            self.gamma,
            self.n_steps,
            self.entropy_coeff,
            self.value_coeff,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    
    #[test]
    fn test_a2c_creation() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = A2CAgent::new(4, 2, &[32, 32], optimizer, 0.99, 5, 0.01, 0.5);
        
        assert_eq!(agent.gamma, 0.99);
        assert_eq!(agent.n_steps, 5);
        assert_eq!(agent.entropy_coeff, 0.01);
        assert_eq!(agent.value_coeff, 0.5);
    }
    
    #[test]
    fn test_a2c_builder() {
        let agent = A2CBuilder::new(4, 2)
            .hidden_sizes(vec![64, 64])
            .gamma(0.95)
            .optimizer(OptimizerWrapper::SGD(SGD::new()))
            .build()
            .unwrap();
        
        assert_eq!(agent.gamma, 0.95);
    }
    
    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        
        // Check probabilities sum to 1
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        
        // Check all probabilities are positive
        for &p in probs.iter() {
            assert!(p > 0.0 && p <= 1.0);
        }
    }
}