use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Proximal Policy Optimization (PPO) Agent
/// 
/// PPO is a policy gradient method that uses a clipped surrogate objective
/// to ensure stable policy updates.
#[derive(Serialize, Deserialize, Clone)]
pub struct PPOAgent {
    /// Policy network (actor)
    pub policy: NeuralNetwork,
    /// Value network (critic)
    pub value: NeuralNetwork,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda parameter
    pub gae_lambda: f32,
    /// Clipping parameter for PPO objective
    pub clip_param: f32,
    /// Number of epochs for each update
    pub ppo_epochs: usize,
    /// Entropy coefficient
    pub entropy_coeff: f32,
    /// Value function coefficient
    pub value_coeff: f32,
    /// Maximum gradient norm
    pub max_grad_norm: Option<f32>,
    /// Random number generator
    #[serde(skip)]
    pub rng: ThreadRng,
}

/// Rollout buffer for storing trajectories
#[derive(Clone, Debug)]
pub struct PPORolloutBuffer {
    pub states: Vec<Array1<f32>>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub log_probs: Vec<f32>,
    pub dones: Vec<bool>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

impl PPORolloutBuffer {
    pub fn new() -> Self {
        PPORolloutBuffer {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            dones: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }
    
    pub fn add(
        &mut self,
        state: Array1<f32>,
        action: usize,
        reward: f32,
        value: f32,
        log_prob: f32,
        done: bool,
    ) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
        self.dones.push(done);
    }
    
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.dones.clear();
        self.advantages.clear();
        self.returns.clear();
    }
    
    pub fn len(&self) -> usize {
        self.states.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl PPOAgent {
    /// Create a new PPO agent
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        optimizer: OptimizerWrapper,
        gamma: f32,
        gae_lambda: f32,
        clip_param: f32,
        ppo_epochs: usize,
    ) -> Self {
        // Build policy network
        let mut policy_sizes = vec![state_size];
        policy_sizes.extend_from_slice(hidden_sizes);
        policy_sizes.push(action_size);
        
        let policy_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let policy = NeuralNetwork::new(&policy_sizes, &policy_activations, optimizer.clone());
        
        // Build value network
        let mut value_sizes = vec![state_size];
        value_sizes.extend_from_slice(hidden_sizes);
        value_sizes.push(1);
        
        let value_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let value = NeuralNetwork::new(&value_sizes, &value_activations, optimizer);
        
        PPOAgent {
            policy,
            value,
            gamma,
            gae_lambda,
            clip_param,
            ppo_epochs,
            entropy_coeff: 0.01,
            value_coeff: 0.5,
            max_grad_norm: Some(0.5),
            rng: thread_rng(),
        }
    }
    
    /// Select action using current policy
    pub fn act(&mut self, state: ArrayView1<f32>) -> Result<(usize, f32, f32)> {
        let logits = self.policy.forward(state);
        let probs = softmax(&logits);
        let value = self.value.forward(state)[0];
        
        // Sample action
        let action = self.sample_action(&probs)?;
        let log_prob = probs[action].ln();
        
        Ok((action, log_prob, value))
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
        
        Ok(probs.len() - 1)
    }
    
    /// Compute GAE advantages
    pub fn compute_gae(&self, buffer: &mut PPORolloutBuffer, last_value: f32) {
        let n = buffer.rewards.len();
        buffer.advantages = vec![0.0; n];
        buffer.returns = vec![0.0; n];
        
        let mut gae = 0.0;
        
        for i in (0..n).rev() {
            let next_value = if i == n - 1 {
                if buffer.dones[i] { 0.0 } else { last_value }
            } else {
                if buffer.dones[i] { 0.0 } else { buffer.values[i + 1] }
            };
            
            let delta = buffer.rewards[i] + self.gamma * next_value - buffer.values[i];
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - buffer.dones[i] as i32 as f32);
            
            buffer.advantages[i] = gae;
            buffer.returns[i] = buffer.advantages[i] + buffer.values[i];
        }
        
        // Normalize advantages
        let mean = buffer.advantages.iter().sum::<f32>() / n as f32;
        let variance = buffer.advantages.iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f32>() / n as f32;
        let std = variance.sqrt() + 1e-8;
        
        for adv in buffer.advantages.iter_mut() {
            *adv = (*adv - mean) / std;
        }
    }
    
    /// Update policy using PPO objective
    pub fn update(
        &mut self,
        buffer: &PPORolloutBuffer,
        _learning_rate: f32,
    ) -> Result<(f32, f32, f32)> {
        if buffer.is_empty() {
            return Err(AthenaError::EmptyBuffer("Empty rollout buffer".to_string()));
        }
        
        let batch_size = buffer.len();
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        
        // Convert to batch arrays
        let states = stack_arrays(buffer.states.iter().map(|s| s.view()).collect());
        
        // Multiple epochs of updates
        for _ in 0..self.ppo_epochs {
            // Forward pass
            let policy_outputs = self.policy.forward_batch(states.view());
            let value_outputs = self.value.forward_batch(states.view());
            
            let mut policy_loss = 0.0;
            let mut value_loss = 0.0;
            let mut entropy = 0.0;
            
            for i in 0..batch_size {
                let logits = policy_outputs.row(i).to_owned();
                let probs = softmax(&logits);
                let new_log_prob = probs[buffer.actions[i]].ln();
                
                // PPO clipped objective
                let ratio = (new_log_prob - buffer.log_probs[i]).exp();
                let surr1 = ratio * buffer.advantages[i];
                let surr2 = ratio.clamp(
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * buffer.advantages[i];
                
                policy_loss -= surr1.min(surr2);
                
                // Value loss
                let value_pred = value_outputs[[i, 0]];
                value_loss += (value_pred - buffer.returns[i]).powi(2);
                
                // Entropy
                for &p in probs.iter() {
                    if p > 1e-8 {
                        entropy -= p * p.ln();
                    }
                }
            }
            
            policy_loss /= batch_size as f32;
            value_loss /= batch_size as f32;
            entropy /= batch_size as f32;
            
            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            total_entropy += entropy;
            
            // Combined loss
            let _total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy;
            
            // Update networks (simplified - would need proper gradient computation)
            // In practice, you'd compute gradients and update using the optimizer
        }
        
        Ok((
            total_policy_loss / self.ppo_epochs as f32,
            total_value_loss / self.ppo_epochs as f32,
            total_entropy / self.ppo_epochs as f32,
        ))
    }
    
    /// Save agent to disk
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::serialize(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load agent from disk
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let agent = bincode::deserialize(&data)?;
        Ok(agent)
    }
}

/// Softmax function
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp = exp_logits.sum();
    exp_logits / sum_exp
}

/// Stack 1D arrays into 2D array
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

/// Builder for PPOAgent
pub struct PPOBuilder {
    state_size: usize,
    action_size: usize,
    hidden_sizes: Vec<usize>,
    optimizer: Option<OptimizerWrapper>,
    gamma: f32,
    gae_lambda: f32,
    clip_param: f32,
    ppo_epochs: usize,
    entropy_coeff: f32,
    value_coeff: f32,
}

impl PPOBuilder {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        PPOBuilder {
            state_size,
            action_size,
            hidden_sizes: vec![64, 64],
            optimizer: None,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_param: 0.2,
            ppo_epochs: 10,
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
    
    pub fn gae_lambda(mut self, lambda: f32) -> Self {
        self.gae_lambda = lambda;
        self
    }
    
    pub fn clip_param(mut self, clip: f32) -> Self {
        self.clip_param = clip;
        self
    }
    
    pub fn ppo_epochs(mut self, epochs: usize) -> Self {
        self.ppo_epochs = epochs;
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
    
    pub fn build(self) -> Result<PPOAgent> {
        let optimizer = self.optimizer
            .ok_or_else(|| AthenaError::InvalidParameter {
            name: "optimizer".to_string(),
            reason: "Optimizer not specified".to_string(),
        })?;
        
        let mut agent = PPOAgent::new(
            self.state_size,
            self.action_size,
            &self.hidden_sizes,
            optimizer,
            self.gamma,
            self.gae_lambda,
            self.clip_param,
            self.ppo_epochs,
        );
        
        agent.entropy_coeff = self.entropy_coeff;
        agent.value_coeff = self.value_coeff;
        
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    
    #[test]
    fn test_ppo_creation() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = PPOAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.95, 0.2, 10);
        
        assert_eq!(agent.gamma, 0.99);
        assert_eq!(agent.gae_lambda, 0.95);
        assert_eq!(agent.clip_param, 0.2);
        assert_eq!(agent.ppo_epochs, 10);
    }
    
    #[test]
    fn test_ppo_builder() {
        let agent = PPOBuilder::new(4, 2)
            .hidden_sizes(vec![64, 64])
            .gamma(0.95)
            .clip_param(0.3)
            .optimizer(OptimizerWrapper::SGD(SGD::new()))
            .build()
            .unwrap();
        
        assert_eq!(agent.gamma, 0.95);
        assert_eq!(agent.clip_param, 0.3);
    }
    
    #[test]
    fn test_rollout_buffer() {
        let mut buffer = PPORolloutBuffer::new();
        assert!(buffer.is_empty());
        
        buffer.add(
            Array1::zeros(4),
            0,
            1.0,
            0.5,
            -0.693,
            false,
        );
        
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        
        buffer.clear();
        assert!(buffer.is_empty());
    }
}