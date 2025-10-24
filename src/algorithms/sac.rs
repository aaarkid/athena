use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Soft Actor-Critic (SAC) Agent for continuous action spaces
/// 
/// SAC is an off-policy actor-critic algorithm that maximizes both
/// expected return and entropy for improved exploration.
#[derive(Serialize, Deserialize, Clone)]
pub struct SACAgent {
    /// Actor network (policy)
    pub actor: NeuralNetwork,
    /// First Q-network
    pub q1: NeuralNetwork,
    /// Second Q-network (for twin delayed)
    pub q2: NeuralNetwork,
    /// Target Q-network 1
    pub q1_target: NeuralNetwork,
    /// Target Q-network 2
    pub q2_target: NeuralNetwork,
    /// Temperature parameter (can be learned)
    pub alpha: f32,
    /// Whether to automatically tune temperature
    pub auto_alpha: bool,
    /// Target entropy for automatic temperature tuning
    pub target_entropy: f32,
    /// Log alpha for automatic tuning
    pub log_alpha: f32,
    /// Discount factor
    pub gamma: f32,
    /// Soft update coefficient
    pub tau: f32,
    /// Random number generator
    #[serde(skip)]
    pub rng: ThreadRng,
}

/// Experience for SAC (continuous actions)
#[derive(Clone, Debug)]
pub struct SACExperience {
    pub state: Array1<f32>,
    pub action: Array1<f32>,
    pub reward: f32,
    pub next_state: Array1<f32>,
    pub done: bool,
}

impl SACAgent {
    /// Create a new SAC agent
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        optimizer: OptimizerWrapper,
        gamma: f32,
        tau: f32,
        alpha: f32,
        auto_alpha: bool,
    ) -> Self {
        // Actor network outputs mean and log_std for each action dimension
        let mut actor_sizes = vec![state_size];
        actor_sizes.extend_from_slice(hidden_sizes);
        actor_sizes.push(action_size * 2); // mean and log_std
        
        let actor_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let actor = NeuralNetwork::new(&actor_sizes, &actor_activations, optimizer.clone());
        
        // Q-networks take state and action as input
        let mut q_sizes = vec![state_size + action_size];
        q_sizes.extend_from_slice(hidden_sizes);
        q_sizes.push(1);
        
        let q_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let q1 = NeuralNetwork::new(&q_sizes, &q_activations, optimizer.clone());
        let q2 = NeuralNetwork::new(&q_sizes, &q_activations, optimizer.clone());
        let q1_target = q1.clone();
        let q2_target = q2.clone();
        
        let target_entropy = -(action_size as f32);
        
        SACAgent {
            actor,
            q1,
            q2,
            q1_target,
            q2_target,
            alpha,
            auto_alpha,
            target_entropy,
            log_alpha: alpha.ln(),
            gamma,
            tau,
            rng: thread_rng(),
        }
    }
    
    /// Select action using current policy
    pub fn act(&mut self, state: ArrayView1<f32>, deterministic: bool) -> Result<Array1<f32>> {
        let output = self.actor.forward(state);
        let action_size = output.len() / 2;
        
        let mean = output.slice(ndarray::s![..action_size]).to_owned();
        let log_std = output.slice(ndarray::s![action_size..]).to_owned();
        
        if deterministic {
            // Return mean action
            Ok(mean.mapv(|x| x.tanh()))
        } else {
            // Sample from Gaussian and apply tanh squashing
            let std = log_std.mapv(|x| x.exp());
            let mut action = Array1::zeros(action_size);
            
            for i in 0..action_size {
                let normal = Normal::new(mean[i], std[i])
                    .map_err(|e| AthenaError::NumericalError(e.to_string()))?;
                let sample: f32 = self.rng.sample(normal);
                action[i] = sample.tanh();
            }
            
            Ok(action)
        }
    }
    
    /// Compute log probability of action under current policy
    fn log_prob(&mut self, state: ArrayView1<f32>, action: ArrayView1<f32>) -> Result<(Array1<f32>, f32)> {
        let output = self.actor.forward(state);
        let action_size = output.len() / 2;
        
        let mean = output.slice(ndarray::s![..action_size]).to_owned();
        let log_std = output.slice(ndarray::s![action_size..]).to_owned();
        let std = log_std.mapv(|x| x.exp());
        
        // Compute log probability with tanh correction
        let mut log_prob = 0.0;
        let mut sampled_action = Array1::zeros(action_size);
        
        for i in 0..action_size {
            let _normal = Normal::new(mean[i], std[i])
                .map_err(|e| AthenaError::NumericalError(e.to_string()))?;
            
            // Inverse tanh to get original sample
            let atanh_action = 0.5 * ((1.0 + action[i]) / (1.0 - action[i])).ln();
            sampled_action[i] = atanh_action;
            
            // Log probability with Jacobian correction for tanh
            let normal_log_prob = -0.5 * ((atanh_action - mean[i]) / std[i]).powi(2) 
                - log_std[i] - 0.5 * (2.0 * std::f32::consts::PI).ln();
            let tanh_correction = (1.0 - action[i].powi(2)).ln();
            
            log_prob += normal_log_prob - tanh_correction;
        }
        
        Ok((sampled_action, log_prob))
    }
    
    /// Update networks using SAC algorithm
    pub fn update(
        &mut self,
        batch: &[SACExperience],
        learning_rate: f32,
    ) -> Result<(f32, f32, f32)> {
        if batch.is_empty() {
            return Err(AthenaError::EmptyBuffer("Empty batch".to_string()));
        }
        
        let batch_size = batch.len();
        
        // Prepare batch data
        let states = stack_arrays(batch.iter().map(|e| e.state.view()).collect());
        let actions = stack_arrays(batch.iter().map(|e| e.action.view()).collect());
        let rewards = batch.iter().map(|e| e.reward).collect::<Vec<_>>();
        let next_states = stack_arrays(batch.iter().map(|e| e.next_state.view()).collect());
        let dones = batch.iter().map(|e| e.done).collect::<Vec<_>>();
        
        // Update Q-networks
        let mut q_loss = 0.0;
        
        for i in 0..batch_size {
            let state = states.row(i);
            let action = actions.row(i);
            let next_state = next_states.row(i);
            
            // Sample next action from policy
            let next_action = self.act(next_state, false)?;
            let (_, next_log_prob) = self.log_prob(next_state, next_action.view())?;
            
            // Compute target Q-value
            let sa_concat = concatenate(next_state, next_action.view());
            let target_q1 = self.q1_target.forward(sa_concat.view())[0];
            let target_q2 = self.q2_target.forward(sa_concat.view())[0];
            let target_q = target_q1.min(target_q2) - self.alpha * next_log_prob;
            
            let target_value = rewards[i] + self.gamma * target_q * (1.0 - dones[i] as i32 as f32);
            
            // Current Q-values
            let sa_concat = concatenate(state, action);
            let q1_value = self.q1.forward(sa_concat.view())[0];
            let q2_value = self.q2.forward(sa_concat.view())[0];
            
            q_loss += (q1_value - target_value).powi(2) + (q2_value - target_value).powi(2);
        }
        
        q_loss /= batch_size as f32;
        
        // Update policy
        let mut policy_loss = 0.0;
        let mut alpha_loss = 0.0;
        
        for i in 0..batch_size {
            let state = states.row(i);
            
            // Sample action from current policy
            let action = self.act(state, false)?;
            let (_, log_prob) = self.log_prob(state, action.view())?;
            
            // Compute Q-values for sampled action
            let sa_concat = concatenate(state, action.view());
            let q1_value = self.q1.forward(sa_concat.view())[0];
            let q2_value = self.q2.forward(sa_concat.view())[0];
            let q_value = q1_value.min(q2_value);
            
            // Policy loss
            policy_loss += self.alpha * log_prob - q_value;
            
            // Temperature loss (if auto-tuning)
            if self.auto_alpha {
                alpha_loss += -self.log_alpha * (log_prob + self.target_entropy);
            }
        }
        
        policy_loss /= batch_size as f32;
        alpha_loss /= batch_size as f32;
        
        // Update temperature
        if self.auto_alpha {
            self.log_alpha -= learning_rate * alpha_loss;
            self.alpha = self.log_alpha.exp();
        }
        
        // Soft update target networks
        self.soft_update();
        
        Ok((q_loss, policy_loss, alpha_loss))
    }
    
    /// Soft update target networks
    fn soft_update(&mut self) {
        // Update Q1 target
        for (target, source) in self.q1_target.layers.iter_mut().zip(self.q1.layers.iter()) {
            target.weights = &target.weights * (1.0 - self.tau) + &source.weights * self.tau;
            target.biases = &target.biases * (1.0 - self.tau) + &source.biases * self.tau;
        }
        
        // Update Q2 target
        for (target, source) in self.q2_target.layers.iter_mut().zip(self.q2.layers.iter()) {
            target.weights = &target.weights * (1.0 - self.tau) + &source.weights * self.tau;
            target.biases = &target.biases * (1.0 - self.tau) + &source.biases * self.tau;
        }
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

/// Concatenate state and action arrays
fn concatenate(state: ArrayView1<f32>, action: ArrayView1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(state.len() + action.len());
    result.slice_mut(ndarray::s![..state.len()]).assign(&state);
    result.slice_mut(ndarray::s![state.len()..]).assign(&action);
    result
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

/// Builder for SACAgent
pub struct SACBuilder {
    state_size: usize,
    action_size: usize,
    hidden_sizes: Vec<usize>,
    optimizer: Option<OptimizerWrapper>,
    gamma: f32,
    tau: f32,
    alpha: f32,
    auto_alpha: bool,
}

impl SACBuilder {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        SACBuilder {
            state_size,
            action_size,
            hidden_sizes: vec![256, 256],
            optimizer: None,
            gamma: 0.99,
            tau: 0.005,
            alpha: 0.2,
            auto_alpha: true,
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
    
    pub fn tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }
    
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
    
    pub fn auto_alpha(mut self, auto: bool) -> Self {
        self.auto_alpha = auto;
        self
    }
    
    pub fn build(self) -> Result<SACAgent> {
        let optimizer = self.optimizer
            .ok_or_else(|| AthenaError::InvalidParameter {
            name: "optimizer".to_string(),
            reason: "Optimizer not specified".to_string(),
        })?;
        
        Ok(SACAgent::new(
            self.state_size,
            self.action_size,
            &self.hidden_sizes,
            optimizer,
            self.gamma,
            self.tau,
            self.alpha,
            self.auto_alpha,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    
    #[test]
    fn test_sac_creation() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = SACAgent::new(4, 2, &[64, 64], optimizer, 0.99, 0.005, 0.2, true);
        
        assert_eq!(agent.gamma, 0.99);
        assert_eq!(agent.tau, 0.005);
        assert_eq!(agent.alpha, 0.2);
        assert!(agent.auto_alpha);
    }
    
    #[test]
    fn test_sac_builder() {
        let agent = SACBuilder::new(4, 2)
            .hidden_sizes(vec![128, 128])
            .gamma(0.95)
            .tau(0.01)
            .optimizer(OptimizerWrapper::SGD(SGD::new()))
            .build()
            .unwrap();
        
        assert_eq!(agent.gamma, 0.95);
        assert_eq!(agent.tau, 0.01);
    }
    
    #[test]
    fn test_concatenate() {
        let state = Array1::from_vec(vec![1.0, 2.0]);
        let action = Array1::from_vec(vec![3.0, 4.0]);
        let result = concatenate(state.view(), action.view());
        
        assert_eq!(result, Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
    }
}