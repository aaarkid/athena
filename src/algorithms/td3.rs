use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent
/// 
/// TD3 improves upon DDPG by using twin Q-networks, delayed policy updates,
/// and target policy smoothing to address overestimation bias.
#[derive(Serialize, Deserialize, Clone)]
pub struct TD3Agent {
    /// Actor network (deterministic policy)
    pub actor: NeuralNetwork,
    /// Actor target network
    pub actor_target: NeuralNetwork,
    /// First critic network
    pub critic1: NeuralNetwork,
    /// Second critic network
    pub critic2: NeuralNetwork,
    /// First critic target network
    pub critic1_target: NeuralNetwork,
    /// Second critic target network
    pub critic2_target: NeuralNetwork,
    /// Discount factor
    pub gamma: f32,
    /// Soft update coefficient
    pub tau: f32,
    /// Policy update delay
    pub policy_delay: usize,
    /// Target policy smoothing noise
    pub policy_noise: f32,
    /// Noise clipping range
    pub noise_clip: f32,
    /// Exploration noise
    pub exploration_noise: f32,
    /// Action bounds
    pub action_low: f32,
    pub action_high: f32,
    /// Update counter
    update_counter: usize,
    /// Random number generator
    #[serde(skip)]
    pub rng: ThreadRng,
}

/// Experience for TD3 (continuous actions)
#[derive(Clone, Debug)]
pub struct TD3Experience {
    pub state: Array1<f32>,
    pub action: Array1<f32>,
    pub reward: f32,
    pub next_state: Array1<f32>,
    pub done: bool,
}

impl TD3Agent {
    /// Create a new TD3 agent
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        optimizer: OptimizerWrapper,
        gamma: f32,
        tau: f32,
        policy_delay: usize,
        action_low: f32,
        action_high: f32,
    ) -> Self {
        // Build actor network
        let mut actor_sizes = vec![state_size];
        actor_sizes.extend_from_slice(hidden_sizes);
        actor_sizes.push(action_size);
        
        let actor_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Tanh)) // Tanh for bounded actions
            .collect::<Vec<_>>();
        
        let actor = NeuralNetwork::new(&actor_sizes, &actor_activations, optimizer.clone());
        let actor_target = actor.clone();
        
        // Build critic networks (take state and action as input)
        let mut critic_sizes = vec![state_size + action_size];
        critic_sizes.extend_from_slice(hidden_sizes);
        critic_sizes.push(1);
        
        let critic_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();
        
        let critic1 = NeuralNetwork::new(&critic_sizes, &critic_activations, optimizer.clone());
        let critic2 = NeuralNetwork::new(&critic_sizes, &critic_activations, optimizer);
        let critic1_target = critic1.clone();
        let critic2_target = critic2.clone();
        
        TD3Agent {
            actor,
            actor_target,
            critic1,
            critic2,
            critic1_target,
            critic2_target,
            gamma,
            tau,
            policy_delay,
            policy_noise: 0.2,
            noise_clip: 0.5,
            exploration_noise: 0.1,
            action_low,
            action_high,
            update_counter: 0,
            rng: thread_rng(),
        }
    }
    
    /// Select action using current policy
    pub fn act(&mut self, state: ArrayView1<f32>, add_noise: bool) -> Result<Array1<f32>> {
        let mut action = self.actor.forward(state);
        
        // Scale from [-1, 1] to [action_low, action_high]
        action.mapv_inplace(|a| {
            (a + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
        });
        
        if add_noise {
            // Add Gaussian noise for exploration
            let noise_std = self.exploration_noise * (self.action_high - self.action_low);
            let normal = Normal::new(0.0, noise_std)
                .map_err(|e| AthenaError::NumericalError(e.to_string()))?;
            
            for i in 0..action.len() {
                let noise: f32 = self.rng.sample(normal);
                action[i] = (action[i] + noise).clamp(self.action_low, self.action_high);
            }
        }
        
        Ok(action)
    }
    
    /// Update networks using TD3 algorithm
    pub fn update(
        &mut self,
        batch: &[TD3Experience],
        _actor_lr: f32,
        _critic_lr: f32,
    ) -> Result<(f32, Option<f32>)> {
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
        
        // Update critics
        let mut critic_loss = 0.0;
        
        for i in 0..batch_size {
            let state = states.row(i);
            let action = actions.row(i);
            let next_state = next_states.row(i);
            
            // Compute target actions with smoothing
            let mut next_action = self.actor_target.forward(next_state);
            
            // Add clipped noise for smoothing
            let noise_std = self.policy_noise;
            let normal = Normal::new(0.0, noise_std)
                .map_err(|e| AthenaError::NumericalError(e.to_string()))?;
            
            for j in 0..next_action.len() {
                let noise: f32 = self.rng.sample(normal);
                let clipped_noise = noise.clamp(-self.noise_clip, self.noise_clip);
                next_action[j] = (next_action[j] + clipped_noise).clamp(-1.0, 1.0);
            }
            
            // Scale to action bounds
            next_action.mapv_inplace(|a| {
                (a + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
            });
            
            // Compute target Q-values
            let sa_concat = concatenate(next_state, next_action.view());
            let target_q1 = self.critic1_target.forward(sa_concat.view())[0];
            let target_q2 = self.critic2_target.forward(sa_concat.view())[0];
            let target_q = target_q1.min(target_q2);
            
            let target_value = rewards[i] + self.gamma * target_q * (1.0 - dones[i] as i32 as f32);
            
            // Current Q-values
            let sa_concat = concatenate(state, action);
            let q1_value = self.critic1.forward(sa_concat.view())[0];
            let q2_value = self.critic2.forward(sa_concat.view())[0];
            
            critic_loss += (q1_value - target_value).powi(2) + (q2_value - target_value).powi(2);
        }
        
        critic_loss /= batch_size as f32;
        
        // Update actor (delayed)
        let mut actor_loss = None;
        self.update_counter += 1;
        
        if self.update_counter % self.policy_delay == 0 {
            let mut policy_loss = 0.0;
            
            for i in 0..batch_size {
                let state = states.row(i);
                
                // Compute action from current policy
                let mut action = self.actor.forward(state);
                
                // Scale to action bounds
                action.mapv_inplace(|a| {
                    (a + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
                });
                
                // Compute Q-value
                let sa_concat = concatenate(state, action.view());
                let q_value = self.critic1.forward(sa_concat.view())[0];
                
                // Policy loss is negative Q-value (we want to maximize Q)
                policy_loss -= q_value;
            }
            
            policy_loss /= batch_size as f32;
            actor_loss = Some(policy_loss);
            
            // Soft update target networks
            self.soft_update();
        }
        
        Ok((critic_loss, actor_loss))
    }
    
    /// Soft update target networks
    fn soft_update(&mut self) {
        // Update actor target
        for (target, source) in self.actor_target.layers.iter_mut().zip(self.actor.layers.iter()) {
            target.weights = &target.weights * (1.0 - self.tau) + &source.weights * self.tau;
            target.biases = &target.biases * (1.0 - self.tau) + &source.biases * self.tau;
        }
        
        // Update critic1 target
        for (target, source) in self.critic1_target.layers.iter_mut().zip(self.critic1.layers.iter()) {
            target.weights = &target.weights * (1.0 - self.tau) + &source.weights * self.tau;
            target.biases = &target.biases * (1.0 - self.tau) + &source.biases * self.tau;
        }
        
        // Update critic2 target
        for (target, source) in self.critic2_target.layers.iter_mut().zip(self.critic2.layers.iter()) {
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

/// Builder for TD3Agent
pub struct TD3Builder {
    state_size: usize,
    action_size: usize,
    hidden_sizes: Vec<usize>,
    optimizer: Option<OptimizerWrapper>,
    gamma: f32,
    tau: f32,
    policy_delay: usize,
    action_low: f32,
    action_high: f32,
    policy_noise: f32,
    noise_clip: f32,
    exploration_noise: f32,
}

impl TD3Builder {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        TD3Builder {
            state_size,
            action_size,
            hidden_sizes: vec![256, 256],
            optimizer: None,
            gamma: 0.99,
            tau: 0.005,
            policy_delay: 2,
            action_low: -1.0,
            action_high: 1.0,
            policy_noise: 0.2,
            noise_clip: 0.5,
            exploration_noise: 0.1,
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
    
    pub fn policy_delay(mut self, delay: usize) -> Self {
        self.policy_delay = delay;
        self
    }
    
    pub fn action_bounds(mut self, low: f32, high: f32) -> Self {
        self.action_low = low;
        self.action_high = high;
        self
    }
    
    pub fn noise_params(mut self, policy_noise: f32, noise_clip: f32, exploration_noise: f32) -> Self {
        self.policy_noise = policy_noise;
        self.noise_clip = noise_clip;
        self.exploration_noise = exploration_noise;
        self
    }
    
    pub fn build(self) -> Result<TD3Agent> {
        let optimizer = self.optimizer
            .ok_or_else(|| AthenaError::InvalidParameter {
            name: "optimizer".to_string(),
            reason: "Optimizer not specified".to_string(),
        })?;
        
        let mut agent = TD3Agent::new(
            self.state_size,
            self.action_size,
            &self.hidden_sizes,
            optimizer,
            self.gamma,
            self.tau,
            self.policy_delay,
            self.action_low,
            self.action_high,
        );
        
        agent.policy_noise = self.policy_noise;
        agent.noise_clip = self.noise_clip;
        agent.exploration_noise = self.exploration_noise;
        
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    
    #[test]
    fn test_td3_creation() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = TD3Agent::new(4, 2, &[64, 64], optimizer, 0.99, 0.005, 2, -1.0, 1.0);
        
        assert_eq!(agent.gamma, 0.99);
        assert_eq!(agent.tau, 0.005);
        assert_eq!(agent.policy_delay, 2);
        assert_eq!(agent.action_low, -1.0);
        assert_eq!(agent.action_high, 1.0);
    }
    
    #[test]
    fn test_td3_builder() {
        let agent = TD3Builder::new(4, 2)
            .hidden_sizes(vec![128, 128])
            .gamma(0.95)
            .action_bounds(-2.0, 2.0)
            .optimizer(OptimizerWrapper::SGD(SGD::new()))
            .build()
            .unwrap();
        
        assert_eq!(agent.gamma, 0.95);
        assert_eq!(agent.action_low, -2.0);
        assert_eq!(agent.action_high, 2.0);
    }
}