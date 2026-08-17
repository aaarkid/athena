use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use rand::rngs::StdRng;
use crate::rng::{default_rng, seeded_rng};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent
///
/// TD3 improves upon DDPG by using twin Q-networks, delayed policy updates,
/// and target policy smoothing to address overestimation bias. It's excellent
/// for continuous control tasks requiring precise actions.
///
/// # Key Features
///
/// 1. **Twin Critics**: Two Q-networks to reduce overestimation
/// 2. **Delayed Policy Updates**: Actor updates less frequently than critics
/// 3. **Target Policy Smoothing**: Adds noise to target actions for regularization
///
/// # Example
///
/// ```rust,no_run
/// use athena::algorithms::{TD3Agent, TD3Builder, TD3Experience};
///
/// // Create TD3 agent with default SGD optimizer
/// let agent = TD3Builder::new(4, 2)
///     .hidden_sizes(vec![256, 256])
///     .action_bounds(-1.0, 1.0)
///     .policy_delay(2)
///     .build()
///     .unwrap();
/// ```
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
    /// Action dimension
    action_size: usize,
    /// Update counter
    update_counter: usize,
    /// Random number generator
    #[serde(skip, default = "crate::rng::default_rng")]
    pub rng: StdRng,
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
    /// Reseed this agent's generator so its randomness repeats.
    ///
    /// Two agents given the same seed and the same inputs follow the same sequence of
    /// sampled actions and exploration noise. Weight initialization is separate; fix
    /// that too when a whole run has to reproduce.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = seeded_rng(seed);
    }

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
        if policy_delay == 0 {
            panic!("policy_delay must be at least 1; the update counter is taken modulo it");
        }

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
            action_size,
            update_counter: 0,
            rng: default_rng(),
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

    /// Get Q-values for a state-action pair
    pub fn get_q_values(&mut self, state: ArrayView1<f32>, action: ArrayView1<f32>) -> (f32, f32) {
        let sa_concat = concatenate(state, action);
        let q1 = self.critic1.forward(sa_concat.view())[0];
        let q2 = self.critic2.forward(sa_concat.view())[0];
        (q1, q2)
    }

    /// Update networks using TD3 algorithm
    pub fn update(
        &mut self,
        batch: &[TD3Experience],
        actor_lr: f32,
        critic_lr: f32,
    ) -> Result<(f32, Option<f32>)> {
        if batch.is_empty() {
            return Err(AthenaError::EmptyBuffer("Empty batch".to_string()));
        }

        let batch_size = batch.len();

        // Prepare batch data
        let states = stack_arrays(batch.iter().map(|e| e.state.view()).collect());
        let actions = stack_arrays(batch.iter().map(|e| e.action.view()).collect());
        let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
        let next_states = stack_arrays(batch.iter().map(|e| e.next_state.view()).collect());
        let dones: Vec<bool> = batch.iter().map(|e| e.done).collect();

        // Critic update: minimize the Bellman error

        // Compute Q targets
        let mut critic1_targets = Array2::zeros((batch_size, 1));
        let mut critic2_targets = Array2::zeros((batch_size, 1));
        let mut critic_inputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let state = states.row(i);
            let action = actions.row(i);
            let next_state = next_states.row(i);

            // Compute target actions with smoothing
            let mut next_action = self.actor_target.forward(next_state);

            // Add clipped noise for smoothing (regularization)
            let normal = Normal::new(0.0, self.policy_noise)
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

            // Compute target Q-values using target networks (twin Q-learning)
            let sa_concat_next = concatenate(next_state, next_action.view());
            let target_q1 = self.critic1_target.forward(sa_concat_next.view())[0];
            let target_q2 = self.critic2_target.forward(sa_concat_next.view())[0];
            let target_q = target_q1.min(target_q2); // Use minimum to avoid overestimation

            let target_value = rewards[i] + self.gamma * target_q * (1.0 - dones[i] as i32 as f32);

            critic1_targets[[i, 0]] = target_value;
            critic2_targets[[i, 0]] = target_value;

            // Store input for critic training
            let sa_concat = concatenate(state, action);
            critic_inputs.push(sa_concat);
        }

        // Convert critic inputs to batch array
        let critic_input_batch = stack_arrays(critic_inputs.iter().map(|a| a.view()).collect());

        // Train critic1 network
        self.critic1.train_minibatch(critic_input_batch.view(), critic1_targets.view(), critic_lr);

        // Train critic2 network
        self.critic2.train_minibatch(critic_input_batch.view(), critic2_targets.view(), critic_lr);

        // Compute critic loss for reporting
        let critic1_outputs = self.critic1.forward_batch(critic_input_batch.view());
        let critic2_outputs = self.critic2.forward_batch(critic_input_batch.view());
        let critic_loss = (&critic1_outputs - &critic1_targets).mapv(|x| x * x).mean().unwrap_or(0.0)
                        + (&critic2_outputs - &critic2_targets).mapv(|x| x * x).mean().unwrap_or(0.0);

        // Actor update, delayed
        //
        // TD3 objective: maximize Q(s, μ(s))
        //
        // We use a simple approach: move the actor output toward
        // actions that result in high Q-values.

        self.update_counter += 1;
        let mut actor_loss = None;

        if self.update_counter % self.policy_delay == 0 {
            let actor_outputs = self.actor.forward_batch(states.view());
            let mut actor_targets = actor_outputs.clone();

            let mut policy_loss = 0.0;

            for i in 0..batch_size {
                let state = states.row(i);

                // Get current action from actor (in [-1, 1] due to tanh)
                let action = actor_outputs.row(i).to_owned();

                // Scale to action bounds
                let scaled_action = action.mapv(|a| {
                    (a + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
                });

                // Compute Q-value for current action
                let sa_concat = concatenate(state, scaled_action.view());
                let q_value = self.critic1.forward(sa_concat.view())[0];
                policy_loss -= q_value;

                // Normalize Q-value as learning signal
                let q_signal = (q_value / 10.0).clamp(-1.0, 1.0);

                // For each action dimension, search for better actions
                for j in 0..self.action_size {
                    // Try action slightly higher and lower
                    let delta = 0.1;
                    let mut action_plus = scaled_action.clone();
                    let mut action_minus = scaled_action.clone();
                    action_plus[j] = (action_plus[j] + delta).clamp(self.action_low, self.action_high);
                    action_minus[j] = (action_minus[j] - delta).clamp(self.action_low, self.action_high);

                    let sa_plus = concatenate(state, action_plus.view());
                    let sa_minus = concatenate(state, action_minus.view());
                    let q_plus = self.critic1.forward(sa_plus.view())[0];
                    let q_minus = self.critic1.forward(sa_minus.view())[0];

                    // Move toward higher Q direction
                    let direction = if q_plus > q_minus { 0.05 } else { -0.05 };

                    // Update target: current output + direction weighted by Q
                    let target = (action[j] + direction * (1.0 + q_signal.abs())).clamp(-1.0, 1.0);
                    actor_targets[[i, j]] = target;
                }
            }

            policy_loss /= batch_size as f32;
            actor_loss = Some(policy_loss);

            // Train actor toward targets
            self.actor.train_minibatch(states.view(), actor_targets.view(), actor_lr);

            // Soft update ALL target networks
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
        let mut agent: Self = bincode::deserialize(&data)?;
        agent.rng = default_rng();
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

impl Default for TD3Builder {
    fn default() -> Self {
        Self::new(4, 2)
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

    #[test]
    fn test_td3_act() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = TD3Agent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 2, -1.0, 1.0);

        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let action = agent.act(state.view(), false).unwrap();

        assert_eq!(action.len(), 2);
        // Actions should be in [action_low, action_high]
        for &a in action.iter() {
            assert!(a >= -1.0 && a <= 1.0);
        }
    }

    #[test]
    fn test_td3_update() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = TD3Agent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 2, -1.0, 1.0);

        let batch: Vec<TD3Experience> = (0..10).map(|i| {
            TD3Experience {
                state: Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3, 0.4]),
                action: Array1::from_vec(vec![0.5, -0.5]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f32 * 0.1, 0.2, 0.3, 0.4]),
                done: i == 9,
            }
        }).collect();

        // First update (critic only, no actor update due to policy_delay=2)
        let result1 = agent.update(&batch, 0.001, 0.001);
        assert!(result1.is_ok());
        let (critic_loss1, actor_loss1) = result1.unwrap();
        assert!(critic_loss1.is_finite());
        assert!(actor_loss1.is_none()); // Actor not updated yet

        // Second update (both critic and actor)
        let result2 = agent.update(&batch, 0.001, 0.001);
        assert!(result2.is_ok());
        let (critic_loss2, actor_loss2) = result2.unwrap();
        assert!(critic_loss2.is_finite());
        assert!(actor_loss2.is_some()); // Actor updated now
    }

    #[test]
    fn test_td3_q_learning() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = TD3Agent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 1, -1.0, 1.0);

        let state = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let action = Array1::from_vec(vec![0.0, 0.0]);

        let (initial_q1, initial_q2) = agent.get_q_values(state.view(), action.view());

        // Train with high rewards
        for _ in 0..30 {
            let batch: Vec<TD3Experience> = (0..10).map(|_| {
                TD3Experience {
                    state: state.clone(),
                    action: action.clone(),
                    reward: 10.0,  // High reward
                    next_state: state.clone(),
                    done: false,
                }
            }).collect();

            agent.update(&batch, 0.01, 0.01).unwrap();
        }

        let (final_q1, final_q2) = agent.get_q_values(state.view(), action.view());

        // Q-values should increase with high rewards
        assert!(final_q1 > initial_q1 || final_q2 > initial_q2,
                "Q should increase. Initial: ({}, {}), Final: ({}, {})",
                initial_q1, initial_q2, final_q1, final_q2);
    }

    #[test]
    fn test_concatenate() {
        let state = Array1::from_vec(vec![1.0, 2.0]);
        let action = Array1::from_vec(vec![3.0, 4.0]);
        let result = concatenate(state.view(), action.view());

        assert_eq!(result, Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
    }
}
