use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use rand::rngs::StdRng;
use crate::rng::{default_rng, seeded_rng};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Proximal Policy Optimization (PPO) Agent
///
/// PPO is a policy gradient method that uses a clipped surrogate objective
/// to ensure stable policy updates. It's one of the most popular and robust
/// RL algorithms, used in training ChatGPT and many game-playing agents.
///
/// # Algorithm Overview
///
/// PPO improves on vanilla policy gradient by:
/// 1. **Clipped objective**: Limits policy updates to prevent too-large changes
/// 2. **GAE**: Uses Generalized Advantage Estimation for lower variance
/// 3. **Multiple epochs**: Reuses collected data for better sample efficiency
///
/// # Example
///
/// ```rust,no_run
/// use athena::algorithms::{PPOAgent, PPOBuilder, PPORolloutBuffer};
/// use ndarray::Array1;
///
/// // Create PPO agent with default SGD optimizer
/// let mut agent = PPOBuilder::new(4, 2)
///     .hidden_sizes(vec![64, 64])
///     .clip_param(0.2)
///     .build()
///     .unwrap();
///
/// // Collect rollout
/// let mut buffer = PPORolloutBuffer::new();
/// // ... collect experience ...
///
/// // Update policy
/// // agent.update(&buffer, 0.0003).unwrap();
/// ```
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
    /// Number of actions
    action_size: usize,
    /// Random number generator
    #[serde(skip, default = "crate::rng::default_rng")]
    pub rng: StdRng,
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

impl Default for PPORolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl PPOAgent {
    /// Reseed this agent's generator so its randomness repeats.
    ///
    /// Two agents given the same seed and the same inputs follow the same sequence of
    /// sampled actions and exploration noise. Weight initialization is separate; fix
    /// that too when a whole run has to reproduce.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = seeded_rng(seed);
    }

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
            action_size,
            rng: default_rng(),
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

    /// Get action probabilities
    pub fn get_action_probs(&mut self, state: ArrayView1<f32>) -> Array1<f32> {
        let logits = self.policy.forward(state);
        softmax(&logits)
    }

    /// Get value estimate for a state
    pub fn get_value(&mut self, state: ArrayView1<f32>) -> f32 {
        self.value.forward(state)[0]
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
            } else if buffer.dones[i] {
                0.0
            } else {
                buffer.values[i + 1]
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
    ///
    /// This implements the PPO-Clip algorithm:
    /// - L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
    /// - r(θ) = π(a|s) / π_old(a|s)
    pub fn update(
        &mut self,
        buffer: &PPORolloutBuffer,
        learning_rate: f32,
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
        for _epoch in 0..self.ppo_epochs {
            // Forward pass
            let policy_outputs = self.policy.forward_batch(states.view());
            let value_outputs = self.value.forward_batch(states.view());

            let mut policy_loss = 0.0;
            let mut value_loss = 0.0;
            let mut entropy = 0.0;

            // Create gradient arrays
            let mut policy_gradients = Array2::zeros((batch_size, self.action_size));
            let mut value_targets = Array2::zeros((batch_size, 1));

            for i in 0..batch_size {
                let logits = policy_outputs.row(i).to_owned();
                let probs = softmax(&logits);
                let new_log_prob = probs[buffer.actions[i]].ln();

                // PPO clipped objective
                let ratio = (new_log_prob - buffer.log_probs[i]).exp();
                let clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param);
                let surr1 = ratio * buffer.advantages[i];
                let surr2 = clipped_ratio * buffer.advantages[i];
                let min_surr = surr1.min(surr2);

                policy_loss -= min_surr;

                // Value loss
                let value_pred = value_outputs[[i, 0]];
                value_loss += (value_pred - buffer.returns[i]).powi(2);
                value_targets[[i, 0]] = buffer.returns[i];

                // Entropy of this row, needed both for reporting and for the entropy
                // gradient below
                let mut row_entropy = 0.0;
                for &p in probs.iter() {
                    if p > 1e-8 {
                        row_entropy -= p * p.ln();
                    }
                }
                entropy += row_entropy;

                // Compute PPO policy gradient with clipping
                // Gradient of min(r*A, clip(r)*A) w.r.t. logits
                // When not clipped (surr1 < surr2), gradient flows through ratio
                // When clipped, gradient is zero (clipped ratio doesn't depend on current params)
                let use_clipped = surr2 < surr1;
                let grad_weight = if use_clipped { 0.0 } else { buffer.advantages[i] };

                // Gradient of ratio w.r.t. log_prob is ratio itself (d/dx e^x = e^x)
                // Gradient of log_prob w.r.t. logits is (one_hot - softmax)
                for j in 0..self.action_size {
                    let one_hot = if j == buffer.actions[i] { 1.0 } else { 0.0 };
                    // Policy gradient (negated for descent)
                    let pg = -grad_weight * ratio * (one_hot - probs[j]);
                    // Entropy gradient. For H = -sum p ln p over softmax logits,
                    // dH/dz_j = -p_j * (ln p_j + H), so minimizing -coeff * H gives
                    // coeff * p_j * (ln p_j + H). The row entropy is what couples the
                    // actions together; a constant 1 there would not sum to zero.
                    let eg = if probs[j] > 1e-8 {
                        self.entropy_coeff * probs[j] * (probs[j].ln() + row_entropy)
                    } else {
                        0.0
                    };
                    policy_gradients[[i, j]] = pg + eg;
                }
            }

            policy_loss /= batch_size as f32;
            value_loss /= batch_size as f32;
            entropy /= batch_size as f32;

            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            total_entropy += entropy;

            // Apply gradients to the networks.
            //
            // The policy and value networks are separate here, so there is no joint loss
            // to weight. value_coeff scales the value network's step instead, which is
            // the same thing the coefficient does in a shared-trunk implementation.
            let value_lr = learning_rate * self.value_coeff;

            match self.max_grad_norm {
                Some(max_norm) => {
                    self.value.train_minibatch_clipped(
                        states.view(),
                        value_targets.view(),
                        value_lr,
                        max_norm,
                    );
                    self.policy.train_policy_gradient_clipped(
                        states.view(),
                        policy_gradients.view(),
                        learning_rate,
                        max_norm,
                    );
                }
                None => {
                    self.value.train_minibatch(states.view(), value_targets.view(), value_lr);
                    self.policy.train_policy_gradient(
                        states.view(),
                        policy_gradients.view(),
                        learning_rate,
                    );
                }
            }
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
        let mut agent: Self = bincode::deserialize(&data)?;
        agent.rng = default_rng();
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

impl Default for PPOBuilder {
    fn default() -> Self {
        Self::new(4, 2)
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

    #[test]
    fn test_ppo_act() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = PPOAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.95, 0.2, 10);

        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (action, log_prob, value) = agent.act(state.view()).unwrap();

        assert!(action < 2);
        assert!(log_prob <= 0.0); // Log probs are always negative
        assert!(value.is_finite());
    }

    #[test]
    fn test_ppo_gae() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = PPOAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.95, 0.2, 10);

        let mut buffer = PPORolloutBuffer::new();

        // Add some experience
        for i in 0..10 {
            buffer.add(
                Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3, 0.4]),
                i % 2,
                1.0,
                0.5,
                -0.5,
                i == 9,
            );
        }

        agent.compute_gae(&mut buffer, 0.0);

        // Check that advantages and returns are computed
        assert_eq!(buffer.advantages.len(), 10);
        assert_eq!(buffer.returns.len(), 10);

        // Check that advantages are normalized (mean ~0, std ~1)
        let mean: f32 = buffer.advantages.iter().sum::<f32>() / 10.0;
        assert!(mean.abs() < 0.1, "Mean should be near 0, got {}", mean);
    }

    #[test]
    fn test_ppo_update() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = PPOAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.95, 0.2, 2);

        let mut buffer = PPORolloutBuffer::new();

        // Add experience
        for i in 0..10 {
            buffer.add(
                Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3, 0.4]),
                i % 2,
                1.0,
                0.5,
                -0.5,
                i == 9,
            );
        }

        agent.compute_gae(&mut buffer, 0.0);

        // Update should work without error
        let result = agent.update(&buffer, 0.001);
        assert!(result.is_ok());

        let (policy_loss, value_loss, entropy) = result.unwrap();
        assert!(policy_loss.is_finite());
        assert!(value_loss.is_finite());
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_ppo_value_learning() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = PPOAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.95, 0.2, 3);

        let state = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let initial_value = agent.get_value(state.view());

        // Ten steps of reward 10, the last one terminal. Ending the episode matters:
        // without it the return bootstraps off the value being trained, the target
        // compounds every iteration, and the run can diverge instead of converging.
        // Terminating makes the target a fixed 10 * sum(gamma^k, k=0..9), about 95.6.
        let expected_return: f32 = (0..10).map(|k| 0.99f32.powi(k) * 10.0).sum();

        for _ in 0..300 {
            let mut buffer = PPORolloutBuffer::new();
            for step in 0..10 {
                buffer.add(
                    state.clone(),
                    0,
                    10.0,
                    agent.get_value(state.view()),
                    -0.5,
                    step == 9,
                );
            }
            agent.compute_gae(&mut buffer, 0.0);
            agent.update(&buffer, 0.05).unwrap();
        }

        let final_value = agent.get_value(state.view());
        assert!(final_value > initial_value,
                "Value should increase with high rewards. Initial: {}, Final: {}",
                initial_value, final_value);
        assert!(final_value > expected_return * 0.5 && final_value < expected_return * 2.0,
                "Value should approach the return of {}, got {}",
                expected_return, final_value);
    }
}
