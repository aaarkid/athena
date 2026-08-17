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
/// expected return and entropy for improved exploration. It's particularly
/// effective for continuous control tasks.
///
/// # Algorithm Overview
///
/// SAC optimizes three objectives:
/// 1. **Q-functions**: Minimize Bellman error with entropy-augmented targets
/// 2. **Policy**: Maximize expected Q-value minus entropy cost
/// 3. **Temperature**: (optionally) Auto-tune entropy weight
///
/// # Example
///
/// ```rust,no_run
/// use athena::algorithms::{SACAgent, SACBuilder, SACExperience};
///
/// // Create SAC agent with default SGD optimizer
/// let agent = SACBuilder::new(4, 2)
///     .hidden_sizes(vec![256, 256])
///     .gamma(0.99)
///     .auto_alpha(true)
///     .build()
///     .unwrap();
/// ```
#[derive(Serialize, Deserialize, Clone)]
pub struct SACAgent {
    /// Actor network (policy) - outputs mean and log_std
    pub actor: NeuralNetwork,
    /// First Q-network
    pub q1: NeuralNetwork,
    /// Second Q-network (for twin delayed)
    pub q2: NeuralNetwork,
    /// Target Q-network 1
    pub q1_target: NeuralNetwork,
    /// Target Q-network 2
    pub q2_target: NeuralNetwork,
    /// Temperature parameter (controls exploration)
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
    /// Action dimension
    action_size: usize,
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
        let q2 = NeuralNetwork::new(&q_sizes, &q_activations, optimizer);
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
            action_size,
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
            // Return mean action (squashed)
            Ok(mean.mapv(|x| x.tanh()))
        } else {
            // Sample from Gaussian and apply tanh squashing
            let std = log_std.mapv(|x| x.clamp(-20.0, 2.0).exp());
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
    fn log_prob(&mut self, state: ArrayView1<f32>, action: ArrayView1<f32>) -> Result<f32> {
        let output = self.actor.forward(state);
        let action_size = output.len() / 2;

        let mean = output.slice(ndarray::s![..action_size]).to_owned();
        let log_std = output.slice(ndarray::s![action_size..]).to_owned();
        let std = log_std.mapv(|x| x.clamp(-20.0, 2.0).exp());

        // Compute log probability with tanh correction
        let mut log_prob = 0.0;

        for i in 0..action_size {
            // Inverse tanh to get original sample (with strict clamping for numerical stability)
            let clamped_action = action[i].clamp(-0.9999, 0.9999);
            let ratio = (1.0 + clamped_action) / (1.0 - clamped_action);
            let atanh_action = if ratio > 0.0 { 0.5 * ratio.ln() } else { 0.0 };

            // Check for NaN and use fallback
            if !atanh_action.is_finite() || !mean[i].is_finite() || !std[i].is_finite() {
                continue; // Skip this dimension if numerical issues
            }

            // Gaussian log probability with clamped std to avoid division by zero
            let std_clamped = std[i].max(1e-6);
            let z = (atanh_action - mean[i]) / std_clamped;
            let normal_log_prob = -0.5 * z.powi(2).min(100.0) // Clamp squared term
                - log_std[i].max(-20.0) - 0.5 * (2.0 * std::f32::consts::PI).ln();

            // Jacobian correction for tanh squashing
            let action_sq = action[i].powi(2).min(0.9999);
            let tanh_correction = (1.0 - action_sq + 1e-6).ln();

            log_prob += normal_log_prob - tanh_correction;
        }

        // Clamp final result for stability
        Ok(log_prob.clamp(-100.0, 100.0))
    }

    /// Get Q-value for a state-action pair
    pub fn get_q_value(&mut self, state: ArrayView1<f32>, action: ArrayView1<f32>) -> (f32, f32) {
        let sa_concat = concatenate(state, action);
        let q1 = self.q1.forward(sa_concat.view())[0];
        let q2 = self.q2.forward(sa_concat.view())[0];
        (q1, q2)
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
        let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
        let next_states = stack_arrays(batch.iter().map(|e| e.next_state.view()).collect());
        let dones: Vec<bool> = batch.iter().map(|e| e.done).collect();

        // Q-network update

        // Compute Q targets
        let mut q1_targets = Array2::zeros((batch_size, 1));
        let mut q2_targets = Array2::zeros((batch_size, 1));
        let mut q1_inputs = Vec::with_capacity(batch_size);
        let mut q2_inputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let state = states.row(i);
            let action = actions.row(i);
            let next_state = next_states.row(i);

            // Sample next action from current policy
            let next_action = self.act(next_state, false)?;
            let next_log_prob = self.log_prob(next_state, next_action.view())?;

            // Compute target Q-value using target networks
            let sa_concat_next = concatenate(next_state, next_action.view());
            let target_q1 = self.q1_target.forward(sa_concat_next.view())[0];
            let target_q2 = self.q2_target.forward(sa_concat_next.view())[0];

            // Use minimum Q-value (clipped double Q-learning) with entropy bonus
            let target_q = target_q1.min(target_q2) - self.alpha * next_log_prob;
            let target_value = rewards[i] + self.gamma * target_q * (1.0 - dones[i] as i32 as f32);

            q1_targets[[i, 0]] = target_value;
            q2_targets[[i, 0]] = target_value;

            // Store inputs for Q-network training
            let sa_concat = concatenate(state, action);
            q1_inputs.push(sa_concat.clone());
            q2_inputs.push(sa_concat);
        }

        // Convert Q inputs to batch array
        let q_inputs = stack_arrays(q1_inputs.iter().map(|a| a.view()).collect());

        // Train Q1 network
        self.q1.train_minibatch(q_inputs.view(), q1_targets.view(), learning_rate);

        // Train Q2 network
        self.q2.train_minibatch(q_inputs.view(), q2_targets.view(), learning_rate);

        // Compute Q-loss for reporting
        let q1_outputs = self.q1.forward_batch(q_inputs.view());
        let q2_outputs = self.q2.forward_batch(q_inputs.view());
        let q_loss = (&q1_outputs - &q1_targets).mapv(|x| x * x).mean().unwrap_or(0.0)
                   + (&q2_outputs - &q2_targets).mapv(|x| x * x).mean().unwrap_or(0.0);

        // Policy update
        //
        // SAC objective: maximize E[Q(s,a) - α * log π(a|s)]
        //
        // Instead of complex reparameterization gradients, we use a simpler approach:
        // - Sample action from current policy
        // - Compute Q-value and log_prob
        // - Create target that moves mean toward high-Q actions

        let actor_outputs = self.actor.forward_batch(states.view());
        let mut actor_targets = actor_outputs.clone();

        let mut policy_loss = 0.0;
        let mut mean_log_prob = 0.0;

        for i in 0..batch_size {
            let state = states.row(i);
            let output = actor_outputs.row(i);

            // Get mean and log_std from actor output
            let mean = output.slice(ndarray::s![..self.action_size]).to_owned();
            let log_std = output.slice(ndarray::s![self.action_size..]).to_owned();
            let _std = log_std.mapv(|x| x.clamp(-20.0, 2.0).exp());

            // Sample action
            let action = self.act(state, false)?;

            // Compute log probability
            let log_prob = self.log_prob(state, action.view())?;
            mean_log_prob += log_prob;

            // Get Q-value
            let sa_concat = concatenate(state, action.view());
            let q1_value = self.q1.forward(sa_concat.view())[0];
            let q2_value = self.q2.forward(sa_concat.view())[0];
            let q_value = q1_value.min(q2_value);

            policy_loss += self.alpha * log_prob - q_value;

            // Actor update: move mean toward actions with high Q - α*log_prob
            // The "advantage" here is Q - α*log_prob (higher is better)
            let advantage = q_value - self.alpha * log_prob;

            // Normalize advantage for stability
            let adv_scale = advantage.clamp(-10.0, 10.0) / 10.0;

            for j in 0..self.action_size {
                // Inverse tanh to get pre-squashed action (with numerical stability)
                let clamped_action = action[j].clamp(-0.9999, 0.9999);
                let ratio = (1.0 + clamped_action) / (1.0 - clamped_action);
                let atanh_action = if ratio > 0.0 { 0.5 * ratio.ln() } else { 0.0 };

                // Check for NaN and use current mean as fallback
                let target_mean = if atanh_action.is_finite() && mean[j].is_finite() {
                    // Move mean toward action if advantage is positive
                    (mean[j] + adv_scale * (atanh_action - mean[j]) * 0.1).clamp(-5.0, 5.0)
                } else {
                    mean[j].clamp(-5.0, 5.0)
                };
                actor_targets[[i, j]] = target_mean;

                // Keep log_std, but encourage exploration when advantage is negative
                let target_log_std = if log_std[j].is_finite() {
                    (log_std[j] + if adv_scale < 0.0 { 0.01 } else { -0.01 }).clamp(-20.0, 2.0)
                } else {
                    0.0
                };
                actor_targets[[i, self.action_size + j]] = target_log_std;
            }
        }

        policy_loss /= batch_size as f32;
        mean_log_prob /= batch_size as f32;

        // Train actor with MSE toward targets (simpler and more stable)
        self.actor.train_minibatch(states.view(), actor_targets.view(), learning_rate);

        // Temperature update, when auto_alpha is set

        let mut alpha_loss = 0.0;
        if self.auto_alpha {
            // Temperature loss: log_alpha * (entropy - target_entropy)
            alpha_loss = -self.log_alpha * (mean_log_prob + self.target_entropy);

            // Update log_alpha with gradient descent
            self.log_alpha -= learning_rate * alpha_loss;
            self.alpha = self.log_alpha.exp().clamp(0.01, 1.0);
        }

        // Soft update of the target networks
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
        let mut agent: Self = bincode::deserialize(&data)?;
        agent.rng = thread_rng();
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

impl Default for SACBuilder {
    fn default() -> Self {
        Self::new(4, 2)
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
    fn test_sac_act() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = SACAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 0.2, true);

        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let action = agent.act(state.view(), false).unwrap();

        assert_eq!(action.len(), 2);
        // Actions should be in [-1, 1] due to tanh
        for &a in action.iter() {
            assert!(a >= -1.0 && a <= 1.0);
        }
    }

    #[test]
    fn test_sac_deterministic_act() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = SACAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 0.2, true);

        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        // Deterministic actions should be consistent
        let action1 = agent.act(state.view(), true).unwrap();
        let action2 = agent.act(state.view(), true).unwrap();

        assert_eq!(action1, action2);
    }

    #[test]
    fn test_sac_update() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = SACAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 0.2, false);

        let batch: Vec<SACExperience> = (0..10).map(|i| {
            SACExperience {
                state: Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3, 0.4]),
                action: Array1::from_vec(vec![0.5, -0.5]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f32 * 0.1, 0.2, 0.3, 0.4]),
                done: i == 9,
            }
        }).collect();

        let result = agent.update(&batch, 0.001);
        assert!(result.is_ok());

        let (q_loss, policy_loss, alpha_loss) = result.unwrap();
        assert!(q_loss.is_finite());
        assert!(policy_loss.is_finite());
        assert!(alpha_loss.is_finite());
    }

    #[test]
    fn test_concatenate() {
        let state = Array1::from_vec(vec![1.0, 2.0]);
        let action = Array1::from_vec(vec![3.0, 4.0]);
        let result = concatenate(state.view(), action.view());

        assert_eq!(result, Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_sac_q_learning() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = SACAgent::new(4, 2, &[32, 32], optimizer, 0.99, 0.005, 0.2, false);

        let state = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let action = Array1::from_vec(vec![0.0, 0.0]);

        let (initial_q1, initial_q2) = agent.get_q_value(state.view(), action.view());

        // Train with high rewards
        for _ in 0..30 {
            let batch: Vec<SACExperience> = (0..10).map(|_| {
                SACExperience {
                    state: state.clone(),
                    action: action.clone(),
                    reward: 10.0,  // High reward
                    next_state: state.clone(),
                    done: false,
                }
            }).collect();

            agent.update(&batch, 0.01).unwrap();
        }

        let (final_q1, final_q2) = agent.get_q_value(state.view(), action.view());

        // Q-values should increase with high rewards
        assert!(final_q1 > initial_q1 || final_q2 > initial_q2,
                "Q should increase. Initial: ({}, {}), Final: ({}, {})",
                initial_q1, initial_q2, final_q1, final_q2);
    }
}
