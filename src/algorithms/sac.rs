use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use rand::rngs::StdRng;
use crate::rng::{default_rng, seeded_rng};

use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// Bounds on the actor's log_std output. Below the lower bound the policy becomes
/// effectively deterministic and log probabilities explode; above the upper bound
/// the samples leave the useful range of tanh.
const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;
/// One sample from the policy, with everything the reparameterized gradient needs
struct PolicySample {
    /// The squashed action, tanh(pre_tanh)
    action: Array1<f32>,
    /// log pi(action | state), tanh correction included
    log_prob: f32,
    /// The pre-squash sample, mean + std * noise. Only the tests read this, to check
    /// that action == tanh(pre_tanh), the invariant the gradient chain rests on.
    #[cfg_attr(not(test), allow(dead_code))]
    pre_tanh: Array1<f32>,
    /// The standard normal draw that produced it
    noise: Array1<f32>,
}

/// Keeps the tanh Jacobian correction finite when an action saturates at +/-1
const TANH_EPS: f32 = 1e-6;
/// Per-element ceiling on the policy gradient, to survive early training
const ACTOR_GRAD_CLIP: f32 = 10.0;

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
    #[serde(skip, default = "crate::rng::default_rng")]
    pub rng: StdRng,
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
    /// Reseed this agent's generator so its randomness repeats.
    ///
    /// Two agents given the same seed and the same inputs follow the same sequence of
    /// sampled actions and exploration noise. Weight initialization is separate; fix
    /// that too when a whole run has to reproduce.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = seeded_rng(seed);
    }

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
        // Target networks are only ever assigned to, so they hold no optimizer state
        let q1_target = q1.clone_as_target();
        let q2_target = q2.clone_as_target();

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
            rng: default_rng(),
        }
    }

    /// Select action using current policy
    pub fn act(&mut self, state: ArrayView1<f32>, deterministic: bool) -> Result<Array1<f32>> {
        let output = self.actor.predict(state);
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

    /// Sample an action and its log probability from a single forward pass.
    ///
    /// Deriving the log probability afterwards would mean inverting the tanh, which
    /// loses precision near +/-1. Keeping the noise lets the policy gradient treat
    /// the action as differentiable in the actor's own outputs.
    fn sample_with_log_prob(&mut self, state: ArrayView1<f32>) -> Result<PolicySample> {
        let output = self.actor.predict(state);
        let action_size = output.len() / 2;

        let mut action = Array1::zeros(action_size);
        let mut pre_tanh = Array1::zeros(action_size);
        let mut noise = Array1::zeros(action_size);
        let mut log_prob = 0.0;

        let normal = Normal::new(0.0f32, 1.0)
            .map_err(|e| AthenaError::NumericalError(e.to_string()))?;

        for i in 0..action_size {
            let mean = output[i];
            let log_std = output[action_size + i].clamp(LOG_STD_MIN, LOG_STD_MAX);
            let std = log_std.exp();

            let eps: f32 = self.rng.sample(normal);
            let u = mean + std * eps;
            let a = u.tanh();

            noise[i] = eps;
            pre_tanh[i] = u;
            action[i] = a;

            // log N(u; mean, std) with (u - mean) / std being eps by construction,
            // then the tanh change-of-variables correction
            let gaussian = -0.5 * eps * eps - log_std - 0.5 * (2.0 * std::f32::consts::PI).ln();
            let squash_correction = (1.0 - a * a + TANH_EPS).ln();
            log_prob += gaussian - squash_correction;
        }

        Ok(PolicySample { action, log_prob, pre_tanh, noise })
    }

    /// Get Q-value for a state-action pair
    pub fn get_q_value(&mut self, state: ArrayView1<f32>, action: ArrayView1<f32>) -> (f32, f32) {
        let sa_concat = concatenate(state, action);
        let q1 = self.q1.predict(sa_concat.view())[0];
        let q2 = self.q2.predict(sa_concat.view())[0];
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

            // Sample next action from the current policy, log probability included
            let next = self.sample_with_log_prob(next_state)?;
            let next_action = next.action;
            let next_log_prob = next.log_prob;

            // Compute target Q-value using target networks
            let sa_concat_next = concatenate(next_state, next_action.view());
            let target_q1 = self.q1_target.predict(sa_concat_next.view())[0];
            let target_q2 = self.q2_target.predict(sa_concat_next.view())[0];

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

        // One forward pass per critic serves both the reported loss and the update
        let q1_outputs = self.q1.forward_batch(q_inputs.view());
        let q1_errors = &q1_outputs - &q1_targets;
        let q1_loss = q1_errors.mapv(|x| x * x).mean().unwrap_or(0.0);
        self.q1.apply_output_errors(q1_errors.view(), learning_rate);

        let q2_outputs = self.q2.forward_batch(q_inputs.view());
        let q2_errors = &q2_outputs - &q2_targets;
        let q2_loss = q2_errors.mapv(|x| x * x).mean().unwrap_or(0.0);
        self.q2.apply_output_errors(q2_errors.view(), learning_rate);

        let q_loss = q1_loss + q2_loss;

        // Policy update, by the reparameterized policy gradient.
        //
        // Objective: minimize L = E[alpha * log pi(a|s) - Q(s,a)], where the action
        // is written as a = tanh(u) with u = mean + std * eps and eps ~ N(0,1) held
        // fixed. That keeps a differentiable in the actor's own outputs, so the
        // gradient flows: L -> a -> u -> {mean, log_std}.
        //
        //   dL/da       = alpha * 2a / (1 - a^2)   -   dQ/da
        //                 \____ tanh correction __/     \_ from the critic _/
        //   dL/du       = dL/da * (1 - a^2)
        //   dL/dmean    = dL/du
        //   dL/dlog_std = dL/du * std * eps  -  alpha
        //                                       \_ the -log(std) term in log pi _/

        // Sample a fresh action per state, keeping the noise that produced it
        let mut sampled_actions = Array2::zeros((batch_size, self.action_size));
        let mut noises = Array2::zeros((batch_size, self.action_size));
        let mut log_probs = vec![0.0f32; batch_size];

        for i in 0..batch_size {
            let sample = self.sample_with_log_prob(states.row(i))?;
            sampled_actions.row_mut(i).assign(&sample.action);
            noises.row_mut(i).assign(&sample.noise);
            log_probs[i] = sample.log_prob;
        }

        // After the sampling loop, so the caches this leaves behind are the ones the
        // actor gradient travels back through. The critic passes below touch different
        // networks and leave them alone.
        let actor_outputs = self.actor.forward_batch(states.view());

        // dQ/da, taken from whichever critic gives the smaller value for that sample.
        // forward_batch first, the backward pass reads its cached pre-activations.
        let sa_batch = concatenate_batch(states.view(), sampled_actions.view());
        let q1_values = self.q1.forward_batch(sa_batch.view());
        let ones = Array2::from_elem((batch_size, 1), 1.0);
        let dq1_dinput = self.q1.input_gradient_batch(ones.view());

        let q2_values = self.q2.forward_batch(sa_batch.view());
        let dq2_dinput = self.q2.input_gradient_batch(ones.view());

        let state_size = states.ncols();
        let mut actor_errors = Array2::zeros((batch_size, self.action_size * 2));
        let mut policy_loss = 0.0;
        let mut mean_log_prob = 0.0;

        for i in 0..batch_size {
            let q1 = q1_values[[i, 0]];
            let q2 = q2_values[[i, 0]];
            let use_q1 = q1 <= q2;
            let q_value = if use_q1 { q1 } else { q2 };

            policy_loss += self.alpha * log_probs[i] - q_value;
            mean_log_prob += log_probs[i];

            for j in 0..self.action_size {
                let a = sampled_actions[[i, j]];
                let log_std = actor_outputs[[i, self.action_size + j]].clamp(LOG_STD_MIN, LOG_STD_MAX);
                let std = log_std.exp();
                let eps = noises[[i, j]];

                // Gradient of the critic with respect to this action dimension
                let dq_da = if use_q1 {
                    dq1_dinput[[i, state_size + j]]
                } else {
                    dq2_dinput[[i, state_size + j]]
                };

                let one_minus_a2 = (1.0 - a * a).max(TANH_EPS);
                let dl_da = self.alpha * 2.0 * a / one_minus_a2 - dq_da;
                let dl_du = dl_da * one_minus_a2;

                let d_mean = dl_du;
                let d_log_std = dl_du * std * eps - self.alpha;

                // Anything non-finite here would poison the actor for good
                actor_errors[[i, j]] = finite_or_zero(d_mean).clamp(-ACTOR_GRAD_CLIP, ACTOR_GRAD_CLIP);
                actor_errors[[i, self.action_size + j]] =
                    finite_or_zero(d_log_std).clamp(-ACTOR_GRAD_CLIP, ACTOR_GRAD_CLIP);
            }
        }

        policy_loss /= batch_size as f32;
        mean_log_prob /= batch_size as f32;

        // Average over the batch, then apply the gradient
        let actor_errors = actor_errors / batch_size as f32;
        self.actor.apply_output_errors(actor_errors.view(), learning_rate);

        // Temperature update, when auto_alpha is set.
        //
        // L(alpha) = -log_alpha * (log pi + target_entropy), so the gradient with
        // respect to log_alpha is just -(log pi + target_entropy). Entropy below
        // target pushes alpha up, which buys back exploration.

        let mut alpha_loss = 0.0;
        if self.auto_alpha {
            let entropy_gap = mean_log_prob + self.target_entropy;
            alpha_loss = -self.log_alpha * entropy_gap;

            if entropy_gap.is_finite() {
                self.log_alpha += learning_rate * entropy_gap;
                self.log_alpha = self.log_alpha.clamp(-10.0, 2.0);
                self.alpha = self.log_alpha.exp();
            }
        }

        // Soft update of the target networks
        self.soft_update();

        Ok((q_loss, policy_loss, alpha_loss))
    }

    /// Soft update target networks
    fn soft_update(&mut self) {
        self.q1_target.soft_update_from(&self.q1, self.tau);
        self.q2_target.soft_update_from(&self.q2, self.tau);
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

/// Concatenate a batch of states with a batch of actions, row by row
fn concatenate_batch(states: ArrayView2<f32>, actions: ArrayView2<f32>) -> Array2<f32> {
    let rows = states.nrows();
    let state_size = states.ncols();
    let action_size = actions.ncols();
    let mut result = Array2::zeros((rows, state_size + action_size));

    result.slice_mut(ndarray::s![.., ..state_size]).assign(&states);
    result.slice_mut(ndarray::s![.., state_size..]).assign(&actions);
    result
}

/// Zero stands in for anything non-finite, so one bad value cannot spread
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
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

    /// The policy gradient has to point the mean at higher-Q actions. Here the critic
    /// is pinned to Q(s,a) = a, so the only correct direction is upward.
    #[test]
    fn policy_gradient_follows_the_critic() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        // No hidden layers, so the critic is a single linear layer we can set by hand
        let mut agent = SACAgent::new(2, 1, &[], optimizer, 0.99, 0.005, 0.01, false);

        // Q(s, a) = a: zero the state weights, put 1.0 on the action input
        for q in [&mut agent.q1, &mut agent.q2, &mut agent.q1_target, &mut agent.q2_target] {
            q.layers[0].weights.fill(0.0);
            q.layers[0].weights[[2, 0]] = 1.0;
            q.layers[0].biases.fill(0.0);
        }

        let state = Array1::from_vec(vec![0.3, -0.4]);
        let mean_before = agent.actor.forward(state.view())[0];

        let batch: Vec<SACExperience> = (0..16)
            .map(|_| SACExperience {
                state: state.clone(),
                action: Array1::from_vec(vec![0.0]),
                reward: 0.0,
                next_state: state.clone(),
                done: true,
            })
            .collect();

        for _ in 0..40 {
            // Restore the critic each round, the Q update would otherwise drift it
            for q in [&mut agent.q1, &mut agent.q2] {
                q.layers[0].weights.fill(0.0);
                q.layers[0].weights[[2, 0]] = 1.0;
                q.layers[0].biases.fill(0.0);
            }
            agent.update(&batch, 0.05).unwrap();
        }

        let mean_after = agent.actor.forward(state.view())[0];
        assert!(mean_after.is_finite(), "mean went non-finite: {mean_after}");
        assert!(mean_after > mean_before + 0.05,
                "mean should climb toward the higher-Q action, went {mean_before} -> {mean_after}");
    }

    #[test]
    fn log_prob_matches_the_sampled_action() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = SACAgent::new(3, 2, &[16], optimizer, 0.99, 0.005, 0.2, false);
        let state = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        for _ in 0..20 {
            let s = agent.sample_with_log_prob(state.view()).unwrap();

            assert_eq!(s.action.len(), 2);
            assert!(s.log_prob.is_finite(), "log_prob was {}", s.log_prob);
            assert!(s.action.iter().all(|a| a.abs() <= 1.0), "tanh output out of range");
            // a = tanh(u) must hold for the gradient chain to be valid
            for j in 0..2 {
                assert!((s.action[j] - s.pre_tanh[j].tanh()).abs() < 1e-5);
            }
            assert!(s.noise.iter().all(|e| e.is_finite()));
        }
    }
}
