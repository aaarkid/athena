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
///
/// # Algorithm Overview
///
/// A2C combines policy gradient methods (actor) with value function estimation (critic):
/// - **Actor**: Learns a policy π(a|s) that maximizes expected returns
/// - **Critic**: Learns a value function V(s) to reduce variance in policy updates
/// - **Advantage**: A(s,a) = R - V(s), used to weight policy gradients
///
/// # Example
///
/// ```rust,no_run
/// use athena::algorithms::{A2CAgent, A2CBuilder, A2CExperience};
/// use athena::optimizer::{OptimizerWrapper, SGD};
/// use ndarray::array;
///
/// // Create agent
/// let agent = A2CBuilder::new(4, 2)
///     .hidden_sizes(vec![64, 64])
///     .gamma(0.99)
///     .optimizer(OptimizerWrapper::SGD(SGD::new()))
///     .build()
///     .unwrap();
/// ```
#[derive(Serialize, Deserialize, Clone)]
pub struct A2CAgent {
    /// Actor network that outputs action logits
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
    /// Number of actions
    action_size: usize,
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
        // Build actor network (outputs logits for each action)
        let mut actor_sizes = vec![state_size];
        actor_sizes.extend_from_slice(hidden_sizes);
        actor_sizes.push(action_size);

        let actor_activations = vec![Activation::Relu; hidden_sizes.len()]
            .into_iter()
            .chain(std::iter::once(Activation::Linear))
            .collect::<Vec<_>>();

        let actor = NeuralNetwork::new(&actor_sizes, &actor_activations, optimizer.clone());

        // Build critic network (outputs single value)
        let mut critic_sizes = vec![state_size];
        critic_sizes.extend_from_slice(hidden_sizes);
        critic_sizes.push(1);

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
            action_size,
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

    /// Get action probabilities without sampling
    pub fn get_action_probs(&mut self, state: ArrayView1<f32>) -> Array1<f32> {
        let logits = self.actor.forward(state);
        softmax(&logits)
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
    ///
    /// This implements the A2C update rule:
    /// - Actor: ∇θ J(θ) = E[∇θ log π(a|s) * A(s,a)]
    /// - Critic: Minimize MSE between V(s) and returns
    pub fn train(
        &mut self,
        experiences: &[A2CExperience],
        learning_rate: f32,
    ) -> Result<(f32, f32)> {
        if experiences.is_empty() {
            return Err(AthenaError::EmptyBuffer("No experiences to train on".to_string()));
        }

        let batch_size = experiences.len();

        // Prepare batch data
        let states = stack_arrays(experiences.iter().map(|e| e.state.view()).collect());
        let actions: Vec<usize> = experiences.iter().map(|e| e.action).collect();
        let rewards: Vec<f32> = experiences.iter().map(|e| e.reward).collect();
        let dones: Vec<bool> = experiences.iter().map(|e| e.done).collect();
        let old_values: Vec<f32> = experiences.iter().map(|e| e.value).collect();

        // The rollout is usually cut short rather than ended, so the last step's return
        // needs V(s') to stand in for the rest of the episode
        let last = &experiences[batch_size - 1];
        let last_value = if last.done {
            0.0
        } else {
            self.critic.forward(last.next_state.view())[0]
        };

        // Compute returns and advantages
        let (returns, advantages) = self.compute_returns(&rewards, &old_values, &dones, last_value);

        // Normalize advantages for stable training
        let adv_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let adv_std = (advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>()
                      / advantages.len() as f32).sqrt() + 1e-8;
        let norm_advantages: Vec<f32> = advantages.iter().map(|a| (a - adv_mean) / adv_std).collect();

        // Critic update: minimize MSE(V(s), returns)

        // Create target values for critic (returns reshaped to match critic output)
        let critic_targets = Array2::from_shape_vec(
            (batch_size, 1),
            returns.clone()
        ).expect("Failed to create critic targets");

        // Train critic network
        self.critic.train_minibatch(states.view(), critic_targets.view(), learning_rate);

        // Compute critic loss for reporting
        let critic_outputs = self.critic.forward_batch(states.view());
        let critic_loss = (&critic_outputs - &critic_targets).mapv(|x| x * x).mean().unwrap_or(0.0);

        // Actor update: policy gradient
        //
        // For discrete actions, the policy gradient is:
        //   ∇θ J(θ) = E[∇θ log π(a|s) * A(s,a)]
        //
        // The gradient of log π(a|s) w.r.t. logits (before softmax) is:
        //   ∇_logits log π(a|s) = one_hot(a) - softmax(logits)
        //
        // We multiply by advantage and pass to train_policy_gradient()

        let actor_outputs = self.actor.forward_batch(states.view());
        let mut policy_gradients = Array2::zeros((batch_size, self.action_size));

        let mut actor_loss = 0.0;
        let mut entropy = 0.0;

        for i in 0..batch_size {
            let logits = actor_outputs.row(i).to_owned();
            let probs = softmax(&logits);

            // Compute current log probability for loss reporting
            let log_prob = probs[actions[i]].ln();
            actor_loss -= log_prob * norm_advantages[i];

            // Entropy of this row, needed both for the reported loss and for the
            // entropy gradient below
            let mut row_entropy = 0.0;
            for &p in probs.iter() {
                if p > 1e-8 {
                    row_entropy -= p * p.ln();
                }
            }
            entropy += row_entropy;

            // Compute policy gradient: advantage * (one_hot(a) - softmax)
            // This is the gradient we want to ASCEND (maximize expected return)
            // So we negate it for gradient descent: -advantage * (one_hot - softmax)
            for j in 0..self.action_size {
                let one_hot = if j == actions[i] { 1.0 } else { 0.0 };
                // Gradient for gradient DESCENT (minimizing negative expected return)
                // = -advantage * (one_hot - softmax)
                // Plus the entropy bonus gradient. For H = -sum p ln p over softmax
                // logits, dH/dz_j = -p_j * (ln p_j + H), so minimizing -coeff * H gives
                // coeff * p_j * (ln p_j + H). The row entropy is what couples the
                // actions together; a constant 1 there would not sum to zero.
                let entropy_grad = if probs[j] > 1e-8 {
                    self.entropy_coeff * probs[j] * (probs[j].ln() + row_entropy)
                } else {
                    0.0
                };
                policy_gradients[[i, j]] = -norm_advantages[i] * (one_hot - probs[j]) + entropy_grad;
            }
        }

        actor_loss /= batch_size as f32;
        entropy /= batch_size as f32;
        let total_actor_loss = actor_loss - self.entropy_coeff * entropy;

        // Train actor using policy gradient method
        self.actor.train_policy_gradient(states.view(), policy_gradients.view(), learning_rate);

        Ok((total_actor_loss, critic_loss * self.value_coeff))
    }

    /// Discounted n-step returns, and the advantage as return minus value.
    ///
    /// This is not GAE: there is no lambda, the return is the full discounted sum to
    /// the end of the rollout. `last_value` is V(s') for the step after the last one,
    /// or 0.0 if the episode ended there.
    fn compute_returns(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        last_value: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        let mut returns = vec![0.0; n];
        let mut advantages = vec![0.0; n];

        for i in (0..n).rev() {
            let next_return = if dones[i] {
                0.0
            } else if i == n - 1 {
                last_value
            } else {
                returns[i + 1]
            };

            returns[i] = rewards[i] + self.gamma * next_return;
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
        let mut agent: Self = bincode::deserialize(&data)?;
        agent.rng = thread_rng();
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

impl Default for A2CBuilder {
    fn default() -> Self {
        Self::new(4, 2)
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
    fn test_a2c_act() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = A2CAgent::new(4, 2, &[32, 32], optimizer, 0.99, 5, 0.01, 0.5);

        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (action, log_prob) = agent.act(state.view()).unwrap();

        assert!(action < 2);
        assert!(log_prob <= 0.0); // Log probs are negative
    }

    #[test]
    fn test_a2c_training() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = A2CAgent::new(4, 2, &[32, 32], optimizer, 0.99, 5, 0.01, 0.5);

        // Create some experiences
        let experiences: Vec<A2CExperience> = (0..10).map(|i| {
            A2CExperience {
                state: Array1::from_vec(vec![i as f32 * 0.1, 0.2, 0.3, 0.4]),
                action: i % 2,
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f32 * 0.1, 0.2, 0.3, 0.4]),
                done: i == 9,
                log_prob: -0.5,
                value: 0.5,
            }
        }).collect();

        // Should not error
        let result = agent.train(&experiences, 0.001);
        assert!(result.is_ok());

        let (actor_loss, critic_loss) = result.unwrap();
        assert!(actor_loss.is_finite());
        assert!(critic_loss.is_finite());
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

    #[test]
    fn test_a2c_value_learning() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = A2CAgent::new(4, 2, &[32, 32], optimizer, 0.99, 5, 0.01, 0.5);

        let state = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let initial_value = agent.get_value(state.view());

        // Train multiple times with high rewards
        for _ in 0..50 {
            let experiences: Vec<A2CExperience> = (0..10).map(|_| {
                A2CExperience {
                    state: state.clone(),
                    action: 0,
                    reward: 10.0,  // High reward
                    next_state: state.clone(),
                    done: false,
                    log_prob: -0.5,
                    value: agent.get_value(state.view()),
                }
            }).collect();

            agent.train(&experiences, 0.01).unwrap();
        }

        let final_value = agent.get_value(state.view());
        // Value should increase toward the high return
        assert!(final_value > initial_value,
                "Value should increase with high rewards. Initial: {}, Final: {}",
                initial_value, final_value);
    }

    #[test]
    fn truncated_rollout_bootstraps_its_last_return() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = A2CAgent::new(4, 2, &[8], optimizer, 0.99, 5, 0.01, 0.5);

        let rewards = [1.0f32; 5];
        let values = [10.0f32; 5];
        let dones = [false; 5];

        let (returns, advantages) = agent.compute_returns(&rewards, &values, &dones, 10.0);

        // The rollout was cut, not ended, so the last return keeps gamma * V(s')
        let expected_last = 1.0 + 0.99 * 10.0;
        assert!(
            (returns[4] - expected_last).abs() < 1e-5,
            "last return was {}, expected {}",
            returns[4],
            expected_last
        );
        assert!(
            (advantages[4] - (expected_last - 10.0)).abs() < 1e-5,
            "last advantage was {}",
            advantages[4]
        );

        // Earlier steps still discount along the rollout
        let expected_third = 1.0 + 0.99 * returns[4];
        assert!((returns[3] - expected_third).abs() < 1e-5);
    }

    #[test]
    fn a_terminal_step_does_not_bootstrap() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = A2CAgent::new(4, 2, &[8], optimizer, 0.99, 5, 0.01, 0.5);

        let rewards = [1.0f32; 3];
        let values = [10.0f32; 3];
        let dones = [false, true, false];

        let (returns, _) = agent.compute_returns(&rewards, &values, &dones, 10.0);

        assert!((returns[1] - 1.0).abs() < 1e-6, "terminal return was {}", returns[1]);
        // Step 0 sees only the terminal step's return, not what follows it
        assert!((returns[0] - (1.0 + 0.99)).abs() < 1e-6);
    }

    #[test]
    fn the_entropy_gradient_sums_to_zero_over_the_logits() {
        // The entropy term is a function of the softmax outputs, so its gradient with
        // respect to the logits has to sum to zero: shifting every logit by a constant
        // leaves the distribution, and the entropy, unchanged
        for probs in [
            vec![0.25f32, 0.25, 0.25, 0.25],
            vec![0.9, 0.05, 0.03, 0.02],
            vec![0.5, 0.3, 0.15, 0.05],
        ] {
            let entropy: f32 = -probs.iter().map(|&p| p * p.ln()).sum::<f32>();
            let coeff = 0.01;

            let total: f32 = probs
                .iter()
                .map(|&p| coeff * p * (p.ln() + entropy))
                .sum();

            assert!(
                total.abs() < 1e-6,
                "entropy gradient summed to {} for {:?}",
                total,
                probs
            );
        }
    }
}
