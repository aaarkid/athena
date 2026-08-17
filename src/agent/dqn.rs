use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::OptimizerWrapper;
use crate::replay_buffer::Experience;
use crate::error::{Result, AthenaError};
use rand::{Rng, rngs::StdRng};
use crate::rng::{default_rng, seeded_rng};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Serialize, Deserialize};

/// Enhanced Deep Q-Network (DQN) Agent with target network and Double DQN support
/// 
/// This agent implements the DQN algorithm with several improvements:
/// - Target network for stable Q-value estimation
/// - Double DQN to reduce overestimation bias
/// - Epsilon-greedy exploration strategy
/// - Experience replay support
/// 
/// # Example
/// 
/// ```rust
/// use athena::agent::DqnAgent;
/// use athena::optimizer::{OptimizerWrapper, Adam};
/// use athena::replay_buffer::{ReplayBuffer, Experience};
/// use ndarray::array;
/// 
/// // Create a DQN agent for CartPole (4 states, 2 actions)
/// let layer_sizes = &[4, 128, 128, 2];
/// let optimizer = OptimizerWrapper::SGD(athena::optimizer::SGD::new());
/// let mut agent = DqnAgent::new(
///     layer_sizes,
///     0.1,      // epsilon (exploration rate)
///     optimizer,
///     1000,     // target_update_freq
///     true      // use_double_dqn
/// );
/// 
/// // Create experience replay buffer
/// let mut replay_buffer = ReplayBuffer::new(10000);
/// 
/// // Training loop example
/// let state = array![0.1, -0.2, 0.3, -0.1];
/// let action = agent.act(state.view()).unwrap();
/// 
/// // After environment step...
/// let next_state = array![0.15, -0.25, 0.35, -0.05];
/// let reward = 1.0;
/// let done = false;
/// 
/// // Store experience
/// replay_buffer.add(Experience {
///     state: state.clone(),
///     action,
///     reward,
///     next_state: next_state.clone(),
///     done,
/// });
/// 
/// // Train on batch when buffer is ready
/// if replay_buffer.len() >= 32 {
///     let batch = replay_buffer.sample(32);
///     let loss = agent.train_on_batch(&batch, 0.99, 0.001).unwrap();
/// }
/// ```
#[derive(Serialize, Deserialize)]
pub struct DqnAgent {
    /// Main network for action selection
    pub q_network: NeuralNetwork,
    
    /// Target network for stable Q-value estimation
    pub target_network: NeuralNetwork,
    
    /// Exploration rate
    pub epsilon: f32,
    
    /// Update frequency for target network
    pub target_update_freq: usize,
    
    /// Counter for updates
    update_counter: usize,
    
    /// Use Double DQN
    pub use_double_dqn: bool,
    
    /// Number of training steps performed
    pub train_steps: usize,
    
    /// Random number generator
    #[serde(skip, default = "default_rng")]
    pub rng: StdRng,
}

impl DqnAgent {
    /// Create a new DQN agent with target network
    pub fn new(
        layer_sizes: &[usize], 
        epsilon: f32, 
        optimizer: OptimizerWrapper,
        target_update_freq: usize,
        use_double_dqn: bool,
    ) -> Self {
        // Validate inputs
        if layer_sizes.len() < 2 {
            panic!("Network must have at least input and output layers");
        }
        if target_update_freq == 0 {
            panic!("target_update_freq must be at least 1; the update counter is taken modulo it");
        }
        
        // Create activations (ReLU for hidden layers, Linear for output)
        let mut activations = vec![Activation::Relu; layer_sizes.len() - 2];
        activations.push(Activation::Linear);
        
        // Create main and target networks
        let q_network = NeuralNetwork::new(layer_sizes, &activations, optimizer);
        // The target network is only ever assigned to, so it holds no optimizer state
        let target_network = q_network.clone_as_target();
        
        let rng = default_rng();
        
        DqnAgent {
            q_network,
            target_network,
            epsilon,
            target_update_freq,
            update_counter: 0,
            use_double_dqn,
            train_steps: 0,
            rng,
        }
    }
    
    /// Reseed this agent's generator so its randomness repeats.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = seeded_rng(seed);
    }

    /// Create an agent whose randomness is reproducible.
    ///
    /// Two agents built with the same seed take the same exploratory actions given the
    /// same states. Weight initialization still comes from the thread generator, so
    /// pair this with a fixed set of weights when a run has to repeat exactly.
    pub fn new_seeded(
        layer_sizes: &[usize],
        epsilon: f32,
        optimizer: OptimizerWrapper,
        target_update_freq: usize,
        use_double_dqn: bool,
        seed: u64,
    ) -> Self {
        let mut agent = Self::new(layer_sizes, epsilon, optimizer, target_update_freq, use_double_dqn);
        agent.rng = seeded_rng(seed);
        agent
    }

    /// Create agent with default architecture
    pub fn new_default(
        state_size: usize, 
        action_size: usize, 
        epsilon: f32, 
        optimizer: OptimizerWrapper
    ) -> Self {
        Self::new(
            &[state_size, 128, 64, action_size], 
            epsilon, 
            optimizer,
            1000,  // Update target network every 1000 steps
            true,  // Use Double DQN by default
        )
    }
    
    /// Select action using epsilon-greedy policy
    pub fn act(&mut self, state: ArrayView1<f32>) -> Result<usize> {
        let num_actions = self.q_network.layers.last()
            .ok_or_else(|| AthenaError::TrainingError("No layers in network".to_string()))?
            .biases.len();
            
        if self.rng.gen::<f32>() < self.epsilon {
            // Exploration: random action
            Ok(self.rng.gen_range(0..num_actions))
        } else {
            // Exploitation: best action from Q-network. try_predict reports a wrong state
            // width as an error; forward would panic inside ndarray, which for a game
            // means the process dies mid-frame. It also writes no caches, so acting
            // costs a matmul and nothing else.
            let q_values = self.q_network.try_predict(state)?;
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .ok_or_else(|| AthenaError::NumericalError("No valid Q-values".to_string()))
        }
    }
    
    /// Update epsilon for exploration decay
    pub fn update_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon.clamp(0.0, 1.0);
    }
    
    /// Update target network weights from main network
    pub fn update_target_network(&mut self) {
        // Assigns into the arrays the target network already owns. Cloning the whole
        // network would also copy the optimizer state and the forward-pass caches.
        self.target_network.copy_parameters_from(&self.q_network);
    }
    
    /// Train the agent on a batch of experiences
    pub fn train_on_batch(
        &mut self,
        experiences: &[&Experience],
        gamma: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        self.train_on_batch_inner(experiences, None, gamma, learning_rate)
    }

    /// Train on a batch where some actions are illegal in the next state.
    ///
    /// `next_masks[i]` says which actions are legal in `experiences[i].next_state`.
    /// Plain `train_on_batch` maximizes over every action, including ones the agent can
    /// never play; those entries are never corrected by experience and drift upward,
    /// then leak into neighbouring states through the bootstrap.
    ///
    /// A next state with no legal action is treated as terminal.
    pub fn train_on_batch_masked(
        &mut self,
        experiences: &[&Experience],
        next_masks: &[Array1<bool>],
        gamma: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        if next_masks.len() != experiences.len() {
            return Err(AthenaError::dimension_mismatch(
                format!("{} masks, one per experience", experiences.len()),
                format!("{} masks", next_masks.len()),
            ));
        }

        let num_actions = self.q_network.output_size();
        for (i, mask) in next_masks.iter().enumerate() {
            if mask.len() != num_actions {
                return Err(AthenaError::dimension_mismatch(
                    format!("mask {} of length {}", i, num_actions),
                    format!("length {}", mask.len()),
                ));
            }
        }

        self.train_on_batch_inner(experiences, Some(next_masks), gamma, learning_rate)
    }

    /// Train on a batch where each sample carries an importance-sampling weight.
    ///
    /// Returns the absolute TD error per sample, which is what a
    /// `PrioritizedReplayBuffer` wants back for `update_priorities`. Pass the weights
    /// from `sample_with_weights`; a weight of 1.0 is the same as `train_on_batch`, and
    /// 0.0 removes that sample from the update.
    pub fn train_on_batch_weighted(
        &mut self,
        experiences: &[&Experience],
        weights: &[f32],
        gamma: f32,
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        if weights.len() != experiences.len() {
            return Err(AthenaError::dimension_mismatch(
                format!("{} weights, one per experience", experiences.len()),
                format!("{} weights", weights.len()),
            ));
        }

        let (_, td_errors) =
            self.train_on_batch_full(experiences, None, Some(weights), gamma, learning_rate)?;
        Ok(td_errors)
    }

    fn train_on_batch_inner(
        &mut self,
        experiences: &[&Experience],
        next_masks: Option<&[Array1<bool>]>,
        gamma: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        let (loss, _) = self.train_on_batch_full(experiences, next_masks, None, gamma, learning_rate)?;
        Ok(loss)
    }

    fn train_on_batch_full(
        &mut self,
        experiences: &[&Experience],
        next_masks: Option<&[Array1<bool>]>,
        weights: Option<&[f32]>,
        gamma: f32,
        learning_rate: f32,
    ) -> Result<(f32, Vec<f32>)> {
        if experiences.is_empty() {
            return Err(AthenaError::EmptyBuffer("No experiences to train on".to_string()));
        }
        
        let batch_size = experiences.len();
        let state_size = self.q_network.input_size();
        let num_actions = self.q_network.output_size();

        // Everything below indexes by action and assigns whole rows, so a mismatched
        // experience would panic. Report it instead.
        for (i, exp) in experiences.iter().enumerate() {
            if exp.state.len() != state_size || exp.next_state.len() != state_size {
                return Err(AthenaError::dimension_mismatch(
                    format!("experience {} state width {}", i, state_size),
                    format!("state {} and next_state {}", exp.state.len(), exp.next_state.len()),
                ));
            }
            if exp.action >= num_actions {
                return Err(AthenaError::InvalidAction {
                    action: exp.action,
                    max_actions: num_actions,
                });
            }
        }
        
        // Stack experiences into batches
        let mut states = Array2::zeros((batch_size, state_size));
        let mut next_states = Array2::zeros((batch_size, state_size));
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        
        for (i, exp) in experiences.iter().enumerate() {
            states.row_mut(i).assign(&exp.state);
            next_states.row_mut(i).assign(&exp.next_state);
            actions.push(exp.action);
            rewards.push(exp.reward);
            dones.push(exp.done);
        }
        
        // Whether action a is legal in the next state. Without a mask every action
        // counts, which is what plain DQN assumes.
        let legal = |i: usize, a: usize| -> bool {
            next_masks.map(|masks| masks[i][a]).unwrap_or(true)
        };

        // The bootstrap passes read no gradients, so they go through predict_batch and
        // leave the main network's caches free for the states pass below
        let mut targets = vec![0.0f32; batch_size];

        if self.use_double_dqn {
            // Double DQN: use main network to select actions, target network to evaluate
            let next_q_values_main = self.q_network.predict_batch(next_states.view());
            let next_q_values_target = self.target_network.predict_batch(next_states.view());

            for i in 0..batch_size {
                if dones[i] {
                    targets[i] = rewards[i];
                    continue;
                }

                // Find best action using main network
                let best_action = next_q_values_main.row(i)
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| legal(i, *idx))
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx);

                // A next state with no legal action is terminal in all but name
                targets[i] = match best_action {
                    Some(a) => rewards[i] + gamma * next_q_values_target[[i, a]],
                    None => rewards[i],
                };
            }
        } else {
            // Standard DQN: use target network for both selection and evaluation
            let next_q_values = self.target_network.predict_batch(next_states.view());

            for i in 0..batch_size {
                if dones[i] {
                    targets[i] = rewards[i];
                    continue;
                }

                let max_next_q = next_q_values.row(i).iter()
                    .enumerate()
                    .filter(|(idx, _)| legal(i, *idx))
                    .map(|(_, &val)| val)
                    .fold(f32::NEG_INFINITY, f32::max);
                targets[i] = if max_next_q.is_finite() {
                    rewards[i] + gamma * max_next_q
                } else {
                    rewards[i]
                };
            }
        }

        // The only forward pass that has to leave caches behind, so that the error below
        // can be backpropagated without repeating it
        let current_q_values = self.q_network.forward_batch(states.view());

        // The target differs from the prediction on the taken action only, so the error
        // matrix is zero on every other column and the TD error is that one difference.
        // Importance-sampling weights scale each sample's contribution to the update but
        // not the TD error the caller gets back for its priorities.
        let mut output_errors = Array2::zeros((batch_size, num_actions));
        let mut td_errors = Vec::with_capacity(batch_size);
        let mut squared_error = 0.0f32;

        for i in 0..batch_size {
            let a = actions[i];
            let td = targets[i] - current_q_values[[i, a]];
            td_errors.push(td.abs());
            squared_error += td * td;

            let weight = weights.map(|w| w[i]).unwrap_or(1.0);
            output_errors[[i, a]] = -weight * td;
        }

        // Mean squared TD error over the batch, measured before the update. The old
        // figure averaged over batch_size * num_actions although only one column per row
        // is ever non-zero, so it read num_actions times too small.
        let loss = squared_error / batch_size as f32;

        self.q_network.apply_output_errors(output_errors.view(), learning_rate);

        // Increment train steps
        self.train_steps += 1;
        
        // Update target network if needed
        self.update_counter += 1;
        if self.update_counter % self.target_update_freq == 0 {
            self.update_target_network();
        }
        
        Ok((loss, td_errors))
    }
    
    /// Save the agent to disk
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

/// Builder pattern for DqnAgent
pub struct DqnAgentBuilder {
    layer_sizes: Vec<usize>,
    activations: Option<Vec<Activation>>,
    epsilon: f32,
    optimizer: Option<OptimizerWrapper>,
    target_update_freq: usize,
    use_double_dqn: bool,
}

impl DqnAgentBuilder {
    pub fn new() -> Self {
        DqnAgentBuilder {
            layer_sizes: vec![],
            activations: None,
            epsilon: 1.0,
            optimizer: None,
            target_update_freq: 1000,
            use_double_dqn: true,
        }
    }
    
    pub fn layer_sizes(mut self, sizes: &[usize]) -> Self {
        self.layer_sizes = sizes.to_vec();
        self
    }
    
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    pub fn optimizer(mut self, optimizer: OptimizerWrapper) -> Self {
        self.optimizer = Some(optimizer);
        self
    }
    
    pub fn target_update_freq(mut self, freq: usize) -> Self {
        self.target_update_freq = freq;
        self
    }
    
    pub fn use_double_dqn(mut self, use_double: bool) -> Self {
        self.use_double_dqn = use_double;
        self
    }
    
    pub fn activations(mut self, activations: &[Activation]) -> Self {
        self.activations = Some(activations.to_vec());
        self
    }
    
    pub fn build(self) -> Result<DqnAgent> {
        if self.layer_sizes.len() < 2 {
            return Err(AthenaError::InvalidParameter {
                name: "layer_sizes".to_string(),
                reason: "Must have at least 2 layers".to_string(),
            });
        }
        
        let optimizer = self.optimizer
            .ok_or_else(|| AthenaError::InvalidParameter {
                name: "optimizer".to_string(),
                reason: "Optimizer must be specified".to_string(),
            })?;
        
        // Use custom activations if provided
        if let Some(activations) = self.activations {
            if activations.len() != self.layer_sizes.len() - 1 {
                return Err(AthenaError::InvalidParameter {
                    name: "activations".to_string(),
                    reason: "Number of activations must match number of layers - 1".to_string(),
                });
            }
            
            // Create networks with custom activations
            let q_network = NeuralNetwork::new(&self.layer_sizes, &activations, optimizer);
            let target_network = q_network.clone_as_target();
            
            Ok(DqnAgent {
                q_network,
                target_network,
                epsilon: self.epsilon,
                target_update_freq: self.target_update_freq,
                update_counter: 0,
                use_double_dqn: self.use_double_dqn,
                train_steps: 0,
                rng: default_rng(),
            })
        } else {
            // Use default activations
            Ok(DqnAgent::new(
                &self.layer_sizes,
                self.epsilon,
                optimizer,
                self.target_update_freq,
                self.use_double_dqn,
            ))
        }
    }
}

impl Default for DqnAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    use ndarray::array;

    fn agent() -> DqnAgent {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        DqnAgent::new(&[4, 16, 3], 0.0, optimizer, 100, false)
    }

    fn experience(state_len: usize, action: usize) -> Experience {
        Experience {
            state: ndarray::Array1::zeros(state_len),
            action,
            reward: 1.0,
            next_state: ndarray::Array1::zeros(state_len),
            done: false,
        }
    }

    #[test]
    fn a_state_of_the_wrong_width_is_an_error_not_a_panic() {
        let mut agent = agent();

        assert!(agent.act(array![1.0, 2.0, 3.0, 4.0].view()).is_ok());
        assert!(agent.act(array![1.0, 2.0, 3.0].view()).is_err());
        assert!(agent.act(array![1.0, 2.0, 3.0, 4.0, 5.0].view()).is_err());
    }

    #[test]
    fn training_rejects_experiences_that_do_not_fit_the_network() {
        let mut agent = agent();

        let good = experience(4, 0);
        assert!(agent.train_on_batch(&[&good], 0.99, 0.01).is_ok());

        // A state one element too wide would panic on row assignment
        let wide = experience(5, 0);
        assert!(agent.train_on_batch(&[&wide], 0.99, 0.01).is_err());

        // An action at or past the output width would panic on indexing
        let bad_action = experience(4, 3);
        assert!(agent.train_on_batch(&[&bad_action], 0.99, 0.01).is_err());

        // A mixed batch is rejected on the offending element, not the first
        let mixed = vec![&good, &wide];
        assert!(agent.train_on_batch(&mixed, 0.99, 0.01).is_err());
    }

    #[test]
    fn an_empty_batch_is_an_error() {
        let mut agent = agent();
        assert!(agent.train_on_batch(&[], 0.99, 0.01).is_err());
    }

    #[test]
    #[should_panic(expected = "target_update_freq must be at least 1")]
    fn a_zero_target_update_frequency_is_rejected_at_construction() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        DqnAgent::new(&[4, 16, 3], 0.0, optimizer, 0, false);
    }

    #[test]
    fn masked_training_does_not_bootstrap_off_an_illegal_action() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        // Plain DQN, so the bootstrap is a straight max over the target network
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        // Make action 3 look extremely good in every state, by hand
        let out = agent.target_network.layers.last_mut().unwrap();
        out.biases.fill(0.0);
        out.biases[3] = 100.0;
        out.weights.fill(0.0);

        let exp = Experience {
            state: array![0.0, 0.0],
            action: 0,
            reward: 1.0,
            next_state: array![0.0, 0.0],
            done: false,
        };

        // Actions 2 and 3 are illegal in the next state
        let mask = array![true, true, false, false];

        let masked_loss = agent
            .train_on_batch_masked(&[&exp], std::slice::from_ref(&mask), 0.99, 0.0)
            .unwrap();
        let unmasked_loss = agent.train_on_batch(&[&exp], 0.99, 0.0).unwrap();

        // Learning rate is zero, so the loss is purely a function of the target. The
        // unmasked target carries 0.99 * 100 from an action the agent may never play.
        assert!(
            unmasked_loss > masked_loss * 10.0,
            "masked loss {} should be far below unmasked {}",
            masked_loss,
            unmasked_loss
        );
    }

    #[test]
    fn a_next_state_with_no_legal_action_is_treated_as_terminal() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        let out = agent.target_network.layers.last_mut().unwrap();
        out.biases.fill(50.0);
        out.weights.fill(0.0);

        let exp = Experience {
            state: array![0.0, 0.0],
            action: 0,
            reward: 1.0,
            next_state: array![0.0, 0.0],
            done: false,
        };

        let dead_end = array![false, false, false, false];
        let terminal = Experience { done: true, ..exp.clone() };

        let dead_end_loss = agent
            .train_on_batch_masked(&[&exp], std::slice::from_ref(&dead_end), 0.99, 0.0)
            .unwrap();
        let terminal_loss = agent.train_on_batch(&[&terminal], 0.99, 0.0).unwrap();

        assert!(
            (dead_end_loss - terminal_loss).abs() < 1e-3,
            "dead end {} should match terminal {}",
            dead_end_loss,
            terminal_loss
        );
    }

    #[test]
    fn masked_training_rejects_masks_that_do_not_fit() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        let exp = Experience {
            state: array![0.0, 0.0],
            action: 0,
            reward: 1.0,
            next_state: array![0.0, 0.0],
            done: false,
        };

        // One mask short
        assert!(agent.train_on_batch_masked(&[&exp], &[], 0.99, 0.01).is_err());

        // Mask of the wrong width
        let narrow = array![true, true];
        assert!(agent
            .train_on_batch_masked(&[&exp], std::slice::from_ref(&narrow), 0.99, 0.01)
            .is_err());
    }

    #[test]
    fn weighted_training_with_zero_weights_leaves_the_network_alone() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        let exp = Experience {
            state: array![0.5, -0.5],
            action: 1,
            reward: 100.0,
            next_state: array![0.1, 0.2],
            done: true,
        };

        let before = agent.q_network.layers[0].weights.clone();
        agent
            .train_on_batch_weighted(&[&exp], &[0.0], 0.99, 0.1)
            .unwrap();
        let after = agent.q_network.layers[0].weights.clone();

        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-6, "a zero weight should produce no update");
        }
    }

    #[test]
    fn weighted_training_reports_the_td_error() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        // Terminal, so the target is exactly the reward and the TD error is
        // reward - Q(s, a)
        let exp = Experience {
            state: array![0.5, -0.5],
            action: 1,
            reward: 50.0,
            next_state: array![0.1, 0.2],
            done: true,
        };

        let predicted = agent.q_network.forward(exp.state.view())[1];
        let expected = (50.0 - predicted).abs();

        let td = agent
            .train_on_batch_weighted(&[&exp], &[1.0], 0.99, 0.0)
            .unwrap();

        assert_eq!(td.len(), 1);
        assert!(
            (td[0] - expected).abs() < 1e-3,
            "reported TD error {} against expected {}",
            td[0],
            expected
        );
    }

    #[test]
    fn weighted_training_rejects_a_mismatched_weight_count() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&[2, 8, 4], 0.0, optimizer, 1000, false);

        let exp = Experience {
            state: array![0.0, 0.0],
            action: 0,
            reward: 1.0,
            next_state: array![0.0, 0.0],
            done: false,
        };

        assert!(agent.train_on_batch_weighted(&[&exp], &[], 0.99, 0.01).is_err());
        assert!(agent
            .train_on_batch_weighted(&[&exp], &[1.0, 1.0], 0.99, 0.01)
            .is_err());
    }
}
