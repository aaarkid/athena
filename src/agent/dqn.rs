use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::OptimizerWrapper;
use crate::replay_buffer::Experience;
use crate::error::{Result, AthenaError};
use rand::{Rng, rngs::ThreadRng};
use ndarray::{Array2, ArrayView1};
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
/// let action = agent.act(state.view());
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
    #[serde(skip)]
    pub rng: ThreadRng,
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
        
        // Create activations (ReLU for hidden layers, Linear for output)
        let mut activations = vec![Activation::Relu; layer_sizes.len() - 2];
        activations.push(Activation::Linear);
        
        // Create main and target networks
        let q_network = NeuralNetwork::new(layer_sizes, &activations, optimizer.clone());
        let target_network = NeuralNetwork::new(layer_sizes, &activations, optimizer);
        
        let rng = rand::thread_rng();
        
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
            // Exploitation: best action from Q-network
            let q_values = self.q_network.forward(state);
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
        self.epsilon = epsilon.max(0.0).min(1.0);
    }
    
    /// Update target network weights from main network
    pub fn update_target_network(&mut self) {
        self.target_network = self.q_network.clone();
    }
    
    /// Train the agent on a batch of experiences
    pub fn train_on_batch(
        &mut self, 
        experiences: &[&Experience], 
        gamma: f32, 
        learning_rate: f32
    ) -> Result<f32> {
        if experiences.is_empty() {
            return Err(AthenaError::EmptyBuffer("No experiences to train on".to_string()));
        }
        
        let batch_size = experiences.len();
        let state_size = experiences[0].state.len();
        let _num_actions = self.q_network.layers.last()
            .ok_or_else(|| AthenaError::TrainingError("No layers in network".to_string()))?
            .biases.len();
        
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
        
        // Get current Q-values
        let current_q_values = self.q_network.forward_batch(states.view());
        
        // Calculate target Q-values
        let mut target_q_values = current_q_values.clone();
        
        if self.use_double_dqn {
            // Double DQN: use main network to select actions, target network to evaluate
            let next_q_values_main = self.q_network.forward_batch(next_states.view());
            let next_q_values_target = self.target_network.forward_batch(next_states.view());
            
            for i in 0..batch_size {
                if !dones[i] {
                    // Find best action using main network
                    let best_action = next_q_values_main.row(i)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    
                    // Use target network to evaluate that action
                    let target_value = rewards[i] + gamma * next_q_values_target[[i, best_action]];
                    target_q_values[[i, actions[i]]] = target_value;
                } else {
                    target_q_values[[i, actions[i]]] = rewards[i];
                }
            }
        } else {
            // Standard DQN: use target network for both selection and evaluation
            let next_q_values = self.target_network.forward_batch(next_states.view());
            
            for i in 0..batch_size {
                if !dones[i] {
                    let max_next_q = next_q_values.row(i).iter()
                        .fold(f32::NEG_INFINITY, |max, &val| max.max(val));
                    let target_value = rewards[i] + gamma * max_next_q;
                    target_q_values[[i, actions[i]]] = target_value;
                } else {
                    target_q_values[[i, actions[i]]] = rewards[i];
                }
            }
        }
        
        // Train the network
        self.q_network.train_minibatch(states.view(), target_q_values.view(), learning_rate);
        
        // Calculate loss for monitoring
        let predictions = self.q_network.forward_batch(states.view());
        let loss = (&predictions - &target_q_values).mapv(|x| x * x).mean()
            .unwrap_or(f32::INFINITY);
        
        // Increment train steps
        self.train_steps += 1;
        
        // Update target network if needed
        self.update_counter += 1;
        if self.update_counter % self.target_update_freq == 0 {
            self.update_target_network();
        }
        
        Ok(loss)
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
        agent.rng = rand::thread_rng();
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
            let q_network = NeuralNetwork::new(&self.layer_sizes, &activations, optimizer.clone());
            let target_network = NeuralNetwork::new(&self.layer_sizes, &activations, optimizer);
            
            Ok(DqnAgent {
                q_network,
                target_network,
                epsilon: self.epsilon,
                target_update_freq: self.target_update_freq,
                update_counter: 0,
                use_double_dqn: self.use_double_dqn,
                train_steps: 0,
                rng: rand::thread_rng(),
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