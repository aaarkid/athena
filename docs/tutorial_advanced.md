# Advanced Athena Tutorial

This tutorial covers advanced topics for experienced users who want to push the boundaries of what's possible with Athena.

## Table of Contents

1. [Custom Environments](#custom-environments)
2. [Multi-Agent Reinforcement Learning](#multi-agent-reinforcement-learning)
3. [Hierarchical RL](#hierarchical-rl)
4. [Custom Neural Network Architectures](#custom-neural-network-architectures)
5. [Advanced Training Techniques](#advanced-training-techniques)
6. [Performance Optimization](#performance-optimization)
7. [Research Applications](#research-applications)

## Custom Environments

### Creating Complex Environments

Let's build a multi-objective environment with continuous and discrete elements:

```rust
use athena::types::{State, Action, StateSpace, ActionSpace};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Complex environment with mixed action spaces
pub struct MultiObjectiveEnv {
    // Continuous state components
    position: Array1<f32>,
    velocity: Array1<f32>,
    
    // Discrete state components
    mode: usize,
    inventory: HashMap<String, i32>,
    
    // Environment dynamics
    physics: PhysicsEngine,
    objectives: Vec<Box<dyn Objective>>,
    
    // Rendering
    renderer: Option<Box<dyn Renderer>>,
}

impl Environment for MultiObjectiveEnv {
    type State = ComplexState;
    type Action = MixedAction;
    
    fn reset(&mut self) -> Self::State {
        // Reset all components
        self.position = Array1::zeros(3);
        self.velocity = Array1::zeros(3);
        self.mode = 0;
        self.inventory.clear();
        
        // Return complex state
        ComplexState {
            continuous: self.get_continuous_state(),
            discrete: self.get_discrete_state(),
            graph: self.get_graph_state(),
        }
    }
    
    fn step(&mut self, action: Self::Action) -> (Self::State, f32, bool) {
        // Process continuous actions
        if let Some(forces) = action.continuous {
            self.physics.apply_forces(&forces);
        }
        
        // Process discrete actions
        if let Some(mode_action) = action.discrete {
            self.process_discrete_action(mode_action);
        }
        
        // Update physics
        self.physics.step(self.dt);
        
        // Calculate multi-objective reward
        let rewards: Vec<f32> = self.objectives
            .iter()
            .map(|obj| obj.calculate_reward(self))
            .collect();
        
        // Weighted sum or Pareto optimization
        let total_reward = self.combine_rewards(&rewards);
        
        // Check termination conditions
        let done = self.check_termination();
        
        (self.get_state(), total_reward, done)
    }
}

/// Custom state representation
#[derive(Clone, Debug)]
pub struct ComplexState {
    pub continuous: Array1<f32>,
    pub discrete: Vec<i32>,
    pub graph: GraphState,
}

impl State for ComplexState {
    fn to_array(&self) -> Array1<f32> {
        // Flatten all components into a single array
        let mut result = self.continuous.clone();
        
        // Append one-hot encoded discrete states
        for &val in &self.discrete {
            let mut one_hot = Array1::zeros(self.discrete_dim);
            one_hot[val as usize] = 1.0;
            result.append(Axis(0), one_hot.view()).unwrap();
        }
        
        // Append graph features
        result.append(Axis(0), self.graph.to_array().view()).unwrap();
        
        result
    }
    
    fn dim(&self) -> usize {
        self.continuous.len() + 
        self.discrete.len() * self.discrete_dim +
        self.graph.dim()
    }
}
```

### Environment Wrappers

Create wrappers to modify environment behavior:

```rust
/// Wrapper that adds curriculum learning
pub struct CurriculumWrapper<E: Environment> {
    env: E,
    difficulty: f32,
    schedule: CurriculumSchedule,
}

impl<E: Environment> CurriculumWrapper<E> {
    pub fn new(env: E, schedule: CurriculumSchedule) -> Self {
        Self {
            env,
            difficulty: 0.0,
            schedule,
        }
    }
    
    pub fn update_difficulty(&mut self, performance: f32) {
        self.difficulty = self.schedule.get_difficulty(performance);
        self.env.set_difficulty(self.difficulty);
    }
}

/// Wrapper for multi-agent environments
pub struct MultiAgentWrapper<E: Environment> {
    envs: Vec<E>,
    communication: CommunicationProtocol,
}

impl<E: Environment> MultiAgentWrapper<E> {
    pub fn step_all(&mut self, actions: Vec<E::Action>) -> Vec<(E::State, f32, bool)> {
        // Collect all actions
        let joint_action = self.combine_actions(&actions);
        
        // Step all environments
        let mut results = Vec::new();
        for (i, env) in self.envs.iter_mut().enumerate() {
            // Each agent sees different observations
            let local_obs = self.get_local_observation(i);
            let result = env.step(actions[i].clone());
            results.push(result);
        }
        
        // Handle communication between agents
        self.communication.exchange_messages(&mut self.envs);
        
        results
    }
}
```

## Multi-Agent Reinforcement Learning

### Centralized Training, Decentralized Execution (CTDE)

```rust
use athena::algorithms::{PPOAgent, PPOBuilder};
use athena::network::NeuralNetwork;

/// Multi-agent coordinator using CTDE
pub struct CTDECoordinator {
    agents: Vec<PPOAgent>,
    critic: NeuralNetwork,  // Centralized critic
    communication: Option<CommNetwork>,
}

impl CTDECoordinator {
    pub fn new(n_agents: usize, obs_dim: usize, action_dim: usize) -> Result<Self> {
        let mut agents = Vec::new();
        
        // Create decentralized actors
        for i in 0..n_agents {
            let agent = PPOBuilder::new()
                .input_dim(obs_dim)
                .action_dim(action_dim)
                .hidden_dims(vec![256, 256])
                .build()?;
            agents.push(agent);
        }
        
        // Create centralized critic
        let critic_input = obs_dim * n_agents;  // All observations
        let critic = NeuralNetwork::new(
            &[critic_input, 512, 512, 1],
            &[Activation::Relu, Activation::Relu, Activation::Linear],
            OptimizerWrapper::Adam(Adam::new(3e-4, 0.9, 0.999, 1e-8)),
        );
        
        Ok(Self {
            agents,
            critic,
            communication: None,
        })
    }
    
    pub fn act(&mut self, observations: &[Array1<f32>]) -> Result<Vec<usize>> {
        let mut actions = Vec::new();
        
        for (i, obs) in observations.iter().enumerate() {
            // Each agent acts based on local observation
            let action = self.agents[i].act(obs)?;
            actions.push(action);
        }
        
        Ok(actions)
    }
    
    pub fn train(&mut self, batch: &MultiAgentBatch) -> Result<()> {
        // Centralized value estimation
        let joint_obs = self.concatenate_observations(&batch.observations);
        let values = self.critic.forward_batch(joint_obs.view());
        
        // Decentralized policy updates
        for (i, agent) in self.agents.iter_mut().enumerate() {
            let agent_batch = batch.get_agent_batch(i);
            agent.train_with_values(
                &agent_batch,
                values.slice(s![.., i]).to_owned(),
            )?;
        }
        
        // Update centralized critic
        let critic_loss = self.compute_critic_loss(&batch, &values);
        self.critic.backward_batch(critic_loss.view());
        
        Ok(())
    }
}

/// Communication network for agent coordination
pub struct CommNetwork {
    hidden_dim: usize,
    message_dim: usize,
    attention: MultiHeadAttention,
}

impl CommNetwork {
    pub fn communicate(&mut self, observations: &[Array1<f32>]) -> Vec<Array1<f32>> {
        // Generate messages from each agent
        let messages = observations.iter()
            .map(|obs| self.encode_message(obs))
            .collect::<Vec<_>>();
        
        // Apply attention mechanism
        let attended_messages = self.attention.attend(&messages);
        
        // Aggregate messages for each agent
        observations.iter().zip(attended_messages.iter())
            .map(|(obs, msg)| concatenate![Axis(0), *obs, *msg])
            .collect()
    }
}
```

### Competitive Multi-Agent Training

```rust
/// Self-play training system
pub struct SelfPlayTrainer {
    main_agent: DqnAgent,
    opponent_pool: Vec<DqnAgent>,
    elo_ratings: HashMap<usize, f32>,
    matchmaking: MatchmakingSystem,
}

impl SelfPlayTrainer {
    pub fn train_episode(&mut self) -> Result<()> {
        // Select opponent based on skill level
        let opponent_id = self.matchmaking.select_opponent(
            self.get_main_elo(),
            &self.elo_ratings,
        );
        
        let opponent = &mut self.opponent_pool[opponent_id];
        
        // Play match
        let result = self.play_match(&mut self.main_agent, opponent)?;
        
        // Update ELO ratings
        self.update_elo(result);
        
        // Add to opponent pool if main agent improved significantly
        if self.should_snapshot() {
            let snapshot = self.main_agent.clone();
            self.opponent_pool.push(snapshot);
            self.elo_ratings.insert(
                self.opponent_pool.len() - 1,
                self.get_main_elo(),
            );
        }
        
        Ok(())
    }
    
    fn play_match(&mut self, agent1: &mut DqnAgent, agent2: &mut DqnAgent) 
        -> Result<MatchResult> {
        let mut env = CompetitiveEnv::new();
        let mut state = env.reset();
        
        loop {
            // Agent 1 acts
            let action1 = agent1.act(state.player1_view())?;
            
            // Agent 2 acts
            let action2 = agent2.act(state.player2_view())?;
            
            // Environment step
            let (next_state, rewards, done) = env.step(action1, action2);
            
            // Store experiences for both agents
            self.store_experience(agent1, state.player1_view(), action1, rewards.0);
            self.store_experience(agent2, state.player2_view(), action2, rewards.1);
            
            if done {
                return Ok(MatchResult::from_rewards(rewards));
            }
            
            state = next_state;
        }
    }
}
```

## Hierarchical RL

### Options Framework

```rust
/// High-level option (temporally extended action)
pub trait Option {
    fn initiation_set(&self, state: &Array1<f32>) -> bool;
    fn policy(&mut self, state: &Array1<f32>) -> Result<usize>;
    fn termination(&self, state: &Array1<f32>) -> f32;
}

/// Hierarchical agent with options
pub struct HierarchicalAgent {
    meta_controller: DqnAgent,      // Selects options
    options: Vec<Box<dyn Option>>,  // Low-level policies
    current_option: Option<usize>,
}

impl HierarchicalAgent {
    pub fn act(&mut self, state: &Array1<f32>) -> Result<usize> {
        // Check if current option should terminate
        if let Some(option_id) = self.current_option {
            let termination_prob = self.options[option_id].termination(state);
            if rand::random::<f32>() < termination_prob {
                self.current_option = None;
            }
        }
        
        // Select new option if needed
        if self.current_option.is_none() {
            // Get available options
            let available: Vec<usize> = self.options
                .iter()
                .enumerate()
                .filter(|(_, opt)| opt.initiation_set(state))
                .map(|(i, _)| i)
                .collect();
            
            // Meta-controller selects from available options
            let option_id = self.meta_controller.act_masked(state, &available)?;
            self.current_option = Some(option_id);
        }
        
        // Execute current option
        let option_id = self.current_option.unwrap();
        self.options[option_id].policy(state)
    }
    
    pub fn train(&mut self, experiences: &[HierarchicalExperience]) -> Result<()> {
        // Train meta-controller on option-level transitions
        let option_transitions = self.extract_option_transitions(experiences);
        self.meta_controller.train_on_batch(&option_transitions, 0.001)?;
        
        // Train each option on primitive transitions
        for (option_id, option) in self.options.iter_mut().enumerate() {
            let option_experiences: Vec<_> = experiences
                .iter()
                .filter(|e| e.option == option_id)
                .collect();
            
            if !option_experiences.is_empty() {
                option.train(&option_experiences)?;
            }
        }
        
        Ok(())
    }
}

/// Learned option using neural networks
pub struct NeuralOption {
    policy: PPOAgent,
    termination_net: NeuralNetwork,
    initiation_net: NeuralNetwork,
}

impl Option for NeuralOption {
    fn initiation_set(&self, state: &Array1<f32>) -> bool {
        let output = self.initiation_net.forward(state.view());
        output[0] > 0.5  // Sigmoid output
    }
    
    fn policy(&mut self, state: &Array1<f32>) -> Result<usize> {
        self.policy.act(state)
    }
    
    fn termination(&self, state: &Array1<f32>) -> f32 {
        let output = self.termination_net.forward(state.view());
        output[0]  // Probability of termination
    }
}
```

## Custom Neural Network Architectures

### Attention Mechanisms

```rust
use athena::layers::{Layer, LayerTrait};

/// Self-attention layer
pub struct SelfAttentionLayer {
    query_projection: Array2<f32>,
    key_projection: Array2<f32>,
    value_projection: Array2<f32>,
    output_projection: Array2<f32>,
    num_heads: usize,
    head_dim: usize,
}

impl LayerTrait for SelfAttentionLayer {
    fn forward(&mut self, input: ArrayView2<f32>) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        
        // Project to Q, K, V
        let queries = input.dot(&self.query_projection);
        let keys = input.dot(&self.key_projection);
        let values = input.dot(&self.value_projection);
        
        // Reshape for multi-head attention
        let queries = self.reshape_for_heads(&queries);
        let keys = self.reshape_for_heads(&keys);
        let values = self.reshape_for_heads(&values);
        
        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = queries.dot(&keys.t()) / scale;
        let attention_weights = self.softmax(&scores);
        
        // Apply attention to values
        let attended = attention_weights.dot(&values);
        
        // Reshape and project output
        let output = self.reshape_from_heads(&attended);
        output.dot(&self.output_projection)
    }
    
    fn backward(&mut self, output_gradient: ArrayView2<f32>) 
        -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        // Implement attention backward pass
        // This is complex - simplified version shown
        let input_gradient = output_gradient.dot(&self.output_projection.t());
        let weight_gradient = self.compute_weight_gradients(&output_gradient);
        let bias_gradient = output_gradient.sum_axis(Axis(0));
        
        (input_gradient, weight_gradient, bias_gradient)
    }
}

/// Transformer block for RL
pub struct TransformerBlock {
    attention: SelfAttentionLayer,
    feed_forward: Vec<Layer>,
    layer_norm1: LayerNormalization,
    layer_norm2: LayerNormalization,
}

impl TransformerBlock {
    pub fn forward(&mut self, input: ArrayView2<f32>) -> Array2<f32> {
        // Self-attention with residual connection
        let attended = self.attention.forward(input);
        let norm1 = self.layer_norm1.forward(&(input + attended));
        
        // Feed-forward with residual connection
        let mut ff_output = norm1.clone();
        for layer in &mut self.feed_forward {
            ff_output = layer.forward(ff_output.view());
        }
        
        self.layer_norm2.forward(&(norm1 + ff_output))
    }
}
```

### Graph Neural Networks

```rust
/// Graph neural network layer for relational reasoning
pub struct GraphConvolutionLayer {
    node_mlp: NeuralNetwork,
    edge_mlp: NeuralNetwork,
    aggregation: AggregationFunction,
}

impl GraphConvolutionLayer {
    pub fn forward(&mut self, 
                   node_features: &Array2<f32>,
                   edge_features: &Array3<f32>,
                   adjacency: &Array2<bool>) -> Array2<f32> {
        let num_nodes = node_features.shape()[0];
        let mut new_features = Array2::zeros(node_features.dim());
        
        // Process each node
        for i in 0..num_nodes {
            let mut messages = Vec::new();
            
            // Collect messages from neighbors
            for j in 0..num_nodes {
                if adjacency[[i, j]] {
                    // Compute edge message
                    let edge_input = concatenate![
                        Axis(0),
                        node_features.row(i),
                        node_features.row(j),
                        edge_features.slice(s![i, j, ..])
                    ];
                    
                    let message = self.edge_mlp.forward(edge_input.view());
                    messages.push(message);
                }
            }
            
            // Aggregate messages
            let aggregated = self.aggregation.aggregate(&messages);
            
            // Update node features
            let node_input = concatenate![
                Axis(0),
                node_features.row(i),
                aggregated.view()
            ];
            
            new_features.row_mut(i).assign(
                &self.node_mlp.forward(node_input.view())
            );
        }
        
        new_features
    }
}

/// Relational Deep RL agent
pub struct RelationalAgent {
    encoder: GraphNeuralNetwork,
    policy_head: NeuralNetwork,
    value_head: NeuralNetwork,
}

impl RelationalAgent {
    pub fn forward(&mut self, graph_state: &GraphState) -> (Array1<f32>, f32) {
        // Encode graph structure
        let encoded = self.encoder.forward(
            &graph_state.node_features,
            &graph_state.edge_features,
            &graph_state.adjacency,
        );
        
        // Global pooling
        let global_features = encoded.mean_axis(Axis(0)).unwrap();
        
        // Compute policy and value
        let policy = self.policy_head.forward(global_features.view());
        let value = self.value_head.forward(global_features.view())[0];
        
        (policy, value)
    }
}
```

## Advanced Training Techniques

### Population-Based Training

```rust
/// Population-based training coordinator
pub struct PopulationBasedTraining {
    population: Vec<TrainableAgent>,
    hyperparameters: Vec<HyperparameterSet>,
    performance_history: Vec<Vec<f32>>,
    exploit_interval: usize,
}

impl PopulationBasedTraining {
    pub fn train_generation(&mut self) -> Result<()> {
        // Train all agents in parallel
        let performances: Vec<f32> = (0..self.population.len())
            .into_par_iter()
            .map(|i| {
                self.train_agent(i).unwrap()
            })
            .collect();
        
        // Record performance
        for (i, perf) in performances.iter().enumerate() {
            self.performance_history[i].push(*perf);
        }
        
        // Exploit and explore
        if self.should_exploit() {
            self.exploit_and_explore()?;
        }
        
        Ok(())
    }
    
    fn exploit_and_explore(&mut self) -> Result<()> {
        let performances = self.get_recent_performances();
        let rankings = self.rank_agents(&performances);
        
        // Bottom 20% copy from top 20%
        let bottom_20_percent = self.population.len() / 5;
        let top_20_percent = self.population.len() / 5;
        
        for i in 0..bottom_20_percent {
            let worst_idx = rankings[i];
            let best_idx = rankings[rankings.len() - 1 - i];
            
            // Copy weights
            self.population[worst_idx] = self.population[best_idx].clone();
            
            // Copy and perturb hyperparameters
            self.hyperparameters[worst_idx] = self.hyperparameters[best_idx].clone();
            self.hyperparameters[worst_idx].perturb();
        }
        
        Ok(())
    }
}

/// Hyperparameter set with perturbation
#[derive(Clone)]
pub struct HyperparameterSet {
    learning_rate: f32,
    batch_size: usize,
    entropy_coeff: f32,
    clip_epsilon: f32,
    // ... other hyperparameters
}

impl HyperparameterSet {
    pub fn perturb(&mut self) {
        let mut rng = rand::thread_rng();
        
        // Perturb with 20% probability
        if rng.gen::<f32>() < 0.2 {
            self.learning_rate *= rng.gen_range(0.8..1.2);
        }
        if rng.gen::<f32>() < 0.2 {
            self.batch_size = (self.batch_size as f32 * rng.gen_range(0.8..1.2)) as usize;
        }
        // ... perturb other parameters
    }
}
```

### Curiosity-Driven Exploration

```rust
/// Intrinsic curiosity module
pub struct CuriosityModule {
    forward_model: NeuralNetwork,
    inverse_model: NeuralNetwork,
    feature_encoder: NeuralNetwork,
}

impl CuriosityModule {
    pub fn compute_intrinsic_reward(
        &mut self,
        state: &Array1<f32>,
        action: usize,
        next_state: &Array1<f32>,
    ) -> f32 {
        // Encode states to feature space
        let phi_s = self.feature_encoder.forward(state.view());
        let phi_s_next = self.feature_encoder.forward(next_state.view());
        
        // Forward model prediction
        let action_one_hot = self.action_to_one_hot(action);
        let predicted_phi_s_next = self.forward_model.forward(
            concatenate![Axis(0), phi_s, action_one_hot].view()
        );
        
        // Prediction error as intrinsic reward
        let prediction_error = (&phi_s_next - &predicted_phi_s_next)
            .mapv(|x| x * x)
            .sum();
        
        prediction_error * self.curiosity_weight
    }
    
    pub fn train(
        &mut self,
        state: &Array1<f32>,
        action: usize,
        next_state: &Array1<f32>,
    ) -> Result<()> {
        // Train forward model
        let phi_s = self.feature_encoder.forward(state.view());
        let phi_s_next = self.feature_encoder.forward(next_state.view());
        let action_one_hot = self.action_to_one_hot(action);
        
        let forward_input = concatenate![Axis(0), phi_s, action_one_hot];
        let forward_loss = self.forward_model.train(
            forward_input.view(),
            phi_s_next.view(),
            0.001,
        );
        
        // Train inverse model
        let inverse_input = concatenate![Axis(0), phi_s, phi_s_next];
        let inverse_loss = self.inverse_model.train(
            inverse_input.view(),
            action_one_hot.view(),
            0.001,
        );
        
        Ok(())
    }
}

/// Agent with curiosity-driven exploration
pub struct CuriousAgent {
    base_agent: DqnAgent,
    curiosity: CuriosityModule,
    extrinsic_weight: f32,
    intrinsic_weight: f32,
}

impl CuriousAgent {
    pub fn train_step(
        &mut self,
        experience: &Experience,
    ) -> Result<()> {
        // Compute intrinsic reward
        let intrinsic_reward = self.curiosity.compute_intrinsic_reward(
            &experience.state,
            experience.action,
            &experience.next_state,
        );
        
        // Combine rewards
        let total_reward = self.extrinsic_weight * experience.reward +
                          self.intrinsic_weight * intrinsic_reward;
        
        // Create modified experience
        let modified_exp = Experience {
            state: experience.state.clone(),
            action: experience.action,
            reward: total_reward,
            next_state: experience.next_state.clone(),
            done: experience.done,
        };
        
        // Train base agent
        self.base_agent.train_on_batch(&[modified_exp], 0.001)?;
        
        // Train curiosity module
        self.curiosity.train(
            &experience.state,
            experience.action,
            &experience.next_state,
        )?;
        
        Ok(())
    }
}
```

## Performance Optimization

### Custom SIMD Operations

```rust
use std::arch::x86_64::*;

/// SIMD-accelerated activation functions
pub struct SimdActivations;

impl SimdActivations {
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(input: &mut [f32]) {
        let zero = _mm256_setzero_ps();
        
        for chunk in input.chunks_exact_mut(8) {
            let values = _mm256_loadu_ps(chunk.as_ptr());
            let result = _mm256_max_ps(values, zero);
            _mm256_storeu_ps(chunk.as_mut_ptr(), result);
        }
        
        // Handle remaining elements
        for x in input.chunks_exact_mut(8).into_remainder() {
            *x = x.max(0.0);
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn tanh_approx_avx2(input: &mut [f32]) {
        // Fast approximation of tanh using rational function
        for chunk in input.chunks_exact_mut(8) {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let x2 = _mm256_mul_ps(x, x);
            
            // Padé approximant of tanh
            let num = _mm256_add_ps(
                x,
                _mm256_mul_ps(
                    _mm256_set1_ps(0.16489087),
                    _mm256_mul_ps(x, x2)
                )
            );
            
            let den = _mm256_add_ps(
                _mm256_set1_ps(1.0),
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_set1_ps(0.49868), x2),
                    _mm256_mul_ps(
                        _mm256_set1_ps(0.03679),
                        _mm256_mul_ps(x2, x2)
                    )
                )
            );
            
            let result = _mm256_div_ps(num, den);
            _mm256_storeu_ps(chunk.as_mut_ptr(), result);
        }
    }
}

/// Cache-friendly matrix operations
pub struct CacheFriendlyOps;

impl CacheFriendlyOps {
    /// Tiled matrix multiplication for better cache usage
    pub fn matmul_tiled(
        a: &Array2<f32>,
        b: &Array2<f32>,
        tile_size: usize,
    ) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        
        let mut c = Array2::zeros((m, n));
        
        // Tile loops for cache efficiency
        for i_tile in (0..m).step_by(tile_size) {
            for j_tile in (0..n).step_by(tile_size) {
                for k_tile in (0..k).step_by(tile_size) {
                    // Compute tile boundaries
                    let i_end = (i_tile + tile_size).min(m);
                    let j_end = (j_tile + tile_size).min(n);
                    let k_end = (k_tile + tile_size).min(k);
                    
                    // Multiply tiles
                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = c[[i, j]];
                            for k in k_tile..k_end {
                                sum += a[[i, k]] * b[[k, j]];
                            }
                            c[[i, j]] = sum;
                        }
                    }
                }
            }
        }
        
        c
    }
}
```

### Memory Pool for Replay Buffer

```rust
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Memory-efficient replay buffer using object pooling
pub struct PooledReplayBuffer {
    experiences: VecDeque<Arc<Experience>>,
    pool: Arc<Mutex<Vec<Box<Experience>>>>,
    capacity: usize,
}

impl PooledReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        // Pre-allocate experience objects
        let pool = Arc::new(Mutex::new(
            (0..capacity)
                .map(|_| Box::new(Experience::default()))
                .collect()
        ));
        
        Self {
            experiences: VecDeque::with_capacity(capacity),
            pool,
            capacity,
        }
    }
    
    pub fn add(&mut self, state: Array1<f32>, action: usize, 
               reward: f32, next_state: Array1<f32>, done: bool) {
        // Get experience from pool or allocate new
        let experience = {
            let mut pool = self.pool.lock().unwrap();
            pool.pop().unwrap_or_else(|| Box::new(Experience::default()))
        };
        
        // Fill experience
        let mut exp = experience;
        exp.state = state;
        exp.action = action;
        exp.reward = reward;
        exp.next_state = next_state;
        exp.done = done;
        
        let exp_arc = Arc::new(*exp);
        
        // Add to buffer
        if self.experiences.len() >= self.capacity {
            if let Some(old_exp) = self.experiences.pop_front() {
                // Return to pool if unique reference
                if let Ok(exp) = Arc::try_unwrap(old_exp) {
                    let mut pool = self.pool.lock().unwrap();
                    pool.push(Box::new(exp));
                }
            }
        }
        
        self.experiences.push_back(exp_arc);
    }
}
```

## Research Applications

### Meta-Learning

```rust
/// Model-Agnostic Meta-Learning (MAML) for RL
pub struct MAMLAgent {
    meta_network: NeuralNetwork,
    inner_lr: f32,
    outer_lr: f32,
    inner_steps: usize,
}

impl MAMLAgent {
    pub fn meta_train(&mut self, task_batch: &[Task]) -> Result<()> {
        let mut meta_gradients = vec![];
        
        for task in task_batch {
            // Clone network for inner loop
            let mut task_network = self.meta_network.clone();
            
            // Inner loop: adapt to specific task
            for _ in 0..self.inner_steps {
                let batch = task.sample_batch(32);
                task_network.train_minibatch(
                    batch.states.view(),
                    batch.targets.view(),
                    self.inner_lr,
                );
            }
            
            // Compute meta-gradient
            let test_batch = task.sample_batch(32);
            let loss = task_network.compute_loss(
                test_batch.states.view(),
                test_batch.targets.view(),
            );
            
            let task_gradients = task_network.backward(loss.view());
            meta_gradients.push(task_gradients);
        }
        
        // Average gradients across tasks
        let avg_gradients = self.average_gradients(&meta_gradients);
        
        // Meta-update
        self.meta_network.apply_gradients(&avg_gradients, self.outer_lr);
        
        Ok(())
    }
    
    pub fn adapt_to_new_task(&self, task: &Task) -> NeuralNetwork {
        let mut adapted_network = self.meta_network.clone();
        
        // Few-shot adaptation
        for _ in 0..self.inner_steps {
            let batch = task.sample_batch(16);  // Small batch for few-shot
            adapted_network.train_minibatch(
                batch.states.view(),
                batch.targets.view(),
                self.inner_lr,
            );
        }
        
        adapted_network
    }
}
```

### Offline RL

```rust
/// Conservative Q-Learning (CQL) for offline RL
pub struct CQLAgent {
    q_network: NeuralNetwork,
    target_network: NeuralNetwork,
    alpha: f32,  // CQL regularization weight
}

impl CQLAgent {
    pub fn train_offline(&mut self, dataset: &OfflineDataset) -> Result<()> {
        let batch = dataset.sample_batch(256);
        
        // Standard Q-learning loss
        let q_values = self.q_network.forward_batch(batch.states.view());
        let next_q_values = self.target_network.forward_batch(batch.next_states.view());
        
        let targets = batch.rewards + 0.99 * next_q_values.max_axis(Axis(1)).unwrap();
        let td_loss = (&q_values - &targets).mapv(|x| x * x).mean().unwrap();
        
        // CQL regularization: minimize Q-values for OOD actions
        let ood_actions = self.sample_ood_actions(&batch.states);
        let ood_q_values = self.evaluate_ood_actions(&batch.states, &ood_actions);
        let cql_loss = ood_q_values.mean().unwrap();
        
        // Combined loss
        let total_loss = td_loss + self.alpha * cql_loss;
        
        // Backward pass
        self.q_network.backward(total_loss);
        
        Ok(())
    }
    
    fn sample_ood_actions(&self, states: &Array2<f32>) -> Array2<usize> {
        // Sample actions not in the dataset
        // Implementation depends on action space
        unimplemented!()
    }
}
```

## Conclusion

These advanced techniques demonstrate the flexibility and power of Athena for cutting-edge RL research and applications. Key takeaways:

1. **Modularity**: Build complex systems from simple components
2. **Extensibility**: Create custom layers, environments, and algorithms
3. **Performance**: Optimize critical paths with SIMD and caching
4. **Research**: Implement state-of-the-art algorithms easily

For more examples and research papers implemented in Athena, visit our [GitHub repository](https://github.com/athena-rl/papers).

Happy researching!