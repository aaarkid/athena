# Multi-Agent Extension Implementation Plan for Athena

## Executive Summary

This document outlines the comprehensive plan for extending Athena with multi-agent capabilities, action masking, and belief states. These extensions will maintain backward compatibility while enabling sophisticated multi-agent reinforcement learning applications.

## Implementation Status

### ✅ Completed Phases (Weeks 1-9)

#### Phase 1: Action Masking ✅
- **MaskedLayer trait**: Generic interface for layers supporting action masking
- **MaskedSoftmax**: Softmax implementation that respects invalid actions
- **MaskedAgent trait**: Extension allowing agents to handle action constraints
- **Integration**: Full DQN support with epsilon-greedy exploration on valid actions only
- **Example**: `masked_cartpole.rs` demonstrating boundary-based action masking

#### Phase 2: Belief States ✅
- **BeliefState trait**: Core abstraction for partial observability
- **HistoryBelief**: Fixed-window observation/action history tracking
- **ParticleFilter**: Monte Carlo belief approximation with systematic resampling
- **BeliefAgent**: Wrapper adding belief tracking to any agent
- **BeliefDqnAgent**: Complete integration with DQN for POMDP solving
- **Example**: `belief_tracking.rs` with partially observable GridWorld

#### Phase 3: Multi-Agent Core ✅
- **MultiAgentEnvironment trait**: Supports both turn-based and simultaneous games
- **TurnBasedWrapper**: Adapts single-agent environments for multi-agent use
- **SelfPlayTrainer**: Population-based training with:
  - ELO rating system for agent ranking
  - Multiple sampling strategies (uniform, prioritized, league)
  - Experience replay per agent
- **CommunicationChannel trait**: Message passing infrastructure
- **BroadcastChannel**: Point-to-point and broadcast messaging
- **CommunicatingAgent**: Agents that encode/decode messages

## Feature Flags Configuration

```toml
# Cargo.toml additions
[features]
default = []
action-masking = []
belief-states = []
multi-agent = ["action-masking", "belief-states"]
cfr = ["multi-agent"]  # Counterfactual Regret Minimization
```

## Phase 1: Action Masking (Week 1-2)

### 1.1 Core Implementation

#### File: `src/layers/masked.rs`
```rust
use ndarray::{Array1, ArrayView1};
use crate::layers::traits::Layer;

/// Layer that applies masking before activation
pub trait MaskedLayer: Layer {
    /// Forward pass with optional action mask
    fn forward_masked(&mut self, input: ArrayView1<f32>, mask: Option<&Array1<bool>>) -> Array1<f32> {
        match mask {
            Some(m) => self.apply_mask(self.forward(input), m),
            None => self.forward(input),
        }
    }
    
    /// Apply mask to output values
    fn apply_mask(&self, output: Array1<f32>, mask: &Array1<bool>) -> Array1<f32>;
}

/// Masked softmax layer for action selection
pub struct MaskedSoftmax {
    temperature: f32,
}

impl MaskedSoftmax {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Layer for MaskedSoftmax {
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        softmax(input, self.temperature)
    }
    
    // Other required methods...
}

impl MaskedLayer for MaskedSoftmax {
    fn apply_mask(&self, mut output: Array1<f32>, mask: &Array1<bool>) -> Array1<f32> {
        // Set masked actions to -inf before softmax
        for (i, &is_valid) in mask.iter().enumerate() {
            if !is_valid {
                output[i] = f32::NEG_INFINITY;
            }
        }
        softmax(&output.view(), self.temperature)
    }
}
```

#### File: `src/agent/masked_agent.rs`
```rust
/// Extension trait for agents with action masking
pub trait MaskedAgent: Agent {
    /// Select action with invalid actions masked out
    fn act_masked(&self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> usize;
    
    /// Get Q-values with masking applied
    fn get_masked_q_values(&self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Array1<f32>;
}

/// Implementation for DQN agent
impl MaskedAgent for DqnAgent {
    fn act_masked(&self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> usize {
        let q_values = self.q_network.forward(state);
        
        // Apply mask
        let mut masked_q = q_values.clone();
        for (i, &is_valid) in action_mask.iter().enumerate() {
            if !is_valid {
                masked_q[i] = f32::NEG_INFINITY;
            }
        }
        
        // Epsilon-greedy with valid actions only
        if self.rng.gen::<f32>() < self.epsilon {
            // Random valid action
            let valid_actions: Vec<usize> = action_mask.iter()
                .enumerate()
                .filter(|(_, &valid)| valid)
                .map(|(i, _)| i)
                .collect();
            valid_actions[self.rng.gen_range(0..valid_actions.len())]
        } else {
            // Greedy from masked Q-values
            masked_q.argmax().unwrap()
        }
    }
    
    fn get_masked_q_values(&self, state: ArrayView1<f32>, action_mask: &Array1<bool>) -> Array1<f32> {
        let mut q_values = self.q_network.forward(state);
        for (i, &is_valid) in action_mask.iter().enumerate() {
            if !is_valid {
                q_values[i] = f32::NEG_INFINITY;
            }
        }
        q_values
    }
}
```

### 1.2 Integration Points

1. Update `src/lib.rs` to include masked module
2. Add tests in `src/agent/tests/masked_agent_tests.rs`
3. Create example: `examples/masked_cartpole.rs`

### 1.3 Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_masked_action_selection() {
        let agent = DqnAgent::new(&[4, 32, 32, 3], 0.0, optimizer, 100, false);
        let state = array![1.0, 2.0, 3.0, 4.0];
        let mask = array![true, false, true];  // Only actions 0 and 2 valid
        
        // Should never select action 1
        for _ in 0..100 {
            let action = agent.act_masked(state.view(), &mask);
            assert!(action != 1);
        }
    }
}
```

## Phase 2: Belief States (Week 3-5)

### 2.1 Core Abstractions

#### File: `src/belief/mod.rs`
```rust
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};

/// Core trait for belief state representations
pub trait BeliefState: Send + Sync {
    type Observation;
    type State;
    
    /// Update belief based on action and observation
    fn update(&mut self, action: usize, observation: &Self::Observation);
    
    /// Sample a concrete state from belief distribution
    fn sample(&self) -> Self::State;
    
    /// Get belief as feature vector for neural network
    fn to_feature_vector(&self) -> Array1<f32>;
    
    /// Reset belief to initial distribution
    fn reset(&mut self);
    
    /// Get entropy of belief distribution (uncertainty measure)
    fn entropy(&self) -> f32;
}

/// Belief state that tracks history
#[derive(Clone, Serialize, Deserialize)]
pub struct HistoryBelief {
    max_history: usize,
    action_history: Vec<usize>,
    observation_history: Vec<Array1<f32>>,
    embedding_size: usize,
}

impl HistoryBelief {
    pub fn new(max_history: usize, embedding_size: usize) -> Self {
        Self {
            max_history,
            action_history: Vec::new(),
            observation_history: Vec::new(),
            embedding_size,
        }
    }
}
```

#### File: `src/belief/particle_filter.rs`
```rust
/// Particle filter for belief state representation
pub struct ParticleFilter<S: Clone> {
    particles: Vec<S>,
    weights: Array1<f32>,
    transition_fn: Box<dyn Fn(&S, usize) -> S>,
    observation_fn: Box<dyn Fn(&S) -> Array1<f32>>,
    resampling_threshold: f32,
}

impl<S: Clone + Send + Sync> ParticleFilter<S> {
    pub fn new(
        num_particles: usize,
        initial_state_fn: impl Fn() -> S,
        transition_fn: impl Fn(&S, usize) -> S + 'static,
        observation_fn: impl Fn(&S) -> Array1<f32> + 'static,
    ) -> Self {
        let particles = (0..num_particles)
            .map(|_| initial_state_fn())
            .collect();
        let weights = Array1::ones(num_particles) / num_particles as f32;
        
        Self {
            particles,
            weights,
            transition_fn: Box::new(transition_fn),
            observation_fn: Box::new(observation_fn),
            resampling_threshold: 0.5,
        }
    }
    
    /// Update particles based on action and observation
    pub fn update_particles(&mut self, action: usize, observation: &Array1<f32>) {
        // Transition particles
        for particle in &mut self.particles {
            *particle = (self.transition_fn)(particle, action);
        }
        
        // Update weights based on observation likelihood
        for (i, particle) in self.particles.iter().enumerate() {
            let predicted_obs = (self.observation_fn)(particle);
            self.weights[i] *= observation_likelihood(&predicted_obs, observation);
        }
        
        // Normalize weights
        let sum = self.weights.sum();
        if sum > 0.0 {
            self.weights /= sum;
        }
        
        // Resample if effective sample size is low
        if self.effective_sample_size() < self.resampling_threshold * self.particles.len() as f32 {
            self.resample();
        }
    }
    
    fn resample(&mut self) {
        // Systematic resampling
        let n = self.particles.len();
        let mut new_particles = Vec::with_capacity(n);
        let cumsum = cumulative_sum(&self.weights);
        
        let step = 1.0 / n as f32;
        let mut u = rand::random::<f32>() * step;
        
        for _ in 0..n {
            let idx = cumsum.iter().position(|&w| w > u).unwrap_or(n - 1);
            new_particles.push(self.particles[idx].clone());
            u += step;
        }
        
        self.particles = new_particles;
        self.weights.fill(1.0 / n as f32);
    }
}
```

### 2.2 Belief-Aware Agents

#### File: `src/belief/belief_agent.rs`
```rust
/// Agent that maintains belief state
pub struct BeliefAgent<B: BeliefState, A: Agent> {
    belief: B,
    base_agent: A,
    belief_encoder: NeuralNetwork,
}

impl<B: BeliefState, A: Agent> BeliefAgent<B, A> {
    pub fn new(belief: B, base_agent: A, encoding_dim: usize) -> Self {
        let belief_size = belief.to_feature_vector().len();
        let belief_encoder = NeuralNetwork::new(&[belief_size, 128, encoding_dim]);
        
        Self {
            belief,
            base_agent,
            belief_encoder,
        }
    }
    
    /// Act based on belief state
    pub fn act_with_belief(&mut self, observation: &B::Observation) -> usize {
        // Get belief representation
        let belief_vector = self.belief.to_feature_vector();
        let encoded_belief = self.belief_encoder.forward(belief_vector.view());
        
        // Use encoded belief as state for base agent
        let action = self.base_agent.act(encoded_belief.view());
        
        // Update belief with taken action
        self.belief.update(action, observation);
        
        action
    }
}
```

## Phase 3: Multi-Agent Core (Week 6-9)

### 3.1 Environment Abstractions

#### File: `src/multi_agent/environment.rs`
```rust
/// Multi-agent environment trait
pub trait MultiAgentEnvironment: Send + Sync {
    type State;
    type Action;
    type Observation;
    
    /// Get number of agents
    fn num_agents(&self) -> usize;
    
    /// Get current active agent(s)
    fn active_agents(&self) -> Vec<usize>;
    
    /// Get observation for specific agent
    fn get_observation(&self, agent_id: usize) -> Self::Observation;
    
    /// Get legal actions for agent
    fn legal_actions(&self, agent_id: usize) -> Array1<bool>;
    
    /// Step environment with actions from active agents
    fn step(&mut self, actions: &[(usize, Self::Action)]) -> MultiAgentTransition<Self::Observation>;
    
    /// Check if episode is done
    fn is_terminal(&self) -> bool;
    
    /// Reset environment
    fn reset(&mut self) -> Vec<Self::Observation>;
}

/// Transition information for multi-agent step
#[derive(Clone, Debug)]
pub struct MultiAgentTransition<O> {
    pub observations: HashMap<usize, O>,
    pub rewards: HashMap<usize, f32>,
    pub done: bool,
    pub info: HashMap<String, Box<dyn Any>>,
}

/// Wrapper to use single-agent environments in multi-agent setting
pub struct TurnBasedWrapper<E: Environment> {
    env: E,
    num_agents: usize,
    current_player: usize,
}
```

### 3.2 Multi-Agent Training

#### File: `src/multi_agent/trainer.rs`
```rust
/// Self-play trainer for multi-agent systems
pub struct SelfPlayTrainer<A: Agent + Clone> {
    agent_pool: Vec<A>,
    update_interval: usize,
    sampling_strategy: SamplingStrategy,
    elo_ratings: Vec<f32>,
}

pub enum SamplingStrategy {
    Uniform,
    Prioritized { temperature: f32 },
    League { main_prob: f32, main_exploit_prob: f32 },
}

impl<A: Agent + Clone> SelfPlayTrainer<A> {
    pub fn train<E: MultiAgentEnvironment>(
        &mut self,
        env: &mut E,
        episodes: usize,
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut replay_buffers: Vec<ReplayBuffer> = (0..env.num_agents())
            .map(|_| ReplayBuffer::new(100000))
            .collect();
        
        for episode in 0..episodes {
            // Sample opponents
            let agents = self.sample_agents(env.num_agents());
            
            // Play episode
            let mut observations = env.reset();
            let mut episode_experiences = vec![Vec::new(); env.num_agents()];
            
            while !env.is_terminal() {
                let active = env.active_agents();
                let mut actions = Vec::new();
                
                for &agent_id in &active {
                    let obs = env.get_observation(agent_id);
                    let mask = env.legal_actions(agent_id);
                    let action = agents[agent_id].act_masked(obs.view(), &mask);
                    actions.push((agent_id, action));
                }
                
                let transition = env.step(&actions);
                
                // Store experiences
                for &agent_id in &active {
                    episode_experiences[agent_id].push(Experience {
                        state: observations[agent_id].clone(),
                        action: actions.iter().find(|(id, _)| *id == agent_id).unwrap().1,
                        reward: transition.rewards[&agent_id],
                        next_state: transition.observations[&agent_id].clone(),
                        done: transition.done,
                    });
                }
                
                observations = transition.observations.into_values().collect();
            }
            
            // Add to replay buffers
            for (buffer, experiences) in replay_buffers.iter_mut().zip(episode_experiences) {
                for exp in experiences {
                    buffer.add(exp);
                }
            }
            
            // Update agents
            if episode % self.update_interval == 0 {
                self.update_agents(&replay_buffers, batch_size);
            }
        }
        
        TrainingMetrics::default()
    }
}
```

### 3.3 Communication and Coordination

#### File: `src/multi_agent/communication.rs`
```rust
/// Message passing between agents
pub trait CommunicationChannel {
    type Message;
    
    fn send(&mut self, from: usize, to: usize, message: Self::Message);
    fn receive(&mut self, agent_id: usize) -> Vec<(usize, Self::Message)>;
    fn broadcast(&mut self, from: usize, message: Self::Message);
}

/// Agent with communication capabilities
pub struct CommunicatingAgent<A: Agent> {
    base_agent: A,
    message_encoder: NeuralNetwork,
    message_decoder: NeuralNetwork,
    comm_channel: Box<dyn CommunicationChannel<Message = Array1<f32>>>,
}

impl<A: Agent> CommunicatingAgent<A> {
    pub fn act_with_communication(&mut self, state: ArrayView1<f32>) -> (usize, Option<Array1<f32>>) {
        // Receive messages
        let messages = self.comm_channel.receive(self.id);
        
        // Encode messages
        let encoded_messages = self.encode_messages(&messages);
        
        // Combine with state
        let augmented_state = concatenate![Axis(0), state, encoded_messages.view()];
        
        // Get action
        let action = self.base_agent.act(augmented_state.view());
        
        // Generate message if needed
        let message = self.generate_message(state, action);
        
        (action, message)
    }
}
```

## Phase 4: Advanced Features - Detailed Implementation Plan

### Overview
Phase 4 represents the cutting-edge multi-agent RL features that will make Athena competitive with state-of-the-art frameworks. This phase focuses on game-theoretic solutions, evolutionary approaches, and large-scale training systems.

### 4.1 Counterfactual Regret Minimization (CFR)

#### Step 1: Create CFR Module Structure
```bash
src/multi_agent/cfr/
├── mod.rs           # Module exports
├── game_tree.rs     # Extensive form game representation  
├── solver.rs        # Core CFR algorithm
├── variants.rs      # CFR+, Linear CFR, Deep CFR
└── strategies.rs    # Strategy computation and storage
```

#### Step 2: Define Extensive Form Game Trait
```rust
// src/multi_agent/cfr/game_tree.rs
pub trait ExtensiveFormGame: Send + Sync {
    type State: Clone + Hash + Eq;
    type Action: Clone + Hash + Eq;
    type InfoSet: Clone + Hash + Eq;
    
    fn initial_state(&self) -> Self::State;
    fn is_terminal(&self, state: &Self::State) -> bool;
    fn current_player(&self, state: &Self::State) -> Option<usize>;
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;
    fn apply_action(&self, state: &Self::State, action: Self::Action) -> Self::State;
    fn utility(&self, state: &Self::State, player: usize) -> f32;
    fn information_set(&self, state: &Self::State, player: usize) -> Self::InfoSet;
}
```

#### Step 3: Implement Core CFR Solver
```rust
// src/multi_agent/cfr/solver.rs
pub struct CFRSolver<G: ExtensiveFormGame> {
    regret_sum: HashMap<G::InfoSet, HashMap<G::Action, f32>>,
    strategy_sum: HashMap<G::InfoSet, HashMap<G::Action, f32>>,
    iterations: usize,
    _phantom: PhantomData<G>,
}

impl<G: ExtensiveFormGame> CFRSolver<G> {
    pub fn solve(&mut self, game: &G, iterations: usize) -> Strategy<G> {
        for _ in 0..iterations {
            for player in 0..game.num_players() {
                self.cfr_iteration(game, game.initial_state(), player, vec![1.0; game.num_players()]);
            }
        }
        self.compute_average_strategy()
    }
}
```

#### Step 4: Implement CFR Variants
- **CFR+**: Only accumulate positive regrets
- **Linear CFR**: Weight recent iterations more heavily
- **Deep CFR**: Use neural networks for large state spaces

#### Step 5: Create Example Game Implementations
- Kuhn Poker (simple bluffing game)
- Leduc Hold'em (larger poker variant)
- Simplified Coup (using game rules from earlier)

#### Testing Requirements:
- Verify Nash equilibrium convergence on known games
- Compare against analytical solutions where available
- Benchmark performance on different game sizes

### 4.2 Advanced Population Training

#### Step 1: Create Population Module
```bash
src/multi_agent/population/
├── mod.rs
├── genetic.rs       # Genetic algorithm operations
├── evolution.rs     # Evolution strategies
├── diversity.rs     # Diversity metrics and bonuses
└── speciation.rs    # NEAT-style speciation
```

#### Step 2: Implement Genetic Operations
```rust
// src/multi_agent/population/genetic.rs
pub trait GeneticOperator<A: Agent> {
    fn crossover(&self, parent1: &A, parent2: &A) -> A;
    fn mutate(&self, agent: &mut A, mutation_rate: f32);
    fn fitness(&self, agent: &A, environment: &dyn Any) -> f32;
}

pub struct NeuralNetworkCrossover {
    crossover_points: usize,
    weight_mixing: WeightMixingStrategy,
}

pub enum WeightMixingStrategy {
    Uniform,
    LayerWise,
    NeuronWise,
}
```

#### Step 3: Implement Evolution Strategies
```rust
// src/multi_agent/population/evolution.rs
pub struct EvolutionStrategy<A: Agent> {
    population_size: usize,
    elite_fraction: f32,
    mutation_power: f32,
    fitness_shaping: FitnessShaping,
}

pub enum FitnessShaping {
    Rank,
    Proportional,
    Sigma { target_std: f32 },
}
```

#### Step 4: Add Diversity Mechanisms
```rust
// src/multi_agent/population/diversity.rs
pub trait DiversityMetric<A: Agent> {
    fn compute_diversity(&self, population: &[A]) -> f32;
    fn pairwise_distance(&self, agent1: &A, agent2: &A) -> f32;
}

pub struct BehavioralDiversity {
    trajectory_length: usize,
    distance_metric: DistanceMetric,
}

pub struct ParameterDiversity {
    layer_weights: Vec<f32>,  // Weight importance of each layer
}
```

#### Step 5: Implement Speciation (NEAT-style)
```rust
// src/multi_agent/population/speciation.rs
pub struct Species<A: Agent> {
    members: Vec<A>,
    representative: A,
    fitness_history: VecDeque<f32>,
    stagnation_counter: usize,
}

pub struct Speciation<A: Agent> {
    species: Vec<Species<A>>,
    compatibility_threshold: f32,
    stagnation_limit: usize,
}
```

#### Testing Requirements:
- Verify population convergence on multi-modal optimization tasks
- Test diversity maintenance over generations
- Benchmark against standard genetic algorithm benchmarks

### 4.3 League Play Systems

#### Step 1: Create League Infrastructure
```bash
src/multi_agent/league/
├── mod.rs
├── league.rs        # Core league structure
├── matchmaking.rs   # Opponent selection algorithms
├── exploiters.rs    # Exploiter agent creation
└── payoff.rs        # Payoff matrix tracking
```

#### Step 2: Implement League Structure
```rust
// src/multi_agent/league/league.rs
pub struct League<A: Agent> {
    main_agents: Vec<(A, AgentStats)>,
    main_exploiters: HashMap<usize, Vec<(A, AgentStats)>>,
    league_exploiters: Vec<(A, AgentStats)>,
    past_agents: CircularBuffer<A>,
    payoff_matrix: PayoffMatrix,
    generation: usize,
}

pub struct AgentStats {
    elo_rating: f32,
    games_played: usize,
    win_rate: f32,
    generation_added: usize,
}
```

#### Step 3: Advanced Matchmaking
```rust
// src/multi_agent/league/matchmaking.rs
pub enum MatchmakingStrategy {
    PrioritizedFictitiousSelfPlay {
        window_size: usize,
        weighting: PFSPWeighting,
    },
    NashResponseMode {
        mix_iterations: usize,
    },
    DiversityMatchmaking {
        novelty_threshold: f32,
    },
}

pub struct Matchmaker {
    strategy: MatchmakingStrategy,
    history: MatchHistory,
    constraints: MatchConstraints,
}
```

#### Step 4: Exploiter Generation
```rust
// src/multi_agent/league/exploiters.rs
pub struct ExploiterGenerator<A: Agent> {
    base_architecture: fn() -> A,
    training_budget: usize,
    exploration_bonus: f32,
}

impl<A: Agent> ExploiterGenerator<A> {
    pub fn create_exploiter(&self, target: &A, target_history: &[Episode]) -> A {
        // Create agent specifically trained to beat target
    }
}
```

#### Step 5: Payoff Matrix Tracking
```rust
// src/multi_agent/league/payoff.rs
pub struct PayoffMatrix {
    winrates: Array2<f32>,
    games_played: Array2<usize>,
    last_updated: Array2<Instant>,
}

impl PayoffMatrix {
    pub fn nash_averaging(&self) -> Array1<f32> {
        // Compute Nash equilibrium over agent mixture
    }
    
    pub fn exploitability(&self, agent_idx: usize) -> f32 {
        // Compute how exploitable an agent is
    }
}
```

#### Testing Requirements:
- Verify league produces diverse strategies
- Test exploiter effectiveness against targets
- Benchmark training efficiency vs standard self-play

### 4.4 Performance Optimization

#### Step 1: Parallel Game Execution
```rust
// src/multi_agent/parallel.rs
pub struct ParallelGameExecutor<E: MultiAgentEnvironment> {
    thread_pool: ThreadPool,
    env_factory: Box<dyn Fn() -> E>,
    batch_size: usize,
}

impl<E: MultiAgentEnvironment> ParallelGameExecutor<E> {
    pub fn run_episodes<A: Agent>(&self, agents: &[A], num_episodes: usize) -> Vec<Episode> {
        // Parallel episode collection
    }
}
```

#### Step 2: Vectorized Belief Updates
- Batch particle filter updates across multiple belief states
- SIMD operations for observation likelihood computation
- GPU kernels for large particle counts

#### Step 3: GPU Multi-Agent Training
- Extend existing GPU kernels for batched multi-agent forward passes
- Implement GPU-based experience replay sampling
- Create GPU kernels for CFR regret updates

#### Step 4: Memory-Efficient Experience Storage
```rust
pub struct CompressedReplayBuffer {
    compression_level: CompressionLevel,
    prioritized: bool,
    circular_buffer: CircularBuffer<CompressedExperience>,
}
```

### 4.5 Comprehensive Documentation

#### Step 1: Tutorial Series
1. **Getting Started with Multi-Agent RL**
   - Basic concepts and terminology
   - First multi-agent environment
   - Training with self-play

2. **Partial Observability and Belief States**
   - Understanding POMDPs
   - Implementing custom belief states
   - Particle filter deep dive

3. **Game Theory and CFR**
   - Nash equilibria concepts
   - Using CFR solver
   - Creating custom games

4. **Population-Based Training**
   - Genetic algorithms in RL
   - Maintaining diversity
   - Hyperparameter evolution

5. **League Play at Scale**
   - Setting up leagues
   - Exploiter strategies
   - Analyzing results

#### Step 2: API Documentation Updates
- Add comprehensive examples to every public function
- Create decision trees for choosing components
- Performance characteristics documentation

#### Step 3: Benchmarking Suite
- Standard multi-agent environments
- Performance comparisons with other frameworks
- Scaling studies

### Implementation Schedule

#### Week 10: CFR Implementation
- Days 1-2: Core CFR algorithm
- Days 3-4: CFR variants (CFR+, Linear CFR)
- Day 5: Testing and debugging

#### Week 11: Population Training
- Days 1-2: Genetic operators and evolution strategies
- Days 3-4: Diversity metrics and speciation
- Day 5: Integration testing

#### Week 12: League Play
- Days 1-2: League infrastructure and matchmaking
- Days 3-4: Exploiter generation and payoff tracking
- Day 5: Full system testing

#### Week 13: Performance & Documentation
- Days 1-2: Parallel execution and GPU optimization
- Days 3-4: Documentation and tutorials
- Day 5: Benchmarking and final testing

#### Week 14: Polish & Release
- Days 1-2: Bug fixes and performance tuning
- Days 3-4: Example games and demonstrations
- Day 5: Release preparation

### Success Criteria

1. **CFR**: Converges to known Nash equilibria in test games
2. **Population**: Maintains diverse strategies over 1000+ generations
3. **League**: Produces agents that defeat 90%+ of earlier generations
4. **Performance**: 10x speedup with parallel execution
5. **Documentation**: All examples run without modification

## Integration Examples

### Example 1: Multi-Agent DQN with Masking
```rust
use athena::prelude::*;
use athena::multi_agent::{MultiAgentDQN, SelfPlayTrainer};

fn main() {
    // Create multi-agent system
    let ma_dqn = MultiAgentDQN::builder()
        .num_agents(4)
        .state_dim(128)
        .action_dim(10)
        .hidden_layers(&[256, 256])
        .shared_parameters(true)
        .action_masking(true)
        .build();
    
    // Create environment
    let mut env = CardGameEnvironment::new();
    
    // Train with self-play
    let mut trainer = SelfPlayTrainer::new(ma_dqn);
    trainer.train(&mut env, 100000, 32);
}
```

### Example 2: Belief-Based Agent for Partial Observability
```rust
use athena::belief::{ParticleFilter, BeliefAgent};

fn main() {
    // Create particle filter for tracking hidden state
    let particle_filter = ParticleFilter::new(
        1000,  // particles
        || random_game_state(),
        |state, action| transition(state, action),
        |state| observe(state),
    );
    
    // Create base agent
    let dqn = DqnAgent::new(&[64, 128, 128, 5], 0.1, optimizer, 1000, true);
    
    // Wrap with belief tracking
    let mut belief_agent = BeliefAgent::new(particle_filter, dqn, 64);
    
    // Use in partially observable environment
    let obs = env.reset();
    let action = belief_agent.act_with_belief(&obs);
}
```

### Example 3: Communication in Multi-Agent System
```rust
use athena::multi_agent::{CommunicatingAgent, BroadcastChannel};

fn main() {
    // Create communication channel
    let channel = BroadcastChannel::new(4, 16);  // 4 agents, 16-dim messages
    
    // Create communicating agents
    let agents: Vec<_> = (0..4).map(|id| {
        let base = DqnAgent::new(&[64, 128, 128, 5], 0.1, optimizer, 1000, true);
        CommunicatingAgent::new(id, base, channel.clone())
    }).collect();
    
    // Agents can now share information during episodes
}
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock environments for predictable behavior
- Property-based testing for belief updates

### Integration Tests
- Full episodes with multiple agents
- Verify convergence on known games
- Performance benchmarks

### Example Games
1. **Simple Turn-Based**: Tic-tac-toe, Connect-4
2. **Simultaneous Action**: Rock-Paper-Scissors variants
3. **Partial Information**: Simplified poker, Coup
4. **Communication**: Cooperative navigation tasks

## Documentation Updates

1. Add new sections to tutorials:
   - Multi-agent basics
   - Belief state tutorial
   - Communication protocols

2. Update examples:
   - Multi-agent cartpole
   - Partially observable gridworld
   - Competitive card games

3. API documentation:
   - Comprehensive rustdoc for all new modules
   - Migration guide for existing users

## Performance Considerations

1. **Parallelization**: Use rayon for parallel environment steps
2. **GPU Support**: Extend existing GPU kernels for batch multi-agent forward passes
3. **Memory Optimization**: Shared parameter agents to reduce memory footprint
4. **Caching**: Cache belief computations when possible

## Timeline Summary

- **Week 1-2**: Action masking implementation and testing
- **Week 3-5**: Belief states and particle filtering
- **Week 6-9**: Core multi-agent infrastructure
- **Week 10-12**: Advanced features (CFR, population-based training)
- **Week 13**: Documentation and example games
- **Week 14**: Performance optimization and benchmarking

## Success Metrics

1. All tests passing with >90% coverage
2. Performance within 20% of single-agent for comparable tasks
3. Successfully train agents for all example games
4. Documentation rated helpful by beta users
5. No breaking changes to existing API