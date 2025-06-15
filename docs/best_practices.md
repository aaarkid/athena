# Athena Best Practices Guide

This guide covers best practices for using the Athena reinforcement learning library effectively, including code organization, training strategies, and common pitfalls to avoid.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Code Organization](#code-organization)
3. [Training Best Practices](#training-best-practices)
4. [Debugging Strategies](#debugging-strategies)
5. [Testing and Validation](#testing-and-validation)
6. [Production Deployment](#production-deployment)

## Project Structure

### Recommended Directory Layout

```
my_rl_project/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point
│   ├── environment/         # Environment implementations
│   │   ├── mod.rs
│   │   ├── cartpole.rs
│   │   └── traits.rs
│   ├── training/            # Training loops and utilities
│   │   ├── mod.rs
│   │   ├── trainer.rs
│   │   └── metrics.rs
│   ├── config/              # Configuration management
│   │   ├── mod.rs
│   │   └── hyperparameters.rs
│   └── utils/               # Helper functions
│       ├── mod.rs
│       └── visualization.rs
├── models/                  # Saved models
├── logs/                    # Training logs
├── data/                    # Datasets or recordings
└── tests/                   # Integration tests
```

### Configuration Management

Use a configuration struct for hyperparameters:

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingConfig {
    // Network architecture
    pub layer_sizes: Vec<usize>,
    pub activation: String,
    
    // Training hyperparameters
    pub learning_rate: f32,
    pub batch_size: usize,
    pub buffer_size: usize,
    pub gamma: f32,
    
    // Algorithm specific
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,
    
    // Training schedule
    pub max_episodes: usize,
    pub eval_frequency: usize,
    pub save_frequency: usize,
}

impl TrainingConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }
}
```

## Code Organization

### 1. Environment Abstraction

Define a clean environment interface:

```rust
use athena::error::Result;

pub trait Environment {
    type State;
    type Action;
    
    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> (Self::State, f32, bool);
    fn render(&self) -> Result<()>;
    fn close(&mut self);
    
    fn state_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}

// Implement wrapper for compatibility
pub struct AthenaEnvWrapper<E: Environment> {
    env: E,
}

impl<E: Environment> AthenaEnvWrapper<E> {
    pub fn new(env: E) -> Self {
        Self { env }
    }
    
    pub fn to_athena_state(&self, state: E::State) -> Array1<f32> {
        // Convert environment state to Athena format
    }
}
```

### 2. Training Loop Structure

Organize training code into reusable components:

```rust
pub struct Trainer<E: Environment> {
    agent: DqnAgent,
    env: E,
    buffer: ReplayBuffer,
    config: TrainingConfig,
    metrics: MetricsTracker,
}

impl<E: Environment> Trainer<E> {
    pub fn train(&mut self) -> Result<()> {
        for episode in 0..self.config.max_episodes {
            self.run_episode()?;
            
            if episode % self.config.eval_frequency == 0 {
                self.evaluate()?;
            }
            
            if episode % self.config.save_frequency == 0 {
                self.save_checkpoint(episode)?;
            }
        }
        Ok(())
    }
    
    fn run_episode(&mut self) -> Result<f32> {
        let mut state = self.env.reset();
        let mut total_reward = 0.0;
        
        loop {
            // Select action
            let action = self.agent.act(&state)?;
            
            // Environment step
            let (next_state, reward, done) = self.env.step(action);
            
            // Store experience
            self.buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train if enough samples
            if self.buffer.len() >= self.config.batch_size {
                self.train_step()?;
            }
            
            total_reward += reward;
            state = next_state;
            
            if done {
                break;
            }
        }
        
        self.metrics.add_episode_reward(total_reward);
        Ok(total_reward)
    }
}
```

### 3. Metrics and Logging

Implement comprehensive logging:

```rust
use std::fs::File;
use std::io::Write;

pub struct MetricsLogger {
    log_file: File,
    tensorboard: Option<TensorboardWriter>,
}

impl MetricsLogger {
    pub fn log_scalar(&mut self, name: &str, value: f32, step: usize) {
        // Write to file
        writeln!(self.log_file, "{},{},{}", step, name, value).unwrap();
        
        // Write to tensorboard if available
        if let Some(tb) = &mut self.tensorboard {
            tb.add_scalar(name, value, step);
        }
    }
    
    pub fn log_histogram(&mut self, name: &str, values: &[f32], step: usize) {
        // Log distribution of values
    }
}
```

## Training Best Practices

### 1. Exploration Strategies

Implement proper exploration decay:

```rust
pub struct ExplorationSchedule {
    start: f32,
    end: f32,
    decay_steps: usize,
}

impl ExplorationSchedule {
    pub fn get_epsilon(&self, step: usize) -> f32 {
        if step >= self.decay_steps {
            self.end
        } else {
            let progress = step as f32 / self.decay_steps as f32;
            self.start + (self.end - self.start) * progress
        }
    }
    
    // Alternative: Exponential decay
    pub fn get_epsilon_exp(&self, step: usize) -> f32 {
        let decay_rate = (self.end / self.start).powf(1.0 / self.decay_steps as f32);
        (self.start * decay_rate.powf(step as f32)).max(self.end)
    }
}
```

### 2. Reward Shaping

Design informative rewards:

```rust
pub fn shape_reward(
    state: &State,
    action: Action,
    next_state: &State,
    base_reward: f32,
) -> f32 {
    let mut shaped_reward = base_reward;
    
    // Example: Encourage forward progress
    let progress = next_state.position - state.position;
    shaped_reward += 0.1 * progress;
    
    // Example: Penalize energy usage
    let energy_cost = action.magnitude() * 0.01;
    shaped_reward -= energy_cost;
    
    // Clip to prevent exploitation
    shaped_reward.clamp(-1.0, 1.0)
}
```

### 3. State Preprocessing

Normalize and augment states:

```rust
pub struct StatePreprocessor {
    running_mean: Array1<f32>,
    running_std: Array1<f32>,
    count: usize,
}

impl StatePreprocessor {
    pub fn update(&mut self, state: &Array1<f32>) {
        // Update running statistics
        self.count += 1;
        let delta = state - &self.running_mean;
        self.running_mean += &delta / self.count as f32;
        let delta2 = state - &self.running_mean;
        self.running_std = ((self.running_std.mapv(|x| x * x) * (self.count - 1) as f32
            + &delta * &delta2) / self.count as f32).mapv(f32::sqrt);
    }
    
    pub fn normalize(&self, state: &Array1<f32>) -> Array1<f32> {
        (state - &self.running_mean) / (&self.running_std + 1e-8)
    }
}
```

### 4. Curriculum Learning

Gradually increase task difficulty:

```rust
pub struct CurriculumSchedule {
    stages: Vec<TaskDifficulty>,
    success_threshold: f32,
    window_size: usize,
}

impl CurriculumSchedule {
    pub fn get_current_difficulty(&self, recent_rewards: &[f32]) -> TaskDifficulty {
        if recent_rewards.len() < self.window_size {
            return self.stages[0].clone();
        }
        
        let avg_reward = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
        
        for (i, stage) in self.stages.iter().enumerate() {
            if avg_reward < stage.threshold {
                return stage.clone();
            }
        }
        
        self.stages.last().unwrap().clone()
    }
}
```

## Debugging Strategies

### 1. Sanity Checks

Implement debugging utilities:

```rust
pub fn sanity_check_agent(agent: &mut DqnAgent) -> Result<()> {
    // Check 1: Network outputs are in valid range
    let dummy_state = Array1::zeros(agent.state_dim());
    let q_values = agent.q_network.forward(dummy_state.view());
    assert!(q_values.iter().all(|&v| v.is_finite()));
    
    // Check 2: Agent can act
    let action = agent.act(dummy_state.view())?;
    assert!(action < agent.action_dim());
    
    // Check 3: Training doesn't explode
    let mut buffer = ReplayBuffer::new(100);
    for _ in 0..100 {
        buffer.add(Experience {
            state: Array1::zeros(agent.state_dim()),
            action: 0,
            reward: 1.0,
            next_state: Array1::zeros(agent.state_dim()),
            done: false,
        });
    }
    
    let batch = buffer.sample(32);
    agent.train_on_batch(&batch, 0.001)?;
    
    println!("All sanity checks passed!");
    Ok(())
}
```

### 2. Visualization Tools

Create debugging visualizations:

```rust
pub fn plot_q_values(agent: &mut DqnAgent, states: &[Array1<f32>]) {
    let mut q_values = vec![];
    
    for state in states {
        let qs = agent.q_network.forward(state.view());
        q_values.push(qs);
    }
    
    // Plot Q-value distributions
    // Use a plotting library like plotters
}

pub fn visualize_policy(agent: &mut DqnAgent, env: &impl Environment) {
    // Create a grid of states
    // For each state, show the preferred action
    // Useful for 2D environments
}
```

### 3. Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Exploding Q-values | Loss → ∞, NaN values | Reduce learning rate, clip gradients |
| No learning | Flat reward curve | Increase exploration, check reward scale |
| Catastrophic forgetting | Performance drops suddenly | Larger replay buffer, reduce learning rate |
| Slow convergence | Very gradual improvement | Increase learning rate, better initialization |
| High variance | Unstable training curve | Larger batch size, target network |

## Testing and Validation

### 1. Unit Tests for Components

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(100);
        
        for i in 0..150 {
            buffer.add(create_dummy_experience(i));
        }
        
        assert_eq!(buffer.len(), 100);
        assert_eq!(buffer.sample(1)[0].state[0], 50.0);
    }
    
    #[test]
    fn test_exploration_schedule() {
        let schedule = ExplorationSchedule {
            start: 1.0,
            end: 0.01,
            decay_steps: 1000,
        };
        
        assert_eq!(schedule.get_epsilon(0), 1.0);
        assert_eq!(schedule.get_epsilon(1000), 0.01);
        assert!(schedule.get_epsilon(500) < 1.0);
        assert!(schedule.get_epsilon(500) > 0.01);
    }
}
```

### 2. Integration Tests

```rust
#[test]
fn test_full_training_loop() {
    let config = TrainingConfig {
        max_episodes: 10,
        // ... other config
    };
    
    let env = DummyEnvironment::new();
    let agent = create_test_agent();
    let mut trainer = Trainer::new(agent, env, config);
    
    let result = trainer.train();
    assert!(result.is_ok());
    
    // Check that agent improved
    let final_reward = trainer.evaluate().unwrap();
    assert!(final_reward > 0.0);
}
```

### 3. Benchmark Tests

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_forward_pass(c: &mut Criterion) {
    let network = create_test_network();
    let input = Array1::zeros(100);
    
    c.bench_function("forward_pass", |b| {
        b.iter(|| {
            black_box(network.forward(input.view()));
        });
    });
}

criterion_group!(benches, benchmark_forward_pass);
criterion_main!(benches);
```

## Production Deployment

### 1. Model Serialization

```rust
use std::path::Path;

pub fn save_training_state(
    agent: &DqnAgent,
    optimizer_state: &OptimizerState,
    episode: usize,
    path: &Path,
) -> Result<()> {
    let state = TrainingState {
        agent: agent.clone(),
        optimizer: optimizer_state.clone(),
        episode,
        timestamp: SystemTime::now(),
    };
    
    let encoded = bincode::serialize(&state)?;
    std::fs::write(path, encoded)?;
    
    // Also save metadata
    let metadata = StateMetadata {
        version: env!("CARGO_PKG_VERSION"),
        git_hash: env!("GIT_HASH"),
        hyperparameters: config.clone(),
    };
    
    let meta_path = path.with_extension("meta.json");
    std::fs::write(meta_path, serde_json::to_string_pretty(&metadata)?)?;
    
    Ok(())
}
```

### 2. Inference Optimization

```rust
pub struct InferenceAgent {
    network: NeuralNetwork,
    preprocessor: StatePreprocessor,
}

impl InferenceAgent {
    pub fn from_checkpoint(path: &str) -> Result<Self> {
        let agent = DqnAgent::load(path)?;
        Ok(Self {
            network: agent.q_network,
            preprocessor: StatePreprocessor::load(path)?,
        })
    }
    
    pub fn act(&mut self, raw_state: &Array1<f32>) -> usize {
        // Preprocess
        let state = self.preprocessor.normalize(raw_state);
        
        // Forward pass
        let q_values = self.network.forward(state.view());
        
        // Argmax (no exploration in production)
        q_values.argmax().unwrap()
    }
}
```

### 3. Monitoring in Production

```rust
pub struct ProductionMonitor {
    action_counts: HashMap<usize, usize>,
    response_times: Vec<Duration>,
    state_statistics: StateStatistics,
}

impl ProductionMonitor {
    pub fn log_inference(&mut self, state: &Array1<f32>, action: usize, duration: Duration) {
        *self.action_counts.entry(action).or_insert(0) += 1;
        self.response_times.push(duration);
        self.state_statistics.update(state);
        
        // Alert if anomalies detected
        if self.is_anomalous(state) {
            self.send_alert("Anomalous state detected");
        }
    }
    
    fn is_anomalous(&self, state: &Array1<f32>) -> bool {
        // Check if state is outside normal range
        let z_scores = (state - &self.state_statistics.mean) / &self.state_statistics.std;
        z_scores.iter().any(|&z| z.abs() > 3.0)
    }
}
```

## Code Quality Checklist

- [ ] All public APIs have documentation
- [ ] Complex algorithms have inline comments
- [ ] Error handling uses `Result` types
- [ ] No `.unwrap()` in production code
- [ ] Constants are named and documented
- [ ] Tests cover edge cases
- [ ] Benchmarks track performance
- [ ] Examples demonstrate usage
- [ ] README includes quickstart
- [ ] CHANGELOG tracks versions

## Summary

Following these best practices will help you:
1. Build maintainable RL projects
2. Debug issues efficiently
3. Train agents effectively
4. Deploy models safely
5. Monitor production systems

Remember that RL is experimental by nature - always validate your results and be prepared to iterate on your approach!