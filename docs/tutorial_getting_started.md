# Getting Started with Athena

This tutorial will guide you through the basics of using Athena for reinforcement learning. We'll start with a simple example and gradually introduce more advanced concepts.

## Table of Contents

1. [Installation](#installation)
2. [Your First Agent](#your-first-agent)
3. [Understanding the Components](#understanding-the-components)
4. [Training Your Agent](#training-your-agent)
5. [Evaluating Performance](#evaluating-performance)
6. [Next Steps](#next-steps)

## Installation

First, add Athena to your `Cargo.toml`:

```toml
[dependencies]
athena = "0.1.0"
ndarray = "0.15"
rand = "0.8"
```

## Your First Agent

Let's create a simple DQN agent to solve a basic grid world environment:

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, Adam};
use athena::replay_buffer::{ReplayBuffer, Experience};
use ndarray::{Array1, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the problem dimensions
    let state_dim = 4;      // e.g., agent x, y, goal x, y
    let action_dim = 4;     // up, down, left, right
    
    // Create a neural network architecture
    let layer_sizes = &[state_dim, 128, 128, action_dim];
    
    // Choose an optimizer
    let optimizer = OptimizerWrapper::Adam(Adam::new(
        0.001,  // learning rate
        0.9,    // beta1
        0.999,  // beta2
        1e-8    // epsilon
    ));
    
    // Create the DQN agent
    let mut agent = DqnAgent::new(
        layer_sizes,
        0.1,        // initial exploration rate (epsilon)
        optimizer,
        1000,       // target network update frequency
        true        // use Double DQN
    );
    
    // Create a replay buffer
    let mut buffer = ReplayBuffer::new(10000);
    
    println!("Agent created successfully!");
    Ok(())
}
```

## Understanding the Components

### 1. Neural Network Architecture

The neural network is the brain of your agent. The architecture is defined by layer sizes:

```rust
let layer_sizes = &[input_dim, hidden1, hidden2, output_dim];
```

- **Input dimension**: Size of your state representation
- **Hidden layers**: Process the information (typically 64-512 neurons)
- **Output dimension**: Number of possible actions

### 2. Exploration vs Exploitation

The epsilon parameter controls exploration:

```rust
agent.epsilon = 0.1;  // 10% random actions, 90% greedy
```

During training, you typically start with high exploration and decay it:

```rust
// Decay epsilon over time
agent.epsilon *= 0.995;  // Exponential decay
agent.epsilon = agent.epsilon.max(0.01);  // Minimum exploration
```

### 3. Experience Replay

Experience replay stores past experiences and samples from them randomly:

```rust
// Store an experience
let experience = Experience {
    state: current_state.clone(),
    action: action,
    reward: reward,
    next_state: next_state.clone(),
    done: episode_finished,
};
buffer.add(experience);

// Sample and train
if buffer.len() >= batch_size {
    let batch = buffer.sample(batch_size);
    agent.train_on_batch(&batch, learning_rate)?;
}
```

## Training Your Agent

Here's a complete training loop for a simple grid world:

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::replay_buffer::{ReplayBuffer, Experience};
use athena::metrics::MetricsTracker;
use ndarray::{Array1, array};
use rand::Rng;

// Simple grid world environment
struct GridWorld {
    agent_pos: (i32, i32),
    goal_pos: (i32, i32),
    size: i32,
}

impl GridWorld {
    fn new(size: i32) -> Self {
        GridWorld {
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
            size,
        }
    }
    
    fn reset(&mut self) -> Array1<f32> {
        self.agent_pos = (0, 0);
        self.get_state()
    }
    
    fn get_state(&self) -> Array1<f32> {
        array![
            self.agent_pos.0 as f32 / self.size as f32,
            self.agent_pos.1 as f32 / self.size as f32,
            self.goal_pos.0 as f32 / self.size as f32,
            self.goal_pos.1 as f32 / self.size as f32,
        ]
    }
    
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        // Move based on action
        match action {
            0 => self.agent_pos.1 = (self.agent_pos.1 - 1).max(0),           // Up
            1 => self.agent_pos.1 = (self.agent_pos.1 + 1).min(self.size - 1), // Down
            2 => self.agent_pos.0 = (self.agent_pos.0 - 1).max(0),           // Left
            3 => self.agent_pos.0 = (self.agent_pos.0 + 1).min(self.size - 1), // Right
            _ => {}
        }
        
        // Calculate reward
        let distance_before = ((self.goal_pos.0 - self.agent_pos.0).abs() + 
                              (self.goal_pos.1 - self.agent_pos.1).abs()) as f32;
        
        let done = self.agent_pos == self.goal_pos;
        let reward = if done {
            100.0  // Large reward for reaching goal
        } else {
            -1.0   // Small penalty for each step
        };
        
        (self.get_state(), reward, done)
    }
}

fn train_grid_world_agent() -> Result<(), Box<dyn std::error::Error>> {
    // Create environment and agent
    let mut env = GridWorld::new(5);
    let layer_sizes = &[4, 64, 64, 4];
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(layer_sizes, 1.0, optimizer, 100, true);
    
    // Training components
    let mut buffer = ReplayBuffer::new(10000);
    let mut metrics = MetricsTracker::new();
    
    // Hyperparameters
    let episodes = 1000;
    let batch_size = 32;
    let learning_rate = 0.001;
    let epsilon_decay = 0.995;
    let min_epsilon = 0.01;
    
    // Training loop
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut steps = 0;
        
        loop {
            // Select action
            let action = agent.act(state.view())?;
            
            // Environment step
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;
            steps += 1;
            
            // Store experience
            buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            // Train if enough samples
            if buffer.len() >= batch_size {
                let batch = buffer.sample(batch_size);
                agent.train_on_batch(&batch, learning_rate)?;
            }
            
            state = next_state;
            
            if done || steps > 100 {
                break;
            }
        }
        
        // Update metrics
        metrics.add_episode_reward(episode_reward);
        
        // Decay exploration
        agent.epsilon = (agent.epsilon * epsilon_decay).max(min_epsilon);
        
        // Print progress
        if episode % 100 == 0 {
            let avg_reward = metrics.get_average_reward(100);
            println!("Episode {}: Avg Reward = {:.2}, Epsilon = {:.3}",
                    episode, avg_reward, agent.epsilon);
        }
    }
    
    // Save trained model
    agent.save("models/grid_world_agent.bin")?;
    println!("Training complete! Model saved.");
    
    Ok(())
}
```

## Evaluating Performance

After training, evaluate your agent's performance:

```rust
fn evaluate_agent(agent: &mut DqnAgent, episodes: usize) -> f32 {
    let mut env = GridWorld::new(5);
    let mut total_reward = 0.0;
    
    // Disable exploration for evaluation
    let original_epsilon = agent.epsilon;
    agent.epsilon = 0.0;
    
    for _ in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        
        loop {
            let action = agent.act(state.view()).unwrap();
            let (next_state, reward, done) = env.step(action);
            
            episode_reward += reward;
            state = next_state;
            
            if done {
                break;
            }
        }
        
        total_reward += episode_reward;
    }
    
    // Restore epsilon
    agent.epsilon = original_epsilon;
    
    total_reward / episodes as f32
}

// Load and evaluate
let mut agent = DqnAgent::load("models/grid_world_agent.bin")?;
let avg_reward = evaluate_agent(&mut agent, 100);
println!("Average evaluation reward: {:.2}", avg_reward);
```

## Advanced Concepts

### 1. Using Different Algorithms

Athena provides several RL algorithms. Here's how to use PPO instead of DQN:

```rust
use athena::algorithms::{PPOAgent, PPOBuilder};

let agent = PPOBuilder::new()
    .input_dim(state_dim)
    .action_dim(action_dim)
    .hidden_dims(vec![64, 64])
    .optimizer(optimizer)
    .clip_epsilon(0.2)
    .build()?;
```

### 2. Custom Network Architectures

You can create custom network architectures:

```rust
use athena::network::NeuralNetwork;
use athena::layers::{Layer, BatchNormLayer, DropoutLayer};
use athena::activations::Activation;

// Create layers manually
let mut layers = Vec::new();

// First hidden layer with batch norm
layers.push(Layer::new(state_dim, 128, Activation::Relu));
layers.push(Layer::BatchNorm(BatchNormLayer::new(128)));

// Second hidden layer with dropout
layers.push(Layer::new(128, 64, Activation::Relu));
layers.push(Layer::Dropout(DropoutLayer::new(64, 0.2)));

// Output layer
layers.push(Layer::new(64, action_dim, Activation::Linear));
```

### 3. Hyperparameter Tuning

Key hyperparameters to tune:

```rust
// Learning rate scheduling
use athena::optimizer::LearningRateScheduler;

let scheduler = LearningRateScheduler::exponential(
    initial_lr: 0.001,
    decay_rate: 0.99,
    decay_steps: 1000,
);

// Gradient clipping
use athena::optimizer::GradientClipper;

let clipper = GradientClipper::new(max_norm: 1.0);
```

### 4. Monitoring Training

Track and visualize training progress:

```rust
use athena::metrics::MetricsTracker;
use athena::visualization::plot_rewards;

let mut metrics = MetricsTracker::new();

// During training
metrics.add_episode_reward(episode_reward);
metrics.add_loss(loss);
metrics.add_q_value(avg_q);

// After training
plot_rewards(&metrics.get_episode_rewards(), "training_progress.png")?;
```

## Common Pitfalls and Solutions

### 1. Exploding Q-values

**Problem**: Q-values grow without bound
**Solution**: 
- Reduce learning rate
- Clip gradients
- Normalize rewards

```rust
// Clip rewards
let clipped_reward = reward.clamp(-1.0, 1.0);

// Use gradient clipping
let clipper = GradientClipper::new(0.5);
```

### 2. No Learning Progress

**Problem**: Agent doesn't improve
**Solution**:
- Increase exploration
- Check reward scale
- Verify state representation

```rust
// Increase initial exploration
agent.epsilon = 1.0;

// Scale rewards
let scaled_reward = reward / 100.0;
```

### 3. Slow Training

**Problem**: Training takes too long
**Solution**:
- Increase batch size
- Use a faster optimizer (Adam vs SGD)
- Simplify network architecture

## Next Steps

Now that you understand the basics, try:

1. **Different Environments**: Implement your own environment following the GridWorld example
2. **Advanced Algorithms**: Try PPO or SAC for more complex tasks
3. **Continuous Actions**: Use SAC or TD3 for continuous control
4. **Custom Rewards**: Design reward functions for your specific problem
5. **Hyperparameter Search**: Systematically tune hyperparameters

### Resources

- [API Documentation](../api/index.html)
- [Examples](../examples/)
- [Algorithm Guide](algorithms_guide.md)
- [Performance Guide](performance_guide.md)
- [Best Practices](best_practices.md)

### Example Projects

1. **CartPole Balance**: Classic control problem
2. **Mountain Car**: Sparse reward environment
3. **Lunar Lander**: Continuous control
4. **Atari Games**: High-dimensional observations
5. **Robot Control**: Real-world applications

Happy learning with Athena!