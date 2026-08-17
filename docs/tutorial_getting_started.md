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
athena = "0.4"
ndarray = "0.15"
rand = "0.8"
```

## Your First Agent

Let's create a simple DQN agent to solve a basic grid world environment:

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::ReplayBuffer;

let state_dim = 4;      // agent x, y, goal x, y
let action_dim = 4;     // up, down, left, right

// Adam::new takes (layers, beta1, beta2, epsilon). It holds no learning rate:
// that is an argument to each training call. An empty slice is fine, it grows
// its per-layer state on first use.
let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));

let mut agent = DqnAgent::new(
    &[state_dim, 128, 128, action_dim],
    1.0,        // initial exploration rate (epsilon)
    optimizer,
    1000,       // training steps between target network refreshes
    true,       // Double DQN
);

let mut buffer = ReplayBuffer::new(10_000);
# let _ = (&mut agent, &mut buffer);
```

## Understanding the Components

### 1. Neural Network Architecture

The neural network is the brain of your agent. The architecture is defined by layer sizes:

```rust
let (input_dim, hidden1, hidden2, output_dim) = (4, 128, 128, 4);
let layer_sizes = &[input_dim, hidden1, hidden2, output_dim];
# assert_eq!(layer_sizes.len(), 4);
```

- **Input dimension**: Size of your state representation
- **Hidden layers**: Process the information (typically 64-512 neurons)
- **Output dimension**: Number of possible actions

### 2. Exploration vs Exploitation

The epsilon parameter controls exploration:

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let mut agent = DqnAgent::new(&[4, 8, 4], 1.0, OptimizerWrapper::SGD(SGD::new()), 100, true);
agent.update_epsilon(0.1);  // 10% random actions, 90% greedy
```

During training, you typically start with high exploration and decay it:

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let mut agent = DqnAgent::new(&[4, 8, 4], 1.0, OptimizerWrapper::SGD(SGD::new()), 100, true);
// Once per episode: multiply by the rate, stop at the floor
agent.decay_epsilon(0.995, 0.01);
```

### 3. Experience Replay

Experience replay stores past experiences and samples from them randomly:

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# use athena::replay_buffer::{Experience, ReplayBuffer};
# use athena::rng::seeded_rng;
# use ndarray::Array1;
# let mut agent = DqnAgent::new(&[4, 8, 4], 0.1, OptimizerWrapper::SGD(SGD::new()), 100, true);
# let mut buffer = ReplayBuffer::new(1000);
# let mut rng = seeded_rng(1);
# let current_state: Array1<f32> = Array1::zeros(4);
# let next_state: Array1<f32> = Array1::zeros(4);
# let (action, reward, episode_finished) = (0usize, 1.0f32, false);
# let (batch_size, gamma, learning_rate) = (8usize, 0.99f32, 1e-3f32);
buffer.add(Experience {
    state: current_state.clone(),
    action,
    reward,
    next_state: next_state.clone(),
    done: episode_finished,
});

// sample_with takes a generator, so a seeded run reproduces.
// train_on_batch takes (experiences, gamma, learning_rate) in that order.
if buffer.len() >= batch_size {
    let batch = buffer.sample_with(batch_size, &mut rng);
    let _loss = agent.train_on_batch(&batch, gamma, learning_rate).unwrap();
}
```

## Training Your Agent

Here's a complete training loop for a simple grid world:

```rust
use athena::agent::DqnAgent;
use athena::metrics::MetricsTracker;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::{array, Array1};

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

    // Every component scaled to roughly 0 to 1: a network learns much faster
    // when its inputs share a scale
    fn get_state(&self) -> Array1<f32> {
        array![
            self.agent_pos.0 as f32 / self.size as f32,
            self.agent_pos.1 as f32 / self.size as f32,
            self.goal_pos.0 as f32 / self.size as f32,
            self.goal_pos.1 as f32 / self.size as f32,
        ]
    }

    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        match action {
            0 => self.agent_pos.1 = (self.agent_pos.1 - 1).max(0),
            1 => self.agent_pos.1 = (self.agent_pos.1 + 1).min(self.size - 1),
            2 => self.agent_pos.0 = (self.agent_pos.0 - 1).max(0),
            3 => self.agent_pos.0 = (self.agent_pos.0 + 1).min(self.size - 1),
            _ => {}
        }

        let done = self.agent_pos == self.goal_pos;
        // Keep the reward scale small. A goal reward of 100 against a step cost
        // of 1 makes the Q-values large enough that SGD diverges.
        let reward = if done { 1.0 } else { -0.01 };

        (self.get_state(), reward, done)
    }
}

fn train_grid_world_agent() -> athena::error::Result<DqnAgent> {
    let mut env = GridWorld::new(5);

    // Adam, not SGD: a squared error on Q-values of this magnitude diverges
    // under a plain gradient step
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = DqnAgent::new(&[4, 64, 64, 4], 1.0, optimizer, 100, true);

    let mut buffer = ReplayBuffer::new(10_000);

    // MetricsTracker::new(num_layers, history_size)
    let mut metrics = MetricsTracker::new(3, 100);
    let mut rng = seeded_rng(7);

    let episodes = 300;
    let batch_size = 32;
    let gamma = 0.95;
    let learning_rate = 0.002;

    for episode in 0..episodes {
        let mut state = env.reset();
        let mut steps = 0;
        metrics.start_episode();

        loop {
            let action = agent.act(state.view())?;
            let (next_state, reward, done) = env.step(action);
            metrics.step(reward);
            steps += 1;

            buffer.add(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            if buffer.len() >= batch_size {
                let batch = buffer.sample_with(batch_size, &mut rng);
                let loss = agent.train_on_batch(&batch, gamma, learning_rate)?;
                metrics.record_loss(loss);
            }

            state = next_state;

            if done || steps > 100 {
                break;
            }
        }

        metrics.end_episode();
        agent.decay_epsilon(0.99, 0.01);

        if episode % 100 == 0 {
            let avg = metrics.avg_episode_reward(100).unwrap_or(0.0);
            println!(
                "Episode {}: avg reward {:.2}, epsilon {:.3}",
                episode, avg, agent.epsilon
            );
        }
    }

    Ok(agent)
}
# // Kept short so the doctest stays fast; the full run is examples/game_loop_dqn.rs
# let _ = train_grid_world_agent;
```

`examples/game_loop_dqn.rs` is this same loop, runnable, with saving and reloading:
`cargo run --release --example game_loop_dqn`.

## Evaluating Performance

After training, evaluate your agent's performance:

Evaluate with `predict`, not `act`. `act` draws from the agent's generator and can
return a random action, so an evaluation that calls it measures the exploration schedule
rather than the policy. `predict` also takes `&self`, so the evaluation cannot disturb
the agent.

```rust,no_run
use athena::agent::DqnAgent;
use athena::network::InferenceBuffers;
use ndarray::Array1;

fn greedy_action(agent: &DqnAgent, state: &Array1<f32>, buffers: &mut InferenceBuffers) -> usize {
    let q_values = agent.q_network.predict_into(state.view(), buffers);
    q_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

let agent = DqnAgent::load("models/grid_world_agent.bin").unwrap();
let mut buffers = InferenceBuffers::new();
let state: Array1<f32> = Array1::zeros(4);
let action = greedy_action(&agent, &state, &mut buffers);
println!("greedy action: {}", action);
```

## Advanced Concepts

### 1. Using Different Algorithms

Athena provides several RL algorithms. Here's how to use PPO instead of DQN:

```rust
use athena::algorithms::PPOBuilder;
use athena::optimizer::{Adam, OptimizerWrapper};

let (state_dim, action_dim) = (4, 4);
let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));

// The dimensions go in new; the builder sizes the hidden layers.
// .optimizer is required: build() errors without it.
let agent = PPOBuilder::new(state_dim, action_dim)
    .hidden_sizes(vec![64, 64])
    .optimizer(optimizer)
    .clip_param(0.2)
    .build()
    .unwrap();
# let _ = agent;
```

### 2. Custom Network Architectures

`NeuralNetwork` holds dense layers only. `Layer` is an alias for `DenseLayer`, not an
enum, so batch norm and dropout cannot be pushed into the same vector. What you can vary
is the width, the depth and the activation of each layer:

```rust
use athena::activations::Activation;
use athena::layers::Layer;
use athena::network::NeuralNetwork;
use athena::optimizer::{Adam, OptimizerWrapper};

let state_dim = 8;
let action_dim = 4;

let layers = vec![
    Layer::new(state_dim, 128, Activation::Relu),
    Layer::new(128, 64, Activation::Tanh),
    Layer::new(64, action_dim, Activation::Linear),
];

let network = NeuralNetwork::new_empty()
    .with_layers(layers);
let _ = network;
```

For batch norm, dropout, conv or pooling, compose the layers by hand against
`LayerTrait`. `examples/conv_shapes.rs` is the worked reference, and
[`docs/conventions.md`](conventions.md) has the full table of what goes where.

### 3. Hyperparameter Tuning

Key hyperparameters to tune:

Both of these are enums, not sets of constructors.

```rust
use athena::optimizer::{GradientClipper, LearningRateScheduler};

let scheduler = LearningRateScheduler::ExponentialDecay {
    initial_lr: 0.001,
    decay_rate: 0.999,
};
let lr = scheduler.get_lr(1000);
assert!(lr < 0.001);

let _clipper = GradientClipper::ClipByGlobalNorm { max_norm: 1.0 };
```

For a whole network at once, `NeuralNetwork::train_minibatch_clipped` applies the global
norm and returns the norm before clipping, which is worth logging when training
diverges.

### 4. Monitoring Training

Track and visualize training progress:

```rust
use athena::metrics::MetricsTracker;
use athena::visualization::plot_reward_history;

// (num_layers, history_size)
let mut metrics = MetricsTracker::new(3, 100);

// During training
metrics.start_episode();
metrics.step(1.0);          // one reward
metrics.record_loss(0.42);
metrics.record_q_value(1.7);
metrics.end_episode();

// After training: a text plot, returned as a String
let plot = plot_reward_history(metrics.metrics(), 80, 20);
println!("{}", plot);
```

## Common Pitfalls and Solutions

### 1. Exploding Q-values

**Problem**: Q-values grow without bound
**Solution**: 
- Reduce learning rate
- Clip gradients
- Normalize rewards

```rust
use athena::optimizer::GradientClipper;

let reward = 12.5f32;

// f32::clamp passes NaN straight through, so check first
let clipped_reward = if reward.is_finite() { reward.clamp(-1.0, 1.0) } else { 0.0 };
assert_eq!(clipped_reward, 1.0);

let _clipper = GradientClipper::ClipByGlobalNorm { max_norm: 0.5 };
```

### 2. No Learning Progress

**Problem**: Agent doesn't improve
**Solution**:
- Increase exploration
- Check reward scale
- Verify state representation

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let mut agent = DqnAgent::new(&[4, 8, 4], 0.1, OptimizerWrapper::SGD(SGD::new()), 100, true);
# let reward = 250.0f32;
// Increase initial exploration
agent.update_epsilon(1.0);

// Scale rewards
let scaled_reward = reward / 100.0;
# assert_eq!(scaled_reward, 2.5);
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

- API documentation: run `cargo doc --open`
- [Examples](../examples/)
- [Algorithm Guide](algorithms_guide.md)
- [Performance Guide](performance_guide.md)
- [Best Practices](best_practices.md)

### Examples in this repository

- `grid_navigation.rs` - DQN on a small grid world
- `cartpole_simple.rs` - classic control
- `mountain_car_working.rs` - sparse reward environment
- `cartpole_ppo.rs` - PPO
- `pendulum_sac.rs` - SAC on continuous control
- `masked_cartpole.rs` - action masking, needs `--features action-masking`
- `game_loop_dqn.rs` - the canonical path: act, learn, decay, save, reload

Run any of them with `cargo run --release --example <name>`.