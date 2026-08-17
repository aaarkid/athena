# Athena

![Athena Logo](assets/favicon.png)

Athena is a deep learning library for Rust, with a focus on reinforcement learning. It covers
network construction and training, the common RL algorithms, and deployment through Python
bindings and WebAssembly.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating a Neural Network](#creating-a-neural-network)
  - [Training a Neural Network](#training-a-neural-network)
  - [Using a DQN Agent](#using-a-dqn-agent)
  - [Replay Buffer](#replay-buffer)
  - [Optimizers](#optimizers)
- [Documentation](#documentation)
- [Examples](#examples)
- [License](#license)

## Features

- Customizable Neural Network architecture
- Support for various activation functions (ReLU, Sigmoid, Tanh, etc.)
- Layer types: Convolutional, Pooling, BatchNorm, Dropout, Embedding
- Recurrent layers (LSTM, GRU) with backpropagation through time, trained through
  `RecurrentNetwork`
- Multiple RL algorithms: DQN, A2C, PPO, SAC, TD3
- A Replay Buffer for experience replay
- Different Optimizers (SGD, Adam, RMSProp)
- GPU acceleration support (Intel Arc priority via OpenCL)
- Save and load trained Neural Networks
- Parallelism support using Rayon and ndarray

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
athena = "0.2.0"

# Optional features
athena = { version = "0.2.0", features = ["gpu"] }       # GPU support through OpenCL
athena = { version = "0.2.0", features = ["gpu-mock"] }  # GPU API without the OpenCL dependency
```

### GPU support

GPU work goes through OpenCL. The backend picks an Intel Arc device first, then falls
back to NVIDIA, AMD, or whatever else the platform reports. Matrix multiplication,
element-wise ops and the activation functions have kernels; everything else stays on CPU.

- `gpu` needs OpenCL drivers installed
- `gpu-mock` gives you the same API without the OpenCL dependency, which is what you
  want for CI or for building on a machine with no SDK

OpenCL on Windows is fiddly; see the [Windows Setup Guide](docs/WINDOWS_SETUP.md).

## Usage

### Creating a Neural Network

```rust
use athena::network::{NeuralNetwork, Activation};
use athena::optimizer::{OptimizerWrapper, SGD};

// Create a neural network with 2 layers and ReLU activation functions
let layer_sizes = &[4, 8, 4];
let activations = &[Activation::Relu, Activation::Linear];
let optimizer = OptimizerWrapper::SGD(SGD::new());
let nn = NeuralNetwork::new(layer_sizes, activations, optimizer);
```

### Training a Neural Network

```rust
// Train the neural network on a minibatch of inputs and targets
nn.train_minibatch(inputs.view(), targets.view(), learning_rate);
```

### Using a DQN Agent

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};

// Create a DQN agent
let layer_sizes = &[4, 8, 2];
let epsilon = 0.1;
let optimizer = OptimizerWrapper::SGD(SGD::new());
let agent = DqnAgent::new(layer_sizes, epsilon, optimizer);

// Train the agent on a batch of experiences
agent.train_on_batch(&experiences, gamma, learning_rate);
```

### Replay Buffer

```rust
use athena::replay_buffer::{ReplayBuffer, Experience};

// Create a replay buffer
let capacity = 1000;
let mut buffer = ReplayBuffer::new(capacity);

// Add experiences to the buffer
buffer.add(experience);

// Sample a batch of experiences
let batch_size = 32;
let sampled_experiences = buffer.sample(batch_size);
```

### Optimizers

- Stochastic Gradient Descent (SGD)
- Adam

```rust
use athena::optimizer::{OptimizerWrapper, SGD, Adam};
use athena::network::{NeuralNetwork, Layer};

// Create an optimizer
let optimizer = OptimizerWrapper::SGD(SGD::new()); // or OptimizerWrapper::Adam(Adam::new(layers, beta1, beta2, epsilon))
```

### Training an agent to play a simple game

This example outlines how you would set up an agent to learn to play a simple game with Athena.

Let's assume the game is very simple: The agent has two actions (move left or right), and it gets a reward of +1 if it moves right, and a reward of -1 if it moves left. The state of the game is a single number: the agent's current position.

First, you would define the state and action sizes for the agent:

```rust
let state_size = 1; // The agent's position
let action_size = 2; // Move left or right
```

Next, create the agent:

```rust
let layer_sizes = &[state_size, 64, action_size]; // Network structure
let epsilon = 1.0; // Starting value of epsilon
let optimizer = OptimizerWrapper::SGD(SGD::new()); // We use SGD as our optimizer
let mut agent = DqnAgent::new(layer_sizes, epsilon, optimizer);
```

Now, you would play many games, allowing the agent to learn from its actions:

```rust
let mut replay_buffer = ReplayBuffer::new(10000);
let mut state = 0.0; // Start at position 0
let mut total_reward = 0.0;

for _ in 0..10000 {
    let action = agent.act(array![state].view()); // The agent decides on an action
    let reward = if action == 0 { -1.0 } else { 1.0 }; // The game gives a reward
    let next_state = state + (2 * action as f32) - 1.0; // The game updates its state
    total_reward += reward;
    
    // Add this experience to the replay buffer
    let experience = Experience {
        state: array![state],
        action,
        reward,
        next_state: array![next_state],
        done: false,
    };
    replay_buffer.add(experience);
    
    // Train the agent with a batch from the replay buffer
    if replay_buffer.len() > 32 {
        let experiences = replay_buffer.sample(32);
        agent.train_on_batch(&experiences, 0.99, 0.001);
    }
    
    // Prepare for the next game iteration
    state = next_state;
}
```

In this example, the agent learns to always move right in order to maximize its reward. This might seem like a trivial problem, but the process is similar for more complex games. The agent explores the game, learns from the rewards it gets, and adjusts its strategy over time.

## Documentation

Run `cargo doc --open` for the API reference. The guides live in `docs/`:

- [Getting Started](docs/tutorial_getting_started.md) - the basics
- [Advanced Tutorial](docs/tutorial_advanced.md) - custom layers, multi-agent, performance
- [Algorithms Guide](docs/algorithms_guide.md) - what each algorithm is for
- [Performance Guide](docs/performance_guide.md) - optimization tips
- [Best Practices](docs/best_practices.md) - recommended patterns

## Examples

Everything in `examples/` runs with `cargo run --example <name>`:

- `grid_navigation` - DQN on a small grid world
- `cartpole_simple` - classic control
- `mountain_car_working` - sparse reward environment
- `cartpole_ppo` - PPO
- `pendulum_sac` - SAC on continuous control
- `masked_cartpole` - action masking
- `belief_tracking` - partially observable environment

## License

Dual licensed under either

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
