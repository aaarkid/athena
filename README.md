# Athena

![Athena Logo](assets/favicon.png)

Athena is a deep learning library for Rust, with a focus on reinforcement learning for
games. It covers network construction and training, the common RL algorithms, and
deployment through Python bindings and WebAssembly.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [What the pieces are](#what-the-pieces-are)
- [Feature flags](#feature-flags)
- [Documentation](#documentation)
- [Examples](#examples)
- [License](#license)

## Features

- Dense networks with ReLU, Sigmoid, Tanh, Linear, LeakyReLU, ELU and GELU
- Conv1D, Conv2D, pooling, batch norm, dropout and embedding layers, composed by hand
- LSTM and GRU with backpropagation through time, trained through `RecurrentNetwork`
- RL algorithms: DQN (with Double DQN), A2C, PPO, SAC, TD3
- Uniform and prioritized replay buffers
- SGD, Adam and RMSProp, with global gradient norm clipping and learning rate schedules
- A cache-free inference path that takes `&self`, so one network serves many entities
- Versioned save and load
- GPU acceleration through OpenCL, plus a CPU mock for building without it

## Installation

```toml
[dependencies]
athena = "0.4"
ndarray = "0.15"
```

## Quickstart

```bash
cargo run --release --example game_loop_dqn
```

```text
training on a 7x7 grid, 400 episodes
episode 100  epsilon 0.221  greedy walk 12 steps
episode 400  epsilon 0.050  greedy walk 12 steps

trained greedy walk: Some(12) steps, shortest possible is 12
saved to models/game_loop_dqn.bin
reloaded greedy walk: Some(12) steps
```

The whole path, from `examples/game_loop_dqn.rs`:

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::Array1;

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let mut agent = DqnAgent::new(&[2, 64, 64, 4], 1.0, optimizer, 200, true);
let mut buffer = ReplayBuffer::new(20_000);
let mut rng = seeded_rng(11);

let state: Array1<f32> = Array1::zeros(2);
let action = agent.act(state.view()).unwrap();

// ... apply the action to the world ...
let (reward, next_state, done) = (1.0, Array1::<f32>::zeros(2), false);

buffer.add(Experience { state, action, reward, next_state, done });

if buffer.len() >= 64 {
    let batch = buffer.sample_with(64, &mut rng);
    agent.train_on_batch(&batch, 0.95, 0.002).unwrap();
}

agent.decay_epsilon(0.985, 0.05);
```

[`docs/quickstart.md`](docs/quickstart.md) walks through each step, including evaluation,
saving and reloading. [`docs/conventions.md`](docs/conventions.md) has the rules that are
not visible from a signature: shapes, the weight orientation, and what can be stacked.

## What the pieces are

### A neural network

```rust
use athena::activations::Activation;
use athena::network::NeuralNetwork;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::Array2;

let mut network = NeuralNetwork::new(
    &[4, 8, 4],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
);

let inputs = Array2::from_shape_fn((16, 4), |(i, j)| (i + j) as f32 * 0.05);
let targets: Array2<f32> = Array2::zeros((16, 4));
network.train_minibatch(inputs.view(), targets.view(), 0.01);
```

Inference that writes no caches, so an `Arc<NeuralNetwork>` can serve many callers:

```rust
use athena::activations::Activation;
use athena::network::NeuralNetwork;
use athena::optimizer::{OptimizerWrapper, SGD};

let network = NeuralNetwork::new(
    &[4, 8, 4],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
);

let state = ndarray::array![0.1, 0.2, 0.3, 0.4];
let q_values = network.predict(state.view());
assert_eq!(q_values.len(), 4);
```

### Optimizers

`Adam::new(&[], ..)` grows its per-layer state on first use and holds no learning rate:
the rate is an argument to each training call.

```rust
use athena::optimizer::{Adam, OptimizerWrapper, RMSProp, SGD};

let _sgd = OptimizerWrapper::SGD(SGD::new());
let _adam = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let _rmsprop = OptimizerWrapper::RMSProp(RMSProp::new(&[], 0.9, 1e-8));
```

### Replay buffers

```rust
use athena::replay_buffer::{Experience, PriorityMethod, PrioritizedReplayBuffer, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::array;

let mut buffer = ReplayBuffer::new(1000);
buffer.add(Experience {
    state: array![0.0, 1.0],
    action: 0,
    reward: 1.0,
    next_state: array![1.0, 1.0],
    done: false,
});

let mut rng = seeded_rng(3);
let batch = buffer.sample_with(1, &mut rng);
assert_eq!(batch.len(), 1);

// Prioritized sampling returns stable slot ids to hand back to update_priorities
let mut prioritized =
    PrioritizedReplayBuffer::new(1000, PriorityMethod::Proportional { alpha: 0.6 });
prioritized.add_with_priority(
    Experience {
        state: array![0.0, 1.0],
        action: 0,
        reward: 1.0,
        next_state: array![1.0, 1.0],
        done: false,
    },
    1.0,
);
let (_experiences, _weights, slots) = prioritized.sample_with_weights(1, 0.4);
prioritized.update_priorities(&slots, &[2.0]);
```

### Recurrent networks

LSTM and GRU do not implement the `Layer` trait, so they cannot go into a
`NeuralNetwork`. Train them through `RecurrentNetwork`:

```rust
use athena::activations::Activation;
use athena::layers::LSTMLayer;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::recurrent::{RecurrentCell, RecurrentNetwork};

let mut model = RecurrentNetwork::new(
    RecurrentCell::Lstm(LSTMLayer::new(3, 16, false)),
    &[16, 1],
    &[Activation::Linear],
    OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)),
);

// Per frame, carrying the hidden state
let output = model.step(ndarray::array![0.1, 0.2, 0.3].view());
assert_eq!(output.len(), 1);
model.reset();
```

## Feature flags

| Feature | What it turns on |
| --- | --- |
| `gpu` | OpenCL backend. Needs OpenCL drivers installed. |
| `gpu-mock` | The same API without OpenCL. All math runs on the CPU. |
| `action-masking` | `MaskedLayer`, `MaskedSoftmax`, `DqnAgent::train_on_batch_masked` |
| `belief-states` | POMDP belief states and a particle filter |
| `multi-agent` | `SelfPlayTrainer`, communication channels |
| `python` | PyO3 bindings |
| `wasm` | `wasm-bindgen` bindings |

`--all-features` does not link without OpenCL installed; use `--features gpu-mock` to
compile the GPU API instead.

### GPU

Work goes through OpenCL. The backend picks an Intel Arc device first, then falls back to
NVIDIA, AMD, or whatever else the platform reports. Matrix multiplication, elementwise
ops and the activation functions have kernels; everything else stays on the CPU.

Under `gpu-mock` **every operation runs on the CPU**, `device_type` reports `IntelGpu` and
`device_info` returns a fabricated device string. It is for compiling and for API-shape
tests; its timings mean nothing.

OpenCL on Windows is fiddly; see the [Windows Setup Guide](docs/WINDOWS_SETUP.md).

## Documentation

`cargo doc --open` for the API reference. The guides live in `docs/`:

- [Quickstart](docs/quickstart.md) - act, learn, save, reload
- [Conventions](docs/conventions.md) - shapes, weight orientation, what can be stacked
- [Getting Started](docs/tutorial_getting_started.md) - the basics at more length
- [Algorithms Guide](docs/algorithms_guide.md) - what each algorithm is for
- [Performance Guide](docs/performance_guide.md) - what costs what
- [Best Practices](docs/best_practices.md) - recommended patterns
- [Advanced Tutorial](docs/tutorial_advanced.md) - design sketches beyond the crate

## Examples

Unless a feature is listed, `cargo run --release --example <name>`:

| Example | What it shows |
| --- | --- |
| `game_loop_dqn` | The canonical path: act, learn, decay, save, reload |
| `background_training` | Training on a worker thread, off the frame thread |
| `grid_navigation` | DQN on a small grid world |
| `cartpole_simple` | Classic control |
| `mountain_car_working` | Sparse reward environment |
| `cartpole_ppo` | PPO |
| `pendulum_sac` | SAC on continuous control |
| `conv_shapes` | The conv and pooling backward passes |
| `masked_cartpole` | Action masking. `--features action-masking` |
| `belief_tracking` | Partial observability. `--features belief-states` |
| `gpu_test` | The GPU backend. `--features gpu` or `--features gpu-mock` |

## License

Dual licensed under either

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
