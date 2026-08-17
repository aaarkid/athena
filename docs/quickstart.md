# Quickstart

`cargo run --release --example game_loop_dqn`

```text
training on a 7x7 grid, 400 episodes
episode 100  epsilon 0.221  greedy walk 12 steps
episode 400  epsilon 0.050  greedy walk 12 steps

trained greedy walk: Some(12) steps, shortest possible is 12
saved to models/game_loop_dqn.bin
reloaded greedy walk: Some(12) steps
```

The source is [`examples/game_loop_dqn.rs`](../examples/game_loop_dqn.rs). Everything
below is quoted from it.

## Dependency

```toml
[dependencies]
athena = "0.4"
ndarray = "0.15"
```

## Building the agent

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));

let mut agent = DqnAgent::new(
    &[2, 64, 64, 4],  // observation width, hidden layers, number of actions
    1.0,              // starting epsilon
    optimizer,
    200,              // training steps between target network refreshes
    true,             // Double DQN
);

let mut buffer = ReplayBuffer::new(20_000);
```

`Adam::new(&[], ..)` grows its per-layer state on first use, so it does not need the
layers up front. It holds no learning rate: that is a per-call argument to
`train_on_batch`.

## The observation

A fixed-width `Array1<f32>`, with the components on a similar scale.

```rust
use ndarray::Array1;

fn observe(x: usize, y: usize, grid: usize) -> Array1<f32> {
    Array1::from_vec(vec![
        x as f32 / (grid - 1) as f32,
        y as f32 / (grid - 1) as f32,
    ])
}
```

## Per frame

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# use athena::replay_buffer::{Experience, ReplayBuffer};
# use athena::rng::seeded_rng;
# use ndarray::Array1;
# let mut agent = DqnAgent::new(&[2, 8, 4], 0.5, OptimizerWrapper::SGD(SGD::new()), 100, true);
# let mut buffer = ReplayBuffer::new(1000);
# let mut rng = seeded_rng(1);
# let state: Array1<f32> = Array1::zeros(2);
# let (reward, done, next_state) = (0.0f32, false, Array1::<f32>::zeros(2));
# const BATCH_SIZE: usize = 8;
# const GAMMA: f32 = 0.95;
# const LEARNING_RATE: f32 = 0.001;
let action = agent.act(state.view()).expect("state width matches the network");

// ... apply the action to the world, collect the reward ...

buffer.add(Experience { state, action, reward, next_state, done });

if buffer.len() >= BATCH_SIZE {
    let batch = buffer.sample_with(BATCH_SIZE, &mut rng);
    agent.train_on_batch(&batch, GAMMA, LEARNING_RATE).expect("shapes match");
}
```

`act` takes `&mut self`: it draws from the agent's own generator to decide whether to
explore. `train_on_batch` takes `(experiences, gamma, learning_rate)` in that order and
returns the mean squared TD error over the batch.

## Per episode

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let mut agent = DqnAgent::new(&[2, 8, 4], 1.0, OptimizerWrapper::SGD(SGD::new()), 100, true);
agent.decay_epsilon(0.985, 0.05);
```

## Evaluating

Use `predict`, not `act`. `act` samples: an evaluation that calls it measures the
exploration schedule rather than the policy.

```rust
# use athena::agent::DqnAgent;
# use athena::network::InferenceBuffers;
# use athena::optimizer::{OptimizerWrapper, SGD};
# use ndarray::Array1;
# let agent = DqnAgent::new(&[2, 8, 4], 0.0, OptimizerWrapper::SGD(SGD::new()), 100, true);
# let state: Array1<f32> = Array1::zeros(2);
let mut buffers = InferenceBuffers::new();
let q_values = agent.q_network.predict_into(state.view(), &mut buffers);

let action = q_values
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(i, _)| i)
    .unwrap();
```

`predict` takes `&self` and writes no caches, so one `Arc<NeuralNetwork>` can serve every
entity in a scene. `InferenceBuffers` sizes itself on the first call and is reused after
that, so the per-frame call allocates nothing. Give each thread its own.

## Saving and loading

```rust,no_run
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let agent = DqnAgent::new(&[2, 8, 4], 0.1, OptimizerWrapper::SGD(SGD::new()), 100, true);
agent.save("models/agent.bin").expect("could not write the model");

let mut loaded = DqnAgent::load("models/agent.bin").expect("could not read it back");
loaded.update_epsilon(0.0);
```

The file carries the weights, both networks and the optimizer state. It does not carry
the random generator, so a loaded agent explores from a fresh one, and it does not carry
the forward-pass caches. `load` refuses a file whose header does not match this build.

## Where to go next

- [`docs/conventions.md`](conventions.md) - shapes, weight orientation, what can be
  stacked, and the rules that are easy to get wrong
- [`examples/background_training.rs`](../examples/background_training.rs) - moving
  training off the frame thread
- [`docs/algorithms_guide.md`](algorithms_guide.md) - choosing between DQN, A2C, PPO,
  SAC and TD3
