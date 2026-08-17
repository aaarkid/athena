# Best Practices

Advice that has cost this repository something to learn, with code that compiles.

Every Rust block here is a doctest. If one drifts from the API, `cargo test` fails.

## Contents

1. [Project layout](#project-layout)
2. [Configuration](#configuration)
3. [The training loop](#the-training-loop)
4. [Exploration](#exploration)
5. [Rewards](#rewards)
6. [Observations](#observations)
7. [Debugging: what to check first](#debugging-what-to-check-first)
8. [Testing an agent](#testing-an-agent)
9. [Checkpoints](#checkpoints)
10. [Inference in a shipped build](#inference-in-a-shipped-build)

## Project layout

```text
my_rl_project/
├── Cargo.toml
├── src/
│   ├── main.rs              # entry point
│   ├── environment/         # your environments, and wrappers over them
│   ├── training/            # the loop, and metrics
│   └── config/              # hyperparameters
├── models/                  # saved agents
└── tests/
```

The one non-obvious thing: keep the environment and the agent in separate modules with no
shared types beyond `Array1<f32>` and `usize`. Every time they get entangled, swapping the
algorithm becomes a rewrite.

## Configuration

Put the hyperparameters in one struct and serialize it next to the model. Six months later
the only way to know what produced a checkpoint is to have written it down.

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingConfig {
    pub layer_sizes: Vec<usize>,

    pub learning_rate: f32,
    pub batch_size: usize,
    pub buffer_size: usize,
    pub gamma: f32,

    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,

    pub max_episodes: usize,
    pub eval_frequency: usize,
    /// Fixed, so a run reproduces
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            layer_sizes: vec![4, 64, 64, 2],
            learning_rate: 1e-3,
            batch_size: 64,
            buffer_size: 20_000,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay: 0.995,
            max_episodes: 1000,
            eval_frequency: 50,
            seed: 7,
        }
    }
}

let config = TrainingConfig::default();
assert_eq!(config.layer_sizes.first(), Some(&4));
```

## The training loop

Four things in order, every step: act, store, sample, train. Then once per episode, decay.

```rust
use athena::agent::DqnAgent;
use athena::metrics::MetricsTracker;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::Array1;

// Adam, not SGD. A squared error on Q-values of any real magnitude diverges under a
// plain gradient step; Pendulum's return of around -1600 does it within a few thousand
// steps.
let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let mut agent = DqnAgent::new_seeded(&[4, 64, 64, 2], 1.0, optimizer, 200, true, 7);

let mut buffer = ReplayBuffer::new(20_000);
let mut metrics = MetricsTracker::new(3, 100);
let mut rng = seeded_rng(11);

for _episode in 0..2 {
    let mut state: Array1<f32> = Array1::zeros(4);
    metrics.start_episode();

    for _step in 0..8 {
        let action = agent.act(state.view()).expect("state width matches");

        // ... your environment ...
        let (next_state, reward, done) = (Array1::<f32>::zeros(4), 1.0f32, false);
        metrics.step(reward);

        buffer.add(Experience {
            state: state.clone(),
            action,
            reward,
            next_state: next_state.clone(),
            done,
        });

        // (experiences, gamma, learning_rate). Getting gamma and the rate the wrong way
        // round is silent: both are small floats.
        if buffer.len() >= 4 {
            let batch = buffer.sample_with(4, &mut rng);
            let loss = agent.train_on_batch(&batch, 0.99, 1e-3).expect("shapes match");
            metrics.record_loss(loss);
        }

        state = next_state;
        if done {
            break;
        }
    }

    metrics.end_episode();
    agent.decay_epsilon(0.995, 0.05);
}

assert!(metrics.avg_episode_reward(10).is_some());
```

Two things that are easy to get wrong here:

- **`sample_with` takes a generator.** `sample` uses a fresh one, so a run does not
  reproduce. Pass a seeded generator when you need it to.
- **Train every step, not every episode.** One gradient step per environment step is the
  usual ratio. Training once per episode wastes most of the buffer.

## Exploration

Decay per episode, and keep a floor above zero. `DqnAgent::decay_epsilon` does both.

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{OptimizerWrapper, SGD};

let mut agent = DqnAgent::new(&[4, 16, 2], 1.0, OptimizerWrapper::SGD(SGD::new()), 100, true);

for _episode in 0..1000 {
    agent.decay_epsilon(0.995, 0.05);
}

// Exponential decay reaches the floor and stops there
assert!((agent.epsilon - 0.05).abs() < 1e-6);
```

If you want linear decay instead, compute it and call `update_epsilon`:

```rust
# use athena::agent::DqnAgent;
# use athena::optimizer::{OptimizerWrapper, SGD};
# let mut agent = DqnAgent::new(&[4, 16, 2], 1.0, OptimizerWrapper::SGD(SGD::new()), 100, true);
let (start, end, decay_episodes) = (1.0f32, 0.05f32, 500.0f32);

for episode in 0..1000 {
    let progress = (episode as f32 / decay_episodes).min(1.0);
    agent.update_epsilon(start + (end - start) * progress);
}
assert!((agent.epsilon - end).abs() < 1e-6);
```

The floor matters more than the schedule. At epsilon 0 a policy that has settled on a bad
action never sees the alternative again.

## Rewards

Keep the scale small. A goal reward of 100 against a step cost of 1 makes the Q-values
large enough that training becomes fragile; the same task at 1.0 and 0.01 learns in a
fraction of the episodes.

```rust
/// Reward, plus a shaping term, clipped.
fn shape_reward(position: f32, next_position: f32, base_reward: f32, effort: f32) -> f32 {
    let mut reward = base_reward;

    // Encourage progress toward the goal
    reward += 0.1 * (next_position - position);

    // Discourage flailing
    reward -= 0.01 * effort;

    // f32::clamp passes NaN straight through, so check finiteness first or a single bad
    // step poisons the network permanently
    if reward.is_finite() {
        reward.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

assert_eq!(shape_reward(0.0, 1.0, 1.0, 0.0), 1.0);
assert_eq!(shape_reward(0.0, 0.0, f32::NAN, 0.0), 0.0);
```

Shaping terms are also how an agent learns to exploit you. Clip, and check that the
shaped total still peaks at the behaviour you wanted.

## Observations

Get every component onto a similar scale. This is the single most common reason an agent
does not learn, and it looks identical to a broken algorithm.

Mountain Car in this repository is the worked example: velocity spans plus or minus 0.07
against a position range of 1.8, and unscaled it was invisible to the network. Scaling it
turned flat noise into solving the task in about 100 steps.

```rust
use ndarray::Array1;

/// Running mean and variance, Welford's method, so it works in one pass.
pub struct Preprocessor {
    mean: Array1<f32>,
    var: Array1<f32>,
    count: f32,
}

impl Preprocessor {
    pub fn new(width: usize) -> Self {
        Preprocessor {
            mean: Array1::zeros(width),
            var: Array1::ones(width),
            count: 1e-4,
        }
    }

    pub fn observe(&mut self, state: &Array1<f32>) {
        self.count += 1.0;
        let delta = state - &self.mean;
        self.mean = &self.mean + &(&delta / self.count);
        let delta2 = state - &self.mean;
        self.var = &self.var + &((&delta * &delta2 - &self.var) / self.count);
    }

    /// Standardize. The floor on the variance keeps a constant component from
    /// producing NaN.
    pub fn normalize(&self, state: &Array1<f32>) -> Array1<f32> {
        (state - &self.mean) / self.var.mapv(|v| v.max(1e-8).sqrt())
    }
}

let mut preprocessor = Preprocessor::new(2);
for value in [1.0f32, 2.0, 3.0, 4.0] {
    preprocessor.observe(&Array1::from_vec(vec![value, value * 100.0]));
}

let normalized = preprocessor.normalize(&Array1::from_vec(vec![2.5, 250.0]));
assert!(normalized.iter().all(|v| v.is_finite() && v.abs() < 5.0));
```

If the ranges are known ahead of time, dividing by them is simpler and does not drift.

## Debugging: what to check first

In order, because each rules out the ones below it.

1. **Does the reference test still pass?** `src/tests/test_learning.rs` trains DQN, A2C,
   SAC, TD3 and PPO on tiny tasks with known optimal policies. If those pass, the library
   is fine and the problem is in your environment, your observation, or your reward.
2. **Are you evaluating with `predict` or with `act`?** `act` samples. Evaluating with it
   measures the exploration schedule, not the policy. This was worth a 10x difference on
   one example in this repository: a PPO agent scoring 11.7 was actually scoring 223.
3. **Are the observation components on the same scale?**
4. **Is the reward scale small?**
5. **Is Adam being used, not SGD?**
6. **Does the episode terminate?** A bootstrapped value target with no terminal state
   feeds on itself and grows every iteration.

Then a sanity check that runs in a second:

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::Array1;

fn sanity_check(agent: &mut DqnAgent) -> athena::error::Result<()> {
    // The agent does not carry its own dimensions; the network does
    let state_width = agent.q_network.input_size();
    let action_count = agent.q_network.output_size();

    // 1. Outputs are finite
    let probe: Array1<f32> = Array1::zeros(state_width);
    let q_values = agent.q_network.predict(probe.view());
    assert!(q_values.iter().all(|v| v.is_finite()), "Q-values are not finite");
    assert_eq!(q_values.len(), action_count);

    // 2. A wrong width is reported, not a crash mid-frame
    assert!(agent.q_network.try_predict(Array1::zeros(state_width + 1).view()).is_err());

    // 3. Acting works and stays in range
    let action = agent.act(probe.view())?;
    assert!(action < action_count);

    // 4. Training does not explode
    let mut buffer = ReplayBuffer::new(100);
    let mut rng = seeded_rng(3);
    for i in 0..64 {
        buffer.add(Experience {
            state: Array1::from_elem(state_width, i as f32 * 0.01),
            action: i % action_count,
            reward: 0.5,
            next_state: Array1::zeros(state_width),
            done: i % 8 == 0,
        });
    }
    for _ in 0..20 {
        let batch = buffer.sample_with(32, &mut rng);
        let loss = agent.train_on_batch(&batch, 0.99, 1e-3)?;
        assert!(loss.is_finite(), "loss went non-finite");
    }

    Ok(())
}

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let mut agent = DqnAgent::new(&[4, 32, 3], 0.1, optimizer, 100, true);
sanity_check(&mut agent).expect("sanity check failed");
```

For a text plot of what happened, `athena::visualization::plot_reward_history` takes
`metrics.metrics()` and returns a `String`. `athena::metrics::statistics` has the norms
and a dead-neuron check; `athena::debug::numerical_check` finds NaN and infinity in a
weight set.

## Testing an agent

Assert that the agent reaches a known optimum, not that a call returned. A test that only
checks a loss is finite passes on a network that has learned nothing.

```rust
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::Array1;

// A corridor: action 1 moves right, action 0 left. The right end pays 1.0 and ends the
// episode. The optimal policy is to always move right, so the test can assert exactly
// how many steps a trained agent takes.
const LENGTH: usize = 6;

fn state_at(position: usize) -> Array1<f32> {
    let mut state = Array1::zeros(LENGTH);
    state[position] = 1.0;
    state
}

fn step(position: usize, action: usize) -> (usize, f32, bool) {
    let next = if action == 1 { (position + 1).min(LENGTH - 1) } else { position.saturating_sub(1) };
    if next == LENGTH - 1 { (next, 1.0, true) } else { (next, 0.0, false) }
}

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let mut agent = DqnAgent::new_seeded(&[LENGTH, 32, 2], 1.0, optimizer, 50, true, 9);
let mut buffer = ReplayBuffer::new(2000);
let mut rng = seeded_rng(9);

for episode in 0..300 {
    // Anneal, or the greedy policy is never exercised
    agent.update_epsilon((1.0 - episode as f32 / 200.0).max(0.05));

    let mut position = 0;
    for _ in 0..(LENGTH * 3) {
        let state = state_at(position);
        let action = agent.act(state.view()).unwrap();
        let (next, reward, done) = step(position, action);

        buffer.add(Experience {
            state,
            action,
            reward,
            next_state: state_at(next),
            done,
        });

        if buffer.len() >= 32 {
            let batch = buffer.sample_with(32, &mut rng);
            agent.train_on_batch(&batch, 0.95, 0.005).unwrap();
        }

        position = next;
        if done {
            break;
        }
    }
}

// Greedy, through predict: LENGTH - 1 steps is optimal
let mut position = 0;
let mut steps = 0;
for _ in 0..(LENGTH * 4) {
    let q_values = agent.q_network.predict(state_at(position).view());
    let action = if q_values[1] > q_values[0] { 1 } else { 0 };
    let (next, _, done) = step(position, action);
    steps += 1;
    if done {
        break;
    }
    position = next;
}

assert_eq!(steps, LENGTH - 1, "greedy policy took {} steps", steps);
```

Some tests depend on random weight initialization. If one fails, run it a few times before
assuming a regression, and fix the flakiness rather than loosening the assertion. Four
flaky tests were found in this repository and every one was an assertion that was true on
average rather than always. Probe a new stochastic test 8 to 15 times before trusting it.

## Checkpoints

`save` and `load`. `DqnAgent` does not implement `Clone`, so a checkpoint helper that
clones the agent will not compile.

```rust,no_run
use athena::agent::DqnAgent;
use athena::optimizer::{Adam, OptimizerWrapper};

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let agent = DqnAgent::new(&[4, 64, 2], 0.1, optimizer, 200, true);

// Include the episode number in the filename: the file itself does not carry it
std::fs::create_dir_all("models").unwrap();
agent.save("models/agent_ep1000.bin").expect("could not write the model");

let mut restored = DqnAgent::load("models/agent_ep1000.bin").expect("could not read it back");

// Epsilon is saved, so an agent reloaded for evaluation still explores unless you say
// otherwise
restored.update_epsilon(0.0);
```

What a saved file does and does not carry:

- **Carried:** both networks' weights and biases, the optimizer state including Adam's
  moment estimates, epsilon, `train_steps`, and the Double DQN and target-update settings.
- **Not carried:** the random generator, so a loaded agent explores from a fresh one; the
  forward-pass caches; and anything about your environment or config.

The file begins with `ATHN` and a format version. A file from 0.3.x, or from another
program, is reported rather than decoded into nonsense.

## Inference in a shipped build

Use `predict`. It takes `&self`, so one network behind an `Arc` serves every entity, and
it writes no caches, so it allocates nothing per frame once the buffers are sized.

```rust
use athena::agent::DqnAgent;
use athena::network::InferenceBuffers;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::Array1;
use std::sync::Arc;

let agent = DqnAgent::new(&[4, 64, 2], 0.0, OptimizerWrapper::SGD(SGD::new()), 200, true);

// Share the trained network; drop the rest of the agent
let policy = Arc::new(agent.q_network);

// One set of buffers per thread
let mut buffers = InferenceBuffers::new();
let state: Array1<f32> = Array1::zeros(4);

let q_values = policy.predict_into(state.view(), &mut buffers);
let action = q_values
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(i, _)| i)
    .unwrap();

assert!(action < 2);
```

There is no `argmax` on an ndarray without `ndarray-stats`, which is not a dependency of
this crate. The fold above is the whole of it.

Two more things worth doing before shipping:

- **`try_predict`, not `predict`, for anything reading state from a file, a network or a
  save.** It returns an error on a wrong width; `predict` multiplies straight into the
  first layer's weights, which for a game means the process dies mid-frame.
- **Move training off the frame thread.** `examples/background_training.rs` runs the
  learner on a worker and passes experiences over a channel.

## Where to go next

- [`conventions.md`](conventions.md) for shapes, weight orientation and what can be
  stacked
- [`quickstart.md`](quickstart.md) for the end-to-end path
- [`tutorial_advanced.md`](tutorial_advanced.md) for writing a layer, multi-agent and
  partial observability
