# Advanced Athena Tutorial

Everything here compiles against the crate as it stands. Where a feature flag is needed
it is named.

This file used to be about twice this length. The removed sections described attention
layers, graph networks, MAML, CQL, population-based training and a curiosity module, none
of which the crate contains, and claimed runnable versions lived in `examples/`, where
they never did. They are working notes now, not documentation.

## Contents

1. [Wrapping an environment](#wrapping-an-environment)
2. [Writing a layer](#writing-a-layer)
3. [Multi-agent training](#multi-agent-training)
4. [Partial observability](#partial-observability)
5. [Reusing buffers](#reusing-buffers)
6. [Splitting work across cores](#splitting-work-across-cores)

## Wrapping an environment

The crate does not define an environment trait: agents take an `ArrayView1<f32>` and
return an action, and what produces those is yours. Defining a trait yourself is still
worth it once you have more than one environment, because wrappers compose.

```rust
use ndarray::Array1;

/// What every environment in your project provides.
pub trait Environment {
    fn reset(&mut self) -> Array1<f32>;
    /// Returns the next observation, the reward, and whether the episode ended.
    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool);
    fn action_count(&self) -> usize;
}

/// Ends an episode after a fixed number of steps.
///
/// Worth having explicitly: an episode that never terminates makes a bootstrapped value
/// target feed on itself, and the target grows every iteration.
pub struct TimeLimit<E> {
    inner: E,
    limit: usize,
    steps: usize,
}

impl<E: Environment> TimeLimit<E> {
    pub fn new(inner: E, limit: usize) -> Self {
        TimeLimit { inner, limit, steps: 0 }
    }
}

impl<E: Environment> Environment for TimeLimit<E> {
    fn reset(&mut self) -> Array1<f32> {
        self.steps = 0;
        self.inner.reset()
    }

    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        let (observation, reward, done) = self.inner.step(action);
        self.steps += 1;
        (observation, reward, done || self.steps >= self.limit)
    }

    fn action_count(&self) -> usize {
        self.inner.action_count()
    }
}

/// Keeps a running mean and variance of each observation component and standardizes it.
///
/// Unscaled inputs are the single most common reason an agent does not learn. A component
/// spanning 0.07 next to one spanning 1.8 is invisible to the network.
pub struct Normalized<E> {
    inner: E,
    mean: Array1<f32>,
    var: Array1<f32>,
    count: f32,
}

impl<E: Environment> Normalized<E> {
    pub fn new(inner: E, width: usize) -> Self {
        Normalized {
            inner,
            mean: Array1::zeros(width),
            var: Array1::ones(width),
            count: 1e-4,
        }
    }

    fn standardize(&mut self, observation: Array1<f32>) -> Array1<f32> {
        self.count += 1.0;
        let delta = &observation - &self.mean;
        self.mean = &self.mean + &(&delta / self.count);
        let delta2 = &observation - &self.mean;
        self.var = &self.var + &((&delta * &delta2 - &self.var) / self.count);

        // is_finite before the divide: a zero variance would produce NaN, and f32::clamp
        // passes NaN straight through
        (&observation - &self.mean) / self.var.mapv(|v| v.max(1e-8).sqrt())
    }
}

impl<E: Environment> Environment for Normalized<E> {
    fn reset(&mut self) -> Array1<f32> {
        let observation = self.inner.reset();
        self.standardize(observation)
    }

    fn step(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        let (observation, reward, done) = self.inner.step(action);
        (self.standardize(observation), reward, done)
    }

    fn action_count(&self) -> usize {
        self.inner.action_count()
    }
}
```

## Writing a layer

`LayerTrait` has eleven required methods and two with defaults. A layer that implements
it can be composed by hand with the conv and pooling layers, but **it cannot go into a
`NeuralNetwork`**: that holds `Vec<DenseLayer>` concretely. See
[`conventions.md`](conventions.md).

Here is the whole trait, implemented. The layer scales its input by a learned vector,
which is enough to exercise every method without obscuring the shape of the impl.

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use athena::layers::LayerTrait;

/// Multiplies each input component by a learned scale. One parameter per feature.
#[derive(Clone)]
pub struct Scale {
    /// The trait wants a weight *matrix*, so the scales live on the diagonal of a
    /// 1-by-n one. A layer whose parameters do not fit that shape cannot use this
    /// trait at all, which is why LSTM and GRU do not.
    weights: Array2<f32>,
    biases: Array1<f32>,
    cached_input: Option<Array2<f32>>,
}

impl Scale {
    pub fn new(size: usize) -> Self {
        Scale {
            weights: Array2::ones((1, size)),
            biases: Array1::zeros(size),
            cached_input: None,
        }
    }

    fn scales(&self) -> ArrayView1<f32> {
        self.weights.row(0)
    }
}

impl LayerTrait for Scale {
    // 1. Single input. The batch form does the work.
    fn forward(&mut self, input: ArrayView1<f32>) -> Array1<f32> {
        let batch = self.forward_batch(input.insert_axis(Axis(0)));
        batch.row(0).to_owned()
    }

    // 2. Batch forward. Caches whatever the backward pass needs.
    fn forward_batch(&mut self, inputs: ArrayView2<f32>) -> Array2<f32> {
        self.cached_input = Some(inputs.to_owned());
        let mut output = inputs.to_owned();
        output *= &self.scales();
        output += &self.biases;
        output
    }

    // 3. Single backward. Returns parameter gradients only, so it cannot be chained.
    fn backward(&self, output_error: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        let (_, weight_grads, bias_grads) =
            self.backward_batch(output_error.insert_axis(Axis(0)));
        (weight_grads, bias_grads)
    }

    // 4. Batch backward. Returns (input_gradients, weight_gradients, bias_gradients),
    //    all summed over the batch: NeuralNetwork divides, a hand-rolled caller must too.
    fn backward_batch(&self, output_errors: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let inputs = self
            .cached_input
            .as_ref()
            .expect("forward_batch must run before backward_batch");

        // d(out)/d(scale_j) = input_j, summed down the batch
        let weight_grads = (&output_errors.to_owned() * inputs)
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));
        let bias_grads = output_errors.sum_axis(Axis(0));

        // d(out)/d(in_j) = scale_j
        let mut input_grads = output_errors.to_owned();
        input_grads *= &self.scales();

        (input_grads, weight_grads, bias_grads)
    }

    // 5 to 8. Parameter access, by reference and by mutable reference.
    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.biases
    }

    fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn biases(&self) -> &Array1<f32> {
        &self.biases
    }

    // 9 and 10. Shapes. This layer does not change the width.
    fn output_size(&self) -> usize {
        self.weights.ncols()
    }

    fn input_size(&self) -> usize {
        self.weights.ncols()
    }

    // 11. Cloning behind a trait object.
    fn clone_box(&self) -> Box<dyn LayerTrait> {
        Box::new(self.clone())
    }

    // forward_batch_into and forward_into have defaults that clone the layer and run
    // forward_batch on the copy. Override them if the layer can write straight into a
    // caller's buffer, as DenseLayer does.
}

// Checking a layer against finite differences is the only way to know its backward pass
// is right. This is the shape of that check.
let mut layer = Scale::new(3);
let inputs = ndarray::array![[1.0f32, 2.0, -1.0], [0.5, -0.5, 2.0]];
let targets = ndarray::array![[0.5f32, 0.5, 0.5], [0.0, 0.0, 0.0]];

let loss = |layer: &mut Scale| -> f32 {
    let outputs = layer.forward_batch(inputs.view());
    (&outputs - &targets).mapv(|e| e * e).sum()
};

let outputs = layer.forward_batch(inputs.view());
let errors = (&outputs - &targets) * 2.0;
let (_, analytic, _) = layer.backward_batch(errors.view());

const EPS: f32 = 1e-2;
for j in 0..3 {
    let original = layer.weights[[0, j]];

    layer.weights[[0, j]] = original + EPS;
    let plus = loss(&mut layer);
    layer.weights[[0, j]] = original - EPS;
    let minus = loss(&mut layer);
    layer.weights[[0, j]] = original;

    let numerical = (plus - minus) / (2.0 * EPS);
    assert!(
        (analytic[[0, j]] - numerical).abs() < 5e-2 * analytic[[0, j]].abs().max(1.0),
        "scale {}: analytic {} vs numerical {}",
        j,
        analytic[[0, j]],
        numerical
    );
}
```

## Multi-agent training

Behind `--features multi-agent`. `SelfPlayTrainer` keeps a pool of past agents, samples
an opponent for each episode, and tracks Elo.

```rust,ignore
// cargo run --features multi-agent
use athena::agent::DqnAgent;
use athena::multi_agent::{SamplingStrategy, SelfPlayTrainer};
use athena::optimizer::{Adam, OptimizerWrapper};

let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
let agent = DqnAgent::new(&[8, 64, 64, 4], 0.3, optimizer, 500, true);

let mut trainer = SelfPlayTrainer::new(
    agent,
    8,    // agents kept in the pool
    100,  // episodes between pool updates
    SamplingStrategy::League { main_prob: 0.5, main_exploit_prob: 0.25 },
);

// Your environment implements athena::multi_agent::MultiAgentEnvironment
// let metrics = trainer.train(&mut env, 10_000, 64);

let ratings = trainer.get_elo_ratings();
println!("pool Elo: {:?}", ratings);
```

The three sampling strategies are `Uniform`, `Prioritized { temperature }` and
`League { main_prob, main_exploit_prob }`. `examples/self_play.rs` is the runnable
version.

Centralized training with decentralized execution is not implemented. Each agent in the
pool is an independent `DqnAgent` with its own observation.

## Partial observability

Behind `--features belief-states`. When the observation does not determine the state, the
agent needs a belief over states rather than the raw observation.

Two things are provided: `BeliefState`, a trait you implement for your problem, and
`ParticleFilter`, a general implementation of it.

```rust,ignore
// cargo run --features belief-states
use athena::belief::BeliefState;
use ndarray::Array1;

/// A belief over which of four rooms holds the target.
struct RoomBelief {
    probabilities: Array1<f32>,
}

impl BeliefState for RoomBelief {
    type Observation = usize;  // the room just looked in
    type State = usize;        // the room the target is in

    fn update(&mut self, _action: usize, observation: &usize) {
        // Looked in this room and did not find it, so its probability goes to zero
        self.probabilities[*observation] = 0.0;
        let total: f32 = self.probabilities.sum();
        if total > 0.0 {
            self.probabilities /= total;
        }
    }

    fn sample(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// This is what the agent actually sees, so it goes in as the state vector
    fn to_feature_vector(&self) -> Array1<f32> {
        self.probabilities.clone()
    }

    fn reset(&mut self) {
        self.probabilities.fill(0.25);
    }

    /// Uncertainty. Useful as a reward shaping term when the task is to find out.
    fn entropy(&self) -> f32 {
        -self
            .probabilities
            .iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| p * p.ln())
            .sum::<f32>()
    }
}
```

`BeliefDqnAgent` wraps a `DqnAgent` so it reads `to_feature_vector()` instead of the raw
observation. `examples/belief_tracking.rs` is the runnable version.

The other option for partial observability is a recurrent policy: `RecurrentNetwork::step`
carries a hidden state across frames, and `reset` clears it at an episode boundary. That
needs no feature flag.

## Reusing buffers

Two separate things, for two separate problems.

`InferenceBuffers` is for the forward pass and needs no pool:

```rust
use athena::activations::Activation;
use athena::network::{InferenceBuffers, NeuralNetwork};
use athena::optimizer::{OptimizerWrapper, SGD};

let network = NeuralNetwork::new(
    &[8, 64, 4],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
);

// Sized on the first call, reused after: a per-frame call allocates nothing.
// One instance per thread.
let mut buffers = InferenceBuffers::new();
let state = ndarray::array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

for _frame in 0..3 {
    let q_values = network.predict_into(state.view(), &mut buffers);
    assert_eq!(q_values.len(), 4);
}
```

`ArrayPool` is for scratch arrays your own code needs:

```rust
use athena::memory_optimization::ArrayPool;

let mut pool = ArrayPool::new(16);

let scratch = pool.get_array_2d((32, 64));
assert_eq!(scratch.dim(), (32, 64));
pool.return_array_2d(scratch);

// Served from the pool, and zeroed
let again = pool.get_array_2d((32, 64));
assert!(again.iter().all(|&v| v == 0.0));
```

`GradientAccumulator` is for a batch too large to hold at once. It takes the mean over
`accumulate` calls, weighting each equally, so keep the mini-batches the same size.

## Splitting work across cores

`athena::parallel` splits a batch across the rayon pool. It is worth it from a few hundred
rows upward. Below roughly a hundred, the handoff costs more than the work, and for one
action per frame a single-threaded `predict` wins.

```rust
use athena::activations::Activation;
use athena::network::NeuralNetwork;
use athena::optimizer::{OptimizerWrapper, SGD};
use athena::parallel::{ParallelGradients, ParallelNetwork};
use ndarray::Array2;

let mut network = NeuralNetwork::new(
    &[16, 64, 8],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
);

let inputs = Array2::from_shape_fn((1024, 16), |(i, j)| ((i + j) as f32 * 0.01).sin());
let targets = Array2::zeros((1024, 8));

// Inference. Borrows the network, so every thread reads one set of weights.
let parallel = ParallelNetwork::from_network(&network, 4);
let outputs = parallel.forward_batch_parallel(inputs.view());
assert_eq!(outputs.dim(), (1024, 8));

// Training. The same gradients backward_batch would give, computed across threads.
let (weight_grads, bias_grads) =
    ParallelGradients::compute_batch_gradients(&network, inputs.view(), targets.view());
let pairs: Vec<_> = weight_grads.into_iter().zip(bias_grads).collect();
network.apply_gradients(pairs, 0.01);
```

`examples/parallel_training.rs` prints the crossover on your machine. Measured here,
against the same cache-free path on one thread: 1.36x at batch 32, 1.92x at 256, 7.63x at
2048.

## Where to go next

- [`conventions.md`](conventions.md) for shapes, weight orientation and what can be
  stacked
- [`quickstart.md`](quickstart.md) for the end-to-end path
- [`algorithms_guide.md`](algorithms_guide.md) for choosing between the five algorithms
