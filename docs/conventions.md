# Conventions

The rules that are not visible from a signature. Read this before writing against the
library, or before assuming a shape.

## Scalars

Everything is `f32`. There is no generic scalar parameter and no `f64` path.

`f32::clamp` passes NaN straight through, so it is not a NaN guard. Check `is_finite()`
explicitly before a value reaches a training target:

```rust
# let value = 1.0f32;
# let (lo, hi) = (-10.0f32, 10.0f32);
let guarded = if value.is_finite() { value.clamp(lo, hi) } else { 0.0 };
# assert_eq!(guarded, 1.0);
```

## Shapes

The batch is always axis 0.

| Kind | Shape |
| --- | --- |
| One observation | `Array1<f32>`, length `input_size` |
| A batch | `Array2<f32>`, `(batch_size, features)` |
| A sequence batch | `Array3<f32>`, `(batch_size, sequence_length, input_size)` |
| Conv1D | `Array3<f32>`, `(batch_size, channels, length)` |
| Conv2D and pooling | `Array4<f32>`, `(batch_size, channels, height, width)` |

`NeuralNetwork::forward` takes an `Array1` and inserts the axis for you; `forward_batch`
takes the `Array2` directly.

## Dense weight orientation

**Weights are stored `(input_size, output_size)`.**

So the pre-activation is `input.dot(&weights)`, and a delta propagates back as
`adjusted_error.dot(&weights.t())`. Reversing this caused two panicking bugs in this
repository. Check any new `.dot` against it.

```rust
use athena::activations::Activation;
use athena::layers::Layer;

let layer = Layer::new(4, 3, Activation::Relu);
assert_eq!(layer.weights.dim(), (4, 3));  // (input_size, output_size)
assert_eq!(layer.biases.len(), 3);        // one per output
```

## Two forward passes, and why

`forward` and `forward_batch` take `&mut self`. Every dense layer stores its inputs and
its pre-activation output so the backward pass can read them, and that write is what
forces the mutable borrow.

`predict` and `predict_batch` take `&self` and store nothing. Use them for anything that
is not about to train:

```rust
use athena::activations::Activation;
use athena::network::{InferenceBuffers, NeuralNetwork};
use athena::optimizer::{OptimizerWrapper, SGD};
use std::sync::Arc;

let net = Arc::new(NeuralNetwork::new(
    &[4, 16, 2],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
));

// One network, many callers, no locking
let mut buffers = InferenceBuffers::new();
let state = ndarray::array![0.1, 0.2, 0.3, 0.4];
let output = net.predict_into(state.view(), &mut buffers);
assert_eq!(output.len(), 2);
```

`InferenceBuffers` sizes itself on the first call and is reused after that, so a
per-frame call allocates nothing. One instance serves one call at a time: give each
thread its own.

## Backward reads what forward wrote

`backward_batch`, `input_gradient_batch` and `apply_output_errors` all read the caches
the last `forward_batch` left behind. Calling one without a matching forward pass
immediately before panics rather than returning something wrong, and a forward pass on
*different* inputs in between silently produces the gradient for those other inputs.

The pattern, when you hold the outputs already:

```rust
use athena::activations::Activation;
use athena::network::NeuralNetwork;
use athena::optimizer::{OptimizerWrapper, SGD};
use ndarray::Array2;

let mut net = NeuralNetwork::new(
    &[3, 8, 2],
    &[Activation::Relu, Activation::Linear],
    OptimizerWrapper::SGD(SGD::new()),
);

let inputs = Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as f32 * 0.1);
let targets: Array2<f32> = Array2::zeros((4, 2));

let outputs = net.forward_batch(inputs.view());
let errors = &outputs - &targets;
let loss = errors.mapv(|e| e * e).mean().unwrap();
net.apply_output_errors(errors.view(), 0.01);   // no second forward pass
assert!(loss.is_finite());
```

`train_minibatch`, `train_policy_gradient` and `train_with_output_errors` are this with a
forward pass in front.

## Gradient scale

Each layer's `backward_batch` **sums** over the batch. `NeuralNetwork::backward_batch`
then divides by the batch size, so the gradient the optimizer sees is a **mean**.

A caller driving layers directly has to divide too, or its effective learning rate scales
with batch size.

## What can be stacked

`NeuralNetwork` holds `Vec<Layer>`, and `Layer` is an alias for `DenseLayer`. It is not
an enum. Conv, pooling, batch norm, dropout and embedding layers **cannot** be pushed
into a `NeuralNetwork`.

| Layer | How to use it |
| --- | --- |
| `DenseLayer` | `NeuralNetwork::new`, or `with_layers` |
| `LSTMLayer`, `GRULayer` | `recurrent::RecurrentNetwork`, or `forward_sequence` / `backward_sequence` / `apply_gradients` directly |
| `Conv1DLayer`, `Conv2DLayer`, pooling | Compose by hand against `LayerTrait`. `examples/conv_shapes.rs` is the worked reference. |
| `BatchNormLayer`, `DropoutLayer` | Compose by hand against `LayerTrait` |
| `EmbeddingLayer` | Forward only through `LayerTrait`: its `backward_batch` returns a zero gradient, so it does not train that way |

LSTM and GRU do not implement `LayerTrait` at all. The trait carries one weight matrix
and one bias vector per layer; an LSTM has eight and four.

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
let observation = ndarray::array![0.1, 0.2, 0.3];
let output = model.step(observation.view());
assert_eq!(output.len(), 1);
model.reset();   // at an episode boundary
```

`RecurrentNetwork` feeds only the final hidden state to its head, so sequence-to-sequence
output is not supported through it. The recurrent layers update with plain SGD: the
optimizers key their state by layer index and assume one weight matrix per layer.

## Optimizers

`Adam::new(&[], beta1, beta2, epsilon)` grows its per-layer state lazily, so one instance
clones safely into several networks and does not need the layers up front. It holds no
learning rate: that is an argument to each training call.

**RL on tasks with large returns needs Adam, not SGD.** Pendulum's return is around -1600
and a squared error at that magnitude diverges under a plain gradient step within a few
thousand steps.

## Target networks

A target network is assigned to, never trained. Build it with `clone_as_target`, which
drops the caches and gives it a stateless optimizer, and update it with
`copy_parameters_from` (hard) or `soft_update_from` (Polyak). Cloning the whole network
copies the optimizer's moment estimates for every parameter, for a network that never
uses them.

## Errors

Anything reading input from a game, a file or a network should use the checking form:
`try_forward`, `try_predict`, `DqnAgent::act`, `train_on_batch`. They return
`Result<_, AthenaError>` and report a wrong width. The unchecked forms multiply straight
into the first layer's weights, which for a game means the process dies mid-frame.

## Randomness and reproducibility

Agents hold a seedable `StdRng` (see `athena::rng`), so they are `Send` and a run
reproduces. Use `DqnAgent::new_seeded` or `set_seed`. Weight initialization still comes
from the thread generator, so pair a seed with a fixed set of weights when a run has to
repeat exactly. The generator is not serialized: a loaded agent explores from a fresh
one.

## Features

| Feature | What it turns on | Try it with |
| --- | --- | --- |
| *(default)* | Everything above | `cargo run --release --example game_loop_dqn` |
| `gpu` | OpenCL backend. Needs OpenCL drivers installed. | `cargo run --example gpu_test --features gpu` |
| `gpu-mock` | The same API with no OpenCL. All math runs on the CPU. | `cargo run --example gpu_test --features gpu-mock` |
| `action-masking` | `MaskedLayer`, `MaskedSoftmax`, `DqnAgent::train_on_batch_masked` | `cargo run --example masked_cartpole --features action-masking` |
| `belief-states` | POMDP belief states and a particle filter | `cargo run --example belief_tracking --features belief-states` |
| `multi-agent` | `SelfPlayTrainer`, communication channels | `cargo run --example self_play --features multi-agent` |
| `python` | PyO3 bindings | `cargo build --features python` |
| `wasm` | `wasm-bindgen` bindings | `cargo build --features wasm --target wasm32-unknown-unknown` |

`--all-features` does not link without OpenCL installed. Use `--features gpu-mock` to
compile the GPU API instead.

## Development

- `cargo test --lib` is the fast loop, a few seconds.
- `cargo test` also runs the doctests, which take several minutes.
- `cargo test --all-targets` runs the benches in the debug profile and does not finish in
  any reasonable time. Use `cargo build --examples` and `cargo test --test '*'` instead.
- `cargo clippy` reports style lints only.

## Known limitations

- `RecurrentNetwork` emits only the final hidden state.
- The recurrent layers train with plain SGD.
- `EmbeddingLayer` returns a zero gradient through `LayerTrait::backward_batch`.
- `export::json` writes weights but the importer does not rebuild a network from them;
  the working round trip is `save` and `load`.
- Under `gpu-mock` every operation runs on the CPU and `device_info` is fabricated, so
  its timings mean nothing.
- Several examples take illustrative shortcuts, labelled as such in their comments.
