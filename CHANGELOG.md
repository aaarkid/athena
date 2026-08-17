# Changelog

## 0.4.0

Breaking changes to the API and to the on-disk format.

### Breaking

- **Saved files carry a header.** Every `save` now writes `ATHN` plus a `u32` format
  version ahead of the bincode payload, and `load` refuses anything else. Files written
  by 0.3.x cannot be read. `load` also runs `NeuralNetwork::validate`, so a file whose
  layers do not chain is reported instead of panicking on the first forward pass.
- **Forward-pass caches are no longer serialized.** They were as large as the last batch
  and held fragments of the training data, so a saved `[4,128,64,2]` network was bigger
  than the model itself and its size depended on whether `act` or `train` ran last.
- **`LSTMLayer` and `GRULayer` no longer implement the `Layer` trait.** The trait carries
  one weight matrix and one bias vector per layer; an LSTM has eight and four, so its
  gradients could not travel through it and it silently returned zeros. Train them
  through `crate::recurrent::RecurrentNetwork`, or through `forward_sequence`,
  `backward_sequence` and `apply_gradients` directly.
- **`NeuralNetwork::backward_batch` averages over the batch** instead of summing. A
  learning rate that was stable at batch 32 was taking steps 32 times larger at batch
  1024. Learning rates tuned against 0.3.x may need raising by roughly the batch size.
- **`DqnAgent::act`, `act_masked` and `get_masked_q_values` return `Result`** rather than
  panicking inside ndarray on a state of the wrong width.
- **`MaskedLayer::apply_mask` and `MaskedSoftmax::forward_masked` return `Result`.** An
  all-false mask used to return an all-zero vector under a comment promising a uniform
  distribution; it is now an error.
- **`DqnAgent::train_on_batch` reports a different loss.** It is now the mean squared TD
  error over the batch, measured before the update. It used to be measured after the
  update and divided by the number of actions, so it read `num_actions` times too small.
- **The loss functions report a mean over samples and features.** `MSE::gradient_batch`
  was divided by the batch size while `compute_batch` was divided by `2 * batch *
  features`, so the gradient was not the derivative of the value.

### Added

- `NeuralNetwork::predict`, `predict_batch`, `predict_into`, `predict_batch_into`,
  `try_predict` and `try_predict_batch`: a forward pass that writes no caches and takes
  `&self`, so one `Arc<NeuralNetwork>` serves many entities. With `InferenceBuffers` a
  per-frame call allocates nothing.
- `NeuralNetwork::apply_output_errors` and `apply_output_errors_clipped`: backpropagate
  through the caches an earlier `forward_batch` wrote, without repeating it.
- `NeuralNetwork::soft_update_from`, `copy_parameters_from`, `clone_as_target` and
  `validate`.
- `NeuralNetwork::train_minibatch_clipped` and `train_policy_gradient_clipped`, global
  gradient norm clipping across all layers.
- `RecurrentNetwork::step`, `step_batch` and `reset`: advance one time step carrying the
  cell state, instead of re-feeding the whole history every frame.
- `DqnAgent::new_seeded`, `set_seed`, `train_on_batch_masked` and
  `train_on_batch_weighted`.
- `Conv1DLayer::backward_batch` and `MaxPool1DLayer::backward_batch`, both checked
  against finite differences.
- `crate::rng` with `default_rng` and `seeded_rng`, and `crate::serialization` with the
  file format.
- `LayerTrait::forward_batch_into` and `forward_into`, cache-free forward passes with
  defaults for every implementation.

### Fixed

- LSTM and GRU dropped the gradient from every step but the last when
  `return_sequences` was off, so backpropagation through time only ever saw one step.
- TD3's actor searched for its update by finite differences instead of following the
  deterministic policy gradient.
- The entropy bonus in A2C and PPO differentiated each action independently, so the
  gradient did not sum to zero across the softmax.
- A2C bootstrapped nothing at the end of a truncated rollout.
- `BatchNormLayer` took a different branch in the backward pass than the forward pass
  had taken, and its inference `grad_gamma` was a copy of `grad_beta`.
- `Conv2DLayer::compute_input_gradients` shaped the gradient from the output, which was
  wrong whenever the stride did not divide the input evenly.
- `max_grad_norm` and `value_coeff` were stored by A2C and PPO and never read.
- `PrioritizedReplayBuffer` indices shifted under eviction, so `update_priorities` wrote
  to the wrong entry. Sampling now returns stable slot ids.
- Agents hold a seedable `StdRng` instead of `ThreadRng`, so they are `Send` and a run
  reproduces.

### Performance

Measured single-threaded in release on one machine; treat the ratios, not the absolutes.

- `ReplayBuffer::sample(32)` at 100k entries: 933 us to 0.47 us, and now flat in
  capacity.
- `predict` against `forward` on `[64,256,256,16]`: 25.6 us to 11.9 us. `DqnAgent::act`
  on the same shape: 25.2 us to 12.1 us.
- Adam's overhead over SGD on 86,544 parameters at batch 32: 921 us to 413 us. RMSProp's:
  492 us to 197 us.
- `LSTMLayer::forward_step` at 16 inputs and 64 units, batch 1: 11.3 us to 5.9 us.
- Target network updates are written in place, and target networks no longer carry a
  copy of the trained network's optimizer state.

## 0.3.0

See the git history.
