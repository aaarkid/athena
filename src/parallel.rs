//! Rayon-backed helpers for work that is wide enough to be worth splitting.
//!
//! # When this is worth using
//!
//! Rayon costs a few microseconds to hand work to a thread pool. A dense forward pass on
//! a game-sized network is a few microseconds in total, so **splitting a batch of tens of
//! rows across threads loses**. These helpers pay off from a few hundred rows upward, and
//! for offline training rather than per-frame inference.
//!
//! For one action per frame, call `NeuralNetwork::predict` directly. It takes `&self` and
//! writes no caches, so many entities can share one network with no locking and no thread
//! pool involved.
//!
//! # What every function here guarantees
//!
//! Each one produces the same numbers as its serial counterpart, and
//! `src/parallel.rs`'s own tests assert exactly that rather than checking output shapes.
//! The shape-only tests that used to be here missed two weight-orientation bugs.

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis, Zip};

#[cfg(test)]
use crate::activations::Activation;
use crate::layers::Layer;
use crate::network::{InferenceBuffers, NeuralNetwork};

/// Rows per chunk when splitting a batch across threads.
///
/// Small enough that a batch of a few hundred still spreads over the pool, large enough
/// that each task is worth more than the handoff.
const CHUNK_ROWS: usize = 64;

/// Runs a network's forward pass over a batch, split across the rayon pool.
///
/// Borrows the network rather than owning a copy: the inference path takes `&self`, so
/// every thread reads the same weights.
pub struct ParallelNetwork<'a> {
    network: &'a NeuralNetwork,
    chunk_rows: usize,
}

impl<'a> ParallelNetwork<'a> {
    /// Borrow a network for parallel inference.
    ///
    /// `num_threads` is not used: rayon's global pool decides how many threads run. Build
    /// the pool yourself with `rayon::ThreadPoolBuilder` if you need to bound it.
    pub fn from_network(network: &'a NeuralNetwork, _num_threads: usize) -> Self {
        ParallelNetwork {
            network,
            chunk_rows: CHUNK_ROWS,
        }
    }

    /// Override how many rows each thread takes at a time.
    pub fn with_chunk_rows(mut self, rows: usize) -> Self {
        self.chunk_rows = rows.max(1);
        self
    }

    /// Forward pass over a batch, one chunk of rows per task.
    ///
    /// Produces the same numbers as `NeuralNetwork::predict_batch` on the same inputs.
    /// Nothing is cached, so this cannot be followed by a backward pass; use
    /// `ParallelGradients::compute_batch_gradients` for training.
    pub fn forward_batch_parallel(&self, inputs: ArrayView2<f32>) -> Array2<f32> {
        let rows = inputs.nrows();
        let mut output = Array2::zeros((rows, self.network.output_size()));

        if rows == 0 {
            return output;
        }

        let network = self.network;
        output
            .axis_chunks_iter_mut(Axis(0), self.chunk_rows)
            .into_par_iter()
            .zip(inputs.axis_chunks_iter(Axis(0), self.chunk_rows).into_par_iter())
            .for_each(|(mut out_chunk, in_chunk)| {
                // One scratch pair per task, reused across the layers of this chunk
                let mut buffers = InferenceBuffers::new();
                let result = network.predict_batch_into(in_chunk, &mut buffers);
                out_chunk.assign(&result);
            });

        output
    }
}

/// Matrix product, through whichever kernel ndarray was built with.
///
/// This is `a.dot(&b)`. ndarray's kernel is already blocked and, with the `rayon`
/// feature, threads large products itself, so there is nothing useful to add here. The
/// function exists because it was public.
pub fn matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.dot(&b)
}

/// Former name of [`matmul`].
///
/// It never did anything in parallel that `dot` was not already doing.
#[deprecated(since = "0.4.0", note = "renamed to `matmul`; it was always a plain dot product")]
pub fn parallel_matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    matmul(a, b)
}

/// Direct 2D convolution, one image per task.
pub struct ParallelConv2D;

impl ParallelConv2D {
    /// Convolve a batch against a kernel stack, splitting the batch across threads.
    ///
    /// Zero padding, no dilation, no bias. `Conv2DLayer` is the trainable version; this
    /// is a bare correlation for cases that only need the forward result.
    pub fn convolve2d_parallel(
        input: ArrayView4<f32>,
        kernels: ArrayView4<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Array4<f32> {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (out_channels, kernel_in_channels, kernel_height, kernel_width) = kernels.dim();
        assert_eq!(
            in_channels, kernel_in_channels,
            "input has {} channels but the kernels expect {}",
            in_channels, kernel_in_channels
        );

        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));

        Zip::from(output.axis_iter_mut(Axis(0)))
            .and(input.axis_iter(Axis(0)))
            .par_for_each(|mut out_image, in_image| {
                for oc in 0..out_channels {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            // Position in the padded input, then translated back. A tap
                            // landing in the padding contributes nothing.
                            let h_start = oh * stride.0;
                            let w_start = ow * stride.1;

                            let mut sum = 0.0;
                            for ic in 0..in_channels {
                                for kh in 0..kernel_height {
                                    let h = h_start + kh;
                                    if h < padding.0 {
                                        continue;
                                    }
                                    let h_idx = h - padding.0;
                                    if h_idx >= in_height {
                                        continue;
                                    }

                                    for kw in 0..kernel_width {
                                        let w = w_start + kw;
                                        if w < padding.1 {
                                            continue;
                                        }
                                        let w_idx = w - padding.1;
                                        if w_idx >= in_width {
                                            continue;
                                        }

                                        sum += in_image[[ic, h_idx, w_idx]]
                                            * kernels[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                            out_image[[oc, oh, ow]] = sum;
                        }
                    }
                }
            });

        output
    }
}

/// Per-layer parameter gradients, accumulated in place.
///
/// One entry per layer, in the network's own order. The shapes match each layer's
/// weights and biases.
#[derive(Clone, Debug)]
pub struct GradientAccumulator {
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
}

impl GradientAccumulator {
    /// Zero gradients shaped for `network`.
    pub fn zeros_for(network: &NeuralNetwork) -> Self {
        GradientAccumulator {
            weights: network
                .layers
                .iter()
                .map(|layer| Array2::zeros(layer.weights.dim()))
                .collect(),
            biases: network
                .layers
                .iter()
                .map(|layer| Array1::zeros(layer.biases.len()))
                .collect(),
        }
    }

    /// Add another accumulator into this one, in place.
    pub fn add_assign(&mut self, other: &GradientAccumulator) {
        for (target, source) in self.weights.iter_mut().zip(other.weights.iter()) {
            *target += source;
        }
        for (target, source) in self.biases.iter_mut().zip(other.biases.iter()) {
            *target += source;
        }
    }

    /// Divide every gradient by `divisor`, in place.
    pub fn scale(&mut self, divisor: f32) {
        if divisor == 0.0 {
            return;
        }
        let factor = 1.0 / divisor;
        for weights in self.weights.iter_mut() {
            *weights *= factor;
        }
        for biases in self.biases.iter_mut() {
            *biases *= factor;
        }
    }

    /// Pair the gradients up the way `NeuralNetwork::apply_gradients` wants them.
    pub fn into_pairs(self) -> Vec<(Array2<f32>, Array1<f32>)> {
        self.weights.into_iter().zip(self.biases).collect()
    }
}

/// Squared-error gradients for a batch, computed across threads.
pub struct ParallelGradients;

impl ParallelGradients {
    /// Mean squared-error gradients for a batch, split across the rayon pool.
    ///
    /// Matches `NeuralNetwork::forward_batch` followed by `backward_batch` on
    /// `outputs - targets`: the same forward arithmetic, the same activation
    /// derivatives, and the same averaging over the batch. It takes `&NeuralNetwork`
    /// rather than `&mut`, and never clones the network, because it carries its own
    /// activations instead of using the layers' caches.
    ///
    /// Returns `(weight_gradients, bias_gradients)`, one entry per layer.
    pub fn compute_batch_gradients(
        network: &NeuralNetwork,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let accumulated = Self::accumulate(network, inputs, targets, CHUNK_ROWS);
        (accumulated.weights, accumulated.biases)
    }

    /// `compute_batch_gradients` returning the accumulator, for callers that want to add
    /// several batches together before applying anything.
    pub fn accumulate(
        network: &NeuralNetwork,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        chunk_rows: usize,
    ) -> GradientAccumulator {
        assert_eq!(
            inputs.nrows(),
            targets.nrows(),
            "{} inputs against {} targets",
            inputs.nrows(),
            targets.nrows()
        );

        let batch_size = inputs.nrows();
        let mut total = GradientAccumulator::zeros_for(network);
        if batch_size == 0 || network.layers.is_empty() {
            return total;
        }

        let chunk_rows = chunk_rows.max(1);

        // Each task folds its chunks into one accumulator, so the number of allocations
        // follows the thread count rather than the batch size
        let summed = inputs
            .axis_chunks_iter(Axis(0), chunk_rows)
            .into_par_iter()
            .zip(targets.axis_chunks_iter(Axis(0), chunk_rows).into_par_iter())
            .fold(
                || GradientAccumulator::zeros_for(network),
                |mut accumulator, (input_chunk, target_chunk)| {
                    chunk_gradients(network, input_chunk, target_chunk, &mut accumulator);
                    accumulator
                },
            )
            .reduce(
                || GradientAccumulator::zeros_for(network),
                |mut left, right| {
                    left.add_assign(&right);
                    left
                },
            );

        total.add_assign(&summed);
        // Each layer's backward pass sums over its rows, so the batch mean is the sum
        // over the whole batch divided by its size. NeuralNetwork::backward_batch does
        // the same division, which is what makes the two agree.
        total.scale(batch_size as f32);
        total
    }
}

/// Forward then backward over one chunk of rows, summing into `accumulator`.
///
/// Written against explicit activation buffers rather than the layers' caches, so it
/// needs only `&NeuralNetwork` and can run on several chunks at once.
fn chunk_gradients(
    network: &NeuralNetwork,
    inputs: ArrayView2<f32>,
    targets: ArrayView2<f32>,
    accumulator: &mut GradientAccumulator,
) {
    let depth = network.layers.len();

    // activations[i] is what layer i reads; pre_activations[i] is what its activation
    // function was applied to, which is what the derivative needs
    let mut activations: Vec<Array2<f32>> = Vec::with_capacity(depth + 1);
    let mut pre_activations: Vec<Array2<f32>> = Vec::with_capacity(depth);

    activations.push(inputs.to_owned());

    for (i, layer) in network.layers.iter().enumerate() {
        // Weights are (input_size, output_size), so the input goes on the left
        let mut pre_activation = activations[i].dot(&layer.weights);
        pre_activation += &layer.biases;

        let mut activated = pre_activation.clone();
        layer.activation.apply_batch(&mut activated);

        pre_activations.push(pre_activation);
        activations.push(activated);
    }

    let outputs = &activations[depth];
    let mut error = outputs - &targets;

    for i in (0..depth).rev() {
        let layer = &network.layers[i];

        let derivative = layer.activation.derivative_batch(pre_activations[i].view());
        let adjusted = &error * &derivative;

        accumulator.weights[i] += &activations[i].t().dot(&adjusted);
        accumulator.biases[i] += &adjusted.sum_axis(Axis(0));

        if i > 0 {
            error = adjusted.dot(&layer.weights.t());
        }
    }
}

/// A replay buffer whose entries can be handed to other threads.
///
/// `crate::replay_buffer::ReplayBuffer` is the one the agents use; this one exists for
/// arbitrary payloads that are `Send + Sync`.
pub struct ParallelReplayBuffer<T: Send + Sync> {
    pub buffer: Vec<T>,
    capacity: usize,
    position: usize,
}

impl<T: Send + Sync + Clone> ParallelReplayBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        ParallelReplayBuffer {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn add(&mut self, experience: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
            self.position = (self.position + 1) % self.capacity;
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Draw `batch_size` distinct entries, using the given generator.
    ///
    /// Costs O(batch_size), not O(capacity): the previous version shuffled every index
    /// in the buffer and then threw all but `batch_size` of them away, which at 100k
    /// entries dominated the call.
    pub fn sample_with<R: rand::Rng + ?Sized>(&self, batch_size: usize, rng: &mut R) -> Vec<T> {
        let available = self.buffer.len();
        if available == 0 {
            return Vec::new();
        }

        let wanted = batch_size.min(available);
        rand::seq::index::sample(rng, available, wanted)
            .into_iter()
            .map(|index| self.buffer[index].clone())
            .collect()
    }

    /// `sample_with` on a fresh generator.
    ///
    /// The clone of each entry is the cost here, and for the batch sizes this is called
    /// with it is not worth a thread handoff, so the gather is sequential despite the
    /// type's name.
    pub fn sample_parallel(&self, batch_size: usize) -> Vec<T> {
        let mut rng = crate::rng::default_rng();
        self.sample_with(batch_size, &mut rng)
    }
}

/// Image augmentation, one image per task.
pub struct ParallelAugmentation;

impl ParallelAugmentation {
    /// Apply random augmentations to a batch of images, splitting the batch across
    /// threads.
    ///
    /// Each image draws from the thread's own generator, so the result is not
    /// reproducible. Use `augment_batch_with` for a run that has to repeat.
    pub fn augment_batch(images: ArrayView4<f32>) -> Array4<f32> {
        let mut result = images.to_owned();

        Zip::from(result.axis_iter_mut(Axis(0)))
            .and(images.axis_iter(Axis(0)))
            .par_for_each(|mut output_image, input_image| {
                let mut rng = crate::rng::default_rng();
                let augmented = Self::augment_single(input_image, &mut rng);
                output_image.assign(&augmented);
            });

        result
    }

    /// `augment_batch` seeded per image, so the whole batch reproduces.
    ///
    /// Image `i` draws from `seed + i`, which keeps the result independent of how rayon
    /// happened to schedule the work.
    pub fn augment_batch_with(images: ArrayView4<f32>, seed: u64) -> Array4<f32> {
        let mut result = images.to_owned();

        Zip::indexed(result.axis_iter_mut(Axis(0)))
            .and(images.axis_iter(Axis(0)))
            .par_for_each(|index, mut output_image, input_image| {
                let mut rng = crate::rng::seeded_rng(seed.wrapping_add(index as u64));
                let augmented = Self::augment_single(input_image, &mut rng);
                output_image.assign(&augmented);
            });

        result
    }

    fn augment_single<R: rand::Rng + ?Sized>(image: ArrayView3<f32>, rng: &mut R) -> Array3<f32> {
        let mut result = image.to_owned();

        // Random horizontal flip
        if rng.gen_bool(0.5) {
            let width = result.dim().2;
            for mut plane in result.axis_iter_mut(Axis(0)) {
                for mut row in plane.axis_iter_mut(Axis(0)) {
                    for w in 0..width / 2 {
                        let mirrored = width - 1 - w;
                        let left = row[w];
                        row[w] = row[mirrored];
                        row[mirrored] = left;
                    }
                }
            }
        }

        // Random brightness adjustment
        if rng.gen_bool(0.3) {
            let factor = rng.gen_range(0.8..1.2);
            result.mapv_inplace(|x| (x * factor).clamp(0.0, 1.0));
        }

        result
    }
}

impl Layer {
    /// The layer's output before its activation function.
    ///
    /// Weights are `(input_size, output_size)`, so the input goes on the left. Getting
    /// that backwards is what caused two panicking bugs in this file.
    pub fn forward_pre_activation(&self, input: ArrayView1<f32>) -> Array1<f32> {
        input.dot(&self.weights) + &self.biases
    }
}

/// Activations reachable from this module, kept so the parity tests can cover more than
/// one derivative shape.
#[cfg(test)]
const TEST_ACTIVATIONS: [Activation; 3] = [Activation::Relu, Activation::Tanh, Activation::Linear];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Conv2DLayer;
    use crate::optimizer::{OptimizerWrapper, SGD};
    use crate::rng::seeded_rng;

    fn network() -> NeuralNetwork {
        NeuralNetwork::new(&[10, 20, 15, 10], &TEST_ACTIVATIONS, OptimizerWrapper::SGD(SGD::new()))
    }

    fn batch(rows: usize, width: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, width), |(i, j)| ((i * width + j) as f32 * 0.13).sin())
    }

    #[test]
    fn matmul_is_the_dot_product_it_says_it_is() {
        let a = batch(100, 50);
        let b = batch(50, 30);

        let through_helper = matmul(a.view(), b.view());
        let direct = a.dot(&b);

        assert_eq!(through_helper.dim(), (100, 30));
        for (x, y) in through_helper.iter().zip(direct.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn the_parallel_forward_pass_matches_the_serial_one() {
        let net = network();
        let parallel = ParallelNetwork::from_network(&net, 4);

        // Several batch sizes, including ones that do not divide the chunk width
        for rows in [1usize, 7, 64, 65, 200] {
            let inputs = batch(rows, 10);

            let from_parallel = parallel.forward_batch_parallel(inputs.view());
            let serial = net.predict_batch(inputs.view());

            assert_eq!(from_parallel.dim(), serial.dim(), "shape differs at {} rows", rows);
            for (a, b) in from_parallel.iter().zip(serial.iter()) {
                assert!((a - b).abs() < 1e-5, "{} rows: {} vs {}", rows, a, b);
            }
        }
    }

    #[test]
    fn a_narrow_chunk_width_gives_the_same_answer() {
        let net = network();
        let inputs = batch(37, 10);
        let serial = net.predict_batch(inputs.view());

        // One row per task exercises every boundary there is
        let parallel = ParallelNetwork::from_network(&net, 4).with_chunk_rows(1);
        let from_parallel = parallel.forward_batch_parallel(inputs.view());

        for (a, b) in from_parallel.iter().zip(serial.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    #[test]
    fn an_empty_batch_forwards_to_an_empty_result() {
        let net = network();
        let parallel = ParallelNetwork::from_network(&net, 4);
        let empty = Array2::zeros((0, 10));
        assert_eq!(parallel.forward_batch_parallel(empty.view()).dim(), (0, 10));
    }

    #[test]
    fn the_parallel_gradients_match_backward_batch() {
        // This is the test that would have caught the weight-orientation bugs: it
        // compares against the gradient the network itself computes, not against a
        // shape.
        let mut net = network();

        for rows in [1usize, 8, 64, 130] {
            let inputs = batch(rows, 10);
            let targets = Array2::from_shape_fn((rows, 10), |(i, j)| ((i + j) as f32 * 0.07).cos());

            let (weight_grads, bias_grads) =
                ParallelGradients::compute_batch_gradients(&net, inputs.view(), targets.view());

            let outputs = net.forward_batch(inputs.view());
            let errors = &outputs - &targets;
            let reference = net.backward_batch(errors.view());

            assert_eq!(weight_grads.len(), reference.len());
            for (i, (weights, biases)) in weight_grads.iter().zip(bias_grads.iter()).enumerate() {
                assert_eq!(weights.dim(), reference[i].0.dim(), "layer {} weight shape", i);
                assert_eq!(biases.len(), reference[i].1.len(), "layer {} bias shape", i);

                for (a, b) in weights.iter().zip(reference[i].0.iter()) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "{} rows, layer {} weight gradient: {} vs {}",
                        rows,
                        i,
                        a,
                        b
                    );
                }
                for (a, b) in biases.iter().zip(reference[i].1.iter()) {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "{} rows, layer {} bias gradient: {} vs {}",
                        rows,
                        i,
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn the_chunk_width_does_not_change_the_gradient() {
        let net = network();
        let inputs = batch(50, 10);
        let targets = Array2::from_shape_fn((50, 10), |(i, j)| ((i * 3 + j) as f32 * 0.11).sin());

        let wide = ParallelGradients::accumulate(&net, inputs.view(), targets.view(), 1024);
        let narrow = ParallelGradients::accumulate(&net, inputs.view(), targets.view(), 1);

        for (layer, (a, b)) in wide.weights.iter().zip(narrow.weights.iter()).enumerate() {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-4, "layer {}: {} vs {}", layer, x, y);
            }
        }
    }

    #[test]
    fn training_on_the_parallel_gradients_lowers_the_loss() {
        let mut net = network();
        let inputs = batch(96, 10);
        let targets = Array2::from_shape_fn((96, 10), |(i, j)| ((i + 2 * j) as f32 * 0.05).sin());

        let loss = |net: &NeuralNetwork| -> f32 {
            let outputs = net.predict_batch(inputs.view());
            (&outputs - &targets).mapv(|e| e * e).mean().unwrap_or(f32::INFINITY)
        };

        let before = loss(&net);
        for _ in 0..40 {
            let gradients =
                ParallelGradients::accumulate(&net, inputs.view(), targets.view(), CHUNK_ROWS);
            net.apply_gradients(gradients.into_pairs(), 0.05);
        }
        let after = loss(&net);

        assert!(after < before, "loss went from {} to {}", before, after);
    }

    #[test]
    fn the_parallel_convolution_matches_conv2d_layer() {
        // Conv2DLayer is the trainable version, so its forward pass is the reference
        let mut layer = Conv2DLayer::new(2, 3, (3, 3), (1, 1), (1, 1), Activation::Linear);
        layer.biases.fill(0.0);

        let input = Array4::from_shape_fn((4, 2, 7, 7), |(b, c, h, w)| {
            ((b * 98 + c * 49 + h * 7 + w) as f32 * 0.03).sin()
        });

        let from_layer = layer.forward_batch(input.view());
        let from_parallel = ParallelConv2D::convolve2d_parallel(
            input.view(),
            layer.kernels.view(),
            (1, 1),
            (1, 1),
        );

        assert_eq!(from_layer.dim(), from_parallel.dim());
        for (a, b) in from_layer.iter().zip(from_parallel.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
    }

    #[test]
    fn a_strided_convolution_still_matches() {
        let mut layer = Conv2DLayer::new(1, 2, (2, 2), (2, 2), (0, 0), Activation::Linear);
        layer.biases.fill(0.0);

        // 5 is not a multiple of the stride, which is where an output-derived shape goes
        // wrong
        let input = Array4::from_shape_fn((2, 1, 5, 5), |(b, _, h, w)| {
            ((b * 25 + h * 5 + w) as f32 * 0.17).cos()
        });

        let from_layer = layer.forward_batch(input.view());
        let from_parallel = ParallelConv2D::convolve2d_parallel(
            input.view(),
            layer.kernels.view(),
            (2, 2),
            (0, 0),
        );

        assert_eq!(from_layer.dim(), from_parallel.dim());
        for (a, b) in from_layer.iter().zip(from_parallel.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
    }

    #[test]
    fn forward_pre_activation_matches_the_batch_form() {
        let layer = Layer::new(6, 4, Activation::Relu);
        let input = Array1::from_shape_fn(6, |i| (i as f32) * 0.3 - 1.0);

        let single = layer.forward_pre_activation(input.view());
        let expected = input.dot(&layer.weights) + &layer.biases;

        for (a, b) in single.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn sampling_draws_distinct_entries_and_respects_the_buffer_length() {
        let mut buffer = ParallelReplayBuffer::new(1000);
        for i in 0..1000usize {
            buffer.add(i);
        }

        let mut rng = seeded_rng(5);
        let drawn = buffer.sample_with(32, &mut rng);
        assert_eq!(drawn.len(), 32);

        let unique: std::collections::HashSet<usize> = drawn.iter().copied().collect();
        assert_eq!(unique.len(), 32, "sample_with returned a repeat");

        // Asking for more than the buffer holds returns the buffer
        assert_eq!(buffer.sample_with(4000, &mut rng).len(), 1000);

        let empty: ParallelReplayBuffer<usize> = ParallelReplayBuffer::new(10);
        assert!(empty.sample_with(4, &mut rng).is_empty());
    }

    #[test]
    fn a_seeded_augmentation_repeats() {
        let images = Array4::from_shape_fn((6, 2, 4, 4), |(b, c, h, w)| {
            ((b * 32 + c * 16 + h * 4 + w) as f32 * 0.05).abs().min(1.0)
        });

        let first = ParallelAugmentation::augment_batch_with(images.view(), 42);
        let second = ParallelAugmentation::augment_batch_with(images.view(), 42);

        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a, b, "the same seed gave a different batch");
        }

        // Every value stays in range and finite whichever augmentations fired
        assert!(first.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }
}
