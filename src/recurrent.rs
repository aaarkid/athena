//! # Recurrent Network Module
//!
//! Pairs a recurrent layer with a feedforward head so a sequence can be trained
//! end to end. `NeuralNetwork` holds dense layers only, and the `Layer` trait
//! carries one weight matrix per layer, which is why LSTM and GRU need their own
//! container rather than being stacked into a `NeuralNetwork`.
//!
//! The recurrent layer reads the whole sequence and the head reads its final
//! hidden state, which covers sequence classification and sequence-to-value
//! regression. Per-time-step outputs are not supported here; drive
//! `forward_sequence` and `backward_sequence` directly for that.
//!
//! ## Example
//!
//! ```rust,no_run
//! use athena::recurrent::{RecurrentCell, RecurrentNetwork};
//! use athena::layers::LSTMLayer;
//! use athena::activations::Activation;
//! use athena::optimizer::{Adam, OptimizerWrapper};
//! use ndarray::Array3;
//!
//! let cell = RecurrentCell::Lstm(LSTMLayer::new(2, 8, false));
//! let mut model = RecurrentNetwork::new(
//!     cell,
//!     &[8, 1],
//!     &[Activation::Linear],
//!     OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)),
//! );
//!
//! let inputs = Array3::zeros((4, 6, 2));
//! let targets = ndarray::Array2::zeros((4, 1));
//! let loss = model.train_batch(inputs.view(), targets.view(), 0.01);
//! ```

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::activations::Activation;
use crate::layers::{GRULayer, LSTMLayer};
use crate::network::NeuralNetwork;
use crate::optimizer::OptimizerWrapper;

/// The recurrent layer a `RecurrentNetwork` runs over the sequence.
// Both variants are large and within 1.4x of each other, so boxing one buys nothing
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RecurrentCell {
    Lstm(LSTMLayer),
    Gru(GRULayer),
}

impl RecurrentCell {
    /// Size of the hidden state, which is the input width of the head.
    pub fn hidden_size(&self) -> usize {
        match self {
            RecurrentCell::Lstm(layer) => layer.hidden_size,
            RecurrentCell::Gru(layer) => layer.hidden_size,
        }
    }

    /// Width of one time step of the input.
    pub fn input_size(&self) -> usize {
        match self {
            RecurrentCell::Lstm(layer) => layer.input_size,
            RecurrentCell::Gru(layer) => layer.input_size,
        }
    }

    /// The head reads the final hidden state, so the cell must emit only that step.
    fn use_last_step_only(&mut self) {
        match self {
            RecurrentCell::Lstm(layer) => layer.return_sequences = false,
            RecurrentCell::Gru(layer) => layer.return_sequences = false,
        }
    }

    fn reset(&mut self) {
        match self {
            RecurrentCell::Lstm(layer) => layer.reset_states(),
            RecurrentCell::Gru(layer) => layer.reset_state(),
        }
    }

    /// Advance one time step, carrying the state forward and writing no BPTT cache.
    fn forward_step(&mut self, input: ArrayView2<f32>) -> Array2<f32> {
        match self {
            RecurrentCell::Lstm(layer) => layer.forward_step(input),
            RecurrentCell::Gru(layer) => layer.forward_step(input),
        }
    }

    fn forward_sequence(&mut self, inputs: ArrayView3<f32>) -> Array3<f32> {
        match self {
            RecurrentCell::Lstm(layer) => layer.forward_sequence(inputs),
            RecurrentCell::Gru(layer) => layer.forward_sequence(inputs),
        }
    }

    /// Runs backward through time and applies the result, returning the gradient
    /// with respect to the inputs.
    fn backward_and_apply(&mut self, output_grad: ArrayView3<f32>, learning_rate: f32) -> Array3<f32> {
        match self {
            RecurrentCell::Lstm(layer) => {
                let gradients = layer.backward_sequence(output_grad);
                layer.apply_gradients(&gradients, learning_rate);
                gradients.dx
            }
            RecurrentCell::Gru(layer) => {
                let gradients = layer.backward_sequence(output_grad);
                layer.apply_gradients(&gradients, learning_rate);
                gradients.dx
            }
        }
    }
}

/// A recurrent layer followed by a dense head.
pub struct RecurrentNetwork {
    pub cell: RecurrentCell,
    pub head: NeuralNetwork,
}

impl RecurrentNetwork {
    /// Build a network from a cell and the head's layer sizes and activations.
    ///
    /// The first entry of `head_sizes` must equal the cell's hidden size.
    pub fn new(
        mut cell: RecurrentCell,
        head_sizes: &[usize],
        head_activations: &[Activation],
        optimizer: OptimizerWrapper,
    ) -> Self {
        cell.use_last_step_only();
        assert_eq!(
            head_sizes[0],
            cell.hidden_size(),
            "head input must match the cell's hidden size"
        );

        let head = NeuralNetwork::new(head_sizes, head_activations, optimizer);
        RecurrentNetwork { cell, head }
    }

    /// Run a batch of sequences through the cell and the head.
    ///
    /// Input shape `(batch_size, sequence_length, input_size)`, output shape
    /// `(batch_size, head_output_size)`. The cell state is reset first, so each
    /// call treats its input as a fresh sequence.
    pub fn forward_batch(&mut self, inputs: ArrayView3<f32>) -> Array2<f32> {
        let hidden = self.final_hidden_state(inputs);
        self.head.forward_batch(hidden.view())
    }

    /// One training step on a batch of sequences, returning the mean squared error
    /// before the update.
    pub fn train_batch(
        &mut self,
        inputs: ArrayView3<f32>,
        targets: ArrayView2<f32>,
        learning_rate: f32,
    ) -> f32 {
        let hidden = self.final_hidden_state(inputs);

        let outputs = self.head.forward_batch(hidden.view());
        let output_errors = &outputs - &targets;
        let loss = output_errors.mapv(|e| e * e).mean().unwrap_or(0.0);

        // The head's gradient with respect to its own input is the error signal for
        // the cell's final hidden state. Read it before training the head, so the
        // cached pre-activations still belong to this forward pass.
        let hidden_grad = self.head.input_gradient_batch(output_errors.view());
        // The forward pass above already wrote the head's caches, so the error goes
        // straight back through them
        self.head.apply_output_errors(output_errors.view(), learning_rate);

        // The cell returned only the last step, so its gradient has one step too
        let cell_grad = hidden_grad.insert_axis(Axis(1));
        self.cell.backward_and_apply(cell_grad.view(), learning_rate);

        loss
    }

    /// Advance one time step and read the head, carrying the cell state forward.
    ///
    /// This is the per-frame entry point. `forward_batch` resets the cell and reads a
    /// whole sequence, so using it every frame means re-feeding the entire history.
    /// Call `reset` at an episode boundary.
    ///
    /// No BPTT cache is written, so training still goes through `train_batch`.
    pub fn step(&mut self, observation: ArrayView1<f32>) -> Array1<f32> {
        let output = self.step_batch(observation.insert_axis(Axis(0)));
        let width = output.shape()[1];
        output
            .into_shape((width,))
            .expect("the head returned more than one row for one observation")
    }

    /// Batch form of `step`: one time step for each of several sequences at once.
    pub fn step_batch(&mut self, observations: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            observations.shape()[1],
            self.cell.input_size(),
            "input width must match the cell's input size"
        );

        let hidden = self.cell.forward_step(observations);
        self.head.predict_batch(hidden.view())
    }

    /// Clear the carried hidden state, at an episode boundary.
    ///
    /// `step` carries state from one call to the next; without this the first frame of a
    /// new episode continues the previous one.
    pub fn reset(&mut self) {
        self.cell.reset();
    }

    /// Mean squared error on a batch, without updating anything.
    pub fn loss(&mut self, inputs: ArrayView3<f32>, targets: ArrayView2<f32>) -> f32 {
        let outputs = self.forward_batch(inputs);
        (&outputs - &targets).mapv(|e| e * e).mean().unwrap_or(0.0)
    }

    fn final_hidden_state(&mut self, inputs: ArrayView3<f32>) -> Array2<f32> {
        assert_eq!(
            inputs.shape()[2],
            self.cell.input_size(),
            "input width must match the cell's input size"
        );

        self.cell.reset();
        let hidden = self.cell.forward_sequence(inputs);
        let last = hidden.shape()[1] - 1;
        hidden.index_axis(Axis(1), last).to_owned()
    }
}
