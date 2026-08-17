//! Numerical gradient checks for the recurrent layers.
//!
//! LSTM and GRU carry their own `backward_sequence`, outside the `Layer` trait,
//! and nothing else in the crate exercises it. These tests compare it against
//! finite differences of the loss.

use ndarray::{Array3, ArrayView3};

use crate::layers::{GRUGradients, GRULayer, LSTMGradients, LSTMLayer};

const EPS: f32 = 1e-2;
// f32 finite differences on a recurrent graph accumulate noise across time steps
const TOLERANCE: f32 = 5e-2;
// The finite difference is quantized by the f32 resolution of the loss itself,
// roughly 1e-7 / (2 * EPS), so components below this cannot be resolved at all
const ABS_FLOOR: f32 = 3e-5;

fn deterministic_seq(batch: usize, seq: usize, features: usize) -> Array3<f32> {
    Array3::from_shape_fn((batch, seq, features), |(b, t, f)| {
        ((b * 7 + t * 3 + f * 5) as f32 * 0.37).sin()
    })
}

fn relative_error(analytic: f32, numerical: f32) -> f32 {
    let scale = (analytic.abs() + numerical.abs()).max(1e-3);
    (analytic - numerical).abs() / scale
}

fn agrees(analytic: f32, numerical: f32) -> bool {
    (analytic - numerical).abs() < ABS_FLOOR || relative_error(analytic, numerical) < TOLERANCE
}

fn squared_loss(output: &Array3<f32>, target: &Array3<f32>) -> f32 {
    0.5 * (output - target).mapv(|v| v * v).sum()
}

fn lstm_loss(layer: &mut LSTMLayer, input: ArrayView3<f32>, target: &Array3<f32>) -> f32 {
    layer.reset_states();
    let output = layer.forward_sequence(input);
    squared_loss(&output, target)
}

fn gru_loss(layer: &mut GRULayer, input: ArrayView3<f32>, target: &Array3<f32>) -> f32 {
    layer.reset_state();
    let output = layer.forward_sequence(input);
    squared_loss(&output, target)
}

/// Checks every weight matrix and bias vector of an LSTM against finite differences.
fn check_lstm(return_sequences: bool) {
    let (batch, seq, input_size, hidden) = (2, 4, 3, 3);
    let mut layer = LSTMLayer::new(input_size, hidden, return_sequences);
    let input = deterministic_seq(batch, seq, input_size);

    layer.reset_states();
    let output = layer.forward_sequence(input.view());
    let target = Array3::from_shape_fn(output.raw_dim(), |(b, t, h)| {
        ((b + t * 2 + h * 3) as f32 * 0.21).cos() * 0.5
    });
    let output_grad = &output - &target;
    let grads = layer.backward_sequence(output_grad.view());

    type Weight = fn(&mut LSTMLayer) -> &mut ndarray::Array2<f32>;
    type WeightGrad = fn(&LSTMGradients) -> &ndarray::Array2<f32>;
    let weights: [(&str, Weight, WeightGrad); 8] = [
        ("w_ii", |l| &mut l.w_ii, |g| &g.dw_ii),
        ("w_hi", |l| &mut l.w_hi, |g| &g.dw_hi),
        ("w_if", |l| &mut l.w_if, |g| &g.dw_if),
        ("w_hf", |l| &mut l.w_hf, |g| &g.dw_hf),
        ("w_ig", |l| &mut l.w_ig, |g| &g.dw_ig),
        ("w_hg", |l| &mut l.w_hg, |g| &g.dw_hg),
        ("w_io", |l| &mut l.w_io, |g| &g.dw_io),
        ("w_ho", |l| &mut l.w_ho, |g| &g.dw_ho),
    ];

    for (name, field, grad_field) in weights {
        let analytic_all = grad_field(&grads).clone();
        for (i, j) in [(0usize, 0usize), (1, 2), (2, 1)] {
            let original = field(&mut layer)[[i, j]];

            field(&mut layer)[[i, j]] = original + EPS;
            let plus = lstm_loss(&mut layer, input.view(), &target);
            field(&mut layer)[[i, j]] = original - EPS;
            let minus = lstm_loss(&mut layer, input.view(), &target);
            field(&mut layer)[[i, j]] = original;

            let numerical = (plus - minus) / (2.0 * EPS);
            let analytic = analytic_all[[i, j]];
            let error = relative_error(analytic, numerical);
            assert!(
                agrees(analytic, numerical),
                "LSTM {} [{}, {}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
                name, i, j, return_sequences, analytic, numerical, error
            );
        }
    }

    type Bias = fn(&mut LSTMLayer) -> &mut ndarray::Array1<f32>;
    type BiasGrad = fn(&LSTMGradients) -> &ndarray::Array1<f32>;
    let biases: [(&str, Bias, BiasGrad); 4] = [
        ("b_i", |l| &mut l.b_i, |g| &g.db_i),
        ("b_f", |l| &mut l.b_f, |g| &g.db_f),
        ("b_g", |l| &mut l.b_g, |g| &g.db_g),
        ("b_o", |l| &mut l.b_o, |g| &g.db_o),
    ];

    for (name, field, grad_field) in biases {
        let analytic_all = grad_field(&grads).clone();
        for i in 0..hidden {
            let original = field(&mut layer)[i];

            field(&mut layer)[i] = original + EPS;
            let plus = lstm_loss(&mut layer, input.view(), &target);
            field(&mut layer)[i] = original - EPS;
            let minus = lstm_loss(&mut layer, input.view(), &target);
            field(&mut layer)[i] = original;

            let numerical = (plus - minus) / (2.0 * EPS);
            let analytic = analytic_all[i];
            let error = relative_error(analytic, numerical);
            assert!(
                agrees(analytic, numerical),
                "LSTM {} [{}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
                name, i, return_sequences, analytic, numerical, error
            );
        }
    }

    // Gradient with respect to the inputs, which is what a preceding layer receives
    for (b, t, f) in [(0usize, 0usize, 0usize), (1, 2, 1), (0, 3, 2)] {
        let mut perturbed = input.clone();
        perturbed[[b, t, f]] = input[[b, t, f]] + EPS;
        let plus = lstm_loss(&mut layer, perturbed.view(), &target);
        perturbed[[b, t, f]] = input[[b, t, f]] - EPS;
        let minus = lstm_loss(&mut layer, perturbed.view(), &target);

        let numerical = (plus - minus) / (2.0 * EPS);
        let analytic = grads.dx[[b, t, f]];
        let error = relative_error(analytic, numerical);
        assert!(
            agrees(analytic, numerical),
            "LSTM dx [{}, {}, {}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
            b, t, f, return_sequences, analytic, numerical, error
        );
    }
}

/// Checks every weight matrix and bias vector of a GRU against finite differences.
fn check_gru(return_sequences: bool) {
    let (batch, seq, input_size, hidden) = (2, 4, 3, 3);
    let mut layer = GRULayer::new(input_size, hidden, return_sequences);
    let input = deterministic_seq(batch, seq, input_size);

    layer.reset_state();
    let output = layer.forward_sequence(input.view());
    let target = Array3::from_shape_fn(output.raw_dim(), |(b, t, h)| {
        ((b + t * 2 + h * 3) as f32 * 0.21).cos() * 0.5
    });
    let output_grad = &output - &target;
    let grads = layer.backward_sequence(output_grad.view());

    type Weight = fn(&mut GRULayer) -> &mut ndarray::Array2<f32>;
    type WeightGrad = fn(&GRUGradients) -> &ndarray::Array2<f32>;
    let weights: [(&str, Weight, WeightGrad); 6] = [
        ("w_ir", |l| &mut l.w_ir, |g| &g.dw_ir),
        ("w_hr", |l| &mut l.w_hr, |g| &g.dw_hr),
        ("w_iz", |l| &mut l.w_iz, |g| &g.dw_iz),
        ("w_hz", |l| &mut l.w_hz, |g| &g.dw_hz),
        ("w_in", |l| &mut l.w_in, |g| &g.dw_in),
        ("w_hn", |l| &mut l.w_hn, |g| &g.dw_hn),
    ];

    for (name, field, grad_field) in weights {
        let analytic_all = grad_field(&grads).clone();
        for (i, j) in [(0usize, 0usize), (1, 2), (2, 1)] {
            let original = field(&mut layer)[[i, j]];

            field(&mut layer)[[i, j]] = original + EPS;
            let plus = gru_loss(&mut layer, input.view(), &target);
            field(&mut layer)[[i, j]] = original - EPS;
            let minus = gru_loss(&mut layer, input.view(), &target);
            field(&mut layer)[[i, j]] = original;

            let numerical = (plus - minus) / (2.0 * EPS);
            let analytic = analytic_all[[i, j]];
            let error = relative_error(analytic, numerical);
            assert!(
                agrees(analytic, numerical),
                "GRU {} [{}, {}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
                name, i, j, return_sequences, analytic, numerical, error
            );
        }
    }

    type Bias = fn(&mut GRULayer) -> &mut ndarray::Array1<f32>;
    type BiasGrad = fn(&GRUGradients) -> &ndarray::Array1<f32>;
    let biases: [(&str, Bias, BiasGrad); 3] = [
        ("b_r", |l| &mut l.b_r, |g| &g.db_r),
        ("b_z", |l| &mut l.b_z, |g| &g.db_z),
        ("b_n", |l| &mut l.b_n, |g| &g.db_n),
    ];

    for (name, field, grad_field) in biases {
        let analytic_all = grad_field(&grads).clone();
        for i in 0..hidden {
            let original = field(&mut layer)[i];

            field(&mut layer)[i] = original + EPS;
            let plus = gru_loss(&mut layer, input.view(), &target);
            field(&mut layer)[i] = original - EPS;
            let minus = gru_loss(&mut layer, input.view(), &target);
            field(&mut layer)[i] = original;

            let numerical = (plus - minus) / (2.0 * EPS);
            let analytic = analytic_all[i];
            let error = relative_error(analytic, numerical);
            assert!(
                agrees(analytic, numerical),
                "GRU {} [{}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
                name, i, return_sequences, analytic, numerical, error
            );
        }
    }

    for (b, t, f) in [(0usize, 0usize, 0usize), (1, 2, 1), (0, 3, 2)] {
        let mut perturbed = input.clone();
        perturbed[[b, t, f]] = input[[b, t, f]] + EPS;
        let plus = gru_loss(&mut layer, perturbed.view(), &target);
        perturbed[[b, t, f]] = input[[b, t, f]] - EPS;
        let minus = gru_loss(&mut layer, perturbed.view(), &target);

        let numerical = (plus - minus) / (2.0 * EPS);
        let analytic = grads.dx[[b, t, f]];
        let error = relative_error(analytic, numerical);
        assert!(
            agrees(analytic, numerical),
            "GRU dx [{}, {}, {}] (return_sequences={}): analytic {} vs numerical {}, relative error {}",
            b, t, f, return_sequences, analytic, numerical, error
        );
    }
}

#[test]
fn lstm_backward_matches_finite_differences_over_sequences() {
    check_lstm(true);
}

#[test]
fn lstm_backward_matches_finite_differences_last_step_only() {
    check_lstm(false);
}

#[test]
fn gru_backward_matches_finite_differences_over_sequences() {
    check_gru(true);
}

#[test]
fn gru_backward_matches_finite_differences_last_step_only() {
    check_gru(false);
}

/// Sequences of one feature per step, target is the mean of the sequence, which
/// cannot be read off any single step.
fn mean_task(count: usize, seq: usize) -> (Array3<f32>, ndarray::Array2<f32>) {
    let inputs = Array3::from_shape_fn((count, seq, 1), |(n, t, _)| {
        ((n * 5 + t * 11) as f32 * 0.29).sin()
    });
    let targets = ndarray::Array2::from_shape_fn((count, 1), |(n, _)| {
        (0..seq).map(|t| inputs[[n, t, 0]]).sum::<f32>() / seq as f32
    });
    (inputs, targets)
}

fn train_mean_task(cell: crate::recurrent::RecurrentCell) -> (f32, f32) {
    use crate::activations::Activation;
    use crate::optimizer::{Adam, OptimizerWrapper};
    use crate::recurrent::RecurrentNetwork;

    let hidden = cell.hidden_size();
    let mut model = RecurrentNetwork::new(
        cell,
        &[hidden, 1],
        &[Activation::Linear],
        OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8)),
    );

    let (inputs, targets) = mean_task(16, 5);
    let initial = model.loss(inputs.view(), targets.view());
    for _ in 0..400 {
        model.train_batch(inputs.view(), targets.view(), 0.02);
    }
    let final_loss = model.loss(inputs.view(), targets.view());
    (initial, final_loss)
}

#[test]
fn recurrent_network_learns_with_an_lstm_cell() {
    let cell = crate::recurrent::RecurrentCell::Lstm(LSTMLayer::new(1, 8, false));
    let (initial, final_loss) = train_mean_task(cell);
    assert!(
        final_loss < initial * 0.2,
        "LSTM cell did not learn: loss went from {} to {}",
        initial, final_loss
    );
}

#[test]
fn recurrent_network_learns_with_a_gru_cell() {
    let cell = crate::recurrent::RecurrentCell::Gru(GRULayer::new(1, 8, false));
    let (initial, final_loss) = train_mean_task(cell);
    assert!(
        final_loss < initial * 0.2,
        "GRU cell did not learn: loss went from {} to {}",
        initial, final_loss
    );
}
